import os
import uuid
import time
import json
import json
import threading
import time
import traceback
import re
import sys
import logging
from flask import Flask, request, jsonify, send_file
import torch
from modelscope.hub.snapshot_download import snapshot_download
from funasr import AutoModel
from audio_downloader import download_audio
from audio_converter import convert_to_wav
from crypto_utils import save_encrypted_cookie, decrypt_data
from srt_utils import generate_smart_srt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("converter_app_remote.log"),
        logging.StreamHandler(sys.stdout)
    ],
    force=True 
)
logger = logging.getLogger(__name__)

# ================= 配置区 =================
# 你用到的三个模型 ID 和版本
MODEL_CONFIG = {
    "asr":  {"id": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", "ver": "v2.0.4"},
    "vad":  {"id": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",  "ver": "v2.0.4"},
    "punc": {"id": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", "ver": "v2.0.4"},
}

# ================= 全局变量 =================
# 存储任务状态和结果
tasks = {}

# ================= 模型加载 =================
def check_and_download_models():
    """检查并下载模型"""
    local_paths = {}
    logger.info("----- 开始检查模型文件 -----")
    try:
        # 遍历三个模型进行检查
        for key, cfg in MODEL_CONFIG.items():
            # snapshot_download 会自动判断本地缓存
            # 如果本地存在，它不会发起网络请求，直接返回路径，速度极快
            path = snapshot_download(model_id=cfg["id"], revision=cfg["ver"])
            local_paths[key] = path
            logger.info(f"✅ {key.upper()} 模型就绪: {path}")
    except Exception as e:
        logger.error(f"模型下载失败，请检查网络或代理设置！\n报错信息: {e}")
        raise e
    return local_paths

def load_funasr_engine(device_select="cuda"):
    """加载FunASR引擎"""
    # 1. 先确保文件都在（引用上面的函数）
    paths = check_and_download_models()

    # 2. 初始化重型对象
    logger.info("🚀 正在初始化 FunASR AutoModel...")
    model = AutoModel(
        model=paths["asr"],
        model_revision=MODEL_CONFIG["asr"]["ver"],

        vad_model=paths["vad"],
        vad_model_revision=MODEL_CONFIG["vad"]["ver"],

        punc_model=paths["punc"],
        punc_model_revision=MODEL_CONFIG["punc"]["ver"],

        device=device_select,
        num_workers=0, # 避免多线程报错
    )
    logger.info("🎉 模型初始化完毕！")
    return model

# ================= 初始化 =================
# 检测设备
if torch.cuda.is_available():
    device_select = "cuda"
# elif torch.backends.mps.is_available(): # 实测Apple M4的mps稳定性不太行所以先注掉了
#     device_select = "mps"
else:
    device_select = "cpu"

# 加载模型（服务启动时一次性加载）
logger.info(f"⚙️ 检测到计算设备: {device_select}")
try:
    model_instance = load_funasr_engine(device_select)
    logger.info("✅ 模型加载成功")
except Exception as e:
    logger.error(f"❌ 模型加载失败: {e}")
    model_instance = None

# ================= 工具函数 =================
def clean_url(url):
    """清理URL"""
    # 如果是 Bilibili (包含 'bilibili')
    if "bilibili" in url:
        match = re.search(r'(BV[a-zA-Z0-9]+)', url)
        if match:
            return f"https://www.bilibili.com/video/{match.group(1)}"
    # 如果是 YouTube，通常不需要去参数，或者只去除无关参数 (yt-dlp 通常能自动处理)
    # 但为了保险，可以去掉 & 及其后面的内容 (YouTube ID 在 ?v= 之后，不能切 ?)
    if "youtube" in url or "youtu.be" in url:
        return url.split('&')[0]
    return url.split('?')[0]

# ================= 核心处理函数 =================
def process_video(task_id, video_url, cookie_file=None, encrypted_cookie_data=None):
    """处理视频的核心函数"""
    global tasks

    # 如果提供了加密的cookie数据，将其保存为临时文件
    temp_cookie_file = None
    cookie_status = 0
    if encrypted_cookie_data:
        try:
            cookie_file = save_encrypted_cookie(encrypted_cookie_data)
            temp_cookie_file = cookie_file
            logger.info(f"[{task_id}] 加密cookie数据已解密并保存为临时文件: {cookie_file}")
            cookie_status = 1
        except Exception as e:
            logger.error(f"[{task_id}] 解密cookie数据失败: {e}")
            cookie_status = 2
            # 继续处理，但不使用cookie

    # 获取是否需要保留音频
    keep_audio = tasks[task_id].get("keep_audio", False)
    audio_file_path = None

    try:
        logger.info(f"[{task_id}] 开始处理视频: {video_url}")

        # 更新任务状态
        tasks[task_id]["query_count"] = 0
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["progress"] = "开始处理..."

        # 步骤1: 下载音频
        tasks[task_id]["progress"] = "正在下载音频 (yt-dlp)..."
        logger.info(f"[{task_id}] 正在下载音频...")
        audio_file, video_title, uploader = download_audio(video_url, cookie_file, extra_info=True)
        tasks[task_id]["title"] = video_title
        tasks[task_id]["uploader"] = uploader
        tasks[task_id]["progress"] = f"✅ 下载完成: {os.path.basename(audio_file)}"
        logger.info(f"[{task_id}] 音频下载完成: {audio_file}")

        # 如果需要保留音频，复制到永久位置
        if keep_audio:
            try:
                import shutil
                # 创建音频存储目录
                audio_dir = os.path.join(os.path.dirname(__file__), "audio_files")
                if not os.path.exists(audio_dir):
                    os.makedirs(audio_dir)

                # 生成唯一文件名
                filename = f"{task_id}_{os.path.basename(audio_file)}"
                audio_file_path = os.path.join(audio_dir, filename)

                # 复制文件
                shutil.copy2(audio_file, audio_file_path)
                tasks[task_id]["audio_file_path"] = audio_file_path
                logger.info(f"[{task_id}] 音频文件已保存: {audio_file_path}")
            except Exception as e:
                logger.error(f"[{task_id}] 保存音频文件失败: {e}")

        # 步骤2: 转换音频格式
        tasks[task_id]["progress"] = "正在转换音频格式 (ffmpeg)..."
        logger.info(f"[{task_id}] 正在转换音频格式...")
        wav_file = convert_to_wav(audio_file)
        tasks[task_id]["progress"] = f"✅ 格式转换完成: {os.path.basename(wav_file)}"
        logger.info(f"[{task_id}] 音频格式转换完成: {wav_file}")

        # 步骤3: 执行语音识别
        tasks[task_id]["progress"] = "正在进行语音识别 (Inference)..."
        logger.info(f"[{task_id}] 正在执行语音识别...")
        res = model_instance.generate(input=wav_file, return_sentence_timestamp=True)
        logger.info(f"[{task_id}] 语音识别结果结构: {type(res)}")
        if res:
            logger.info(f"[{task_id}] 语音识别结果长度: {len(res)}")
            if len(res) > 0:
                logger.info(f"[{task_id}] 第一个结果的键: {list(res[0].keys())}")
        tasks[task_id]["progress"] = "✅ 识别推理结束"
        logger.info(f"[{task_id}] 语音识别完成")

        # 步骤4: 生成结果
        logger.info(f"[{task_id}] 正在生成结果...")
        transcription_result = res[0]['text']
        srt_result = generate_smart_srt(res)

        # 提取时间戳数据
        timestamp_data = []
        if res and len(res) > 0 and 'timestamp' in res[0]:
            timestamp_data = res[0]['timestamp']
            logger.info(f"[{task_id}] 时间戳数据提取完成，共 {len(timestamp_data)} 个时间点")
        else:
            logger.warning(f"[{task_id}] 未找到时间戳数据")

        # 步骤5: 清理临时文件
        tasks[task_id]["progress"] = "🧹 清理临时文件..."
        logger.info(f"[{task_id}] 正在清理临时文件...")
        try:
            os.remove(wav_file)
            # 如果不需要保留音频且不是要保存的音频文件，则删除
            if not keep_audio or audio_file != audio_file_path:
                os.remove(audio_file)
            # 如果创建了临时cookie文件，也要清理
            if temp_cookie_file and os.path.exists(temp_cookie_file):
                os.remove(temp_cookie_file)
                logger.info(f"[{task_id}] 临时cookie文件已清理: {temp_cookie_file}")
            logger.info(f"[{task_id}] 临时文件清理完成")
        except Exception as e:
            logger.warning(f"[{task_id}] 清理文件警告: {e}")

        # 步骤6: 保存结果
        tasks[task_id]["result"] = {
            "transcription": transcription_result,
            "srt": srt_result,
            "raw": res,
            "timestamp": timestamp_data,
            "cookie_status": cookie_status
        }
        # 如果保留了音频，添加音频URL
        if keep_audio and audio_file_path:
            tasks[task_id]["result"]["audio_url"] = f"/api/audio/{task_id}"

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["progress"] = "🎉 处理全部完成！"
        logger.info(f"[{task_id}] 处理全部完成！")

    except Exception as e:
        error_msg = f"处理失败: {str(e)}"
        error_traceback = traceback.format_exc()
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = error_msg
        tasks[task_id]["progress"] = f"❌ 处理失败: {str(e)}"
        logger.error(f"[{task_id}] {error_msg}\n{error_traceback}")
        # 清理可能创建的临时cookie文件
        if temp_cookie_file and os.path.exists(temp_cookie_file):
            try:
                os.remove(temp_cookie_file)
                logger.info(f"[{task_id}] 异常处理时临时cookie文件已清理: {temp_cookie_file}")
            except Exception as cleanup_error:
                logger.warning(f"[{task_id}] 清理临时cookie文件失败: {cleanup_error}")

# ================= Flask 应用 =================
app = Flask(__name__)


@app.route('/api/process', methods=['POST'])
def start_processing():
    """开始处理视频"""
    global tasks

    try:
        if False:
            # === 记录完整的HTTP请求信息用于调试 ===

            # 生成唯一的请求ID
            request_id = str(time.time()).replace('.', '_')
            debug_dir = "/tmp/request_debug"
            os.makedirs(debug_dir, exist_ok=True)
            debug_file = f"{debug_dir}/request_{request_id}.txt"

            # 记录完整的请求信息
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(f"=== HTTP请求调试信息 ===\n")
                f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"请求ID: {request_id}\n")
                f.write(f"远程地址: {request.remote_addr}\n")
                f.write(f"请求方法: {request.method}\n")
                f.write(f"请求URL: {request.url}\n")
                f.write(f"请求路径: {request.path}\n")
                f.write(f"请求查询字符串: {request.query_string.decode() if request.query_string else ''}\n")
                f.write(f"\n--- 请求头 ---\n")
                for key, value in request.headers:
                    f.write(f"{key}: {value}\n")
                f.write(f"\n--- 请求体 ---\n")
                f.write(f"原始数据长度: {len(request.data) if request.data else 0}\n")
                f.write(f"原始数据: {request.data.decode('utf-8', errors='ignore') if request.data else ''}\n")
                f.write(f"\n--- 解析后的JSON ---\n")
                try:
                    data = request.get_json(force=True)
                    f.write(f"JSON数据: {json.dumps(data, indent=2, ensure_ascii=False) if data else 'None'}\n")
                except Exception as e:
                    f.write(f"JSON解析错误: {str(e)}\n")
                f.write(f"\n--- 其他信息 ---\n")
                f.write(f"Content-Type: {request.content_type}\n")
                f.write(f"Content-Length: {request.content_length}\n")
                f.write("=" * 50 + "\n")

            logger.info(f"请求调试信息已保存到: {debug_file}")
            # === 调试信息记录结束 ===

        # 获取请求数据
        data = request.get_json(force=True)  # 强制解析JSON，即使Content-Type不正确
        if data is None:
            logger.warning("请求数据不是有效的JSON格式")
            return jsonify({"error": "请求数据不是有效的JSON格式"}), 400

        video_url = data.get('url')
        cookie_file = data.get('cookie_file', None)  # 可选的cookie文件路径
        encrypted_cookie_data = data.get('encrypted_cookie_data', None)  # 可选的加密cookie数据
        keep_audio = data.get('keep_audio', False)  # 是否保留音频文件，默认不保留

        if not video_url:
            logger.warning("请求缺少视频URL")
            return jsonify({"error": "缺少视频URL"}), 400

        # 清理URL
        video_url = clean_url(video_url)
        logger.info(f"收到处理请求: {video_url}")

        # 生成任务ID
        task_id = str(uuid.uuid4())

        # 创建任务记录
        tasks[task_id] = {
            "status": "queued",
            "progress": "任务已加入队列...",
            "result": None,
            "error": None,
            "created_at": time.time(),
            "keep_audio": keep_audio,  # 记录是否需要保留音频
            "audio_file_path": None  # 音频文件路径
        }

        # 在后台线程中处理任务
        thread = threading.Thread(
            target=process_video,
            args=(task_id, video_url, cookie_file, encrypted_cookie_data)
        )
        thread.daemon = True
        thread.start()

        # 返回任务ID
        logger.info(f"任务已启动，ID: {task_id}")
        return jsonify({
            "task_id": task_id,
            "message": "任务已启动，请使用任务ID查询处理状态"
        }), 202

    except Exception as e:
        logger.error(f"启动任务失败: {str(e)}")
        return jsonify({"error": f"启动任务失败: {str(e)}"}), 500

@app.route('/api/status/<task_id>', methods=['GET'])
def get_status(task_id):
    """获取任务状态"""
    global tasks

    if task_id not in tasks:
        logger.warning(f"查询不存在的任务: {task_id}")
        return jsonify({"error": "任务不存在"}), 404

    task = tasks[task_id]
    logger.info(f"查询任务状态: {task_id}, 状态: {task['status']}")
    if "query_count" not in task:
        task["query_count"] = 0
    else:
        task["query_count"] += 1
    task_query = task["query_count"]
    task_progress = task["progress"]

    response = {
        "task_id": task_id,
        "status": task["status"],
        "progress": f"[{task_query:5d}] {task_progress}"
    }

    if task["status"] == "completed":
        response["result"] = {
            "title": task.get("title", "未知标题"),
            "uploader": task.get("uploader", "未知作者"),
            "transcription": task["result"]["transcription"],
            "srt": task["result"]["srt"],
            "timestamp": task["result"]["timestamp"],
            "cookie_status": task["result"]["cookie_status"],
        }
        # 如果有音频URL，也返回
        if "audio_url" in task["result"]:
            response["result"]["audio_url"] = task["result"]["audio_url"]
    elif task["status"] == "failed":
        response["error"] = task["error"]

    return jsonify(response)

@app.route('/api/audio/<task_id>')
def download_audio_file(task_id):
    """下载音频文件"""
    global tasks

    if task_id not in tasks:
        logger.warning(f"请求不存在的任务音频: {task_id}")
        return jsonify({"error": "任务不存在"}), 404

    task = tasks[task_id]

    # 检查任务是否完成
    if task["status"] != "completed":
        logger.warning(f"请求未完成任务的音频: {task_id}")
        return jsonify({"error": "任务尚未完成"}), 400

    # 检查是否有音频文件路径
    audio_file_path = task.get("audio_file_path")
    if not audio_file_path or not os.path.exists(audio_file_path):
        logger.warning(f"任务无音频文件或文件不存在: {task_id}")
        return jsonify({"error": "音频文件不存在"}), 404

    try:
        # 确定文件MIME类型
        import mimetypes
        mime_type, _ = mimetypes.guess_type(audio_file_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'

        # 返回文件
        logger.info(f"提供音频文件下载: {audio_file_path}")
        return send_file(audio_file_path, mimetype=mime_type)
    except Exception as e:
        logger.error(f"提供音频文件失败: {e}")
        return jsonify({"error": "无法提供音频文件"}), 500

@app.route('/api/audio/<task_id>', methods=['DELETE'])
def delete_audio_file(task_id):
    """删除指定任务的音频文件"""
    global tasks

    if task_id not in tasks:
        logger.warning(f"尝试删除不存在的任务音频: {task_id}")
        return jsonify({"error": "任务不存在"}), 404

    task = tasks[task_id]

    # 检查任务是否完成
    if task["status"] != "completed":
        logger.warning(f"尝试删除未完成任务的音频: {task_id}")
        return jsonify({"error": "任务尚未完成"}), 400

    # 检查是否有音频文件路径
    audio_file_path = task.get("audio_file_path")
    if not audio_file_path or not os.path.exists(audio_file_path):
        logger.warning(f"任务无音频文件或文件不存在: {task_id}")
        return jsonify({"error": "音频文件不存在"}), 404

    try:
        # 删除音频文件
        os.remove(audio_file_path)
        # 清除任务记录中的音频文件路径
        task["audio_file_path"] = None
        if "result" in task and "audio_url" in task["result"]:
            del task["result"]["audio_url"]

        logger.info(f"音频文件已删除: {audio_file_path}")
        return jsonify({"message": "音频文件删除成功"}), 200
    except Exception as e:
        logger.error(f"删除音频文件失败: {e}")
        return jsonify({"error": "无法删除音频文件"}), 500

@app.route('/api/tasks', methods=['GET'])
def list_tasks():
    """列出所有任务"""
    logger.info("列出所有任务")
    task_list = []
    for task_id, task in tasks.items():
        task_info = {
            "task_id": task_id,
            "status": task["status"],
            "progress": task["progress"]
        }
        # 如果有音频文件，标记一下
        if task.get("audio_file_path") and os.path.exists(task.get("audio_file_path")):
            task_info["has_audio"] = True
        else:
            task_info["has_audio"] = False
        task_list.append(task_info)
    return jsonify({"tasks": task_list})

@app.route('/api/cleanup', methods=['POST'])
def cleanup_expired_files():
    """清理过期的音频文件"""
    try:
        # 从请求中获取过期时间参数，默认24小时
        data = request.get_json() or {}
        max_age_hours = data.get('max_age_hours', 24)

        deleted_count = cleanup_expired_audio_files_internal(max_age_hours)

        logger.info(f"清理完成，删除了 {deleted_count} 个过期文件")
        return jsonify({
            "message": f"清理完成，删除了 {deleted_count} 个过期文件",
            "deleted_count": deleted_count
        }), 200
    except Exception as e:
        logger.error(f"清理过期文件失败: {e}")
        return jsonify({"error": "清理过期文件失败"}), 500

def cleanup_expired_audio_files_internal(max_age_hours=24):
    """内部函数：清理过期的音频文件"""
    deleted_count = 0
    try:
        audio_dir = os.path.join(os.path.dirname(__file__), "audio_files")
        if not os.path.exists(audio_dir):
            return deleted_count

        import datetime
        now = datetime.datetime.now()
        cutoff_time = now - datetime.timedelta(hours=max_age_hours)

        for filename in os.listdir(audio_dir):
            file_path = os.path.join(audio_dir, filename)
            if os.path.isfile(file_path):
                # 获取文件修改时间
                mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                if mod_time < cutoff_time:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        logger.info(f"已删除过期音频文件: {file_path}")
                    except Exception as e:
                        logger.warning(f"删除过期音频文件失败: {file_path}, {e}")

        return deleted_count
    except Exception as e:
        logger.error(f"清理过期音频文件时出错: {e}")
        return deleted_count

# 启动时清理一次
cleanup_expired_audio_files_internal()

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    logger.info("健康检查请求")
    return jsonify({
        "status": "healthy",
        "device": device_select,
        "model_loaded": model_instance is not None
    })

if __name__ == '__main__':
    logger.info("🚀 启动 Audio2Text 远程服务...")
    logger.info("📝 API 文档:")
    logger.info("   POST /api/process - 开始处理视频")
    logger.info("   GET /api/status/<task_id> - 查询任务状态")
    logger.info("   GET /api/tasks - 列出所有任务")
    logger.info("   GET /api/health - 健康检查")
    logger.info("📭 服务运行在 http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)