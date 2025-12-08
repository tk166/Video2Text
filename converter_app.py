import streamlit as st
import tempfile
import os
import sys
import torch
import time
import re

from audio_downloader import download_audio
from audio_converter import convert_to_wav
from funasr import AutoModel
from modelscope.hub.snapshot_download import snapshot_download
# ================= 配置区 =================
# 你用到的三个模型 ID 和版本
MODEL_CONFIG = {
    "asr":  {"id": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", "ver": "v2.0.4"},
    "vad":  {"id": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",  "ver": "v2.0.4"},
    "punc": {"id": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", "ver": "v2.0.4"},
}
# ================= 预下载/检查 =================
@st.cache_data(show_spinner="正在检查本地模型完整性...")
def check_and_download_models():
    local_paths = {}
    print("----- 开始检查模型文件 -----")
    try:
        # 遍历三个模型进行检查
        for key, cfg in MODEL_CONFIG.items():
            # snapshot_download 会自动判断本地缓存
            # 如果本地存在，它不会发起网络请求，直接返回路径，速度极快
            path = snapshot_download(model_id=cfg["id"], revision=cfg["ver"])
            local_paths[key] = path
            print(f"✅ {key.upper()} 模型就绪: {path}")
            
    except Exception as e:
        st.error(f"模型下载失败，请检查网络或代理设置！\n报错信息: {e}")
        st.stop() # 停止运行后续代码
        
    return local_paths
# ================= 加载进显存（防卡顿核心） =================
@st.cache_resource(show_spinner="正在加载神经网络到显存 (只加载一次)...")
def load_funasr_engine(device_select="cuda"):
    # 1. 先确保文件都在（引用上面的函数）
    paths = check_and_download_models()
    
    # 2. 初始化重型对象
    print("🚀 正在初始化 FunASR AutoModel...")
    model = AutoModel(
        model=paths["asr"],
        model_revision=MODEL_CONFIG["asr"]["ver"],
        
        vad_model=paths["vad"],
        vad_model_revision=MODEL_CONFIG["vad"]["ver"],
        
        punc_model=paths["punc"],
        punc_model_revision=MODEL_CONFIG["punc"]["ver"],
        
        device=device_select,
        num_workers=0, # 避免 Streamlit 多线程报错
    )
    print("🎉 模型初始化完毕！")
    return model

if torch.cuda.is_available():
    device_select = "cuda"
# elif torch.backends.mps.is_available(): # 实测Apple M4的mps稳定性不太行所以先注掉了
#     device_select = "mps"
else:
    device_select = "cpu"
model_instance = load_funasr_engine(device_select)

# --- 核心组件：日志重定向类 ---
class StreamlitLogger:
    def __init__(self, log_container):
        self.log_container = log_container
        self.log_buffer = []
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # 这是一个能匹配几乎所有 ANSI 转义序列的正则表达式
        # 它能识别颜色 (\x1b[34m) 和光标移动 (\x1b[A) 等
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, message):
        # 1. 仍然输出到后台终端 (保留原始带颜色的格式，方便你在 VSCode 里看)
        self.original_stdout.write(message)
        
        # 2. 清洗数据给 Streamlit 显示
        
        # 第一步：去除 ANSI 颜色和控制符
        clean_message = self.ansi_escape.sub('', message)
        
        # 第二步：处理回车符 \r
        # tqdm 喜欢用 \r 回到行首覆盖进度。在网页上我们把它变成换行 \n，
        # 这样进度条就会变成瀑布流（一行行显示），而不是挤在一起。
        clean_message = clean_message.replace('\r', '\n')
        
        # 第三步：去除一些可能残留的 weird artifact (比如 [A 如果是纯文本形式出现)
        # 有时候 tqdm 的 cursor up 会留下显式的 [A
        clean_message = clean_message.replace('[A', '')

        if clean_message.strip():
            self.log_buffer.append(clean_message)
            
            # --- 性能优化 ---
            # 只保留最后 20 行日志，避免网页越来越卡
            if len(self.log_buffer) > 20:
                self.log_buffer = self.log_buffer[-20:]
            
            # 显示清洗后的日志
            self.log_container.code("".join(self.log_buffer), language="text")

    def flush(self):
        self.original_stdout.flush()
        self.original_stderr.flush()

def format_time(ms):
    """毫秒转SRT时间格式"""
    seconds = ms / 1000
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int(ms % 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def generate_smart_srt(inference_result, min_length=10):
    """
    智能SRT生成：
    - 硬标点 (。？！)：强制换行
    - 软标点 (，、)：只有当前句长度超过 min_length 时才换行，否则合并
    """
    # 1. 提取数据
    data = inference_result[0] if isinstance(inference_result, list) else inference_result
    text = data.get('text', '')
    ts_list = data.get('timestamp', [])
    
    # 2. 定义标点集合
    # 硬断句：句号、问号、感叹号、分号
    hard_break_chars = set("。？！；：?!;:\n")
    # 软断句：逗号、顿号、空格
    soft_break_chars = set("，、, ")
    
    srt_content = ""
    sentence_idx = 1
    ts_index = 0  # 时间戳指针
    
    # 当前行的状态缓存
    curr_text = ""
    curr_start = -1
    curr_end = 0
    
    for char in text:
        # --- A. 处理时间戳 (如果是有效文字) ---
        is_punctuation = char in hard_break_chars or char in soft_break_chars or char.isspace()
        
        if not is_punctuation:
            if ts_index < len(ts_list):
                start, end = ts_list[ts_index]
                # 如果是当前行的第一个字
                if curr_start == -1:
                    curr_start = start
                # 更新当前行的结束时间
                curr_end = end
                ts_index += 1
        
        # --- B. 拼接字符 ---
        curr_text += char
        
        # --- C. 判断是否断句 ---
        should_flush = False
        
        # C1. 硬断句：遇到句号，必须断
        if char in hard_break_chars:
            should_flush = True
            
        # C2. 软断句：遇到逗号，看字数够不够
        elif char in soft_break_chars:
            # 只有当当前句长度 >= 设定的最小长度时，才断开
            # 否则就忽略这个逗号，继续往后拼
            if len(curr_text) >= min_length:
                should_flush = True
        
        # --- D. 执行断句 ---
        if should_flush and curr_text.strip():
            # 防御：万一全是标点或没时间戳
            if curr_start == -1: 
                curr_start = curr_end # 兜底
                
            srt_content += f"{sentence_idx}\n"
            srt_content += f"{format_time(curr_start)} --> {format_time(curr_end)}\n"
            srt_content += f"{curr_text.strip()}\n\n" # strip去掉首尾空格
            
            sentence_idx += 1
            # 重置状态
            curr_text = ""
            curr_start = -1
            
    # --- E. 处理残留文本 ---
    if curr_text.strip():
        if curr_start == -1: curr_start = curr_end
        srt_content += f"{sentence_idx}\n"
        srt_content += f"{format_time(curr_start)} --> {format_time(curr_end)}\n"
        srt_content += f"{curr_text.strip()}\n\n"
        
    return srt_content

def update_srt_by_slider():
    """
    当滑动条变化时触发此函数：
    1. 获取滑动条的新值
    2. 重新计算 SRT
    3. 强制覆盖 text_area 的状态
    """
    # 获取滑动条当前的值 (通过 key 获取)
    min_len = st.session_state.srt_min_len_slider
    
    if "raw_res" in st.session_state:
        # 重新生成内容
        new_content = generate_smart_srt(st.session_state.raw_res, min_length=min_len)
        
        # 💥 关键点：直接修改 session_state 中 text_area 对应的 key
        # 这会强制 Streamlit 在下一次渲染时使用这个新值
        st.session_state.editor_srt = new_content
        st.session_state.srt_result = new_content
# --- 主程序 ---

# 初始化session state
if "transcription_result" not in st.session_state:
    st.session_state.transcription_result = ""
if "srt_result" not in st.session_state:
    st.session_state.srt_result = ""
if "is_processed" not in st.session_state:
    st.session_state.is_processed = False

st.set_page_config(page_title="Video2Text", page_icon="🎧")
st.title("🎧 Video2Text - 语音识别工具")
st.markdown("将YouTube/Bilibili视频转换为文字")

# 视频链接输入
video_url = st.text_input("请输入YouTube或Bilibili视频链接:", placeholder="https://www.youtube.com/watch?v=...")

# 处理按钮
if st.button("开始处理", type="primary") and video_url:
    st.session_state.is_processed = False
    
    # 1. 创建一个日志显示区域（默认折叠）
    with st.expander("查看详细运行日志 (Terminal Output)", expanded=True):
        log_placeholder = st.empty()
    
    # 实例化我们的日志捕获器
    logger = StreamlitLogger(log_placeholder)
    
    # 2. 使用 st.status 创建漂亮的进度容器
    with st.status("正在初始化任务...", expanded=True) as status:
        
        # --- 关键：开始劫持 stdout ---
        sys.stdout = logger 
        sys.stderr = logger
        
        try:
            # 步骤1: 下载音频
            status.update(label="正在下载音频 (yt-dlp)...", state="running")
            st.write("🚀 开始调用下载工具...") # 这行字会显示在日志框里
            
            # 注意：如果 download_audio 内部使用了 print，会被捕获。
            # 如果它使用 subprocess 直接输出到系统终端，可能无法被捕获（见下方说明）。
            audio_file = download_audio(video_url)
            st.write(f"✅ 下载完成: {os.path.basename(audio_file)}")
            
            # 步骤2: 转换音频格式
            status.update(label="正在转换音频格式 (ffmpeg)...", state="running")
            wav_file = convert_to_wav(audio_file)
            st.write(f"✅ 格式转换完成: {os.path.basename(wav_file)}")

            # 步骤3: 加载模型
            status.update(label="正在加载 FunASR 模型...", state="running")
            
            st.write(f"⚙️ 检测到计算设备: {device_select}")
            st.write("✅ 模型加载成功")

            # 步骤4: 执行语音识别
            status.update(label="正在进行语音识别 (Inference)...", state="running")
            
            # FunASR 的 generate 内部通常会有进度条打印，这里会被捕获
            res = model_instance.generate(input=wav_file, return_sentence_timestamp=True)
            st.session_state.raw_res = res 
            st.write("✅ 识别推理结束")

            # 保存结果
            st.session_state.transcription_result = res[0]['text']
            try:
                st.session_state.srt_result = generate_smart_srt(res)
                st.write("✅ SRT 字幕生成完成")
            except Exception as e:
                st.write(f"⚠️ SRT生成警告: {e}")
                st.session_state.srt_result = ""
            st.session_state.is_processed = True

            # 清理临时文件
            try:
                st.write("🧹 清理临时文件...")
                os.remove(audio_file)
                os.remove(wav_file)
            except Exception as e:
                st.write(f"⚠️ 清理文件警告: {e}")

            # 更新最终状态
            status.update(label="🎉 处理全部完成！", state="complete", expanded=False)

        except Exception as e:
            status.update(label="❌ 处理失败", state="error")
            st.error(f"发生错误: {str(e)}")
            # 这里的 print 也会显示在日志框里方便调试
            print(f"Error Details: {e}")
            
        finally:
            # --- 关键：务必恢复 stdout，否则后续 Streamlit 可能报错 ---
            sys.stdout = logger.original_stdout

# 结果展示区 (高难度动态切换版)
if st.session_state.is_processed:
    st.divider()
    
    # 1. 顶部控制栏
    col_ctrl_1, col_ctrl_2 = st.columns([1, 3])
    with col_ctrl_1:
        st.subheader("识别结果")
    with col_ctrl_2:
        # 使用 toggle 开关，默认关闭(纯文本模式)
        is_srt_mode = st.toggle("开启 SRT 字幕模式", value=False)

    # 2. 动态逻辑处理
    if is_srt_mode:
        # --- SRT 模式 ---
        # 创建一个更细致的设置栏
        with st.container():
            col_set_1, col_set_2 = st.columns([2, 1])
            with col_set_1:
                st.info("💡 智能断句：逗号会尝试合并，直到达到最小字数；句号强制换行。")
            with col_set_2:
                # 滑动条：控制断句阈值
                # key="srt_min_len" 会自动记录状态
                min_len = st.slider(
                    "⏱️ 最小字幕字数 (逗号合并阈值)", 
                    min_value=8, 
                    max_value=80, 
                    value=15, 
                    step=1,
                    key="srt_min_len_slider", # 必须给个独立的 key
                    on_change=update_srt_by_slider # 绑定回调函数
                )
        
        
        # 第一次进入 SRT 模式时的初始化逻辑
        if "editor_srt" not in st.session_state:
             # 如果还没生成过，先生成一次默认的
             if "raw_res" in st.session_state:
                 st.session_state.editor_srt = generate_smart_srt(st.session_state.raw_res, min_length=15)
             else:
                 st.session_state.editor_srt = ""
        # current_content = st.session_state.editor_srt
        current_label = f"🎬 SRT 字幕 (每行至少 {min_len} 字)"
        current_filename = "subtitle.srt"
        widget_key = "editor_srt" 
    else:
        # --- 纯文本模式 ---
        if "editor_txt" not in st.session_state:
            st.session_state.editor_txt = st.session_state.transcription_result
        # current_content = st.session_state.transcription_result
        current_label = "📄 纯文本编辑"
        current_filename = "transcription.txt"
        widget_key = "editor_txt"

    # 3. 统一的编辑区域
    # 注意：我们将 session_state 的值赋给 value 作为初始值
    # 用户的修改会自动更新到 st.session_state[widget_key] 中
    edited_content = st.text_area(
        label=current_label,
        height=600,
        key=widget_key 
    )

    # 4. 数据同步回写 (这一步很重要)
    # 当用户编辑时，Streamlit 自动更新了 session_state[widget_key]
    # 但我们需要把它同步回我们自定义的 result 变量，以防下次切换时数据丢失
    if is_srt_mode:
        st.session_state.srt_result = edited_content
    else:
        st.session_state.transcription_result = edited_content

    # 5. 底部操作栏
    col_act_1, col_act_2 = st.columns([3, 1])
    
    with col_act_1:
        # 显示当前模式的状态提示
        if is_srt_mode:
            st.caption("ℹ️ 当前为字幕模式，编辑内容将保存为 .srt 格式")
        else:
            st.caption("ℹ️ 当前为纯文本模式，编辑内容将保存为 .txt 格式")
            
    with col_act_2:
        # 下载按钮也是动态的
        st.download_button(
            label=f"📥 导出 {current_filename}",
            data=edited_content,
            file_name=current_filename,
            mime="text/plain",
            type="primary", # 醒目样式
            use_container_width=True
        )
