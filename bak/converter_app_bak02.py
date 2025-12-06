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

# --- 核心组件：日志重定向类 ---
class StreamlitLogger:
    """
    这个类用于捕获 print 输出并实时显示在 Streamlit 的代码框中
    """
    def __init__(self, log_container):
        self.log_container = log_container
        self.log_buffer = []
        # 保存原始的 stdout 以便恢复
        self.original_stdout = sys.stdout

    def write(self, message):
        # 实时打印到终端（保留原始行为）
        self.original_stdout.write(message)
        
        # 过滤掉空的换行，避免刷屏太快视觉效果不好
        if message.strip():
            self.log_buffer.append(message)
            # 为了性能，只显示最后 20 行日志
            display_text = "".join(self.log_buffer[-20:])
            # 实时更新 Streamlit 容器
            self.log_container.code(display_text, language="bash")

    def flush(self):
        self.original_stdout.flush()

def format_time(ms):
    """毫秒转SRT时间格式"""
    seconds = ms / 1000
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int(ms % 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def generate_srt(inference_result):
    """
    根据字级别 timestamp 和带标点的 text 手动生成 SRT
    """
    # 1. 提取数据
    data = inference_result[0] if isinstance(inference_result, list) else inference_result
    text = data.get('text', '')
    ts_list = data.get('timestamp', [])
    
    # 2. 定义断句标点 (遇到这些符号就换行)
    # 包含中文标点和英文标点
    split_chars = set("，。、？！；：,?!;:")
    
    srt_content = ""
    sentence_idx = 1
    
    # 指针
    ts_index = 0  # 对应 timestamp 列表的索引
    
    # 当前句子的状态
    curr_text = ""
    curr_start = -1
    curr_end = 0
    
    for char in text:
        # 处理标点符号
        if char in split_chars or char.isspace():
            curr_text += char
            # 如果碰到了标点，且当前有内容，就作为一句字幕输出
            # (这里为了字幕观感，逗号也换行，如果想句子长一点可以只在句号换行)
            if curr_start != -1:
                srt_content += f"{sentence_idx}\n"
                srt_content += f"{format_time(curr_start)} --> {format_time(curr_end)}\n"
                srt_content += f"{curr_text}\n\n"
                
                sentence_idx += 1
                # 重置当前句状态
                curr_text = ""
                curr_start = -1
            continue
            
        # 处理普通文字
        curr_text += char
        
        # 尝试匹配时间戳
        if ts_index < len(ts_list):
            start, end = ts_list[ts_index]
            
            # 如果是当前句的第一个字，记录开始时间
            if curr_start == -1:
                curr_start = start
            
            # 不断更新结束时间
            curr_end = end
            
            # 移动时间戳指针
            ts_index += 1
            
    # 处理最后可能剩余的一点文本（如果最后没有标点结尾）
    if curr_text and curr_start != -1:
        srt_content += f"{sentence_idx}\n"
        srt_content += f"{format_time(curr_start)} --> {format_time(curr_end)}\n"
        srt_content += f"{curr_text}\n\n"
        
    return srt_content

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
            
            if torch.cuda.is_available():
                device_select = "cuda"
            elif torch.backends.mps.is_available():
                device_select = "mps"
            else:
                device_select = "cpu"
                
            st.write(f"⚙️ 检测到计算设备: {device_select}")
            
            model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                    vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                    punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                    device=device_select,
                    # 注意：设为0或1，多进程可能导致 print 捕获不到
                    num_workers=0, 
                    )
            st.write("✅ 模型加载成功")

            # 步骤4: 执行语音识别
            status.update(label="正在进行语音识别 (Inference)...", state="running")
            
            # FunASR 的 generate 内部通常会有进度条打印，这里会被捕获
            res = model.generate(input=wav_file, return_sentence_timestamp=True)
            st.write("✅ 识别推理结束")

            # 保存结果
            st.session_state.transcription_result = res[0]['text']
            try:
                st.session_state.srt_result = generate_srt(res)
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
        is_srt_mode = st.toggle("开启 SRT 字幕模式 (Subtitle Mode)", value=False)

    # 2. 动态逻辑处理
    if is_srt_mode:
        # --- SRT 模式 ---
        current_content = st.session_state.srt_result
        current_label = "🎬 SRT 字幕编辑 (包含时间轴)"
        current_filename = "subtitle.srt"
        # 关键：使用唯一的 key，让 streamlit 记住这个模式下的编辑内容
        widget_key = "editor_srt" 
    else:
        # --- 纯文本模式 ---
        current_content = st.session_state.transcription_result
        current_label = "📄 纯文本编辑"
        current_filename = "transcription.txt"
        widget_key = "editor_txt"

    # 3. 统一的编辑区域
    # 注意：我们将 session_state 的值赋给 value 作为初始值
    # 用户的修改会自动更新到 st.session_state[widget_key] 中
    edited_content = st.text_area(
        label=current_label,
        value=current_content, 
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