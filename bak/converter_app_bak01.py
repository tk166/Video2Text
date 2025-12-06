import streamlit as st
import tempfile
import os
import sys
import torch
import time
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

# --- 主程序 ---

# 初始化session state
if "transcription_result" not in st.session_state:
    st.session_state.transcription_result = ""

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
            res = model.generate(input=wav_file)
            st.write("✅ 识别推理结束")

            # 保存结果
            st.session_state.transcription_result = res[0]['text']
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

# 结果展示和编辑区域
if st.session_state.is_processed:
    st.divider()
    st.subheader("📝 转录结果")
    
    col_l, col_r = st.columns([3, 1])
    
    with col_l:
        edited_text = st.text_area("编辑文本", value=st.session_state.transcription_result, height=400, label_visibility="collapsed")
        st.session_state.transcription_result = edited_text

    with col_r:
        st.info("操作栏")
        if st.button("💾 保存修改", use_container_width=True):
            st.toast("文本已保存到内存！")
            
        st.download_button(
            label="📥 导出 TXT",
            data=st.session_state.transcription_result,
            file_name="transcription.txt",
            mime="text/plain",
            use_container_width=True
        )