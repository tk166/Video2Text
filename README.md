# Video2Text 📹➡️📝
**一键视频语音识别与字幕转换工具**

Video2Text 支持**本地一体化部署**与**前后端分离**两种模式。基于 FunASR、PyTorch、FFmpeg 构建，集成了 yt-dlp 视频下载与 Streamlit/Flet 图形界面。

## ✨ 功能特性

- **多平台支持**：支持 YouTube 和 Bilibili 等主流视频平台链接。
- **全自动流程**：自动下载视频/音频 -> 提取音频 -> 语音识别 -> 生成字幕。
- **智能字幕**：基于 NLP 算法对识别结果进行智能断句，生成标准 SRT 字幕文件。
- **硬件加速**：原生支持 CUDA 加速（需 NVIDIA 显卡），大幅提升识别速度。
- **隐私访问**：支持加载浏览器 Cookie（Netscape 格式或本地浏览器直接读取），轻松访问会员或登录限制内容。
- **双模式运行**：
  - **本地模式**：基于 Streamlit 的 WebUI，开箱即用。
  - **Client/Server模式**：Flask 后端 + Flet 独立客户端，适合局域网服务或计算资源分离场景。

## 🛠️ 安装依赖

### 1. 系统级依赖 (FFmpeg)
本工具必须依赖 FFmpeg 进行音频处理，请确保已正确安装并配置环境变量。

*   **Ubuntu**: `sudo apt install ffmpeg`
*   **MacOS**: `brew install ffmpeg`
*   **Windows**: `winget install Gyan.FFmpeg` (安装后需重启终端)

### 2. Python 依赖
建议使用 Python 3.10+ 环境（为了更好的异步支持）。

```bash
# 推荐创建虚拟环境
conda create -n v2t python=3.12
conda activate v2t

# 安装项目依赖
pip install -r requirements.txt
```

## 🚀 使用方式

### 模式一：本地 WebUI (Streamlit)
简单快捷的开箱即用网页模式。

```bash
streamlit run converter_app.py --server.port=8351
```
启动后，浏览器访问 `http://127.0.0.1:8351/` 即可使用。

![网页UI界面](bak/example2.png)

### 模式二：前后端分离 (Remote API + GUI)
更强大的N-Client-to-1-Server模式，以及本地管理功能。

适合将服务部署在一台高性能服务器（Server），在任意多台不同的电脑（Client）上操作。

#### 1. 启动服务端 (API Server)
```bash
python converter_app_remote.py
```
*服务端默认运行在 5001 端口，API 文档详见 [README_REMOTE.md](README_REMOTE.md)。*

#### 2. 启动客户端 (GUI Client)
本仓库仅包含服务端的代码，功能更完备的图形界面客户端（Flet版）请移步：
👉 **[Video2TextGUI 项目仓库](https://github.com/tk166/Video2TextGUI)**

![Client主界面](bak/example.png)

## 🍪 Cookie 支持说明

为了下载高清画质或会员专属视频，本工具支持 Cookie 注入：

1. **本地浏览器读取**（仅Client/Server模式下支持）：支持从 Chrome/Edge/Firefox 自动读取本地 Cookie。
2. **文件上传**：支持标准的 Netscape 格式 `cookies.txt` 文件。

*关于如何导出 Netscape 格式 Cookie，请参阅 [COOKIE_FORMAT.md](COOKIE_FORMAT.md)*

## 📦 主要技术栈

*   **GUI**: [Streamlit](https://streamlit.io/) (Local), [Flet](https://flet.dev/) (Remote Client)
*   **API**: [Flask](https://flask.palletsprojects.com/)
*   **Core**: [FunASR](https://github.com/alibaba-damo-academy/FunASR) (语音识别模型), [PyTorch](https://pytorch.org/)
*   **Utils**: [yt-dlp](https://github.com/yt-dlp/yt-dlp) (下载), [FFmpeg-python](https://github.com/kkroening/ffmpeg-python)

## 🤝 协议

- 代码辅助生成：Gemini-Pro-Preview & Qwen-Coder-Plus
- 核心模型支持：ModelScope Community

本项目采用 **MIT 协议** 开源。
