# Video2Text
一键视频语音识别与字幕转换，本地部署模型版本、基于funasr、pytorch、yt-dlp、streamlit、ffmpeg

使用方式：
```
// 安装 FFMpeg
// Ununtu
sudo apt install ffmpeg
// MacOS
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install ffmpeg
// Windows（运行命令后重启窗口）
winget install Gyan.FFmpeg

pip install -r requirements.txt

streamlit run converter_app.py --server.port=8351
```

使用gemini-3-pro-preview与qwen3-coder-plus生成代码

MIT协议
