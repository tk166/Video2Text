# Video2Text
一键视频语音识别与字幕转换，本地部署模型版本、基于funasr、pytorch、yt-dlp、streamlit、ffmpeg

使用方式：
```
pip install -r requirements.txt

python download_model.py --all

streamlit run converter_app.py
```

使用gemini-3-pro-preview与qwen3-coder-plus生成代码
MIT协议