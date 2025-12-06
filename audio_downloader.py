import yt_dlp
import os
import tempfile

def download_audio(video_url):
    """
    使用yt-dlp下载视频的音频
    
    Args:
        video_url (str): 视频链接
        
    Returns:
        str: 下载的音频文件路径
    """
    # 创建临时目录用于存储下载的文件
    temp_dir = tempfile.mkdtemp()
    
    # yt-dlp配置
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'postprocessor_args': [
            '-ar', '16000'
        ],
        'prefer_ffmpeg': True,
        'keepvideo': False,
        'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # 下载音频
            info_dict = ydl.extract_info(video_url, download=True)
            
            # 获取下载的文件路径
            downloaded_file = ydl.prepare_filename(info_dict)
            # 因为使用了FFmpegExtractAudio后处理器，实际文件扩展名会变为mp3
            mp3_file = downloaded_file.rsplit('.', 1)[0] + '.mp3'
            
            return mp3_file
    except Exception as e:
        raise Exception(f"下载失败: {str(e)}")

# 示例用法（在实际应用中会被主程序调用）
if __name__ == "__main__":
    # 这里只是一个示例，实际使用时不会直接运行
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # 示例链接
    try:
        audio_file = download_audio(url)
        print(f"音频下载成功: {audio_file}")
    except Exception as e:
        print(f"下载失败: {e}")
