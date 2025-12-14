import yt_dlp
import os
import tempfile
import time

def download_audio(video_url, cookiefile=None, extra_info=False):
    """
    使用yt-dlp下载视频的音频

    Args:
        video_url (str): 视频链接
        cookiefile (str, optional): Cookie文件路径，默认为None不使用

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
            'preferredquality': '320',
        }],
        'postprocessor_args': [
            '-ar', '48000'
        ],
        'prefer_ffmpeg': True,
        'keepvideo': False,
        'outtmpl': os.path.join(temp_dir, 'audio.%(ext)s'),
    }

    # 如果提供了cookie文件且文件存在，则添加到配置中
    if cookiefile and os.path.exists(cookiefile):
        # # 调试：保存cookie文件内容用于分析
        # try:
        #     with open(cookiefile, 'r', encoding='utf-8') as f:
        #         cookie_content = f.read()

        #     # 保存到固定位置用于调试
        #     debug_cookie_path = os.path.join(os.getcwd(), 'cookie_debug.txt')
        #     with open(debug_cookie_path, 'w', encoding='utf-8') as f:
        #         f.write(f"# 调试信息\n")
        #         f.write(f"# 原始临时文件: {cookiefile}\n")
        #         f.write(f"# 保存时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        #         f.write(f"# 文件大小: {len(cookie_content)} 字符\n")
        #         f.write("=" * 50 + "\n")
        #         f.write(cookie_content)

        #     print(f"✅ Cookie文件内容已保存到调试文件: {debug_cookie_path}")
        # except Exception as debug_e:
        #     print(f"⚠️  保存cookie调试文件失败: {debug_e}")

        # 验证cookie文件是否可能是有效的Netscape格式
        try:
            # 检查文件大小，空文件或非常小的文件可能无效
            if os.path.getsize(cookiefile) > 0:
                ydl_opts['cookiefile'] = cookiefile
        except Exception:
            # 如果有任何问题，忽略cookie文件并继续
            pass

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # 下载音频
            info_dict = ydl.extract_info(video_url, download=True)

            # 获取下载的文件路径
            downloaded_file = ydl.prepare_filename(info_dict)
            # 因为使用了FFmpegExtractAudio后处理器，实际文件扩展名会变为mp3
            mp3_file = downloaded_file.rsplit('.', 1)[0] + '.mp3'
            
            if extra_info:
                video_title = info_dict.get('title', '未知标题')
                uploader = info_dict.get('uploader', '未知作者')
                return mp3_file, video_title, uploader

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
