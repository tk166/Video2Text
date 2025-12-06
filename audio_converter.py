import librosa
import soundfile as sf
import os

def convert_to_wav(input_file, target_sr=16000):
    """
    将音频文件转换为WAV格式并调整采样率
    
    Args:
        input_file (str): 输入音频文件路径
        target_sr (int): 目标采样率，默认16000Hz
        
    Returns:
        str: 转换后的WAV文件路径
    """
    # 生成输出文件路径
    filename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(os.path.dirname(input_file), f"{filename}_converted.wav")
    
    try:
        # 使用librosa加载音频文件
        audio_data, original_sr = librosa.load(input_file, sr=None)
        
        # 如果原始采样率与目标采样率不同，则重新采样
        if original_sr != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
        
        # 保存为WAV文件
        sf.write(output_file, audio_data, target_sr)
        
        return output_file
    except Exception as e:
        raise Exception(f"音频转换失败: {str(e)}")

# 示例用法
if __name__ == "__main__":
    # 示例输入文件（需要实际存在）
    input_file = "test.mp3"
    if os.path.exists(input_file):
        try:
            output_file = convert_to_wav(input_file)
            print(f"音频转换成功: {output_file}")
        except Exception as e:
            print(f"转换失败: {e}")
    else:
        print(f"输入文件不存在: {input_file}")