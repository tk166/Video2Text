import librosa
import soundfile as sf
import os

def convert_to_wav(input_file, target_sr=16000):
    filename = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(os.path.dirname(input_file), f"{filename}_converted.wav")
    
    try:
        # 1. 强制单声道 (mono=True 是默认值，但显式写出来更保险)
        # librosa 会自动归一化数据到 -1~1 之间的 float32
        audio_data, original_sr = librosa.load(input_file, sr=None, mono=True)
        
        # 2. 重采样
        if original_sr != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=target_sr)
        
        # 3. 保存为 PCM_16 (16位整数) 格式
        # 很多 ASR 前端处理对 float 格式支持可能不如 int16 稳定
        # soundfile 会自动处理 float 到 int16 的量化转换
        sf.write(output_file, audio_data, target_sr, subtype='PCM_16')
        
        return output_file
    except Exception as e:
        raise Exception(f"音频转换失败: {str(e)}")

if __name__ == "__main__":
    # 测试
    input_file = "/var/folders/sx/8rjxr_yj3wl_7cdggzqx_bvr0000gn/T/tmp40g506ey/audio.webm" 
    
    if os.path.exists(input_file):
        try:
            output_file = convert_to_wav(input_file)
            print(f"转换成功，请尝试用此文件运行 FunASR: {output_file}")
        except Exception as e:
            print(f"转换失败: {e}")
