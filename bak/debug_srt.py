#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
调试脚本，用于检查generate_custom_srt函数
"""

def format_time(ms):
    """将毫秒转换为SRT时间格式"""
    seconds = ms / 1000
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int(ms % 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def generate_custom_srt(text, timestamps, chars_per_line=20):
    """
    使用时间戳数据生成自定义SRT字幕
    """
    print(f"调试信息:")
    print(f"  文本长度: {len(text)}")
    print(f"  时间戳长度: {len(timestamps)}")
    print(f"  长度是否匹配: {len(text) == len(timestamps)}")
    
    if not text:
        print("  文本为空")
        return ""
    if not timestamps:
        print("  时间戳为空")
        return ""
    if len(text) != len(timestamps):
        print("  文本和时间戳长度不匹配")
        return ""
    
    srt_lines = []
    line_number = 1
    
    # 按指定字符数分割文本
    for i in range(0, len(text), chars_per_line):
        # 获取当前行的文本
        line_text = text[i:i+chars_per_line]
        print(f"  处理第{i}个字符，文本: '{line_text}'")
        
        # 获取当前行的时间戳范围
        start_idx = i
        end_idx = min(i + chars_per_line - 1, len(text) - 1)
        print(f"  开始索引: {start_idx}, 结束索引: {end_idx}")
        
        if start_idx < len(timestamps) and end_idx < len(timestamps):
            start_time = timestamps[start_idx][0]  # 第一个字符的开始时间
            end_time = timestamps[end_idx][1]      # 最后一个字符的结束时间
            print(f"  开始时间: {start_time}, 结束时间: {end_time}")
            
            # 生成SRT行
            srt_lines.append(str(line_number))
            srt_lines.append(f"{format_time(start_time)} --> {format_time(end_time)}")
            srt_lines.append(line_text)
            srt_lines.append("")  # 空行
            
            line_number += 1
        else:
            print(f"  索引超出范围: start_idx={start_idx}, end_idx={end_idx}, len(timestamps)={len(timestamps)}")
    
    result = "\n".join(srt_lines)
    print(f"  生成的SRT行数: {len(srt_lines)}")
    return result

def main():
    """主函数"""
    # 模拟的文本和时间戳数据（确保长度匹配）
    sample_text = "大家好欢迎观看我的视频今天我们要讲解的是人工智能的发展历程"
    sample_timestamps = [
        [1000, 1500], [1500, 1800], [1800, 2100],  # "大家好"
        [2100, 2400], [2400, 2700], [2700, 3000],  # "欢迎观"
        [3000, 3300], [3300, 3600], [3600, 3900],  # "看我的"
        [3900, 4200], [4200, 4500], [4500, 4800],  # "视频今"
        [4800, 5100], [5100, 5400], [5400, 5700],  # "天我们"
        [5700, 6000], [6000, 6300], [6300, 6600],  # "要讲解"
        [6600, 6900], [6900, 7200], [7200, 7500],  # "的是人"
        [7500, 7800], [7800, 8100], [8100, 8400],  # "工智能"
        [8400, 8700], [8700, 9000], [9000, 9300]   # "的发展历程"
    ]

    # 截取文本以匹配时间戳数量
    sample_text = sample_text[:len(sample_timestamps)]
    
    print(f"原始文本: {sample_text}")
    print(f"文本长度: {len(sample_text)}")
    print(f"时间戳数量: {len(sample_timestamps)}")
    
    # 生成自定义SRT字幕
    custom_srt = generate_custom_srt(sample_text, sample_timestamps, chars_per_line=10)
    print(f"\n生成的自定义SRT字幕:")
    print("=" * 50)
    print(custom_srt if custom_srt else "生成失败")
    print("=" * 50)

if __name__ == "__main__":
    main()