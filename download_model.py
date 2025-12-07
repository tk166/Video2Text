import argparse
import sys
from modelscope import snapshot_download

# 定义模型组 ID
MODEL_GROUPS = {
    "paraformer": [
        "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", # 主模型
        "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",                                  # VAD
        "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"                       # 标点
    ],
    "sensevoice": [
        "iic/SenseVoiceSmall"
    ],
    "whisper": [
        "iic/Whisper-large-v3"
    ]
}

def download_list(model_ids, description):
    """下载一组模型"""
    print(f"\n🚀 开始下载: {description}")
    for mid in model_ids:
        print(f"   ⬇️  正在拉取: {mid}")
        try:
            path = snapshot_download(mid)
            print(f"   ✅ 完成: {path}")
        except Exception as e:
            print(f"   ❌ 失败: {mid}")
            print(f"      错误: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="FunASR 模型批量下载工具 (支持断点续传)"
    )
    
    # 添加命令行参数
    parser.add_argument("--all", action="store_true", help="下载所有模型")
    parser.add_argument("--paraformer", action="store_true", help="仅下载 Paraformer (中文高效)")
    parser.add_argument("--sensevoice", action="store_true", help="仅下载 SenseVoice (多语言/情感)")
    # parser.add_argument("--whisper", action="store_true", help="仅下载 Whisper Large V3 (通用大模型)")

    # 解析参数
    args = parser.parse_args()

    # 如果用户没有输入任何参数，打印帮助信息
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    print("="*50)
    print("📥 FunASR 模型下载器启动")
    print("="*50)

    # 逻辑判断
    if args.all or args.paraformer:
        download_list(MODEL_GROUPS["paraformer"], "Paraformer 中文套餐 (ASR+VAD+PUNC)")
    
    if args.all or args.sensevoice:
        download_list(MODEL_GROUPS["sensevoice"], "SenseVoice 多语言模型")
        
    # if args.all or args.whisper:
    #     download_list(MODEL_GROUPS["whisper"], "Whisper Large V3 (体积较大)")

    print("\n" + "="*50)
    print("🎉 所有请求处理完毕。")

if __name__ == "__main__":
    main()