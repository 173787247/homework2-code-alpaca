"""
预下载模型到本地缓存（避免训练时重复下载）
"""
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="预下载模型到本地缓存")
    parser.add_argument("--model_name", type=str, 
                       default="codellama/CodeLlama-7b-hf",
                       help="模型名称")
    parser.add_argument("--cache_dir", type=str, 
                       default=None,
                       help="缓存目录（默认使用 HuggingFace 默认缓存）")
    args = parser.parse_args()
    
    print("=" * 50)
    print("预下载模型")
    print("=" * 50)
    print(f"模型: {args.model_name}")
    if args.cache_dir:
        print(f"缓存目录: {args.cache_dir}")
    print("=" * 50)
    
    try:
        # 下载 tokenizer
        print("\n正在下载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir
        )
        print("Tokenizer 下载完成！")
        
        # 下载模型（只下载权重，不加载到内存）
        print("\n正在下载模型权重...")
        print("（这可能需要一些时间，模型约 13GB）")
        
        # 使用 from_pretrained 但设置 low_cpu_mem_usage 和 torch_dtype
        # 这样可以下载模型但不会完全加载到内存
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        print("\n" + "=" * 50)
        print("模型下载完成！")
        print("=" * 50)
        print(f"\n模型已保存到缓存目录")
        print("下次训练时可以直接使用，无需重新下载")
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n提示：")
        print("1. 确保网络连接正常")
        print("2. 确保有足够的磁盘空间（至少 15GB）")
        print("3. 如果使用 Docker，确保缓存目录已挂载")


if __name__ == "__main__":
    main()

