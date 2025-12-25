"""
下载 Code Alpaca 数据集
"""
import argparse
from datasets import load_dataset
from pathlib import Path
import json


def main():
    parser = argparse.ArgumentParser(description="下载 Code Alpaca 数据集")
    parser.add_argument("--dataset_name", type=str, 
                       default="sahil2801/codealpaca",
                       help="数据集名称")
    parser.add_argument("--output_dir", type=str, 
                       default="./data/codealpaca",
                       help="输出目录")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("下载 Code Alpaca 数据集")
    print("=" * 50)
    print(f"数据集: {args.dataset_name}")
    print(f"保存到: {output_dir}")
    print("=" * 50)
    
    try:
        # 从 HuggingFace 加载数据集
        print("\n正在加载数据集...")
        dataset = load_dataset(args.dataset_name)
        
        print(f"\n数据集信息:")
        print(f"  训练集: {len(dataset['train'])} 条")
        if 'test' in dataset:
            print(f"  测试集: {len(dataset['test'])} 条")
        
        # 保存为 JSON 文件
        train_file = output_dir / "train.json"
        print(f"\n正在保存训练集到: {train_file}")
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(dataset['train'].to_list(), f, ensure_ascii=False, indent=2)
        
        if 'test' in dataset:
            test_file = output_dir / "test.json"
            print(f"正在保存测试集到: {test_file}")
            with open(test_file, 'w', encoding='utf-8') as f:
                json.dump(dataset['test'].to_list(), f, ensure_ascii=False, indent=2)
        
        # 显示示例
        print("\n数据集示例:")
        example = dataset['train'][0]
        print(f"  Instruction: {example.get('instruction', 'N/A')[:100]}...")
        print(f"  Input: {example.get('input', 'N/A')[:100]}...")
        print(f"  Output: {example.get('output', 'N/A')[:200]}...")
        
        print("\n" + "=" * 50)
        print("下载完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n如果下载失败，可以手动从以下链接下载:")
        print("  https://github.com/sahil280114/codealpaca")
        print("  或使用 HuggingFace: https://huggingface.co/datasets/sahil2801/codealpaca")


if __name__ == "__main__":
    main()

