"""
下载 Code Alpaca 数据集
"""
import argparse
from datasets import load_dataset, Dataset
from pathlib import Path
import json
import urllib.request
import os


def download_from_github(output_dir: Path):
    """从 GitHub 下载 Code Alpaca 数据集"""
    print("\n正在从 GitHub 下载数据集...")
    
    # Code Alpaca 数据集的 GitHub 原始文件链接
    github_url = "https://raw.githubusercontent.com/sahil280114/codealpaca/master/data/code_alpaca_20k.json"
    
    try:
        # 下载文件
        print(f"正在下载: {github_url}")
        urllib.request.urlretrieve(github_url, output_dir / "code_alpaca_20k.json")
        print("下载完成！")
        
        # 读取并处理数据
        print("正在处理数据...")
        with open(output_dir / "code_alpaca_20k.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 分割训练集和测试集（90% 训练，10% 测试）
        total = len(data)
        train_size = int(total * 0.9)
        
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        # 保存训练集
        train_file = output_dir / "train.json"
        print(f"正在保存训练集到: {train_file}")
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        # 保存测试集
        test_file = output_dir / "test.json"
        print(f"正在保存测试集到: {test_file}")
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        # 显示示例
        print("\n数据集示例:")
        example = train_data[0]
        print(f"  Instruction: {example.get('instruction', 'N/A')[:100]}...")
        print(f"  Input: {example.get('input', 'N/A')[:100]}...")
        print(f"  Output: {example.get('output', 'N/A')[:200]}...")
        
        print(f"\n数据集信息:")
        print(f"  训练集: {len(train_data)} 条")
        print(f"  测试集: {len(test_data)} 条")
        
        return True
        
    except Exception as e:
        print(f"从 GitHub 下载失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="下载 Code Alpaca 数据集")
    parser.add_argument("--dataset_name", type=str, 
                       default="sahil2801/codealpaca",
                       help="数据集名称（HuggingFace）")
    parser.add_argument("--output_dir", type=str, 
                       default="./data/codealpaca",
                       help="输出目录")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("下载 Code Alpaca 数据集")
    print("=" * 50)
    print(f"保存到: {output_dir}")
    print("=" * 50)
    
    # 首先尝试从 HuggingFace 下载
    try:
        print("\n正在尝试从 HuggingFace 加载数据集...")
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
        print(f"\n从 HuggingFace 下载失败: {e}")
        print("\n正在尝试从 GitHub 下载...")
        
        # 如果 HuggingFace 失败，尝试从 GitHub 下载
        if download_from_github(output_dir):
            print("\n" + "=" * 50)
            print("下载完成！")
            print("=" * 50)
        else:
            print("\n" + "=" * 50)
            print("下载失败！")
            print("=" * 50)
            print("\n请手动从以下链接下载:")
            print("  https://github.com/sahil280114/codealpaca")
            print("  下载 code_alpaca_20k.json 文件并保存到:", output_dir)
            print("  然后重命名为 train.json 和 test.json")


if __name__ == "__main__":
    main()



