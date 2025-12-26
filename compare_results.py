"""
对比微调前后效果
"""
import argparse
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import os
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.metrics import calculate_bleu, calculate_codebleu


def generate_code(model, tokenizer, instruction: str, input_text: str = "",
                  max_length: int = 256, device: str = "cuda"):
    """生成代码"""
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=3,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:\n" in generated_text:
        generated_code = generated_text.split("### Response:\n")[-1].strip()
    else:
        generated_code = generated_text
    
    return generated_code


def evaluate_model(model, tokenizer, test_dataset, num_samples: int,
                   device: str, max_length: int = 256):
    """评估模型"""
    bleu_scores = []
    codebleu_scores = []
    results = []
    
    num_samples = min(num_samples, len(test_dataset))
    
    for i in tqdm(range(num_samples)):
        example = test_dataset[i]
        instruction = example['instruction']
        input_text = example.get('input', '')
        ground_truth = example['output']
        
        generated = generate_code(
            model, tokenizer, instruction, input_text,
            max_length=max_length, device=device
        )
        
        bleu = calculate_bleu(generated, ground_truth)
        codebleu = calculate_codebleu(generated, ground_truth)
        
        bleu_scores.append(bleu)
        codebleu_scores.append(codebleu)
        
        results.append({
            'instruction': instruction,
            'input': input_text,
            'ground_truth': ground_truth,
            'generated': generated,
            'bleu': bleu,
            'codebleu': codebleu
        })
    
    return {
        'bleu_mean': np.mean(bleu_scores),
        'bleu_std': np.std(bleu_scores),
        'codebleu_mean': np.mean(codebleu_scores),
        'codebleu_std': np.std(codebleu_scores),
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(description="对比微调前后效果")
    parser.add_argument("--before_model", type=str, required=True,
                       help="微调前的模型路径")
    parser.add_argument("--after_model", type=str, required=True,
                       help="微调后的模型路径")
    parser.add_argument("--base_model", type=str, default=None,
                       help="基础模型路径（如果使用 LoRA/QLoRA）")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="配置文件路径")
    parser.add_argument("--num_samples", type=int, default=50,
                       help="评估样本数")
    parser.add_argument("--output_dir", type=str, default="./reports",
                       help="输出目录（默认：./reports，报告文件会提交到GitHub）")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = config['model']['device']
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    # 加载测试数据
    print("正在加载测试数据...")
    if os.path.exists(config['data']['dataset_path']):
        from src.data.dataset import CodeAlpacaDataset
        dataset_loader = CodeAlpacaDataset(dataset_path=config['data']['dataset_path'])
        test_dataset = dataset_loader.get_test_dataset()
    else:
        dataset = load_dataset(config['data']['dataset_name'])
        test_dataset = dataset.get('test', dataset['train'])
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 评估微调前的模型
    print("\n" + "=" * 50)
    print("评估微调前的模型")
    print("=" * 50)
    print("正在加载模型...")
    
    before_model = AutoModelForCausalLM.from_pretrained(
        args.before_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    before_tokenizer = AutoTokenizer.from_pretrained(args.before_model)
    if before_tokenizer.pad_token is None:
        before_tokenizer.pad_token = before_tokenizer.eos_token
    before_model.eval()
    
    before_results = evaluate_model(
        before_model, before_tokenizer, test_dataset,
        args.num_samples, device, config['data']['max_target_length']
    )
    
    print(f"微调前 - BLEU: {before_results['bleu_mean']:.4f} ± {before_results['bleu_std']:.4f}")
    print(f"微调前 - CodeBLEU: {before_results['codebleu_mean']:.4f} ± {before_results['codebleu_std']:.4f}")
    
    # 评估微调后的模型
    print("\n" + "=" * 50)
    print("评估微调后的模型")
    print("=" * 50)
    print("正在加载模型...")
    
    if args.base_model:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        after_model = PeftModel.from_pretrained(base_model, args.after_model)
        after_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    else:
        after_model = AutoModelForCausalLM.from_pretrained(
            args.after_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        after_tokenizer = AutoTokenizer.from_pretrained(args.after_model)
    
    if after_tokenizer.pad_token is None:
        after_tokenizer.pad_token = after_tokenizer.eos_token
    after_model.eval()
    
    after_results = evaluate_model(
        after_model, after_tokenizer, test_dataset,
        args.num_samples, device, config['data']['max_target_length']
    )
    
    print(f"微调后 - BLEU: {after_results['bleu_mean']:.4f} ± {after_results['bleu_std']:.4f}")
    print(f"微调后 - CodeBLEU: {after_results['codebleu_mean']:.4f} ± {after_results['codebleu_std']:.4f}")
    
    # 对比结果
    print("\n" + "=" * 50)
    print("对比结果")
    print("=" * 50)
    bleu_improvement = after_results['bleu_mean'] - before_results['bleu_mean']
    codebleu_improvement = after_results['codebleu_mean'] - before_results['codebleu_mean']
    
    print(f"BLEU 提升: {bleu_improvement:+.4f} "
          f"({bleu_improvement / before_results['bleu_mean'] * 100:+.2f}%)")
    print(f"CodeBLEU 提升: {codebleu_improvement:+.4f} "
          f"({codebleu_improvement / before_results['codebleu_mean'] * 100:+.2f}%)")
    
    # 保存结果
    comparison = {
        'before': {
            'bleu_mean': float(before_results['bleu_mean']),
            'bleu_std': float(before_results['bleu_std']),
            'codebleu_mean': float(before_results['codebleu_mean']),
            'codebleu_std': float(before_results['codebleu_std'])
        },
        'after': {
            'bleu_mean': float(after_results['bleu_mean']),
            'bleu_std': float(after_results['bleu_std']),
            'codebleu_mean': float(after_results['codebleu_mean']),
            'codebleu_std': float(after_results['codebleu_std'])
        },
        'improvement': {
            'bleu': float(bleu_improvement),
            'codebleu': float(codebleu_improvement)
        },
        'examples': []
    }
    
    # 保存示例
    for i in range(min(10, len(before_results['results']))):
        comparison['examples'].append({
            'instruction': before_results['results'][i]['instruction'],
            'input': before_results['results'][i]['input'],
            'ground_truth': before_results['results'][i]['ground_truth'][:500],
            'before_generated': before_results['results'][i]['generated'][:500],
            'after_generated': after_results['results'][i]['generated'][:500],
            'before_bleu': float(before_results['results'][i]['bleu']),
            'after_bleu': float(after_results['results'][i]['bleu']),
            'before_codebleu': float(before_results['results'][i]['codebleu']),
            'after_codebleu': float(after_results['results'][i]['codebleu'])
        })
    
    # 保存 JSON
    with open(output_dir / "comparison.json", 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    # 绘制对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    metrics = ['BLEU', 'CodeBLEU']
    before_means = [before_results['bleu_mean'], before_results['codebleu_mean']]
    after_means = [after_results['bleu_mean'], after_results['codebleu_mean']]
    before_stds = [before_results['bleu_std'], before_results['codebleu_std']]
    after_stds = [after_results['bleu_std'], after_results['codebleu_std']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, before_means, width, yerr=before_stds, 
           label='微调前', alpha=0.8)
    ax1.bar(x + width/2, after_means, width, yerr=after_stds,
           label='微调后', alpha=0.8)
    ax1.set_ylabel('分数')
    ax1.set_title('微调前后对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 改进百分比
    improvements = [bleu_improvement, codebleu_improvement]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax2.bar(metrics, improvements, color=colors, alpha=0.8)
    ax2.set_ylabel('改进')
    ax2.set_title('改进幅度')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png", dpi=300, bbox_inches='tight')
    print(f"\n对比图表已保存到: {output_dir / 'comparison.png'}")
    print(f"详细结果已保存到: {output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()



