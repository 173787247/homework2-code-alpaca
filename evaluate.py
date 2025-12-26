"""
评估脚本
"""
import argparse
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import os
from tqdm import tqdm
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
    
    # 提取生成的代码部分
    if "### Response:\n" in generated_text:
        generated_code = generated_text.split("### Response:\n")[-1].strip()
    else:
        generated_code = generated_text
    
    return generated_code


def main():
    parser = argparse.ArgumentParser(description="评估模型")
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型路径")
    parser.add_argument("--base_model", type=str, default=None,
                       help="基础模型路径（如果使用 LoRA/QLoRA）")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="配置文件路径")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="评估样本数")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = config['model']['device']
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    # 加载模型
    print("正在加载模型...")
    if args.base_model:
        # LoRA/QLoRA 模型
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        model = PeftModel.from_pretrained(base_model, args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    else:
        # 完整模型
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # 加载测试数据
    print("正在加载测试数据...")
    if os.path.exists(config['data']['dataset_path']):
        from src.data.dataset import CodeAlpacaDataset
        dataset_loader = CodeAlpacaDataset(dataset_path=config['data']['dataset_path'])
        test_dataset = dataset_loader.get_test_dataset()
    else:
        dataset = load_dataset(config['data']['dataset_name'])
        test_dataset = dataset.get('test', dataset['train'])
    
    # 评估
    print("正在评估...")
    results = []
    bleu_scores = []
    codebleu_scores = []
    
    num_samples = min(args.num_samples, len(test_dataset))
    
    for i in tqdm(range(num_samples)):
        example = test_dataset[i]
        instruction = example['instruction']
        input_text = example.get('input', '')
        ground_truth = example['output']
        
        # 生成代码
        generated = generate_code(
            model, tokenizer, instruction, input_text,
            max_length=config['data']['max_target_length'],
            device=device
        )
        
        # 计算指标
        bleu = calculate_bleu(generated, ground_truth)
        codebleu = calculate_codebleu(generated, ground_truth)
        
        bleu_scores.append(bleu)
        codebleu_scores.append(codebleu)
        
        results.append({
            'instruction': instruction,
            'input': input_text,
            'ground_truth': ground_truth[:200],
            'generated': generated[:200],
            'bleu': bleu,
            'codebleu': codebleu
        })
    
    # 打印结果
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    print(f"平均 BLEU 分数: {sum(bleu_scores) / len(bleu_scores):.4f}")
    print(f"平均 CodeBLEU 分数: {sum(codebleu_scores) / len(codebleu_scores):.4f}")
    print("=" * 50)
    
    # 显示示例
    print("\n示例结果:")
    for i, result in enumerate(results[:5]):
        print(f"\n示例 {i+1}:")
        print(f"  Instruction: {result['instruction'][:100]}...")
        print(f"  Ground Truth: {result['ground_truth'][:100]}...")
        print(f"  Generated: {result['generated'][:100]}...")
        print(f"  BLEU: {result['bleu']:.4f}, CodeBLEU: {result['codebleu']:.4f}")


if __name__ == "__main__":
    main()


