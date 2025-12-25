"""
推理脚本：使用微调后的模型生成代码
"""
import argparse
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


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
            pad_token_id=tokenizer.pad_token_id,
            temperature=0.7,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:\n" in generated_text:
        generated_code = generated_text.split("### Response:\n")[-1].strip()
    else:
        generated_code = generated_text
    
    return generated_code


def main():
    parser = argparse.ArgumentParser(description="使用微调后的模型进行推理")
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型路径")
    parser.add_argument("--base_model", type=str, default=None,
                       help="基础模型路径（如果使用 LoRA/QLoRA）")
    parser.add_argument("--instruction", type=str, required=True,
                       help="指令")
    parser.add_argument("--input", type=str, default="",
                       help="输入（可选）")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="配置文件路径")
    parser.add_argument("--max_length", type=int, default=256,
                       help="最大生成长度")
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
    
    # 生成代码
    print("\n" + "=" * 50)
    print("生成代码")
    print("=" * 50)
    print(f"Instruction: {args.instruction}")
    if args.input:
        print(f"Input: {args.input}")
    print("=" * 50)
    
    generated = generate_code(
        model, tokenizer, args.instruction, args.input,
        max_length=args.max_length, device=device
    )
    
    print("\n生成的代码:")
    print("-" * 50)
    print(generated)
    print("-" * 50)


if __name__ == "__main__":
    main()

