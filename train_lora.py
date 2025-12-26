"""
LoRA 微调训练脚本
"""
import argparse
import yaml
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
import os
from pathlib import Path

from src.models.lora_model import load_lora_model
from src.data.dataset import preprocess_function


def main():
    parser = argparse.ArgumentParser(description="LoRA 微调训练")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="配置文件路径")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = config['model']['device']
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU")
        device = "cpu"
    
    # 加载数据集
    print("正在加载数据集...")
    if os.path.exists(config['data']['dataset_path']):
        from src.data.dataset import CodeAlpacaDataset
        dataset_loader = CodeAlpacaDataset(dataset_path=config['data']['dataset_path'])
        dataset = {
            'train': dataset_loader.get_train_dataset(),
            'test': dataset_loader.get_test_dataset()
        }
    else:
        dataset = load_dataset(config['data']['dataset_name'])
    
    # 加载模型和 tokenizer
    print("正在加载模型...")
    model, tokenizer = load_lora_model(
        base_model_name=config['model']['base_model'],
        lora_config=config['lora'],
        device=device,
        load_in_8bit=config['model'].get('load_in_8bit', False)
    )
    
    # 预处理数据集
    print("正在预处理数据集...")
    def tokenize_function(examples):
        return preprocess_function(
            examples,
            tokenizer,
            max_length=config['data']['max_length'],
            max_target_length=config['data']['max_target_length']
        )
    
    tokenized_train = dataset['train'].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    # 如果有测试集，用作验证集
    tokenized_eval = None
    if 'test' in dataset and len(dataset['test']) > 0:
        tokenized_eval = dataset['test'].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['test'].column_names
        )
    
    tokenized_dataset = tokenized_train
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 检查是否有验证集
    has_eval_dataset = tokenized_eval is not None and len(tokenized_eval) > 0
    eval_steps = config['training'].get('eval_steps', 0)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'] + "/lora",
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=float(config['training']['learning_rate']),
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        eval_steps=eval_steps if has_eval_dataset else None,
        save_total_limit=config['training']['save_total_limit'],
        fp16=config['training']['fp16'],
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        optim=config['training']['optim'],
        report_to="tensorboard",
        # 如果有验证集且配置了 eval_steps，启用评估策略
        eval_strategy="steps" if has_eval_dataset and eval_steps > 0 else "no",
        # 只有在有评估策略时才启用 load_best_model_at_end
        load_best_model_at_end=True if has_eval_dataset and eval_steps > 0 else False,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_eval if tokenized_eval is not None else None,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("=" * 50)
    print("开始 LoRA 微调训练")
    print("=" * 50)
    print(f"设备: {device}")
    print(f"训练样本数: {len(tokenized_dataset)}")
    print(f"批次大小: {config['training']['per_device_train_batch_size']}")
    print(f"训练轮数: {config['training']['num_train_epochs']}")
    print("=" * 50)
    
    trainer.train()
    
    # 保存模型
    output_dir = Path(config['training']['output_dir']) / "lora" / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    print(f"\n模型已保存到: {output_dir}")
    print("训练完成!")


if __name__ == "__main__":
    main()



