"""
训练工具函数
"""
import torch
from typing import Dict, Any


def print_model_info(model):
    """打印模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 50)
    print("模型信息")
    print("=" * 50)
    print(f"总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    print(f"可训练参数比例: {trainable_params / total_params * 100:.2f}%")
    print("=" * 50)


def save_checkpoint(model, tokenizer, output_dir: str, epoch: int):
    """保存检查点"""
    import os
    from pathlib import Path
    
    checkpoint_dir = Path(output_dir) / f"checkpoint-{epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    
    print(f"检查点已保存到: {checkpoint_dir}")


def load_checkpoint(model, checkpoint_path: str):
    """加载检查点"""
    from peft import PeftModel
    
    if isinstance(model, PeftModel):
        model = PeftModel.from_pretrained(model, checkpoint_path)
    else:
        model.load_state_dict(torch.load(checkpoint_path))
    
    return model

