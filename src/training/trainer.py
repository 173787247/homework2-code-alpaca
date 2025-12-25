"""
训练器封装
"""
from transformers import Trainer, TrainingArguments
from typing import Optional


class CodeAlpacaTrainer:
    """Code Alpaca 训练器"""
    
    def __init__(self, model, tokenizer, train_dataset, eval_dataset=None,
                 training_args: Optional[TrainingArguments] = None):
        """
        初始化训练器
        
        Args:
            model: 模型
            tokenizer: tokenizer
            train_dataset: 训练数据集
            eval_dataset: 验证数据集（可选）
            training_args: 训练参数
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        if training_args is None:
            training_args = TrainingArguments(
                output_dir="./checkpoints",
                num_train_epochs=3,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                learning_rate=2e-4,
                fp16=True,
                logging_steps=10,
                save_steps=500
            )
        
        self.training_args = training_args
        self.trainer = None
    
    def create_trainer(self, data_collator):
        """创建 Trainer 实例"""
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator
        )
        return self.trainer
    
    def train(self, data_collator):
        """开始训练"""
        if self.trainer is None:
            self.create_trainer(data_collator)
        
        return self.trainer.train()
    
    def evaluate(self, data_collator):
        """评估模型"""
        if self.trainer is None:
            self.create_trainer(data_collator)
        
        if self.eval_dataset is None:
            raise ValueError("需要提供验证数据集")
        
        return self.trainer.evaluate()

