"""
评估指标计算
"""
from typing import List
import re


def calculate_bleu(prediction: str, reference: str) -> float:
    """
    计算简化的 BLEU 分数
    
    Args:
        prediction: 预测文本
        reference: 参考文本
        
    Returns:
        BLEU 分数 (0-1)
    """
    # 简化的 BLEU 计算
    # 实际应该使用 nltk.translate.bleu_score
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    
    if len(pred_tokens) == 0:
        return 0.0
    
    # 计算 1-gram 精确度
    matches = sum(1 for token in pred_tokens if token in ref_tokens)
    precision = matches / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    
    # 简化的 BLEU（实际应该考虑 n-gram）
    return precision


def calculate_codebleu(prediction: str, reference: str) -> float:
    """
    计算简化的 CodeBLEU 分数
    
    Args:
        prediction: 预测代码
        reference: 参考代码
        
    Returns:
        CodeBLEU 分数 (0-1)
    """
    # 简化的 CodeBLEU 计算
    # 实际应该使用 codebleu 库
    
    # 提取代码结构（函数名、变量名等）
    def extract_code_elements(code: str) -> set:
        elements = set()
        # 提取函数定义
        functions = re.findall(r'def\s+(\w+)', code)
        elements.update(functions)
        # 提取类定义
        classes = re.findall(r'class\s+(\w+)', code)
        elements.update(classes)
        # 提取变量（简单版本）
        variables = re.findall(r'\b([a-z_][a-z0-9_]*)\s*=', code)
        elements.update(variables)
        return elements
    
    pred_elements = extract_code_elements(prediction)
    ref_elements = extract_code_elements(reference)
    
    if len(ref_elements) == 0:
        return 1.0 if len(pred_elements) == 0 else 0.0
    
    # 计算重叠
    overlap = len(pred_elements & ref_elements)
    codebleu = overlap / len(ref_elements) if len(ref_elements) > 0 else 0.0
    
    return codebleu


def compute_metrics(eval_pred):
    """
    计算评估指标（用于 Trainer）
    
    Args:
        eval_pred: 包含 predictions 和 label_ids 的 EvalPrediction 对象
        
    Returns:
        包含各种指标的字典
    """
    predictions, labels = eval_pred
    
    # 这里应该实现实际的指标计算
    # 由于需要 tokenizer 来解码，这里提供一个简化版本
    # 实际使用时应该在训练脚本中自定义 compute_metrics
    
    return {
        "accuracy": 0.0,  # 占位符，实际需要根据任务计算
    }


