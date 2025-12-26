# 作业2：Code Alpaca LoRA/QLoRA 微调

## 项目概述

本项目基于 Code Alpaca 数据集，使用 LoRA/QLoRA 技术对代码生成模型进行微调，并对比微调前后的效果。

## 参考项目

- Code Alpaca: https://github.com/sahil280114/codealpaca
- 本项目不是从零开始训练，而是使用预训练模型进行 LoRA/QLoRA 微调

## 技术栈

- **框架**: HuggingFace Transformers, PEFT
- **微调技术**: LoRA (Low-Rank Adaptation), QLoRA (Quantized LoRA)
- **量化**: bitsandbytes
- **模型**: CodeLlama, StarCoder, 或其他代码生成模型

## 项目结构

```
homework2-code-alpaca/
├── README.md                 # 项目说明
├── requirements.txt          # 依赖包
├── config.yaml              # 配置文件
├── download_data.py         # 下载 Code Alpaca 数据
├── train_lora.py            # LoRA 微调脚本
├── train_qlora.py           # QLoRA 微调脚本
├── evaluate.py              # 评估脚本
├── inference.py             # 推理脚本
├── compare_results.py       # 对比微调前后效果
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py       # 数据集加载
│   │   └── preprocessing.py # 数据预处理
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lora_model.py    # LoRA 模型封装
│   │   └── qlora_model.py   # QLoRA 模型封装
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py       # 训练器
│   │   └── utils.py         # 训练工具
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluator.py     # 评估器
│       └── metrics.py       # 评估指标
└── notebooks/
    └── demo.ipynb           # 演示笔记本
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 完整流程：LoRA/QLoRA 微调并对比前后效果

#### 1. 下载数据

```bash
python download_data.py
```

#### 2. LoRA 微调

```bash
python train_lora.py --config config.yaml
```

训练完成后，模型会保存在 `./checkpoints/lora/final/`

#### 3. QLoRA 微调

```bash
python train_qlora.py --config config.yaml
```

训练完成后，模型会保存在 `./checkpoints/qlora/final/`

#### 4. 生成对比报告

**LoRA 微调前后对比：**
```bash
python compare_results.py \
  --before_model codellama/CodeLlama-7b-hf \
  --after_model ./checkpoints/lora/final \
  --base_model codellama/CodeLlama-7b-hf \
  --num_samples 50 \
  --output_dir ./reports
```

**QLoRA 微调前后对比：**
```bash
python compare_results.py \
  --before_model codellama/CodeLlama-7b-hf \
  --after_model ./checkpoints/qlora/final \
  --base_model codellama/CodeLlama-7b-hf \
  --num_samples 50 \
  --output_dir ./reports
```

**重要：** 报告文件会保存在 `reports/` 目录中，包括：
- `comparison.json` - 详细的对比数据（微调前后在测试集上的表现、示例代码生成对比）
- `comparison.png` - 性能指标对比图表（BLEU、CodeBLEU 分数对比）

这些报告文件**需要提交到GitHub**供老师查看。

#### 5. 单独评估模型（可选）

```bash
# 评估 LoRA 模型
python evaluate.py \
  --model_path ./checkpoints/lora/final \
  --base_model codellama/CodeLlama-7b-hf \
  --num_samples 100

# 评估 QLoRA 模型
python evaluate.py \
  --model_path ./checkpoints/qlora/final \
  --base_model codellama/CodeLlama-7b-hf \
  --num_samples 100
```

## 微调前后对比

### 评估指标

- **代码生成准确率**: 生成的代码是否能正确执行
- **BLEU 分数**: 与参考代码的相似度
- **CodeBLEU 分数**: 代码特定的 BLEU 分数
- **执行成功率**: 生成代码的执行成功率

### 对比结果

训练完成后，将生成对比报告，包括：
- 微调前后在测试集上的表现
- 示例代码生成对比
- 性能指标对比图表

## 实验配置

### LoRA 配置
- rank: 8-16
- alpha: 16-32
- target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

### QLoRA 配置
- bits: 4
- 使用 bitsandbytes 进行量化
- 其他配置与 LoRA 相同

## 参考文献

- Code Alpaca: https://github.com/sahil280114/codealpaca
- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
- PEFT: https://github.com/huggingface/peft



