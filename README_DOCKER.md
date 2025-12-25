# Docker 使用指南

## 快速开始

### 1. 配置 Docker 镜像加速器

在 Docker Desktop 中配置镜像加速器：
- Settings → Docker Engine → 添加以下配置：

```json
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com"
  ]
}
```

点击 "Apply & Restart" 重启 Docker。

### 2. 拉取基础镜像

```powershell
# 使用清华镜像源拉取
.\docker-pull-tsinghua.ps1

# 或手动拉取
docker pull pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
```

### 3. 构建项目镜像

```powershell
docker-compose -f docker-compose.gpu.yml build
```

### 4. 启动容器

```powershell
docker-compose -f docker-compose.gpu.yml up -d
```

### 5. 进入容器

```powershell
docker exec -it homework2-code-alpaca-gpu bash
```

### 6. 在容器内测试

```bash
# 检查 GPU
python check_gpu.py

# 运行完整测试
python run_tests.py

# 下载数据集（如果需要）
python download_data.py

# LoRA 微调训练
python train_lora.py --config config.yaml

# QLoRA 微调训练
python train_qlora.py --config config.yaml

# 推理
python inference.py --instruction "Write a Python function to calculate factorial"
```

## 镜像源配置说明

### Dockerfile 已配置：

1. **APT 镜像源**：使用清华镜像源加速系统包下载
2. **PIP 镜像源**：使用清华 PyPI 镜像源加速 Python 包下载

### 环境变量：

- `CUDA_VISIBLE_DEVICES=0`：使用第一个 GPU
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`：优化 GPU 内存分配
- `OMP_NUM_THREADS=8`：设置 OpenMP 线程数

## 故障排除

### 问题1：GPU 不可用

检查 Docker Desktop GPU 设置：
- Settings → Resources → Advanced
- 确保 "Use the WSL 2 based engine" 已启用
- 确保 GPU 支持已启用

### 问题2：镜像拉取慢

确保已配置 Docker 镜像加速器（见步骤1）

### 问题3：模型下载慢

模型会从 HuggingFace 下载，如果网络慢可以：
- 使用 HuggingFace 镜像源
- 或预先下载模型到 `./models` 目录

## 参考

- PyTorch Docker 镜像：https://hub.docker.com/r/pytorch/pytorch
- 清华镜像源：https://mirrors.tuna.tsinghua.edu.cn/
- Code Alpaca 数据集：https://github.com/sahil280114/codealpaca

