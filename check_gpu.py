"""
检查 GPU 是否可用
"""
import sys
import io

# 设置输出编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_gpu():
    """检查 GPU 状态"""
    print("=" * 50)
    print("GPU 检查")
    print("=" * 50)
    
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        print(f"CUDA 可用: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU 数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}:")
                print(f"  名称: {torch.cuda.get_device_name(i)}")
                print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
                print(f"  计算能力: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
            
            # 测试 GPU 计算
            print("\n测试 GPU 计算...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("[OK] GPU 计算测试通过")
        else:
            print("\n警告: CUDA 不可用，将使用 CPU")
            print("如果您的系统有 GPU，请检查:")
            print("1. 是否安装了 CUDA 版本的 PyTorch")
            print("2. 是否安装了 NVIDIA 驱动")
            print("3. 在 Docker 中: 是否配置了 GPU 支持")
        
        print("=" * 50)
        return cuda_available
        
    except ImportError:
        print("错误: PyTorch 未安装")
        return False
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = check_gpu()
    sys.exit(0 if success else 1)


