"""
完整的测试脚本：测试作业2的所有功能
"""
import sys
import io
from pathlib import Path

# 设置输出编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def test_imports():
    """测试模块导入"""
    print("=" * 60)
    print("测试1: 模块导入")
    print("=" * 60)
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "test_imports.py"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        print(result.stdout)
        if result.returncode != 0:
            print("[WARN] 部分模块导入失败（可能是缺少依赖）")
        return result.returncode == 0
    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
        return False


def test_gpu():
    """测试 GPU"""
    print("\n" + "=" * 60)
    print("测试2: GPU 检查")
    print("=" * 60)
    
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "check_gpu.py"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        print(result.stdout)
        return result.returncode == 0
    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
        return False


def test_config():
    """测试配置文件"""
    print("\n" + "=" * 60)
    print("测试3: 配置文件检查")
    print("=" * 60)
    
    config_file = Path("config.yaml")
    if not config_file.exists():
        print(f"[ERROR] 配置文件不存在: {config_file}")
        return False
    
    try:
        import yaml
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['data', 'model', 'lora', 'training']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"[ERROR] 配置文件缺少必要的键: {missing_keys}")
            return False
        
        print("[OK] 配置文件格式正确")
        print(f"  基础模型: {config['model'].get('base_model', 'N/A')}")
        print(f"  设备: {config['model'].get('device', 'N/A')}")
        print(f"  训练轮数: {config['training'].get('num_train_epochs', 'N/A')}")
        return True
    except Exception as e:
        print(f"[ERROR] 配置文件读取失败: {e}")
        return False


def test_scripts():
    """测试脚本参数解析"""
    print("\n" + "=" * 60)
    print("测试4: 脚本参数解析")
    print("=" * 60)
    
    scripts = [
        ("train_lora.py", ["--help"]),
        ("train_qlora.py", ["--help"]),
        ("inference.py", ["--help"]),
        ("evaluate.py", ["--help"]),
    ]
    
    all_passed = True
    for script, args in scripts:
        script_path = Path(script)
        if not script_path.exists():
            print(f"[SKIP] {script} 不存在")
            continue
        
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, script] + args,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=10
            )
            if result.returncode == 0:
                print(f"[OK] {script} 参数解析正常")
            else:
                print(f"[WARN] {script} 参数解析有问题")
                all_passed = False
        except Exception as e:
            print(f"[ERROR] {script} 测试失败: {e}")
            all_passed = False
    
    return all_passed


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("作业2：Code Alpaca LoRA/QLoRA 微调 - 完整测试")
    print("=" * 60)
    print()
    
    results = []
    
    # 测试1: 模块导入
    results.append(("模块导入", test_imports()))
    
    # 测试2: GPU 检查
    results.append(("GPU 检查", test_gpu()))
    
    # 测试3: 配置文件
    results.append(("配置文件", test_config()))
    
    # 测试4: 脚本参数解析
    results.append(("脚本参数解析", test_scripts()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"{status} {name}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n[SUCCESS] 所有测试通过！")
        return 0
    else:
        print(f"\n[WARN] {total - passed} 个测试未通过（可能是缺少依赖或数据）")
        print("       在 Docker 环境中运行应该可以解决这些问题")
        return 1


if __name__ == "__main__":
    sys.exit(main())

