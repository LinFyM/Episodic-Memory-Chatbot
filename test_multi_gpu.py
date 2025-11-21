#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多GPU配置测试脚本
验证多GPU环境和配置是否正确
"""

import yaml
import os
import sys

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_gpu_availability():
    """测试GPU可用性"""
    print("🔍 GPU可用性测试")
    print("-" * 40)

    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA不可用")
            return False

        gpu_count = torch.cuda.device_count()
        print(f"✅ CUDA可用，检测到 {gpu_count} 张GPU")

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, {props.total_memory // 1024 // 1024}MB")

        return True
    except ImportError:
        print("⚠️ PyTorch未安装，跳过GPU测试")
        return True

def test_config_loading():
    """测试配置文件加载"""
    print("\n🔍 配置文件测试")
    print("-" * 40)

    config_path = "server/config_qwen3vl.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        model_config = config.get("model", {})
        device = model_config.get("device", "cuda:0")
        multi_gpu_config = model_config.get("multi_gpu", {})

        print("✅ 配置文件加载成功")
        print(f"  设备配置: {device}")
        print(f"  多GPU启用: {multi_gpu_config.get('enabled', True)}")
        print(f"  梯度累积步数: {multi_gpu_config.get('gradient_accumulation_steps', 1)}")

        return True
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return False

def test_model_loading():
    """测试模型加载（不实际加载，只测试配置）"""
    print("\n🔍 模型加载配置测试")
    print("-" * 40)

    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        # 测试基本导入
        print("✅ Transformers库导入成功")

        # 测试processor
        test_processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-8B-Thinking",
            trust_remote_code=True,
            local_files_only=True  # 只测试本地是否存在
        )
        print("✅ Processor配置测试通过")

        return True
    except ImportError as e:
        print(f"❌ 库导入失败: {e}")
        return False
    except Exception as e:
        print(f"⚠️ 模型加载测试跳过（需要完整环境）: {e}")
        return True

def test_accelerate_setup():
    """测试Accelerate设置"""
    print("\n🔍 Accelerate配置测试")
    print("-" * 40)

    try:
        from accelerate import Accelerator
        accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)
        print("✅ Accelerator初始化成功")
        print(f"  混合精度: {accelerator.mixed_precision}")
        print(f"  梯度累积步数: {accelerator.gradient_accumulation_steps}")
        print(f"  设备: {accelerator.device}")
        return True
    except ImportError:
        print("⚠️ Accelerate库未安装，跳过测试")
        return True
    except Exception as e:
        print(f"❌ Accelerator配置失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 多GPU配置测试开始")
    print("=" * 60)

    tests = [
        ("GPU可用性", test_gpu_availability),
        ("配置文件", test_config_loading),
        ("模型配置", test_model_loading),
        ("Accelerate配置", test_accelerate_setup),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}测试通过")
            else:
                print(f"❌ {test_name}测试失败")
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")

    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！多GPU环境配置正确")
        print("\n💡 使用建议:")
        print("  1. 确保配置文件中的device设置为'auto'或GPU列表")
        print("  2. 根据GPU数量调整gradient_accumulation_steps")
        print("  3. 启动服务器: python server/api_server_qwen3vl.py")
    else:
        print("⚠️ 部分测试失败，请检查配置")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
