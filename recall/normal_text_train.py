import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import json
from peft import LoraConfig, get_peft_model, TaskType
from modelscope import AutoModelForCausalLM, AutoTokenizer

class NormalTextDataset(Dataset):
    """普通文本训练数据集"""
    
    def __init__(self, texts, tokenizer, max_length=3000):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # 编码文本
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # 标签就是input_ids，向右移位在损失计算时处理
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

class NormalTextTrainer:
    """普通文本训练器 - 支持多GPU配置"""
    
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.specified_device = device
        
        # 设备处理逻辑 - 与其他训练器保持一致
        if device is None:
            self.use_auto_device = False
            self.primary_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.multi_gpu_list = None
        elif isinstance(device, list):
            # 处理GPU列表
            if len(device) > 0:
                self.use_auto_device = False
                self.primary_device = torch.device(device[0])
                self.multi_gpu_list = device
                print(f"   使用多GPU列表: {device}，主设备: {device[0]}")
            else:
                self.use_auto_device = True
                self.primary_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.multi_gpu_list = None
        elif isinstance(device, str):
            if device == "auto":
                self.use_auto_device = True
                self.primary_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.multi_gpu_list = None
            elif device.startswith('cuda:'):
                # 单GPU配置 - 检查CUDA_VISIBLE_DEVICES
                cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                if cuda_visible and cuda_visible.strip():
                    # CUDA_VISIBLE_DEVICES已设置，使用重新映射后的设备
                    self.primary_device = torch.device("cuda:0")
                    print(f"   CUDA_VISIBLE_DEVICES={cuda_visible}，使用重新映射设备 cuda:0（对应物理GPU {device}）")
                else:
                    # 未设置CUDA_VISIBLE_DEVICES，直接使用物理设备
                    self.primary_device = torch.device(device)
                    print(f"   使用设备 {device}")
                self.use_auto_device = False
                self.multi_gpu_list = None
            else:
                self.use_auto_device = False
                self.primary_device = torch.device(device)
                self.multi_gpu_list = None
        else:
            self.use_auto_device = False
            self.primary_device = device
            self.multi_gpu_list = None
            
        print(f"🤖 初始化普通文本训练器...")
        print(f"   模型: {model_name}")
        print(f"   设备: {device}")
        
        self._load_model()
        
        # 获取实际设备信息
        first_param = next(self.model.parameters())
        self.actual_device = first_param.device
        print(f"   实际模型设备: {self.actual_device}")
        
        self._setup_lora()
        
    def _load_model(self):
        """加载模型和分词器 - 支持多GPU配置"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        try:
            # 根据设备配置选择device_map
            if self.use_auto_device:
                device_map = "auto"
                print("   使用自动设备分配")
            elif hasattr(self, 'multi_gpu_list') and self.multi_gpu_list:
                # 多GPU配置
                device_map = "auto"
                print(f"   使用多GPU自动分配: {self.multi_gpu_list}")
                
                # 设置环境变量限制可见GPU
                import os
                if 'CUDA_VISIBLE_DEVICES' not in os.environ:
                    gpu_indices = [gpu.split(':')[1] for gpu in self.multi_gpu_list if gpu.startswith('cuda:')]
                    if gpu_indices:
                        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_indices)
                        print(f"   设置CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
            elif isinstance(self.specified_device, str) and self.specified_device.startswith('cuda:'):
                # 单GPU指定
                device_index = int(self.specified_device.split(':')[1])
                device_map = {"": device_index}
                print(f"   使用指定单GPU: {self.specified_device}")
            elif self.specified_device == "cpu":
                device_map = {"": "cpu"}
                print(f"   使用CPU设备")
            else:
                # 默认情况
                if hasattr(self, 'primary_device') and self.primary_device.type == 'cuda':
                    device_map = {"": self.primary_device.index}
                else:
                    device_map = "auto"
                print(f"   使用默认设备映射: {device_map}")
            
            print(f"   实际使用设备映射: {device_map}")
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map=device_map,
                trust_remote_code=True
            )
            
            # 获取实际设备信息
            first_param = next(self.model.parameters())
            model_dtype = first_param.dtype
            model_device = first_param.device
            
            print(f"✅ 模型加载成功")
            print(f"   实际设备: {model_device}")
            print(f"   数据类型: {model_dtype}")
            
            # 显示设备映射信息
            if hasattr(self.model, 'hf_device_map'):
                print(f"   设备映射详情: {self.model.hf_device_map}")
                
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            # 回退策略
            print("🔄 尝试回退到单GPU模式...")
            
            try:
                # 确定回退设备
                if hasattr(self, 'multi_gpu_list') and self.multi_gpu_list:
                    fallback_device = self.multi_gpu_list[0]
                elif isinstance(self.specified_device, str) and self.specified_device.startswith('cuda:'):
                    fallback_device = self.specified_device
                else:
                    fallback_device = 'cuda:0'
                
                # 提取设备索引
                if fallback_device.startswith('cuda:'):
                    device_index = int(fallback_device.split(':')[1])
                    device_map = {"": device_index}
                else:
                    device_map = {"": "cpu"}
                
                print(f"   回退设备映射: {device_map}")
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    device_map=device_map,
                    trust_remote_code=True
                )
                
                first_param = next(self.model.parameters())
                print(f"✅ 使用回退设备加载成功: {first_param.device}")
                
            except Exception as fallback_error:
                print(f"❌ 回退加载也失败: {fallback_error}")
                raise RuntimeError(f"模型加载完全失败: 原错误={e}, 回退错误={fallback_error}")
        
    def _setup_lora(self):
        """设置LoRA"""
        print("⚡ 配置LoRA...")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        print("✅ LoRA配置完成")
        
    def create_dataloader(self, texts, batch_size=4, shuffle=True):
        """创建数据加载器"""
        dataset = NormalTextDataset(texts, self.tokenizer)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
    def train_epoch(self, dataloader, optimizer):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc="普通文本训练")
        
        for batch in progress_bar:
            # 使用实际设备
            input_ids = batch['input_ids'].to(self.actual_device)
            attention_mask = batch['attention_mask'].to(self.actual_device)
            labels = batch['labels'].to(self.actual_device)
            
            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
            
        return total_loss / len(dataloader)
        
    def merge_and_save_model(self, save_path):
        """合并LoRA权重并保存模型"""
        print("🔄 合并LoRA权重...")
        
        merged_model = self.model.merge_and_unload()
        
        os.makedirs(save_path, exist_ok=True)
        merged_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        print(f"✅ 模型已保存到: {save_path}")
        return merged_model
        
    def train(self, texts, num_epochs=5, batch_size=4, learning_rate=1e-4, save_path=None):
        """完整训练流程"""
        print(f"\n🚀 开始普通文本训练")
        print(f"   文本数量: {len(texts)}")
        print(f"   训练轮数: {num_epochs}")
        print(f"   批次大小: {batch_size}")
        print(f"   学习率: {learning_rate}")
        print(f"   实际设备: {self.actual_device}")
        
        # 创建数据加载器
        train_loader = self.create_dataloader(texts, batch_size, True)
        
        # 优化器
        optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # 训练循环
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            avg_loss = self.train_epoch(train_loader, optimizer)
            print(f"   平均损失: {avg_loss:.6f}")
            
        # 保存模型
        if save_path:
            final_model = self.merge_and_save_model(save_path)
            
        print("🎉 普通文本训练完成!")
        return avg_loss

def main():
    """测试函数"""
    print("🚀 测试普通文本训练器...")
    
    # 测试多GPU配置
    trainer = NormalTextTrainer(
        "./Qwen2.5-7B-Instruct", 
        device=['cuda:1', 'cuda:2', 'cuda:3']  # 多GPU配置
    )
    
    # 测试数据
    test_texts = [
        "这是一个测试文本，用于验证普通文本训练功能。",
        "多GPU训练可以加速大模型的训练过程。",
        "LoRA是一种参数高效的微调方法。"
    ]
    
    # 开始训练
    trainer.train(
        texts=test_texts,
        num_epochs=2,
        batch_size=2,
        learning_rate=1e-4,
        save_path="./test_normal_trained"
    )
    
    print("✅ 测试完成!")

if __name__ == "__main__":
    main()