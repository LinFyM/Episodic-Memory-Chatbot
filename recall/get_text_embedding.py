import torch
import os
from tqdm import tqdm
from datetime import datetime
from create_text_dataset import load_text_dataset, get_dataset_paths
from modelscope import AutoModelForCausalLM, AutoTokenizer
from recall.model_utils import forward_backbone, ensure_last_hidden_state, build_causal_lm_output

def extract_last_token_embedding(model, tokenizer, text, device):
    """提取文本最后一个token的嵌入向量"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        backbone_outputs = forward_backbone(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        last_hidden_states = ensure_last_hidden_state(backbone_outputs)
        last_token_embedding = last_hidden_states[0, -1, :]
        
        # 获取预测token（只为这个操作转换类型）
        causal_outputs = build_causal_lm_output(model, backbone_outputs)
        logits = causal_outputs.logits[0, -1, :]
        predicted_token_id = torch.argmax(logits.float()).item()
        
    return last_token_embedding, predicted_token_id

class TextEmbeddingExtractor:
    def __init__(self, model_name="./Qwen2.5-7B-Instruct", device=None, verbose=False):
        """
        初始化文本嵌入提取器
        
        Args:
            model_name: 模型名称或路径
            device: 设备配置，支持：
                   - 字符串: 'cuda:0', 'auto', 'cpu'
                   - 列表: ['cuda:0', 'cuda:1', ...]
                   - None: 使用默认设备cuda:5
        """
        self.model_name = model_name
        self.specified_device = device
        self.model = None
        self.tokenizer = None
        self.device = None
        self.verbose = verbose
        
        # 处理多种设备配置
        if device is None:
            # 保持原有默认行为
            self.use_auto_device = False
            self.primary_device = torch.device('cuda:5')
            self.multi_gpu_list = None
        elif isinstance(device, list):
            # 处理GPU列表，使用第一个作为主设备
            if len(device) > 0:
                self.use_auto_device = False
                self.primary_device = torch.device(device[0])  # 使用列表中第一个GPU
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
            else:
                self.use_auto_device = False
                self.primary_device = torch.device(device)
                self.multi_gpu_list = None
        else:
            self.use_auto_device = False
            self.primary_device = device
            self.multi_gpu_list = None
        
        self._load_model()
    
    def _load_model(self):
        """加载模型和分词器 - 支持多GPU配置"""
        if self.verbose:
            print("🤖 正在加载模型...")
            print(f"🎯 指定设备: {self.specified_device}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        
        # 根据设备配置选择device_map
        try:
            if self.use_auto_device:
                device_map = "auto"
                if self.verbose:
                    print("   使用自动设备分配")
            elif hasattr(self, 'multi_gpu_list') and self.multi_gpu_list:
                # 为多GPU创建设备映射
                device_map = "auto"  # 让transformers自动分配到可用GPU
                if self.verbose:
                    print(f"   使用多GPU自动分配: {self.multi_gpu_list}")
                
                # 可选：设置环境变量限制可见GPU
                import os
                if 'CUDA_VISIBLE_DEVICES' not in os.environ:
                    gpu_indices = [gpu.split(':')[1] for gpu in self.multi_gpu_list if gpu.startswith('cuda:')]
                    if gpu_indices:
                        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_indices)
                        if self.verbose:
                            print(f"   设置CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
                        
            elif isinstance(self.specified_device, str) and self.specified_device.startswith('cuda:'):
                # 单GPU指定
                device_index = int(self.specified_device.split(':')[1])
                device_map = {"": device_index}
                if self.verbose:
                    print(f"   使用指定单GPU: {self.specified_device}")
            elif self.specified_device == "cpu":
                # CPU设备
                device_map = {"": "cpu"}
                if self.verbose:
                    print(f"   使用CPU设备")
            else:
                # 默认情况或其他设备字符串
                if hasattr(self, 'primary_device') and self.primary_device.type == 'cuda':
                    device_map = {"": self.primary_device.index}
                else:
                    device_map = "cuda:5"  # 保持原有默认值
                if self.verbose:
                    print(f"   使用默认设备映射: {device_map}")
            
            if self.verbose:
                print(f"   实际使用设备映射: {device_map}")
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map=device_map,
                trust_remote_code=True
            )
            
            # 获取实际设备信息
            self.device = next(self.model.parameters()).device
            model_dtype = next(self.model.parameters()).dtype
            
            if self.verbose:
                print(f"✅ 模型加载成功")
                print(f"   实际设备: {self.device}")
                print(f"   数据类型: {model_dtype}")
            
            # 显示设备映射信息（如果可用）
            if hasattr(self.model, 'hf_device_map'):
                if self.verbose:
                    print(f"   设备映射详情: {self.model.hf_device_map}")
                
        except Exception as e:
            if self.verbose:
                print(f"❌ 模型加载失败: {e}")
                print("🔄 尝试回退到单GPU模式...")
            
            try:
                # 确定回退设备
                if hasattr(self, 'multi_gpu_list') and self.multi_gpu_list:
                    fallback_device = self.multi_gpu_list[0]
                elif isinstance(self.specified_device, str) and self.specified_device.startswith('cuda:'):
                    fallback_device = self.specified_device
                else:
                    fallback_device = 'cuda:5'  # 原有默认值
                
                # 提取设备索引
                if fallback_device.startswith('cuda:'):
                    device_index = int(fallback_device.split(':')[1])
                    device_map = {"": device_index}
                else:
                    device_map = {"": "cpu"}
                
                if self.verbose:
                    print(f"   回退设备映射: {device_map}")
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype="auto",
                    device_map=device_map,
                    trust_remote_code=True
                )
                
                self.device = next(self.model.parameters()).device
                if self.verbose:
                    print(f"✅ 使用回退设备加载成功: {self.device}")
                
            except Exception as fallback_error:
                if self.verbose:
                    print(f"❌ 回退加载也失败: {fallback_error}")
                raise RuntimeError(f"模型加载完全失败: 原错误={e}, 回退错误={fallback_error}")
    
    def create_prompt(self, text):
        """创建提示词"""
        return f'请用一个Token表征"{text}"这句话：'
    
    def extract_embeddings(self, texts):
        """批量提取嵌入向量"""
        if not texts:
            raise ValueError("没有成功提取任何嵌入向量")
        
        if self.verbose:
            print(f"📊 开始处理 {len(texts)} 条文本...")
        
        embeddings_list = []
        texts_list = []
        prompts_list = []
        tokens_list = []
        
        with torch.no_grad():
            for i, text in enumerate(tqdm(texts, desc="提取嵌入向量")):
                try:
                    # 创建提示词
                    prompt = self.create_prompt(text)
                    
                    # 提取嵌入向量
                    embedding, predicted_token_id = extract_last_token_embedding(
                        self.model, self.tokenizer, prompt, self.device
                    )
                    
                    # 解码预测token
                    predicted_token = self.tokenizer.decode([predicted_token_id])
                    
                    # 收集结果
                    embeddings_list.append(embedding)
                    texts_list.append(text)
                    prompts_list.append(prompt)
                    tokens_list.append(predicted_token)
                        
                except Exception as e:
                    print(f"⚠️ 处理文本 {i} 时出错: {e}")
                    continue
        
        if not embeddings_list:
            raise ValueError("没有成功提取任何嵌入向量")
        
        # 堆叠所有embeddings
        embeddings_tensor = torch.stack(embeddings_list)
        
        if self.verbose:
            print(f"✅ 成功提取 {len(embeddings_list)} 个嵌入向量")
            print(f"   向量形状: {embeddings_tensor.shape}")
            print(f"   数据类型: {embeddings_tensor.dtype}")
        
        # 返回符合text_memory_train.py期望的格式
        data = {
            'texts': texts_list,        # 必须有这个字段
            'embeddings': embeddings_tensor, # 必须有这个字段
            'prompts': prompts_list,
            'predicted_tokens': tokens_list,
            'metadata': {
                'model_name': self.model_name,
                'embedding_dim': embeddings_tensor.shape[-1],
                'num_samples': len(texts_list),
                'dtype': str(embeddings_tensor.dtype),
                'created_date': datetime.now().isoformat(),
                'device': str(self.device),
                'device_config': str(self.specified_device)
            }
        }
        
        return data
    
    def save_embeddings(self, data, save_dir="./embeddings", filename="text_embeddings.pt"):
        """保存embeddings数据 - 确保格式匹配"""
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        
        # 确保保存的格式与text_memory_train.py期望的一致
        save_data = {
            'texts': data['texts'],           # text_memory_train.py需要这个字段
            'embeddings': data['embeddings'], # text_memory_train.py需要这个字段
            'prompts': data.get('prompts', []),
            'predicted_tokens': data.get('predicted_tokens', []),
            'metadata': data.get('metadata', {}),
            'created_date': datetime.now().isoformat()
        }
        
        torch.save(save_data, save_path)
        if self.verbose:
            print(f"💾 Embeddings已保存到: {save_path}")
            print(f"   文本数量: {len(save_data['texts'])}")
            print(f"   向量形状: {save_data['embeddings'].shape}")
        
        return save_path
    
    def load_embeddings(self, file_path=None):
        """加载嵌入向量"""
        if file_path is None:
            paths = get_dataset_paths()
            file_path = os.path.join(paths['embeddings_dir'], 'text_embeddings.pt')
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if self.verbose:
            print(f"📖 加载嵌入向量: {file_path}")
        data = torch.load(file_path, map_location='cpu')  # 先加载到CPU
        
        if self.verbose:
            print(f"✅ 加载成功:")
            print(f"   样本数量: {data['metadata']['num_samples']}")
            print(f"   嵌入维度: {data['metadata']['embedding_dim']}")
            print(f"   数据类型: {data['metadata']['dtype']}")
        
        return data

def main():
    """主函数"""
    print("🚀 开始文本嵌入向量提取...")
    
    # 1. 加载文本数据
    try:
        texts = load_text_dataset()
        print(f"📖 加载了 {len(texts)} 条文本")
    except FileNotFoundError:
        print("❌ 请先运行 create_text_dataset.py 创建数据集")
        return
    
    # 2. 初始化提取器 - 可以测试不同的设备配置
    # extractor = TextEmbeddingExtractor()  # 默认设备
    # extractor = TextEmbeddingExtractor(device='auto')  # 自动分配
    extractor = TextEmbeddingExtractor(device=['cuda:5', 'cuda:6', 'cuda:7'])  # 多GPU
    
    # 3. 提取嵌入向量
    try:
        data = extractor.extract_embeddings(texts)
    except Exception as e:
        print(f"❌ 提取失败: {e}")
        return
    
    # 4. 保存数据
    save_path = extractor.save_embeddings(data)
    
    # 5. 验证加载
    print("\n🔬 验证加载...")
    try:
        loaded_data = extractor.load_embeddings()
        
        # 检查数据一致性
        original_shape = data['embeddings'].shape
        loaded_shape = loaded_data['embeddings'].shape
        
        print(f"   原始形状: {original_shape}")
        print(f"   加载形状: {loaded_shape}")
        print(f"   数据一致: {original_shape == loaded_shape}")
        
        # 显示第一个样本
        print(f"\n📝 第一个样本:")
        print(f"   文本: {loaded_data['texts'][0][:100]}...")
        print(f"   预测token: '{loaded_data['predicted_tokens'][0]}'")
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
    
    print("\n🎉 完成!")

if __name__ == "__main__":
    main()