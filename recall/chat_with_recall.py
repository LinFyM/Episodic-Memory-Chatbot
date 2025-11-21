import torch
import torch.nn.functional as F
import os
import numpy as np
from modelscope import AutoModelForCausalLM, AutoTokenizer
from model_utils import forward_backbone, ensure_last_hidden_state, build_causal_lm_output

class MemoryVectorDB:
    """记忆向量数据库，使用PyTorch实现（无需faiss）"""
    
    def __init__(self, embedding_dim=4096, device="cpu"):
        """初始化向量数据库"""
        self.embedding_dim = embedding_dim
        self.embeddings = None  # 存储所有向量的tensor
        self.texts = []
        self.device = device  # 添加设备属性
        
    def add_vectors(self, embeddings, texts=None):
        """添加向量到数据库"""
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        
        # 确保数据类型为bfloat16并移动到正确设备上
        embeddings = embeddings.to(dtype=torch.bfloat16, device=self.device)
        
        # 归一化向量用于余弦相似度
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = torch.cat([self.embeddings, embeddings], dim=0)
        
        if texts:
            self.texts.extend(texts)
        else:
            # 如果没有提供文本，使用占位符
            self.texts.extend([f"Memory_{i}" for i in range(len(self.texts), len(self.texts) + embeddings.shape[0])])
        
        print(f"向量数据库现有 {len(self.texts)} 条记忆")
    
    def search(self, query_embedding, top_k=5, debug=False):
        """搜索最相似的向量，增强版"""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        
        # 确保查询向量和存储向量在同一设备上且为相同数据类型
        if isinstance(query_embedding, np.ndarray):
            query_embedding = torch.from_numpy(query_embedding)
        
        original_device = query_embedding.device
        original_dtype = query_embedding.dtype
        
        if debug:
            print(f"\n[调试] 查询向量原始信息:")
            print(f"  - 设备: {original_device}")
            print(f"  - 数据类型: {original_dtype}")
            print(f"  - 维度: {query_embedding.shape}")
            print(f"  - 范数: {torch.norm(query_embedding).item():.4f}")
            print(f"  - 均值: {torch.mean(query_embedding).item():.4f}")
            print(f"  - 标准差: {torch.std(query_embedding).item():.4f}")
            print(f"  - 最大值: {torch.max(query_embedding).item():.4f}")
            print(f"  - 最小值: {torch.min(query_embedding).item():.4f}")
        
        # 移动查询向量到与存储向量相同的设备上
        query_embedding = query_embedding.to(dtype=torch.bfloat16, device=self.device)
        
        # 确保查询向量有正确的维度
        if query_embedding.dim() == 1:
            # 如果是单个向量 [embed_dim]，添加批次维度
            query_embedding = query_embedding.unsqueeze(0)  # [1, embed_dim]
        
        # 归一化查询向量
        query_embedding_normalized = F.normalize(query_embedding, p=2, dim=-1)
        
        if debug:
            print(f"\n[调试] 归一化后查询向量信息:")
            print(f"  - 范数: {torch.norm(query_embedding_normalized).item():.4f}")
            print(f"  - 均值: {torch.mean(query_embedding_normalized).item():.4f}")
            print(f"  - 标准差: {torch.std(query_embedding_normalized).item():.4f}")
        
        # 计算余弦相似度
        similarities = torch.matmul(query_embedding_normalized, self.embeddings.t())
        
        if debug:
            # 显示相似度分布信息
            sim_mean = torch.mean(similarities).item()
            sim_std = torch.std(similarities).item()
            sim_max = torch.max(similarities).item()
            sim_min = torch.min(similarities).item()
            print(f"\n[调试] 相似度分布:")
            print(f"  - 平均相似度: {sim_mean:.4f}")
            print(f"  - 标准差: {sim_std:.4f}")
            print(f"  - 最大相似度: {sim_max:.4f}")
            print(f"  - 最小相似度: {sim_min:.4f}")
            
            # 计算相似度直方图 - 修复 bfloat16 转 numpy 的问题
            sim_flat = similarities.flatten().cpu().to(torch.float32).numpy()  # 先转换为 float32
            hist_counts = np.histogram(sim_flat, bins=10, range=(float(sim_min), float(sim_max)))[0]
            hist_edges = np.linspace(float(sim_min), float(sim_max), 11)
            print(f"\n[调试] 相似度直方图:")
            for i in range(10):
                bar_len = int(hist_counts[i] / len(sim_flat) * 50)
                print(f"  {hist_edges[i]:.2f}-{hist_edges[i+1]:.2f}: {'#' * bar_len} ({hist_counts[i]})")
        
        # 获取top_k个最相似的结果
        top_k = min(top_k, len(self.embeddings))
        top_scores, top_indices = torch.topk(similarities, top_k, largest=True)
        
        # 处理维度，确保结果可迭代
        if top_scores.dim() == 1:
            top_scores = top_scores.unsqueeze(0)
            top_indices = top_indices.unsqueeze(0)
        
        results = []
        for i, (score, idx) in enumerate(zip(top_scores[0], top_indices[0])):
            memory_text = self.texts[idx.item()] if idx.item() < len(self.texts) else "Unknown memory"
            preview_text = memory_text[:100] + "..." if len(memory_text) > 100 else memory_text
            
            result = {
                'text': memory_text,
                'preview': preview_text,
                'embedding': self.embeddings[idx.item()].clone(),
                'score': float(score.item()),
                'index': int(idx.item())
            }
            results.append(result)
            
            if debug:
                print(f"\n[调试] 匹配结果 #{i+1}:")
                print(f"  - 相似度: {score.item():.4f}")
                print(f"  - 索引: {idx.item()}")
                print(f"  - 预览: {preview_text}")
        
        return results
    
    def load_from_pt(self, pt_file_path):
        """从.pt文件加载向量数据"""
        print(f"从 {pt_file_path} 加载记忆数据...")
        data = torch.load(pt_file_path, map_location='cpu')
        
        if isinstance(data, dict):
            if 'embeddings' in data and 'texts' in data:
                embeddings = data['embeddings']
                texts = data['texts']
            else:
                # 尝试推断键名
                embedding_keys = [k for k in data.keys() if 'embed' in k.lower()]
                text_keys = [k for k in data.keys() if 'text' in k.lower()]
                
                if embedding_keys and text_keys:
                    embeddings = data[embedding_keys[0]]
                    texts = data[text_keys[0]]
                else:
                    raise ValueError(f"无法从数据中识别嵌入向量和文本字段: {list(data.keys())}")
        else:
            # 假设是直接的嵌入向量
            embeddings = data
            texts = [f"Memory_{i}" for i in range(embeddings.shape[0])]
        
        self.add_vectors(embeddings, texts)
        print(f"成功加载 {len(texts)} 条记忆")


class MemoryRecallChat:
    """具有记忆回溯功能的聊天模型 - 优化版"""
    
    def __init__(self, model_name, memory_path=None, device=None):
        """初始化模型"""
        self.model_name = model_name
        
        # 设置设备
        if device is None:
            self.device = "auto"
        else:
            self.device = device
            
        print(f"🤖 初始化记忆回溯聊天模型...")
        print(f"   模型: {model_name}")
        print(f"   设备: {self.device}")
        
        # 初始化对话历史
        self.conversation_history = []
        
        # 加载模型和分词器
        self._load_model()
        
        # 检查特殊token
        self._check_special_tokens()
        
        # 加载记忆数据 - 确保使用正确的设备
        self.memory_db = MemoryVectorDB(device=self.actual_device)  # 传入模型的实际设备
        if memory_path:
            self.memory_db.load_from_pt(memory_path)
    
    def _load_model(self):
        """加载模型和分词器"""
        print("加载模型和分词器...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map=self.device,
            trust_remote_code=True
        )
        
        # 获取模型实际设备
        self.actual_device = next(self.model.parameters()).device
        print(f"模型已加载到设备: {self.actual_device}")

    def _forward_with_backbone(self, **forward_inputs):
        local_inputs = dict(forward_inputs)
        use_cache_flag = local_inputs.pop("use_cache", True)
        backbone_outputs = forward_backbone(
            self.model,
            use_cache=use_cache_flag,
            output_hidden_states=False,
            return_dict=True,
            **local_inputs,
        )
        outputs = build_causal_lm_output(self.model, backbone_outputs)
        outputs.last_hidden_state = ensure_last_hidden_state(backbone_outputs)
        return outputs
    
    def _check_special_tokens(self):
        """检查特殊token是否存在"""
        self.special_tokens = {
            'recall_start': '<recall>',
            'recall': '<|recall|>',  # 这个token可能仍然存在，保留
            'recall_end': '</recall>'
        }
        
        self.special_token_ids = {}
        missing_tokens = []
        
        for name, token in self.special_tokens.items():
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id == self.tokenizer.unk_token_id:
                missing_tokens.append(token)
            else:
                self.special_token_ids[name] = token_id
                print(f"找到特殊token: {token} (ID: {token_id})")
        
        if missing_tokens:
            raise ValueError(f"以下特殊token不存在: {missing_tokens}")
    
    def reset_conversation(self):
        """重置对话历史"""
        self.conversation_history = []
        print("对话历史已重置")
    
    def chat(self, user_message, system_prompt=None, max_new_tokens=2000, temperature=0.7, top_p=0.9, stream=True, force_recall=False):
        """优化的记忆回溯聊天 - 保留KV缓存，支持对话历史
        
        Args:
            user_message: 用户消息
            system_prompt: 系统提示词
            max_new_tokens: 最大生成token数
            temperature: 温度参数
            top_p: top-p采样参数
            stream: 是否流式输出
            force_recall: 是否强制第一个token为<|recall_start|>
        """
        
        # 1. 处理系统提示和对话历史
        # 如果是新对话（没有历史）且提供了系统提示，则添加系统提示
        if not self.conversation_history and system_prompt:
            self.conversation_history.append({"role": "system", "content": system_prompt})
        
        # 添加用户消息到对话历史
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # 2. 使用分词器的chat模板应用整个对话历史
        chat_text = self.tokenizer.apply_chat_template(
            self.conversation_history,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 3. 编码输入
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(self.actual_device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        print(f"\n🧠 开始生成回答..." + (" (强制以回忆开始)" if force_recall else ""))
        
        # 4. 生成结果记录
        generated_ids = []
        past_key_values = None
        in_recall_mode = False
        
        # 5. 生成循环
        for i in range(max_new_tokens):
            # 确定当前处理的token
            current_input = input_ids[:, -1:] if past_key_values is not None else input_ids
            
            # 模型前向传播
            with torch.no_grad():
                outputs = self._forward_with_backbone(
                    input_ids=current_input,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            
            # 更新KV缓存
            past_key_values = outputs.past_key_values
            
            # 获取预测结果
            logits = outputs.logits[:, -1, :]
            
            # 如果是第一个token且启用了强制回忆，则将<|recall_start|>的概率设为最高
            if len(generated_ids) == 0 and force_recall:
                recall_start_id = self.special_token_ids.get('recall_start')
                # 创建一个新的logits tensor，将所有值设为一个非常小的值
                new_logits = torch.full_like(logits, -10000.0)
                # 将<|recall_start|>的logit设为一个很大的值
                new_logits[0, recall_start_id] = 10000.0
                logits = new_logits
                print("强制生成<|recall_start|>作为第一个token")
            
            # 根据是否在回忆模式选择解码策略
            if in_recall_mode:
                # 回忆模式使用贪婪解码
                next_token_id = torch.argmax(logits, dim=-1).item()
            else:
                # 正常模式使用温度采样
                if temperature > 0:
                    logits = logits / temperature
                    
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # 移除低概率token
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = -float('inf')
                    
                    # 采样
                    probs = F.softmax(logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).item()
                else:
                    # 贪婪解码
                    next_token_id = torch.argmax(logits, dim=-1).item()
            
            # 判断是否生成了<|recall_start|>
            if next_token_id == self.special_token_ids.get('recall_start') and not in_recall_mode:
                # 进入记忆回溯模式
                in_recall_mode = True
                
                # 1. 先将<|recall_start|>添加到生成结果
                generated_ids.append(next_token_id)
                
                # 流式输出当前token
                if stream:
                    token_text = self.tokenizer.decode([next_token_id])
                    print(token_text, end="", flush=True)
                
                # 2. 将<|recall_start|>输入模型获取隐藏状态
                recall_start_input = torch.tensor([[next_token_id]], device=self.actual_device)
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones(1, 1, device=self.actual_device)
                ], dim=1)
                
                # 使用KV缓存进行有效计算
                with torch.no_grad():
                    recall_outputs = self._forward_with_backbone(
                        input_ids=recall_start_input,
                        attention_mask=attention_mask[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                
                # 3. 获取<|recall_start|>的隐藏状态作为查询向量
                query_vector = recall_outputs.last_hidden_state[0, -1]
                
                # 4. 更新KV缓存
                past_key_values = recall_outputs.past_key_values
                
                # 5. 使用查询向量在记忆数据库中检索
                if len(self.memory_db.texts) > 0:
                    # 启用调试模式，返回更多结果
                    query_vector = query_vector.to(dtype=torch.bfloat16)
                    
                    print("\n[系统: 开始记忆检索...]")
                    search_results = self.memory_db.search(query_vector, top_k=5, debug=True)
                    
                    if search_results:
                        print("\n[系统: 找到以下记忆匹配结果]")
                        for i, result in enumerate(search_results):
                            print(f"  #{i+1} 相似度: {result['score']:.4f} | {result['preview']}")
                        
                        # 使用最匹配的结果继续
                        result = search_results[0]
                        memory_embedding = result['embedding'].to(self.actual_device)
                        memory_text = result['text']
                        
                        print(f"\n[系统: 使用最佳匹配 (相似度: {result['score']:.4f})]")
                        
                        # 6. 创建嵌入层引用
                        embedding_layer = self.model.get_input_embeddings()
                        
                        # 7. 将记忆向量直接输入模型（跳过嵌入层）
                        memory_embed = memory_embedding.unsqueeze(0).unsqueeze(0)
                        
                        # 确保数据类型匹配
                        memory_dtype = next(self.model.parameters()).dtype
                        memory_embed = memory_embed.to(memory_dtype)
                        
                        # 使用KV缓存进行前向传播 - 贪婪解码
                        with torch.no_grad():
                            memory_outputs = self._forward_with_backbone(
                                inputs_embeds=memory_embed,
                                attention_mask=torch.ones(1, 1, device=self.actual_device),
                                past_key_values=past_key_values,
                                use_cache=True,
                            )
                        
                        # 8. 更新KV缓存
                        past_key_values = memory_outputs.past_key_values
                        
                        # 9. 更新注意力掩码
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.ones(1, 1, device=self.actual_device)
                        ], dim=1)
                        
                        # 10. 输入<|recall|> token
                        recall_token_id = self.special_token_ids['recall']
                        recall_input = torch.tensor([[recall_token_id]], device=self.actual_device)
                        
                        with torch.no_grad():
                            recall_outputs = self._forward_with_backbone(
                                input_ids=recall_input,
                                attention_mask=torch.ones(1, 1, device=self.actual_device),
                                past_key_values=past_key_values,
                                use_cache=True,
                            )
                        
                        # 11. 更新KV缓存和注意力掩码
                        past_key_values = recall_outputs.past_key_values
                        attention_mask = torch.cat([
                            attention_mask,
                            torch.ones(1, 1, device=self.actual_device)
                        ], dim=1)
                        
                        # 12. 添加<|recall|> token到生成结果
                        generated_ids.append(recall_token_id)
                        
                        # 流式输出当前token
                        if stream:
                            token_text = self.tokenizer.decode([recall_token_id])
                            print(token_text, end="", flush=True)
                        
                        # 准备下一次迭代的输入
                        input_ids = recall_input
                        continue
                
                # 如果没有找到记忆或记忆数据库为空
                print("\n[系统: 没有找到相关记忆]")
                input_ids = recall_start_input
                continue
            
            # 处理<|recall_end|>
            elif next_token_id == self.special_token_ids.get('recall_end') and in_recall_mode:
                in_recall_mode = False
            
            # 添加token到生成结果
            generated_ids.append(next_token_id)
            
            # 流式输出当前token
            if stream:
                token_text = self.tokenizer.decode([next_token_id])
                print(token_text, end="", flush=True)
            
            # 检查是否生成了EOS
            if next_token_id == self.tokenizer.eos_token_id:
                break
            
            # 准备下一次迭代的输入
            input_ids = torch.tensor([[next_token_id]], device=self.actual_device)
            
            # 更新注意力掩码
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(1, 1, device=self.actual_device)
            ], dim=1)
        
        # 解码最终结果
        generated_text = self.tokenizer.decode(generated_ids)
        print("\n")
        
        # 将助手回复添加到对话历史
        self.conversation_history.append({"role": "assistant", "content": generated_text})
        
        return generated_text

def main():
    """主函数"""
    print("🧠 优化的记忆回溯聊天系统")
    print("=" * 50)
    
    # 模型和记忆路径
    MODEL_PATH = "./training_workspace/model_cycle_2"
    MEMORY_PATH = "./training_workspace/embeddings/text_embeddings.pt"
    
    # 检查文件
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        return
    
    if not os.path.exists(MEMORY_PATH):
        print(f"❌ 记忆数据文件不存在: {MEMORY_PATH}")
        return
    
    # 初始化聊天模型
    chat_model = MemoryRecallChat(
        model_name=MODEL_PATH,
        memory_path=MEMORY_PATH,
        device="cuda:0"  # 或指定设备，如"cuda:0"
    )
    
    print("\n🤖 记忆回溯聊天已准备就绪！")
    print("输入 'exit' 退出聊天, 输入 'reset' 重置对话历史")
    print("输入 'force-recall' 强制模型以回忆模式开始回答")
    print("=" * 50)
    
    # 设置默认系统提示词
    default_system_prompt = """你是一个有记忆能力的AI助手。你需要根据回忆出的内容回答问题。"""
    
    # 询问用户是否修改系统提示词
    print(f"\n当前系统提示词:\n{default_system_prompt}")
    change_prompt = input("\n是否修改系统提示词? (y/n): ").strip().lower()
    
    if change_prompt == 'y':
        system_prompt = input("请输入新的系统提示词:\n")
    else:
        system_prompt = default_system_prompt
    
    # 默认不强制回忆
    force_recall_mode = True
    
    # 聊天循环
    while True:
        user_input = input("\n用户: ")
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
        elif user_input.lower() == 'reset':
            chat_model.reset_conversation()
            continue
        elif user_input.lower() == 'force-recall':
            force_recall_mode = not force_recall_mode
            print(f"强制回忆模式: {'开启' if force_recall_mode else '关闭'}")
            continue
        
        try:
            # 首次对话传入系统提示词，后续对话不需要
            if len(chat_model.conversation_history) == 0:
                response = chat_model.chat(
                    user_message=user_input,
                    system_prompt=system_prompt,
                    max_new_tokens=500,
                    temperature=0.7,
                    top_p=0.9,
                    stream=True,
                    force_recall=force_recall_mode
                )
            else:
                response = chat_model.chat(
                    user_message=user_input,
                    max_new_tokens=500,
                    temperature=0.7,
                    top_p=0.9,
                    stream=True,
                    force_recall=force_recall_mode
                )
            
        except Exception as e:
            print(f"❌ 出错: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()