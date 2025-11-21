import os
import random
import torch
import json
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any
import torch.distributed as dist

# 导入各个训练组件
from add_special_tokens_wrapper import SpecialTokensManager
from get_text_embedding import TextEmbeddingExtractor
from text_embedding_train import RecallMemoryTrainer
from text_memory_train import EnhancedTextMemoryTrainer
from normal_text_train import NormalTextTrainer

class IntegratedTrainingPipeline:
    """集成训练流水线 - 自动化完成记忆训练全流程"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练流水线
        
        Args:
            config: 配置字典，包含所有训练参数
        """
        self.config = config
        self.device = config.get('device', 'cuda:0')
        self.ddp_enabled = False
        self.local_rank = None
        # 若由 torchrun 启动，启用DDP
        if 'LOCAL_RANK' in os.environ:
            self.local_rank = int(os.environ['LOCAL_RANK'])
            os.environ.setdefault('RANK', os.environ.get('RANK', '0'))
            os.environ.setdefault('WORLD_SIZE', os.environ.get('WORLD_SIZE', '1'))
            torch.cuda.set_device(self.local_rank)
            # 可能被多处初始化，做幂等保护
            if not dist.is_available() or (dist.is_available() and not dist.is_initialized()):
                dist.init_process_group(backend='nccl', timeout=timedelta(minutes=60))
            self.ddp_enabled = True
            # 在DDP下固定每进程单卡
            self.device = f"cuda:{self.local_rank}"
            self.config['device'] = self.device
        
        # 创建工作目录
        self.work_dir = config.get('work_dir', './training_workspace')
        if self.is_main_process():
            os.makedirs(self.work_dir, exist_ok=True)
        
        # 保存配置
        self._save_config()
        
        if self.is_main_process():
            print("🚀 集成训练流水线初始化完成")
            print(f"   工作目录: {self.work_dir}")
            print(f"   设备: {self.device}")
    
    def is_main_process(self) -> bool:
        return (not self.ddp_enabled) or (dist.get_rank() == 0)
        
    def _save_config(self):
        """保存配置到文件"""
        config_path = os.path.join(self.work_dir, 'training_config.json')
        if self.is_main_process():
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"📝 配置已保存到: {config_path}")
        
    def load_and_split_dataset(self, dataset_path: str) -> Tuple[List[str], List[str]]:
        """
        加载数据集并按比例分割 - 支持JSON和CSV格式
        """
        if self.is_main_process():
            print(f"📖 加载数据集: {dataset_path}")
        
        # 检查文件是否存在
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
        
        # 根据文件格式加载数据
        texts = []
        
        if dataset_path.endswith('.csv'):
            # CSV格式支持 - 只读取text1并清洗
            import pandas as pd
            if self.is_main_process():
                print("📊 检测到CSV格式，正在解析...")
            
            try:
                # 读取CSV文件
                df = pd.read_csv(dataset_path)
                if self.is_main_process():
                    print(f"   CSV文件列名: {list(df.columns)}")
                    print(f"   CSV文件行数: {len(df)}")
                
                # 检查是否有text1列
                if 'text1' not in df.columns:
                    raise ValueError(f"CSV文件中没有找到'text1'列，现有列: {list(df.columns)}")
                
                print("   只读取text1列，清洗[SEP]前的答案部分...")
                
                for _, row in df.iterrows():
                    text1 = str(row['text1']).strip()
                    
                    # 处理text1：去掉[SEP]前的答案，只保留后面的上下文
                    if '[SEP]' in text1:
                        # 分割并只取[SEP]后面的部分
                        parts = text1.split('[SEP]', 1)
                        if len(parts) == 2:
                            context = parts[1].strip()  # 只要[SEP]后面的上下文
                            if context:  # 确保上下文不为空
                                texts.append(context)
                    else:
                        # 如果没有[SEP]，直接使用整个text1
                        if text1:
                            texts.append(text1)
                            
            except Exception as e:
                print(f"❌ CSV解析失败: {e}")
                # 尝试手动解析
                if self.is_main_process():
                    print("🔄 尝试手动解析CSV...")
                texts = self._manual_parse_csv_text1_only(dataset_path)
                
        elif dataset_path.endswith('.txt'):
            # 原有的TXT格式支持
            with open(dataset_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
                
        elif dataset_path.endswith('.json'):
            # 原有的JSON格式支持
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    texts = data
                elif isinstance(data, dict):
                    if 'data' in data:
                        data_items = data['data']
                        if isinstance(data_items, list):
                            texts = []
                            for item in data_items:
                                if isinstance(item, dict) and 'text' in item:
                                    texts.append(item['text'])
                                elif isinstance(item, str):
                                    texts.append(item)
                    elif 'texts' in data:
                        texts = data['texts']
                    else:
                        texts = list(data.values())[0] if data else []
                else:
                    raise ValueError(f"JSON文件格式不正确")
        else:
            raise ValueError(f"不支持的文件格式: {dataset_path}")
        
        if self.is_main_process():
            print(f"   成功解析文本数量: {len(texts)}")
        
        # 数据质量检查和过滤
        original_count = len(texts)
        texts = [text for text in texts if text and len(text.strip()) > 20]  # 最少20个字符
        filtered_count = len(texts)
        
        if self.is_main_process():
            print(f"   过滤后文本数量: {filtered_count} (过滤掉 {original_count - filtered_count} 条)")
        
        if len(texts) == 0:
            raise ValueError("过滤后没有有效文本，请检查数据质量")
        
        # 显示数据样本
        if self.is_main_process():
            print("\n📝 清洗后的数据样本预览:")
            for i, text in enumerate(texts[:1]):
                preview = text[:200] + "..." if len(text) > 200 else text
                print(f"   样本 {i+1}: {preview}")
        
        # 按比例分割
        memory_ratio = self.config.get('memory_ratio', 0.3)
        random.shuffle(texts)
        
        split_idx = int(len(texts) * memory_ratio)
        if split_idx == 0:
            split_idx = 1
        
        memory_texts = texts[:split_idx]
        normal_texts = texts[split_idx:]
        
        if self.is_main_process():
            print(f"\n📊 数据分割结果:")
            print(f"   记忆训练文本: {len(memory_texts)} ({len(memory_texts)/len(texts)*100:.1f}%)")
            print(f"   普通训练文本: {len(normal_texts)} ({len(normal_texts)/len(texts)*100:.1f}%)")
        
        return memory_texts, normal_texts

    def _manual_parse_csv_text1_only(self, dataset_path: str) -> List[str]:
        """手动解析CSV文件，只提取text1的上下文部分"""
        texts = []
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                import csv
                reader = csv.reader(f)
                header = next(reader)  # 跳过标题行
                
                print(f"   CSV标题行: {header}")
                
                # 找到text1列的索引
                text1_index = None
                for i, col in enumerate(header):
                    if col.strip().lower() == 'text1':
                        text1_index = i
                        break
                
                if text1_index is None:
                    raise ValueError("未找到text1列")
                
                for row in reader:
                    if len(row) > text1_index:
                        text1 = row[text1_index].strip().strip('"')
                        
                        # 处理[SEP]分隔符
                        if '[SEP]' in text1:
                            parts = text1.split('[SEP]', 1)
                            if len(parts) == 2:
                                context = parts[1].strip()  # 只要[SEP]后面的部分
                                if context:
                                    texts.append(context)
                        else:
                            # 没有[SEP]就直接使用
                            if text1:
                                texts.append(text1)
                            
        except Exception as e:
            print(f"❌ 手动CSV解析失败: {e}")
            raise ValueError(f"无法解析CSV文件: {dataset_path}")
        
        return texts
        
    def step1_extract_embeddings(self, memory_texts: List[str]) -> str:
        """步骤1: 使用原始基础模型提取记忆文本的特征向量"""
        print(f"🎯 使用原始基础模型: {self.config['original_model_path']}")
        print(f"🎯 使用设备: {self.device}")

        save_dir = os.path.join(self.work_dir, 'embeddings')

        if self.ddp_enabled:
            # 1) 各rank分片处理
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            total = len(memory_texts)
            shard_size = (total + world_size - 1) // world_size
            start = rank * shard_size
            end = min(total, (rank + 1) * shard_size)
            shard_texts = memory_texts[start:end]

            print(f"[rank{rank}] 提取分片: {start}:{end} / {total}")
            extractor = TextEmbeddingExtractor(self.config['original_model_path'], device=self.device)
            data = extractor.extract_embeddings(shard_texts)
            os.makedirs(save_dir, exist_ok=True)
            partial_path = os.path.join(save_dir, f'text_embeddings_rank{rank}.pt')
            # 用已有的保存函数但指定文件名
            extractor.save_embeddings(data, save_dir=save_dir, filename=f'text_embeddings_rank{rank}.pt')
            del extractor
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            dist.barrier()

            # 2) rank0 合并
            final_path = os.path.join(save_dir, 'text_embeddings.pt')
            if self.is_main_process():
                print("🔗 合并各rank提取的嵌入...")
                # 收集所有分片
                texts_all = []
                embeds_all = []
                prompts_all = []
                tokens_all = []
                for r in range(world_size):
                    p = os.path.join(save_dir, f'text_embeddings_rank{r}.pt')
                    part = torch.load(p, map_location='cpu')
                    texts_all.extend(part.get('texts', []))
                    if 'embeddings' in part:
                        embeds = part['embeddings']
                        if isinstance(embeds, torch.Tensor):
                            embeds_all.append(embeds.cpu())
                    prompts_all.extend(part.get('prompts', []))
                    tokens_all.extend(part.get('predicted_tokens', []))

                if embeds_all:
                    embeddings_cat = torch.cat(embeds_all, dim=0)
                else:
                    raise ValueError("未发现任何分片的embeddings")

                merged = {
                    'texts': texts_all,
                    'embeddings': embeddings_cat,
                    'prompts': prompts_all,
                    'predicted_tokens': tokens_all,
                    'metadata': {
                        'model_name': self.config['original_model_path'],
                        'embedding_dim': embeddings_cat.shape[-1],
                        'num_samples': len(texts_all),
                        'dtype': str(embeddings_cat.dtype),
                        'created_date': datetime.now().isoformat(),
                        'device': str(self.device),
                        'device_config': f'ddp world_size={world_size}'
                    }
                }
                torch.save(merged, final_path)
                print(f"💾 已合并保存到: {final_path} (总样本: {len(texts_all)})")

                # 清理分片文件
                for r in range(world_size):
                    p = os.path.join(save_dir, f'text_embeddings_rank{r}.pt')
                    try:
                        os.remove(p)
                    except OSError:
                        pass

            dist.barrier()
            return final_path

        else:
            extractor = TextEmbeddingExtractor(self.config['original_model_path'], device=self.device)
            data = extractor.extract_embeddings(memory_texts)
            save_path = extractor.save_embeddings(data, save_dir)
            del extractor
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print("🧹 已清理特征提取器内存")
            print("✅ 步骤1完成: 特征向量提取成功")
            return save_path

    def step2_add_special_tokens(self) -> str:
        """步骤2: 添加特殊token"""
        if self.is_main_process():
            print("\n" + "="*60)
            print("📍 步骤 2/6: 添加记忆相关特殊token")
            print("="*60)
            original_model = self.config['original_model_path']
            token_manager = SpecialTokensManager(original_model, self.device)
            save_path = os.path.join(self.work_dir, 'model_with_special_tokens')
            model_path, token_ids = token_manager.process(
                save_path, 
                self.config.get('token_perturbation_std', 0.02)
            )
            del token_manager
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print("🧹 已清理token管理器内存")
            token_info = {
                'token_ids': token_ids,
                'model_path': model_path,
                'step_completed': 'add_special_tokens'
            }
            with open(os.path.join(self.work_dir, 'token_info.json'), 'w') as f:
                json.dump(token_info, f, indent=2)
            print("✅ 步骤2完成: 特殊token添加成功")
        else:
            model_path = os.path.join(self.work_dir, 'model_with_special_tokens')
        if self.ddp_enabled:
            dist.barrier()
        return model_path
        
    def step3_embedding_training(self, embedding_path: str, model_path: str) -> str:
        """步骤3: 嵌入向量训练（<recall> token）"""
        print("\n" + "="*60)
        print("📍 步骤 3/6: 训练 <recall> token")
        print("="*60)
        print(f"🎯 使用设备: {self.device}")
        
        trainer = RecallMemoryTrainer(model_path, self.device)
        
        embedding_config = self.config.get('embedding_training', {})
        save_path = os.path.join(self.work_dir, 'model_embedding_trained')
        
        trainer.train(
            pt_file_path=embedding_path,
            num_epochs=embedding_config.get('num_epochs', 30),
            batch_size=embedding_config.get('batch_size', 4),
            learning_rate=embedding_config.get('learning_rate', 1e-4),
            save_path=save_path
        )
        
        # 清理trainer占用的内存
        del trainer
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("🧹 已清理嵌入训练器内存")
        
        print("✅ 步骤3完成: 嵌入向量训练成功")
        return save_path

    def step4_memory_training(self, embedding_path: str, model_path: str) -> str:
        """步骤4: 记忆训练（表征向量解码）"""
        print("\n" + "="*60)
        print("📍 步骤 4/6: 训练表征向量解码能力")
        print("="*60)
        print(f"🎯 使用设备: {self.device}")
        
        trainer = EnhancedTextMemoryTrainer(model_path, self.device)
        
        memory_config = self.config.get('memory_training', {})
        save_path = os.path.join(self.work_dir, 'model_memory_trained')
        
        trainer.train(
            pt_file_path=embedding_path,
            num_epochs=memory_config.get('num_epochs', 20),
            batch_size=memory_config.get('batch_size', 4),
            learning_rate=memory_config.get('learning_rate', 1e-4),
            noise_std=memory_config.get('noise_std', 0.0),
            save_path=save_path
        )
        
        # 清理trainer占用的内存
        del trainer
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("🧹 已清理记忆训练器内存")
        
        print("✅ 步骤4完成: 记忆训练成功")
        return save_path

    def step5_normal_training(self, normal_texts: List[str], model_path: str) -> str:
        """步骤5: 普通文本训练"""
        print("\n" + "="*60)
        print("📍 步骤 5/6: 普通文本训练")
        print("="*60)
        print(f"🎯 使用设备: {self.device}")
        
        # NormalTextTrainer 未改造DDP，这里仅在主进程运行，避免重复训练
        if self.is_main_process():
            trainer = NormalTextTrainer(model_path, self.device)
            normal_config = self.config.get('normal_training', {})
            save_path = os.path.join(self.work_dir, 'model_normal_trained')
            trainer.train(
                texts=normal_texts,
                num_epochs=normal_config.get('num_epochs', 5),
                batch_size=normal_config.get('batch_size', 4),
                learning_rate=normal_config.get('learning_rate', 1e-4),
                save_path=save_path
            )
            del trainer
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print("🧹 已清理普通训练器内存")
            print("✅ 步骤5完成: 普通文本训练成功")
        else:
            save_path = os.path.join(self.work_dir, 'model_normal_trained')
        if self.ddp_enabled:
            dist.barrier()
        return save_path
        
    def step6_final_integration(self, final_model_path: str) -> str:
        """步骤6: 最终整合和验证"""
        print("\n" + "="*60)
        print("📍 步骤 6/6: 最终整合和验证")
        print("="*60)
        
        # 仅主进程执行最终整合
        if self.is_main_process():
            final_save_path = os.path.join(self.work_dir, 'final_model')
            import shutil
            if os.path.exists(final_save_path):
                shutil.rmtree(final_save_path)
            shutil.copytree(final_model_path, final_save_path)
            self._generate_training_report(final_save_path)
            print("✅ 步骤6完成: 最终整合成功")
        else:
            final_save_path = os.path.join(self.work_dir, 'final_model')
        if self.ddp_enabled:
            dist.barrier()
        return final_save_path
        
    def _generate_training_report(self, final_model_path: str):
        """生成训练报告"""
        report = {
            'training_completed': datetime.now().isoformat(),
            'final_model_path': final_model_path,
            'config': self.config,
            'steps_completed': [
                'add_special_tokens',
                'extract_embeddings', 
                'embedding_training',
                'memory_training',
                'normal_training',
                'final_integration'
            ]
        }
        
        report_path = os.path.join(self.work_dir, 'training_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print(f"📊 训练报告已保存: {report_path}")
            
    def run_training_cycle(self, embedding_path: str, normal_texts: List[str], 
                        current_model_path: str, cycle_num: int) -> str:
        """运行一个训练周期"""
        print(f"\n🔄 开始训练周期 {cycle_num}")
        
        # 嵌入向量训练
        model_path = self.step3_embedding_training(embedding_path, current_model_path)
        
        # 记忆训练  
        model_path = self.step4_memory_training(embedding_path, model_path)
        
        # 普通文本训练
        model_path = self.step5_normal_training(normal_texts, model_path)
        
        print(f"✅ 训练周期 {cycle_num} 完成")
        return model_path
        
    def run_full_pipeline(self, dataset_path: str):
        """运行完整训练流水线"""
        if self.is_main_process():
            print("🚀 开始集成训练流水线")
            print("="*80)
        
        start_time = datetime.now()
        
        try:
            # 加载和分割数据集
            memory_texts, normal_texts = self.load_and_split_dataset(dataset_path)
            
            # 步骤1: 使用原始模型提取特征向量（只执行一次）
            if self.is_main_process():
                print("\n" + "="*60)
                print("📍 步骤 1/6: 使用原始模型提取特征向量")
                print("="*60)
            embedding_path = self.step1_extract_embeddings(memory_texts)
            
            # 步骤2: 添加特殊token
            model_path = self.step2_add_special_tokens()
            
            # 多轮训练循环
            num_cycles = self.config.get('num_training_cycles', 3)
            if self.is_main_process():
                print(f"\n🔄 将进行 {num_cycles} 个训练周期")
            
            for cycle in range(1, num_cycles + 1):
                model_path = self.run_training_cycle(
                    embedding_path, normal_texts, model_path, cycle  # 传递embedding_path而不是memory_texts
                )
                
                # 每个周期后保存中间模型
                if self.is_main_process():
                    cycle_save_path = os.path.join(self.work_dir, f'model_cycle_{cycle}')
                    import shutil
                    if os.path.exists(cycle_save_path):
                        shutil.rmtree(cycle_save_path)
                    shutil.copytree(model_path, cycle_save_path)
                    print(f"💾 周期 {cycle} 模型已保存: {cycle_save_path}")
                if self.ddp_enabled:
                    dist.barrier()
            
            # 步骤6: 最终整合
            final_model_path = self.step6_final_integration(model_path)
            
            # 计算总时间
            end_time = datetime.now()
            total_time = end_time - start_time
            
            if self.is_main_process():
                print("\n" + "="*80)
                print("🎉 集成训练流水线完成!")
                print("="*80)
                print(f"   总耗时: {total_time}")
                print(f"   最终模型: {final_model_path}")
                print(f"   工作目录: {self.work_dir}")
                print(f"   训练周期: {num_cycles}")
                print("="*80)
            
            return final_model_path
            
        except Exception as e:
            print(f"❌ 训练流水线出错: {e}")
            import traceback
            traceback.print_exc()
            raise

def create_default_config() -> Dict[str, Any]:
    """创建默认配置"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    return {
        # 基础配置
        'original_model_path': os.path.join(project_root, 'Qwen2.5-7B-Instruct'),
        'device': 'cuda:0',
        'work_dir': os.path.join(project_root, 'training_workspace'),
        'memory_ratio': 0.6,  # 记忆训练数据占比
        'num_training_cycles': 3,  # 训练周期数
        
        # 特殊token配置
        'token_perturbation_std': 0.02,
        
        # 嵌入训练配置
        'embedding_training': {
            'num_epochs': 2,
            'batch_size': 2,
            'learning_rate': 1e-4
        },
        
        # 记忆训练配置
        'memory_training': {
            'num_epochs': 2,
            'batch_size': 2,
            'learning_rate': 1e-4,
            'noise_std': 0.0
        },
        
        # 普通训练配置
        'normal_training': {
            'num_epochs': 2,
            'batch_size': 2,
            'learning_rate': 1e-4
        }
    }

def main():
    """主函数"""
    # 创建配置
    config = create_default_config()
    
    # 修改配置
    config['device'] = ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
    config['memory_ratio'] = 0.6  # 30%用于记忆训练
    config['num_training_cycles'] = 8
    # 降内存：Step3批次设为1（DDP下每卡1样本）
    config['embedding_training']['batch_size'] = 2
    
    # 修改数据集路径为CSV文件
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(project_root, "DuReader_robust-QG", "train.csv")  # 绝对路径，避免相对路径失效
    # dataset_path = "./datasets/rich_text_dataset.json"
    
    # 检查数据集文件
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集文件不存在: {dataset_path}")
        print("请确保CSV文件存在，或者修改dataset_path变量")
        return
    
    # 添加pandas依赖检查
    try:
        import pandas as pd
        print("✅ pandas可用，支持CSV解析")
    except ImportError:
        print("❌ 需要安装pandas: pip install pandas")
        return
    
    try:
        # 创建并运行流水线
        pipeline = IntegratedTrainingPipeline(config)
        final_model_path = pipeline.run_full_pipeline(dataset_path)
        
        print(f"\n🎯 训练完成! 最终模型位于: {final_model_path}")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # 优雅关闭进程组，消除NCCL资源告警
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
        except Exception:
            pass

if __name__ == "__main__":
    main()