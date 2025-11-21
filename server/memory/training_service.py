#! -*- coding: utf-8 -*-
"""
记忆训练服务
负责整合聊天记录、提取记忆条目、训练模型并保存
（从 server/memory_training_service.py 迁移）
"""

import os
import json
import shutil
import torch
import torch.nn as nn
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import PIL
import requests
from transformers import AutoProcessor
import random
import logging as _logging

# 导入训练相关组件
import sys
# training_service.py 在 server/memory/ 目录下，需要往上3层到项目根目录
# __file__ = server/memory/training_service.py
# dirname(__file__) = server/memory/
# dirname(dirname(__file__)) = server/
# dirname(dirname(dirname(__file__))) = 项目根目录/
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
recall_dir = os.path.join(project_root, 'recall')
# 确保recall目录在sys.path的最前面
if recall_dir in sys.path:
    sys.path.remove(recall_dir)
sys.path.insert(0, recall_dir)

from recall.model_utils import forward_backbone, ensure_last_hidden_state

# 延迟导入训练器（避免循环导入）
# 注意：这里不预先导入，让_ensure_training_modules_loaded()函数处理
TRAINING_MODULES_AVAILABLE = False


def _try_prepare_recall_paths() -> bool:
    """
    尝试将可能存在的 recall 目录加入 sys.path：
    - <project_root>/recall
    - <project_root>/萝卜子v2.0/recall
    返回是否至少存在一个目录。
    """
    try:
        import sys as _sys, os as _os
        # training_service.py 在 server/memory/ 目录下，需要往上3层到项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # qqbot_new/server/memory -> qqbot_new
        candidates = [
            os.path.join(project_root, "recall"),
            os.path.join(project_root, "萝卜子v2.0", "recall"),
        ]
        found = False
        for p in candidates:
            if os.path.isdir(p):
                found = True
                if p not in _sys.path:
                    _sys.path.insert(0, p)
        return found
    except Exception:
        return False


def _ensure_training_modules_loaded() -> bool:
    """
    确保训练依赖可导入。若可导入则返回True。
    """
    global TRAINING_MODULES_AVAILABLE
    if TRAINING_MODULES_AVAILABLE:
        return True
    
    # 获取logger（如果_log还没有定义，就创建一个临时的）
    try:
        logger = _log
    except NameError:
        logger = logging.getLogger(__name__)
    
    # 尝试准备路径并导入
    _try_prepare_recall_paths()
    # 确保recall目录在sys.path中（即使_try_prepare_recall_paths已经添加，也要确保）
    import sys
    # training_service.py 在 server/memory/ 目录下，需要往上3层到项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    recall_dir = os.path.join(project_root, 'recall')
    
    # 确保recall目录存在且已添加到sys.path（确保在最前面）
    if os.path.isdir(recall_dir):
        if recall_dir in sys.path:
            # 如果已经在sys.path中，先移除再重新插入到最前面
            sys.path.remove(recall_dir)
            logger.debug(f"从sys.path中移除recall目录: {recall_dir}")
        # 插入到sys.path的最前面，确保优先使用
        sys.path.insert(0, recall_dir)
        logger.debug(f"已将recall目录插入到sys.path最前面: {recall_dir}")
    else:
        logger.error(f"recall目录不存在: {recall_dir}")
    
    try:
        global RecallMemoryTrainer, EnhancedTextMemoryTrainer, extract_last_token_embedding  # type: ignore
        # 尝试导入训练模块
        logger.info(f"尝试导入训练模块，当前sys.path前3项: {sys.path[:3]}")
        logger.info(f"recall目录: {recall_dir}")
        logger.info(f"recall目录存在: {os.path.isdir(recall_dir)}")
        logger.info(f"Python executable: {sys.executable}")

        # 检查文件是否存在
        files_to_check = [
            'text_embedding_train.py',
            'text_memory_train.py',
            'get_text_embedding.py'
        ]
        for file_name in files_to_check:
            file_path = os.path.join(recall_dir, file_name)
            exists = os.path.isfile(file_path)
            logger.info(f"  {file_name}: {'存在' if exists else '不存在'}")

        # 先确保torch可用
        try:
            import torch
            logger.info(f"torch版本: {torch.__version__}, CUDA可用: {torch.cuda.is_available()}")
        except ImportError as e:
            logger.error(f"torch不可用: {e}。请确保torch已正确安装。")
            TRAINING_MODULES_AVAILABLE = False
            return False

        # 检查训练模块的语法
        try:
            import importlib.util
            for file_name in files_to_check:
                file_path = os.path.join(recall_dir, file_name)
                if os.path.isfile(file_path):
                    spec = importlib.util.spec_from_file_location(file_name[:-3], file_path)  # 移除.py扩展名
                    if spec is None:
                        logger.warning(f"无法创建spec for {file_name}")
                    else:
                        logger.info(f"spec for {file_name} 创建成功")
        except Exception as e:
            logger.warning(f"spec检查失败: {e}")

        from text_embedding_train import RecallMemoryTrainer  # type: ignore
        from text_memory_train import EnhancedTextMemoryTrainer  # type: ignore
        from get_text_embedding import extract_last_token_embedding  # type: ignore
        TRAINING_MODULES_AVAILABLE = True
        logger.info("✅ 训练模块加载成功")
        return True
    except ImportError as e:
        logger.warning(f"训练依赖不可用：{e}。如需启用训练，请在项目根准备 recall/ 脚本或 萝卜子v2.0/recall/")
        logger.debug(f"当前sys.path中的recall相关路径: {[p for p in sys.path if 'recall' in p]}")
        logger.debug(f"recall目录绝对路径: {recall_dir}")
        logger.debug(f"recall目录是否存在: {os.path.isdir(recall_dir)}")
        if os.path.isdir(recall_dir):
            logger.debug(f"recall目录内容: {os.listdir(recall_dir)[:10]}")
        TRAINING_MODULES_AVAILABLE = False
        return False
    except Exception as e:
        logger.error(f"导入训练模块时发生未知错误: {e}", exc_info=True)
        TRAINING_MODULES_AVAILABLE = False
        return False

from .vector_db import MemoryVectorDB

if 'TRAINING_MODULES_AVAILABLE' not in locals():
    TRAINING_MODULES_AVAILABLE = False

if '_log' not in locals():
    _log = logging.getLogger(__name__)


def _optimize_multi_gpu_allocation(device_list: List[str], max_memory_config: Dict[int, str] = None, cuda_visible_set: bool = False) -> Dict[str, Any]:
    """
    优化多GPU分配策略，确保模型和数据更均匀地分布在多张GPU上
    
    Args:
        device_list: GPU设备列表，如 ["cuda:0", "cuda:1"] 或 ["cuda:6", "cuda:7"]
        max_memory_config: 用户配置的max_memory，格式如 {0: "20GB", 1: "20GB"}（索引是可见GPU的索引，不是物理索引）
        cuda_visible_set: 是否已经设置了CUDA_VISIBLE_DEVICES（如果已设置，需要使用重新映射后的索引）
    
    Returns:
        包含优化后的max_memory和device_map的字典
    """
    if not torch.cuda.is_available():
        return {"device_map": "cpu", "max_memory": None}
    
    num_gpus = len(device_list)
    if num_gpus == 0:
        return {"device_map": "cpu", "max_memory": None}
    
    # 检测每张GPU的可用显存
    gpu_memories = {}
    for i, device in enumerate(device_list):
        if device.startswith("cuda:"):
            try:
                physical_gpu_idx = int(device.split(":")[1])
                
                # 如果CUDA_VISIBLE_DEVICES已经设置，torch只能看到重新映射后的索引
                # 此时需要使用可见GPU的索引（0, 1, 2...），而不是物理索引
                if cuda_visible_set:
                    # 使用可见GPU的索引（i就是重新映射后的索引）
                    visible_gpu_idx = i
                    # 获取GPU总显存（MB）- 使用可见索引
                    total_memory_mb = torch.cuda.get_device_properties(visible_gpu_idx).total_memory // (1024 * 1024)
                    # 获取当前已用显存（MB）
                    torch.cuda.set_device(visible_gpu_idx)
                    allocated_mb = torch.cuda.memory_allocated(visible_gpu_idx) // (1024 * 1024)
                    reserved_mb = torch.cuda.memory_reserved(visible_gpu_idx) // (1024 * 1024)
                    available_mb = total_memory_mb - reserved_mb
                    _log.info(f"🔍 训练模型 GPU {i} (物理索引 {physical_gpu_idx}, 可见索引 {visible_gpu_idx}): 总显存={total_memory_mb}MB, 可用={available_mb}MB, 已保留={reserved_mb}MB")
                else:
                    # CUDA_VISIBLE_DEVICES未设置，使用物理索引
                    # 获取GPU总显存（MB）
                    total_memory_mb = torch.cuda.get_device_properties(physical_gpu_idx).total_memory // (1024 * 1024)
                    # 获取当前已用显存（MB）
                    torch.cuda.set_device(physical_gpu_idx)
                    allocated_mb = torch.cuda.memory_allocated(physical_gpu_idx) // (1024 * 1024)
                    reserved_mb = torch.cuda.memory_reserved(physical_gpu_idx) // (1024 * 1024)
                    available_mb = total_memory_mb - reserved_mb
                    _log.info(f"🔍 训练模型 GPU {i} (物理索引 {physical_gpu_idx}): 总显存={total_memory_mb}MB, 可用={available_mb}MB, 已保留={reserved_mb}MB")
                
                gpu_memories[i] = {
                    "total_mb": total_memory_mb,
                    "available_mb": available_mb,
                    "reserved_mb": reserved_mb,
                    "allocated_mb": allocated_mb
                }
            except Exception as e:
                _log.warning(f"⚠️ 无法检测GPU {i}的显存: {e}")
                # 使用默认值
                gpu_memories[i] = {"total_mb": 24000, "available_mb": 20000, "reserved_mb": 0, "allocated_mb": 0}
    
    # 计算优化的max_memory配置
    optimized_max_memory = {}
    if max_memory_config:
        # 如果用户提供了配置，使用用户配置，但确保所有GPU都有配置
        for i in range(num_gpus):
            if i in max_memory_config:
                optimized_max_memory[i] = max_memory_config[i]
            else:
                # 如果没有配置，使用可用显存的90%（留10%给系统和其他操作）
                if i in gpu_memories:
                    available_gb = gpu_memories[i]["available_mb"] / 1024
                    optimized_max_memory[i] = f"{int(available_gb * 0.9)}GB"
                else:
                    optimized_max_memory[i] = "20GB"  # 默认值
    else:
        # 如果没有用户配置，自动计算：使用每张GPU可用显存的90%
        for i in range(num_gpus):
            if i in gpu_memories:
                available_gb = gpu_memories[i]["available_mb"] / 1024
                optimized_max_memory[i] = f"{int(available_gb * 0.9)}GB"
            else:
                optimized_max_memory[i] = "20GB"  # 默认值
    
    _log.info(f"✅ 训练模型优化的max_memory配置: {optimized_max_memory}")
    
    # 使用 "balanced" device_map，尽可能均匀地分配模型层到所有GPU
    # 这样可以最大化利用所有GPU的显存，避免单张GPU过载
    # 注意：如果遇到OOM，可以考虑使用 "balanced_low_0" 让cuda:0分配更少
    # 参考：https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.device_map
    if num_gpus > 1:
        device_map_strategy = "balanced"
        _log.info(f"🔧 多GPU模式：使用 device_map='balanced'，均匀分配模型层到所有 {num_gpus} 张GPU")
    else:
        device_map_strategy = "auto"
        _log.info(f"🔧 单GPU模式：使用 device_map='auto'")
    
    return {
        "device_map": device_map_strategy,
        "max_memory": optimized_max_memory
    }


class TrainingModelContext:
    """训练模型上下文管理器 - 管理训练模型的生命周期"""

    def __init__(self, model_path: str, device, multi_gpu_config: Dict[str, Any] = None, add_special_tokens: bool = True):
        """
        初始化训练模型上下文管理器
        """
        self.model_path = model_path
        self.device = device
        self.multi_gpu_config = multi_gpu_config or {}
        self.add_special_tokens = add_special_tokens
        self.model = None
        self.processor = None

    def __enter__(self):
        """进入上下文，加载训练模型"""
        _log.info(f"加载训练模型上下文: {self.model_path}")
        self.model, self.processor = self._load_training_model()
        return self.model, self.processor

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，彻底清理模型（支持多GPU）"""
        _log.info("清理训练模型上下文...")

        try:
            # 清理模型（多GPU情况下需要更彻底的清理）
            if self.model is not None:
                try:
                    # 对于多GPU模型，需要先尝试移动到CPU
                    # 如果模型使用了device_map="auto"，可能需要特殊处理
                    if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
                        # 多GPU模型，需要逐个设备清理
                        _log.info("检测到多GPU模型，执行彻底清理...")
                        # 先尝试移动到CPU（可能部分层已经在CPU上）
                        try:
                            self.model.cpu()
                        except Exception as e:
                            _log.warning(f"移动模型到CPU时出现警告: {e}")
                        
                        # 如果模型有accelerator包装，需要先清理accelerator
                        if hasattr(self.model, 'accelerator'):
                            try:
                                self.model.accelerator.free_memory()
                            except:
                                pass
                    else:
                        # 单GPU模型，直接移动到CPU
                        try:
                            self.model.cpu()
                        except:
                            pass
                except Exception as e:
                    _log.warning(f"清理模型时出现警告: {e}")
                
                # 删除模型引用
                del self.model
                self.model = None

            # 清理processor
            if self.processor is not None:
                del self.processor
                self.processor = None

            # 强制垃圾回收和显存清理（多次清理确保彻底）
            import gc
            for _ in range(5):  # 增加清理次数
                gc.collect()

            # 清理所有GPU的显存
            if torch.cuda.is_available():
                # 同步所有GPU
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                
                # 再次清理所有GPU
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                
                _log.info(f"✅ 已清理所有 {torch.cuda.device_count()} 张GPU的显存")

            _log.info("✅ 训练模型上下文清理完成")

        except Exception as cleanup_error:
            _log.warning(f"训练模型上下文清理时出现错误: {cleanup_error}")

        return False  # 不抑制异常

    def _load_training_model(self):
        """加载训练模型（内部方法）"""
        return self.load_training_model(self.model_path, self.device, self.multi_gpu_config, add_special_tokens=self.add_special_tokens)

    @staticmethod
    def load_training_model(model_path: str, device, multi_gpu_config: Dict[str, Any] = None, add_special_tokens: bool = True):
        """加载统一的训练模型（静态方法）"""
        # 使用与initialize_model相同的加载逻辑
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        multi_gpu_config = multi_gpu_config or {}
        multi_gpu_enabled = multi_gpu_config.get("enabled", True)

        # 将相对路径转换为绝对路径
        if not os.path.isabs(model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            model_path = os.path.abspath(os.path.join(project_root, model_path))

        # 检查是否为本地路径
        is_local_path = os.path.exists(model_path) and os.path.isdir(model_path)

        try:
            # 加载processor（使用AutoProcessor而不是AutoTokenizer，因为需要处理图片和视频）
            # 正常推理时使用AutoProcessor，训练时也应该使用AutoProcessor
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=is_local_path
            )

            # 处理多GPU设备配置
            if device == "auto" and multi_gpu_enabled:
                # 自动检测所有可用GPU
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    device = [f"cuda:{i}" for i in range(gpu_count)]
                    _log.info(f"🔧 训练模型: 自动检测到 {gpu_count} 张GPU，使用多GPU模式")

            # 加载模型 - 支持多GPU
            from transformers import Qwen3VLForConditionalGeneration
            load_kwargs = {
                "torch_dtype": torch.bfloat16,
                "trust_remote_code": True,
                "local_files_only": is_local_path
            }

            # 检查CUDA_VISIBLE_DEVICES设置状态（在所有设备配置之前）
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            cuda_visible_set = bool(cuda_visible)
            cuda_visible_devices = cuda_visible

            # 根据设备配置决定device_map
            if isinstance(device, list) and multi_gpu_enabled:
                # 多GPU配置
                # 注意：CUDA_VISIBLE_DEVICES应该在导入torch之前设置（在app.py中已设置）
                # 这里只需要检查是否已经设置，如果没有设置则设置（兼容性处理）
                
                if cuda_visible:
                    _log.info(f"🔧 检测到CUDA_VISIBLE_DEVICES={cuda_visible}（已在导入torch之前设置）")
                else:
                    # 如果未设置，则在这里设置（虽然可能已经太晚了）
                    gpu_indices = []
                    for gpu_device in device:
                        if gpu_device.startswith("cuda:"):
                            try:
                                gpu_idx = int(gpu_device.split(":")[1])
                                gpu_indices.append(str(gpu_idx))
                            except (ValueError, IndexError):
                                _log.warning(f"⚠️ 无效的GPU设备名称: {gpu_device}，跳过")
                                continue
                    if gpu_indices:
                        cuda_visible_devices = ",".join(gpu_indices)
                        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
                        _log.warning(f"⚠️ CUDA_VISIBLE_DEVICES未在导入torch之前设置，现在设置={cuda_visible_devices}（可能无效）")
                        # 注意：如果在这里设置，torch可能已经初始化，所以可能无效
                        # 但为了兼容性，我们仍然设置它

                # 使用优化的多GPU分配策略
                # 注意：如果CUDA_VISIBLE_DEVICES已设置，需要使用重新映射后的索引
                max_memory_config = multi_gpu_config.get("max_memory", {})
                allocation = _optimize_multi_gpu_allocation(device, max_memory_config, cuda_visible_set=cuda_visible_set)
                load_kwargs["device_map"] = allocation["device_map"]
                if allocation["max_memory"]:
                    load_kwargs["max_memory"] = allocation["max_memory"]
                _log.info(f"🔧 训练模型: 指定设备{device}，使用优化的分配策略")
            elif isinstance(device, str) and device.startswith("cuda"):
                # 如果设置了CUDA_VISIBLE_DEVICES，需要使用重新映射后的索引
                if cuda_visible_set and cuda_visible_devices:
                    # CUDA_VISIBLE_DEVICES已设置，使用重新映射后的索引
                    device_map_device = "cuda:0"
                    _log.info(f"🔧 训练模型: 单GPU模式，CUDA_VISIBLE_DEVICES={cuda_visible_devices}，使用重新映射设备 {device_map_device}（对应物理GPU {device}）")
                else:
                    # 未设置CUDA_VISIBLE_DEVICES，直接使用物理设备
                    device_map_device = device
                    _log.info(f"🔧 训练模型: 单GPU模式，设备映射到 {device}")
                load_kwargs["device_map"] = {"": device_map_device}
            else:
                load_kwargs["device_map"] = "auto"  # 默认使用auto
                _log.info("🔧 训练模型: 使用自动设备分配")

            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                **load_kwargs
            )

            if add_special_tokens:
                # 添加特殊token（如果没有的话）
                # 使用MemoryTokenManager，与正常推理时保持一致
                from memory.token_manager import MemoryTokenManager
                token_manager = MemoryTokenManager(model, processor.tokenizer)
                recall_token_ids = token_manager.check_and_add_tokens(perturbation_std=0.02)
                _log.info(f"✅ 特殊token处理完成: {recall_token_ids}")

            _log.info("✅ 训练模型加载成功")
            return model, processor

        except Exception as e:
            _log.error(f"❌ 加载训练模型失败: {e}")
            raise


def _resolve_path(path: Optional[str], project_root: str) -> Optional[str]:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(project_root, path))


class MemoryTrainingService:
    """记忆训练服务"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练服务
        
        Args:
            config: 配置字典，包含训练相关参数
        """
        self.config = config
        self.memory_config = config.get("memory", {}).get("training", {})
        
        script_dir = os.path.dirname(os.path.abspath(__file__))  # server/memory
        server_dir = os.path.dirname(script_dir)                 # server
        project_root = os.path.dirname(server_dir)               # 项目根
        self._project_root = project_root
        
        # 路径配置
        self.base_model_path = _resolve_path(self.memory_config.get("base_model_path"), project_root)
        self.trained_model_dir = _resolve_path(self.memory_config.get("trained_model_dir"), project_root)
        self.memory_db_dir = _resolve_path(self.memory_config.get("memory_db_dir"), project_root)
        self.chat_history_storage_dir = _resolve_path(self.memory_config.get("chat_history_storage_dir"), project_root)
        
        # 训练配置
        self.training_config = self.memory_config.get("training_config", {})
        self.lora_config = self.memory_config.get("lora_config", {})
        self.guides_config = self.memory_config.get("guides", {})
        
        # 设备配置（使用模型配置中的设备）
        model_config = config.get("model", {})
        self.device = model_config.get("device", "cuda:0")
        # 保存原始设备信息（用于训练器日志显示）
        self.original_device = self.device
        
        # 创建必要的目录
        os.makedirs(self.trained_model_dir, exist_ok=True)
        os.makedirs(self.memory_db_dir, exist_ok=True)
        
        if self.chat_history_storage_dir:
            os.makedirs(self.chat_history_storage_dir, exist_ok=True)
        
        _log.info("记忆训练服务初始化完成")
        _log.info(f"  基础模型路径: {self.base_model_path}")
        _log.info(f"  训练模型目录: {self.trained_model_dir}")
        _log.info(f"  记忆数据库目录: {self.memory_db_dir}")
        _log.info(f"  聊天记录目录: {self.chat_history_storage_dir}")
        
        # SFT相关配置
        sft_cfg = self.memory_config.get("sft", {})
        self.sft_enabled = bool(sft_cfg.get("enabled", False))
        self.sft_path = sft_cfg.get("dataset_path")
        self.sft_per_epoch = bool(sft_cfg.get("per_epoch", True))
        self.sft_max_per_epoch = sft_cfg.get("max_per_epoch") or None
        self.sft_seed = int(sft_cfg.get("seed", 42))
        export_cfg = self.memory_config.get("export", {})
        self.export_save_full_vl_assets = bool(export_cfg.get("save_full_vl_assets", True))
        self.export_merge_lora = bool(export_cfg.get("merge_lora", True))
        self._memory_entry_count = None
        self._current_epoch_sample_n = None
        self._saved_history_counts = {
            "group": {},
            "private": {}
        }

    # 下方保留与旧实现一致的大段函数（提取/保存/训练/清理等），为节省篇幅省略重复注释
    # 由于内容较多，这里直接从旧实现完全迁移（逻辑不变）

    def _prepare_output_dir(self, path: str):
        if os.path.isdir(path):
            _log.info(f"清理历史模型目录: {path}")
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    def _get_latest_trained_model_path(self) -> str:
        """
        获取最新的模型路径
        优先级：训练后的模型 > 添加了token的模型 > 基础模型
        """
        # 获取token_added_model_dir配置
        token_added_model_dir = self.memory_config.get("token_added_model_dir", "./server/models/token_added")
        
        # 转换为绝对路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        server_dir = os.path.dirname(script_dir)
        project_root = os.path.dirname(server_dir)
        
        trained_model_dir = self.trained_model_dir
        if not os.path.isabs(trained_model_dir):
            trained_model_dir = os.path.abspath(os.path.join(project_root, trained_model_dir))
        if not os.path.isabs(token_added_model_dir):
            token_added_model_dir = os.path.abspath(os.path.join(project_root, token_added_model_dir))
        
        # 1. 优先查找训练后的模型
        if os.path.exists(trained_model_dir):
            model_dirs = [
                d for d in os.listdir(trained_model_dir)
                if os.path.isdir(os.path.join(trained_model_dir, d)) and d.startswith("model_")
            ]
            if model_dirs:
                model_dirs.sort(reverse=True)
                latest_model = os.path.join(trained_model_dir, model_dirs[0])
                _log.info(f"找到最新训练模型: {latest_model}")
                return latest_model
        
        # 2. 如果没有训练模型，查找添加了token的模型
        if os.path.exists(token_added_model_dir):
            model_dirs = [
                d for d in os.listdir(token_added_model_dir)
                if os.path.isdir(os.path.join(token_added_model_dir, d)) and d.startswith("model_")
            ]
            if model_dirs:
                model_dirs.sort(reverse=True)
                latest_model = os.path.join(token_added_model_dir, model_dirs[0])
                _log.info(f"找到添加了token的模型: {latest_model}")
                return latest_model
        
        # 3. 如果都没有，使用基础模型
        _log.info(f"未找到训练模型或添加了token的模型，使用基础模型: {self.base_model_path}")
        return self.base_model_path

    def _create_trained_model_path(self) -> str:
        """
        创建新的训练模型保存路径，使用时间戳命名格式：model_YYYYMMDD_HHMMSS
        确保与加载逻辑匹配（按时间戳排序选择最新的）
        """
        from datetime import datetime
        if not os.path.isabs(self.trained_model_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))  # memory目录
            server_dir = os.path.dirname(script_dir)  # server目录
            project_root = os.path.dirname(server_dir)  # 项目根目录
            # 路径相对于项目根目录，直接拼接
            trained_model_dir = os.path.abspath(os.path.join(project_root, self.trained_model_dir))
        else:
            trained_model_dir = self.trained_model_dir
        
        os.makedirs(trained_model_dir, exist_ok=True)
        
        # 使用时间戳创建新的模型目录名：model_YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir_name = f"model_{timestamp}"
        model_path = os.path.join(trained_model_dir, model_dir_name)
        
        _log.info(f"创建新的训练模型保存路径: {model_path}")
        return model_path

    def run_training(self, skip_memory_dump: bool = False) -> Optional[str]:
        """
        运行完整的训练流程
        
        Returns:
            最终训练模型的路径
        """
        _log.info("=" * 60)
        _log.info("开始记忆训练流程")
        _log.info("=" * 60)
        
        multi_gpu_config = self.config.get("model", {}).get("multi_gpu", {})

        # 0. 保存内存中的聊天记录到JSON文件（训练前调用）
        if skip_memory_dump:
            _log.info("=" * 60)
            _log.info("步骤0: 已在API中保存聊天记录，跳过此步骤")
            _log.info("=" * 60)
        else:
            _log.info("=" * 60)
            _log.info("步骤0: 保存内存中的聊天记录到JSON文件")
            _log.info("=" * 60)
            self.save_memory_chat_histories_to_storage()

        # 1. 从JSON文件加载聊天记录
        _log.info("=" * 60)
        _log.info("步骤1: 加载JSON文件中的聊天记录")
        _log.info("=" * 60)
        chat_messages = self.load_chat_histories_from_json_only()

        # chat_messages 是一个列表，每个元素是 {"chat_type": ..., "chat_id": ..., "message": ...}
        if len(chat_messages) == 0:
            _log.warning("⚠️ 没有聊天记录可训练，跳过训练")
            _log.info("💡 可能的原因：")
            _log.info("   - 内存中没有聊天记录")
            _log.info("   - chat_history_storage_dir 中没有JSON文件")
            _log.info("   - 请检查聊天记录是否被正确保存")
            return None

        # 统计聊天记录信息（按聊天分组统计）
        chat_groups = {}
        for msg_data in chat_messages:
            chat_type = msg_data.get("chat_type", "unknown")
            chat_id = msg_data.get("chat_id", "unknown")
            key = f"{chat_type}_{chat_id}"
            if key not in chat_groups:
                chat_groups[key] = 0
            chat_groups[key] += 1

        total_messages = len(chat_messages)
        _log.info(f"📊 总共 {total_messages} 条消息，分布在 {len(chat_groups)} 个聊天组中")

        # 2. 使用基础模型提取记忆条目和监督向量
        _log.info("=" * 60)
        _log.info("步骤2: 使用基础模型提取记忆条目和监督向量")
        _log.info("=" * 60)
        _log.info(f"使用基础模型: {self.base_model_path}")
        
        with TrainingModelContext(self.base_model_path, self.device, multi_gpu_config, add_special_tokens=False) as (base_model, base_processor):
            # 提取记忆条目并保存到临时文件（使用基础模型）
            temp_training_data_path = self.extract_memory_entries(chat_messages, base_model, base_processor)

            if temp_training_data_path is None or not os.path.exists(temp_training_data_path):
                _log.warning("⚠️ 没有提取到记忆条目或生成训练数据文件，跳过训练")
                _log.info("💡 可能的原因：")
                _log.info("   - 模型在提取记忆条目时没有识别到值得记忆的内容")
                _log.info("   - 提取过程中出现错误（请查看上面的日志）")
                _log.info("   - 聊天记录中的内容可能不适合提取为记忆条目")
                return None

            # 加载训练数据以获取统计信息
            training_data = torch.load(temp_training_data_path, map_location='cpu')
            num_entries = len(training_data.get('texts', []))
            _log.info(f"📊 提取到 {num_entries} 个记忆条目，已保存到临时文件")
            # 设置本轮SFT每epoch采样参考数（与记忆条目数量等量）
            try:
                self._memory_entry_count = int(num_entries)
                self._current_epoch_sample_n = int(num_entries)
            except Exception:
                self._memory_entry_count = None
                self._current_epoch_sample_n = None

            # 保存监督向量到MemoryVectorDB（从训练数据文件中提取）
            self.save_memory_embeddings_from_file(temp_training_data_path)

            # 同时提取等量的SFT向量用于第一步训练，防止<recall>token过拟合
            sft_vectors_path = self._extract_sft_vectors_for_recall_training(
                num_entries, base_model, base_processor
            )

        # 基础模型上下文自动清理

        # 3. 使用最新的训练模型进行训练
        _log.info("=" * 60)
        _log.info("步骤3: 使用最新的训练模型进行训练")
        _log.info("=" * 60)
        if getattr(self, "_memory_entry_count", None):
            self._current_epoch_sample_n = self._memory_entry_count
        
        # 查找最新的训练模型路径（如果存在），否则使用基础模型
        training_model_path = self._get_latest_trained_model_path()
        _log.info(f"训练模型路径: {training_model_path}")
        
        with TrainingModelContext(training_model_path, self.device, multi_gpu_config) as (training_model, training_processor):
            # 3.5. 清理显存，确保模型处于干净状态
            _log.info("=" * 60)
            _log.info("步骤3.5: 清理显存，准备训练")
            _log.info("=" * 60)
            
            # 确保模型处于eval模式，清除梯度
            training_model.eval()
            with torch.no_grad():
                # 强制清理显存（多次清理确保彻底）
                import gc
                for _ in range(5):
                    gc.collect()
                
                if torch.cuda.is_available():
                    # 清理所有GPU的显存
                    gpu_count = torch.cuda.device_count()
                    _log.info(f"清理 {gpu_count} 张GPU的显存...")
                    
                    # 同步并清理所有GPU
                    for i in range(gpu_count):
                        with torch.cuda.device(i):
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                    
                    # 再次清理所有GPU
                    for i in range(gpu_count):
                        with torch.cuda.device(i):
                            torch.cuda.empty_cache()
                    
                    _log.info(f"✅ 已清理所有 {gpu_count} 张GPU的显存")
            
            _log.info("✅ 显存清理完成，模型已准备就绪")

            # 4. 第一步训练：<recall> token训练（使用最新的训练模型）
            step1_model_path = self.train_recall_token(temp_training_data_path, training_model, training_processor, sft_vectors_path)

            # 5. 第二步训练：记忆解码训练（重新加载第一阶段训练好的模型）
            final_model_path = self.train_memory_decoding(temp_training_data_path, step1_model_path)

            # 6. 按时间戳保存最终模型
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_save_path = os.path.join(self.trained_model_dir, f"model_{timestamp}")

            import shutil
            if os.path.exists(final_save_path):
                shutil.rmtree(final_save_path)
            shutil.copytree(final_model_path, final_save_path)

            # 保存Processor配置到最终模型目录
            self._save_processor_to_path(final_save_path)
            if self.export_save_full_vl_assets:
                self._ensure_full_vl_assets(final_save_path)

            _log.info(f"最终模型保存在: {final_save_path}")

            # 7. 清理训练数据和缓存（训练模型由上下文管理器自动清理）
            self.cleanup_after_training()

            _log.info("=" * 60)
            _log.info("记忆训练流程完成")
            _log.info("=" * 60)

            # 训练完成后清理上传缓存（图片/视频），避免长期占用磁盘
            try:
                script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # server/
                uploaded_images = os.path.join(script_dir, "uploaded_images")
                uploaded_videos = os.path.join(script_dir, "uploaded_videos")
                cleaned = 0
                for d in [uploaded_images, uploaded_videos]:
                    if os.path.isdir(d):
                        for fname in os.listdir(d):
                            fpath = os.path.join(d, fname)
                            try:
                                os.remove(fpath)
                                cleaned += 1
                            except Exception:
                                pass
                if cleaned:
                    _log.info(f"✅ 训练完成后已清空缓存文件 {cleaned} 个（images/videos）")
            except Exception as ce:
                _log.warning(f"⚠️ 清理上传缓存失败: {ce}")

            return final_save_path

    def _save_processor_to_path(self, target_path: str):
        """
        保存完整的Processor配置到目标路径
        确保使用训练后的tokenizer（包含特殊token），同时保留processor的其他配置
        """
        try:
            base_path = self.base_model_path
            if not os.path.isabs(base_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(script_dir)
                base_path = os.path.abspath(os.path.join(project_root, base_path))
            
            # 1. 从基础模型加载processor（包含image_processor、video_processor等配置）
            base_processor = AutoProcessor.from_pretrained(
                base_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            # 2. 从训练后的模型加载tokenizer（包含特殊token）
            trained_tokenizer = None
            if os.path.exists(target_path):
                try:
                    # 尝试加载训练后的tokenizer
                    from transformers import AutoTokenizer
                    trained_tokenizer = AutoTokenizer.from_pretrained(
                        target_path,
                        trust_remote_code=True,
                        local_files_only=True
                    )
                    _log.info("✅ 已加载训练后的tokenizer（包含特殊token）")
                except Exception as e:
                    _log.warning(f"⚠️ 加载训练后的tokenizer失败: {e}，将使用基础模型的tokenizer")
            
            # 3. 如果训练后的tokenizer存在，更新processor的tokenizer
            if trained_tokenizer is not None:
                base_processor.tokenizer = trained_tokenizer

            # 4. 保存完整的processor配置（包含训练后的tokenizer和其他processor组件）
            base_processor.save_pretrained(target_path)

            # 5. 确保所有必要的配置文件都被正确保存（在save_pretrained之后，确保不被覆盖）
            # 这些文件对于Qwen3VLProcessor的正确工作至关重要
            import shutil
            essential_files = [
                "chat_template.json",
                "preprocessor_config.json",
                "video_preprocessor_config.json"
            ]
            for file_name in essential_files:
                source_file = os.path.join(base_path, file_name)
                target_file = os.path.join(target_path, file_name)
                if os.path.exists(source_file):
                    try:
                        shutil.copy2(source_file, target_file)
                        _log.info(f"✅ 已复制{file_name}到: {target_path}")
                    except Exception as e:
                        _log.warning(f"⚠️ 复制{file_name}失败: {e}")
                else:
                    _log.warning(f"⚠️ 基础模型中不存在{file_name}，跳过复制")
            _log.info(f"✅ 已保存Processor配置到: {target_path}（包含训练后的tokenizer）")
            
        except Exception as e:
            _log.warning(f"⚠️ 保存Processor配置失败: {e}")

    def _ensure_full_vl_assets(self, output_dir: str):
        if not self.export_save_full_vl_assets:
            return
        try:
            base_path = self.base_model_path
            if not os.path.isabs(base_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(script_dir)
                base_path = os.path.abspath(os.path.join(project_root, base_path))
            required_files = [
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "preprocessor_config.json",
                "video_preprocessor_config.json",
                "processor_config.json",
            ]
            required_dirs = [
                "image_processor",
                "processor",
            ]
            os.makedirs(output_dir, exist_ok=True)
            for fname in required_files:
                src = os.path.join(base_path, fname)
                if os.path.exists(src):
                    dst = os.path.join(output_dir, fname)
                    if not os.path.exists(dst):
                        try:
                            shutil.copy2(src, dst)
                            _log.info(f"✅ 复制缺失文件: {fname}")
                        except Exception as ce:
                            _log.warning(f"复制文件失败 {fname}: {ce}")
            for dname in required_dirs:
                srcd = os.path.join(base_path, dname)
                dstd = os.path.join(output_dir, dname)
                if os.path.isdir(srcd) and not os.path.exists(dstd):
                    try:
                        shutil.copytree(srcd, dstd)
                        _log.info(f"✅ 复制目录: {dname}")
                    except Exception as ce:
                        _log.warning(f"复制目录失败 {dname}: {ce}")
        except Exception as e:
            _log.warning(f"⚠️ 确保VL资产时出错: {e}")

    def _resolve_dataset_path(self, path_str: str) -> str:
        if not path_str:
            return None
        if os.path.isabs(path_str):
            return path_str
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        abs_path = os.path.abspath(os.path.join(project_root, path_str))
        return abs_path

    def _load_sft_dataset(self) -> List[Dict[str, Any]]:
        """从jsonl加载通用SFT样本，兼容常见schema"""
        dataset_path = self._resolve_dataset_path(self.sft_path)
        if not dataset_path or not os.path.exists(dataset_path):
            _log.warning(f"⚠️ SFT数据集不存在: {dataset_path}")
            return []
        samples = []
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        samples.append(obj)
                    except Exception:
                        continue
        except Exception as e:
            _log.warning(f"加载SFT数据集失败: {e}")
            return []
        _log.info(f"✅ 加载SFT样本: {len(samples)}")
        return samples
    
    def _standardize_sft_messages(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        标准化为messages列表：[{'role': 'user'|'assistant'|'system', 'content': [{'type':'text','text':...}]}]
        仅文本，不处理图片。
        """
        # 1) 如果已有messages
        msgs = sample.get("messages")
        if isinstance(msgs, list) and msgs:
            std = []
            for m in msgs:
                role = m.get("role", "user")
                content = m.get("content") or m.get("text") or ""
                if isinstance(content, str):
                    std.append({"role": role, "content": [{"type": "text", "text": content}]})
                elif isinstance(content, list):
                    # 假设list文本
                    text_join = ""
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_join += item.get("text", "")
                        elif isinstance(item, str):
                            text_join += item
                    std.append({"role": role, "content": [{"type": "text", "text": text_join}]})
            if std:
                return std
        # 2) instruction + output
        inst = sample.get("instruction") or sample.get("input")
        out = sample.get("output") or sample.get("answer")
        if isinstance(inst, str) and isinstance(out, str):
            return [
                {"role": "user", "content": [{"type": "text", "text": inst}]},
                {"role": "assistant", "content": [{"type": "text", "text": out}]},
            ]
        # 3) 单轮问答风格
        q = sample.get("query") or sample.get("question")
        a = sample.get("response") or sample.get("answer")
        if isinstance(q, str) and isinstance(a, str):
            return [
                {"role": "user", "content": [{"type": "text", "text": q}]},
                {"role": "assistant", "content": [{"type": "text", "text": a}]},
            ]
        return []
    
    def _build_simple_sft_batch(self, processor, messages: List[List[Dict[str, Any]]]):
        """
        简单SFT批处理：将messages转成input_ids并直接用自回归标签（不区分mask）。
        如果messages中包含字符串格式的文本（用于<recall> token训练），则只让<recall> token参与训练。
        """
        # 检查是否有字符串格式的文本（用于<recall> token训练）
        recall_token_id = None
        try:
            recall_token_id = processor.tokenizer.convert_tokens_to_ids("<recall>")
        except:
            pass
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for msg in messages:
            # 如果是字符串格式（用于<recall> token训练）
            if isinstance(msg, str):
                # 直接tokenize文本
                encoded = processor.tokenizer(
                    msg,
                    return_tensors="pt",
                    add_special_tokens=True,
                    padding=False,
                    truncation=False
                )
                input_ids = encoded["input_ids"][0]  # [seq_len]
                attention_mask = torch.ones_like(input_ids)
                
                # 创建labels，默认全部mask（-100）
                labels = torch.full_like(input_ids, -100)
                
                # 找到<recall> token的位置，只让这个token参与训练
                if recall_token_id is not None:
                    recall_positions = (input_ids == recall_token_id).nonzero(as_tuple=True)[0]
                    if len(recall_positions) > 0:
                        # 只让最后一个<recall> token参与训练
                        last_recall_pos = recall_positions[-1].item()
                        labels[last_recall_pos] = input_ids[last_recall_pos]
                        _log.debug(f"找到<recall> token位置: {last_recall_pos}, 已设置为参与训练")
                    else:
                        _log.warning(f"⚠️ 文本中未找到<recall> token: {msg}")
                
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_labels.append(labels)
            else:
                # 原有的messages格式处理
                batch_inputs = processor.apply_chat_template(
                    [msg], tokenize=True, add_generation_prompt=False,
                    return_dict=True, return_tensors="pt"
                )
                input_ids = batch_inputs["input_ids"][0]  # [seq_len]
                attention_mask = batch_inputs.get("attention_mask", (input_ids != 0).long())[0]
                labels = input_ids.clone()
                
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_labels.append(labels)
        
        # 对batch进行padding
        max_len = max(len(ids) for ids in batch_input_ids)
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for i in range(len(batch_input_ids)):
            pad_len = max_len - len(batch_input_ids[i])
            padded_input_ids.append(torch.cat([batch_input_ids[i], torch.zeros(pad_len, dtype=batch_input_ids[i].dtype)]))
            padded_attention_mask.append(torch.cat([batch_attention_mask[i], torch.zeros(pad_len, dtype=batch_attention_mask[i].dtype)]))
            padded_labels.append(torch.cat([batch_labels[i], torch.full((pad_len,), -100, dtype=batch_labels[i].dtype)]))
        
        input_ids = torch.stack(padded_input_ids)
        attention_mask = torch.stack(padded_attention_mask)
        labels = torch.stack(padded_labels)
        
        return input_ids, attention_mask, labels
    
    def _run_sft_one_epoch(self, trainer_obj, epoch: int, epoch_sample_n: int):
        """使用训练阶段的LoRA模型，跑1个epoch的通用SFT（重建优化器，权重连续累积）
        
        Args:
            trainer_obj: 训练器对象
            epoch: 当前epoch编号（用于改变随机种子，确保每次采样不同）
            epoch_sample_n: 采样数量（与记忆条目数量相同）
        """
        if not self.sft_enabled or not self.sft_per_epoch:
            return
        try:
            # 在SFT训练前，清理记忆训练可能残留的显存
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            _log.debug("🧹 SFT训练前已清理显存缓存")
            
            # 取出句柄
            handles = getattr(trainer_obj, "expose_training_handles", None)
            if not callable(handles):
                _log.warning("⚠️ 训练器未暴露expose_training_handles，跳过本次SFT")
                return
            handle = trainer_obj.expose_training_handles()
            model = handle.get("model") or handle.get("base_model")
            tokenizer = handle.get("tokenizer")
            accelerator = handle.get("accelerator", None)
            grad_acc_steps = getattr(trainer_obj, "gradient_accumulation_steps", 1)
            if model is None or tokenizer is None:
                _log.warning("⚠️ SFT句柄缺失，跳过")
                return
            # 加载样本与采样
            all_samples = self._load_sft_dataset()
            if not all_samples:
                _log.warning("⚠️ 无SFT数据，跳过")
                return
            # 使用epoch编号来改变随机种子，确保每个epoch采样不同的样本
            random.seed(self.sft_seed + epoch)
            # 采样数量与记忆条目数量相同（epoch_sample_n已传入）
            sample_n = min(epoch_sample_n, len(all_samples)) if epoch_sample_n else len(all_samples)
            if self.sft_max_per_epoch is not None:
                sample_n = min(sample_n, int(self.sft_max_per_epoch))
            picked = random.sample(all_samples, sample_n)
            std_msgs = []
            for s in picked:
                m = self._standardize_sft_messages(s)
                if m:
                    std_msgs.append(m)
            
            if not std_msgs:
                _log.warning("⚠️ 本轮SFT无有效样本，跳过")
                return
            # 获取SFT训练的batch_size（默认为1，保持向后兼容）
            sft_batch_size = self.training_config.get("sft_batch_size", 1)
            _log.info(f"🧪 本epoch插入SFT: {len(std_msgs)} 条 (batch_size={sft_batch_size})")

            # 构建小批数据
            model.train()
            optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(self.training_config.get("learning_rate", 1e-4)))
            accumulation = 0
            total_loss = 0.0
            steps = 0

            # 使用tqdm进度条
            try:
                from tqdm import tqdm
                use_tqdm = True
            except ImportError:
                use_tqdm = False
                progress_interval = max(1, len(std_msgs) // 10)  # 每10%打印一次进度

            if use_tqdm:
                pbar = tqdm(total=len(std_msgs), desc="🧪 SFT训练", unit="样本")

            # 按batch_size分组处理
            for i in range(0, len(std_msgs), sft_batch_size):
                batch_end = min(i + sft_batch_size, len(std_msgs))
                batch_msgs = std_msgs[i:batch_end]
                actual_batch_size = len(batch_msgs)

                input_ids, attention_mask, labels = self._build_simple_sft_batch(tokenizer, batch_msgs)
                device = next(model.parameters()).device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / grad_acc_steps
                if accelerator is not None:
                    accelerator.backward(loss)
                else:
                    loss.backward()
                accumulation += 1
                if accumulation % grad_acc_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optim.step()
                    optim.zero_grad()
                    steps += 1
                    total_loss += loss.item() * grad_acc_steps

                    # 更新进度条或打印进度
                    if use_tqdm:
                        pbar.update(actual_batch_size)
                        pbar.set_postfix({'loss': f'{total_loss/steps:.4f}'})
                    else:
                        # 打印进度（兼容没有tqdm的情况）
                        if steps % progress_interval == 0 or steps == (len(std_msgs) // grad_acc_steps + 1):
                            progress = (steps * grad_acc_steps) / len(std_msgs) * 100
                            avg_loss_so_far = total_loss / steps
                            _log.info(f"🧪 SFT进度: {progress:.1f}% ({steps * grad_acc_steps}/{len(std_msgs)}), 当前loss={avg_loss_so_far:.6f}")

            if accumulation % grad_acc_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
                optim.zero_grad()
                steps += 1
                total_loss += loss.item() * grad_acc_steps

            if use_tqdm:
                pbar.close()

            avg_loss = total_loss / max(steps, 1)
            _log.info(f"✅ 本epoch SFT完成，avg_loss={avg_loss:.6f}, steps={steps}")
            
            # SFT训练结束后，清理SFT数据以释放显存，为下一个epoch的记忆训练做准备
            import torch
            import gc
            # 清理SFT训练中创建的tensor
            del input_ids, attention_mask, labels, outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            _log.debug("🧹 SFT训练后已清理显存缓存")
        except Exception as e:
            _log.warning(f"⚠️ SFT执行失败，已跳过: {e}", exc_info=True)
    
    def save_memory_chat_histories_to_storage(self):
        """
        将内存中的聊天记录保存到JSON文件（训练前调用）
        这样可以确保训练时使用最新的聊天记录，同时清理内存
        
        注意：此函数通过模块导入获取全局变量，可能获取不到运行时的对象。
        如果获取不到数据，请使用api_server的全局函数直接保存。
        """
        _log.info("开始保存内存中的聊天记录到存储...")
        
        # 使用延迟导入避免循环导入
        import importlib
        import sys
        
        # 尝试多种方式导入api_server_qwen3vl模块
        api_server = None
        try:
            # 方式1：直接导入（如果已经在sys.modules中）- 这是最可靠的方式
            if 'api_server_qwen3vl' in sys.modules:
                api_server = sys.modules['api_server_qwen3vl']
                _log.info("从sys.modules获取api_server_qwen3vl模块")
            else:
                # 方式2：使用importlib导入
                api_server = importlib.import_module('api_server_qwen3vl')
                _log.info("使用importlib导入api_server_qwen3vl模块")
        except Exception as e:
            _log.error(f"导入api_server_qwen3vl模块失败: {e}")
            # 方式3：尝试从server目录导入
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                if script_dir not in sys.path:
                    sys.path.insert(0, script_dir)
                api_server = importlib.import_module('api_server_qwen3vl')
                _log.info("从server目录导入api_server_qwen3vl模块")
            except Exception as e2:
                _log.error(f"从server目录导入也失败: {e2}")
                _log.warning("⚠️ 无法导入模块，无法保存内存中的聊天记录")
                return
        
        if api_server is None:
            _log.error("无法导入api_server_qwen3vl模块，跳过保存")
            return
        
        # 获取全局变量（注意：如果模块重新导入，可能获取不到运行时的对象）
        group_chat_histories = getattr(api_server, 'group_chat_histories', {})
        private_chat_histories = getattr(api_server, 'private_chat_histories', {})
        save_chat_history_to_storage = getattr(api_server, 'save_chat_history_to_storage', None)
        
        # 检查是否获取到了运行时的对象（通过检查是否有实际内容）
        # 如果获取到的对象是空字典，可能是模块导入问题，记录警告
        if isinstance(group_chat_histories, dict) and len(group_chat_histories) == 0 and isinstance(private_chat_histories, dict) and len(private_chat_histories) == 0:
            _log.warning("⚠️ 获取到的聊天记录为空，可能是模块导入问题")
            _log.warning("💡 提示：如果确实有聊天记录，请检查模块导入是否正确")
        
        _log.info(f"📊 内存中的聊天记录统计:")
        _log.info(f"   群聊数量: {len(group_chat_histories)}")
        _log.info(f"   私聊数量: {len(private_chat_histories)}")
        
        # 详细统计每个聊天的消息数
        for chat_id, history in group_chat_histories.items():
            _log.info(f"   群聊 {chat_id}: {len(history)} 条消息")
        for chat_id, history in private_chat_histories.items():
            _log.info(f"   私聊 {chat_id}: {len(history)} 条消息")
        
        # 如果获取到的记录为空，直接返回（不保存）
        if len(group_chat_histories) == 0 and len(private_chat_histories) == 0:
            _log.warning("⚠️ 获取到的聊天记录为空，无法保存")
            _log.warning("💡 这可能是因为模块导入问题，请检查日志或使用手动保存API")
            return
        
        # 确保目录存在
        # 优先使用api_server的CHAT_HISTORY_STORAGE_DIR（这是实际保存文件的目录）
        # 如果获取不到，使用training_service配置的目录
        chat_history_storage_dir = getattr(api_server, 'CHAT_HISTORY_STORAGE_DIR', None)
        if not chat_history_storage_dir:
            chat_history_storage_dir = self.chat_history_storage_dir
            _log.warning(f"⚠️ 无法获取api_server的CHAT_HISTORY_STORAGE_DIR，使用配置目录: {chat_history_storage_dir}")
        else:
            _log.info(f"✅ 使用api_server的CHAT_HISTORY_STORAGE_DIR: {chat_history_storage_dir}")
        
        if chat_history_storage_dir:
            os.makedirs(chat_history_storage_dir, exist_ok=True)
            _log.info(f"✅ 确保聊天记录存储目录存在: {chat_history_storage_dir}")
        
        saved_count = 0

        def _get_pending_messages(chat_type_key: str, chat_id: str, history: List[Dict[str, Any]]):
            last_saved = self._saved_history_counts[chat_type_key].get(chat_id, 0)
            if last_saved < 0 or last_saved > len(history):
                last_saved = 0
            if last_saved == len(history):
                return [], len(history)
            return history[last_saved:], len(history)
        
        # 优先使用api_server的save_chat_history_to_storage函数（使用正确的目录）
        if save_chat_history_to_storage:
            # 保存群聊记录
            for chat_id, history in group_chat_histories.items():
                if not history:  # 只保存非空记录
                    continue
                pending_messages, final_len = _get_pending_messages("group", chat_id, history)
                if not pending_messages:
                    continue
                try:
                    save_chat_history_to_storage("group", chat_id, pending_messages)
                    saved_count += len(pending_messages)
                    self._saved_history_counts["group"][chat_id] = final_len
                    _log.info(f"✅ 保存群聊 {chat_id} 的 {len(pending_messages)} 条新消息到 {chat_history_storage_dir}")
                except Exception as e:
                    _log.warning(f"保存群聊 {chat_id} 失败: {e}", exc_info=True)
                    # 如果保存失败，尝试直接保存
                    try:
                        self._save_chat_history_directly("group", chat_id, pending_messages)
                        saved_count += len(pending_messages)
                        self._saved_history_counts["group"][chat_id] = final_len
                        _log.info(f"✅ 使用直接保存方式成功保存群聊 {chat_id} 的 {len(pending_messages)} 条新消息")
                    except Exception as e2:
                        _log.error(f"直接保存也失败: {e2}", exc_info=True)
            
            # 保存私聊记录
            for chat_id, history in private_chat_histories.items():
                if not history:  # 只保存非空记录
                    continue
                pending_messages, final_len = _get_pending_messages("private", chat_id, history)
                if not pending_messages:
                    continue
                try:
                    save_chat_history_to_storage("private", chat_id, pending_messages)
                    saved_count += len(pending_messages)
                    self._saved_history_counts["private"][chat_id] = final_len
                    _log.info(f"✅ 保存私聊 {chat_id} 的 {len(pending_messages)} 条新消息到 {chat_history_storage_dir}")
                except Exception as e:
                    _log.warning(f"保存私聊 {chat_id} 失败: {e}", exc_info=True)
                    # 如果保存失败，尝试直接保存
                    try:
                        self._save_chat_history_directly("private", chat_id, pending_messages)
                        saved_count += len(pending_messages)
                        self._saved_history_counts["private"][chat_id] = final_len
                        _log.info(f"✅ 使用直接保存方式成功保存私聊 {chat_id} 的 {len(pending_messages)} 条新消息")
                    except Exception as e2:
                        _log.error(f"直接保存也失败: {e2}", exc_info=True)
        else:
            # 如果无法使用api_server的函数，直接保存
            _log.warning("无法找到save_chat_history_to_storage函数，使用直接保存方式...")
            for chat_id, history in group_chat_histories.items():
                if history:
                    pending_messages, final_len = _get_pending_messages("group", chat_id, history)
                    if not pending_messages:
                        continue
                    try:
                        self._save_chat_history_directly("group", chat_id, pending_messages)
                        saved_count += len(pending_messages)
                        self._saved_history_counts["group"][chat_id] = final_len
                        _log.info(f"✅ 直接保存群聊 {chat_id} 的 {len(pending_messages)} 条新消息")
                    except Exception as e:
                        _log.warning(f"直接保存群聊 {chat_id} 失败: {e}", exc_info=True)
            for chat_id, history in private_chat_histories.items():
                if history:
                    pending_messages, final_len = _get_pending_messages("private", chat_id, history)
                    if not pending_messages:
                        continue
                    try:
                        self._save_chat_history_directly("private", chat_id, pending_messages)
                        saved_count += len(pending_messages)
                        self._saved_history_counts["private"][chat_id] = final_len
                        _log.info(f"✅ 直接保存私聊 {chat_id} 的 {len(pending_messages)} 条新消息")
                    except Exception as e:
                        _log.warning(f"直接保存私聊 {chat_id} 失败: {e}", exc_info=True)
        
        _log.info(f"✅ 共保存 {saved_count} 条内存中的聊天记录到存储")
    
    def load_chat_histories_from_json_only(self) -> List[Dict[str, Any]]:
        """
        只从JSON文件加载聊天记录（训练时使用，不从内存加载）
        
        Returns:
            所有聊天记录的列表
        """
        all_messages = []
        json_count = 0
        
        # 使用api_server的CHAT_HISTORY_STORAGE_DIR（如果可用）
        import importlib
        import sys
        
        chat_history_storage_dir = self.chat_history_storage_dir
        api_server = None
        if 'api_server_qwen3vl' in sys.modules:
            api_server = sys.modules['api_server_qwen3vl']
        else:
            try:
                api_server = importlib.import_module('api_server_qwen3vl')
            except:
                pass
        
        if api_server:
            api_storage_dir = getattr(api_server, 'CHAT_HISTORY_STORAGE_DIR', None)
            if api_storage_dir and os.path.exists(api_storage_dir):
                chat_history_storage_dir = api_storage_dir
                _log.info(f"✅ 使用api_server的CHAT_HISTORY_STORAGE_DIR: {chat_history_storage_dir}")
        
        # 加载JSON文件
        _log.info(f"检查聊天记录存储目录: {chat_history_storage_dir}")
        if os.path.exists(chat_history_storage_dir):
            json_files = list(Path(chat_history_storage_dir).glob("*.json"))
            _log.info(f"找到 {len(json_files)} 个JSON文件")
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            # 如果data是列表，直接使用
                            json_count += len(data)
                            all_messages.extend(data)
                            _log.info(f"从 {json_file.name} 加载 {len(data)} 条消息（列表格式）")
                        elif isinstance(data, dict) and "messages" in data:
                            # 如果data是字典且包含messages字段
                            messages = data.get("messages", [])
                            json_count += len(messages)
                            # 需要将消息转换为统一格式
                            chat_type = data.get("chat_type", "unknown")
                            chat_id = data.get("chat_id", "unknown")
                            for msg in messages:
                                all_messages.append({
                                    "chat_type": chat_type,
                                    "chat_id": chat_id,
                                    "message": msg
                                })
                            _log.info(f"从 {json_file.name} 加载 {len(messages)} 条消息（字典格式，chat_type={chat_type}, chat_id={chat_id}）")
                        else:
                            _log.warning(f"JSON文件 {json_file.name} 格式不正确，跳过")
                except Exception as e:
                    _log.warning(f"加载 {json_file} 失败: {e}", exc_info=True)
        else:
            _log.warning(f"聊天记录存储目录不存在: {chat_history_storage_dir}")
        
        _log.info(f"总共从JSON文件加载 {len(all_messages)} 条消息")
        return all_messages
    
    def load_chat_histories(self) -> List[Dict[str, Any]]:
        """
        加载所有聊天记录（包括内存中的和历史JSON文件）
        注意：调用此函数前应该先调用save_memory_chat_histories_to_storage()保存内存中的记录
        
        Returns:
            所有聊天记录的列表
        """
        all_messages = []
        
        # 1. 加载内存中的聊天记录（最新的30条）
        # 注意：这些记录应该在训练前已经保存到JSON文件了
        # 使用延迟导入避免循环导入
        import importlib
        import sys
        
        memory_count = 0
        
        # 尝试多种方式导入api_server_qwen3vl模块
        api_server = None
        try:
            # 方式1：直接导入（如果已经在sys.modules中）
            if 'api_server_qwen3vl' in sys.modules:
                api_server = sys.modules['api_server_qwen3vl']
                _log.info("从sys.modules获取api_server_qwen3vl模块（用于加载内存记录）")
            else:
                # 方式2：使用importlib导入
                api_server = importlib.import_module('api_server_qwen3vl')
                _log.info("使用importlib导入api_server_qwen3vl模块（用于加载内存记录）")
        except Exception as e:
            _log.warning(f"导入api_server_qwen3vl模块失败: {e}，将跳过内存记录加载")
            # 方式3：尝试从server目录导入
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                if script_dir not in sys.path:
                    sys.path.insert(0, script_dir)
                api_server = importlib.import_module('api_server_qwen3vl')
                _log.info("从server目录导入api_server_qwen3vl模块（用于加载内存记录）")
            except Exception as e2:
                _log.warning(f"从server目录导入也失败: {e2}，将跳过内存记录加载")
                api_server = None
        
        if api_server is not None:
            group_chat_histories = getattr(api_server, 'group_chat_histories', {})
            private_chat_histories = getattr(api_server, 'private_chat_histories', {})
            
            _log.info(f"📊 内存中的聊天记录统计（加载时）:")
            _log.info(f"   群聊数量: {len(group_chat_histories)}")
            _log.info(f"   私聊数量: {len(private_chat_histories)}")
            
            for chat_id, history in group_chat_histories.items():
                history_len = len(history)
                memory_count += history_len
                _log.info(f"   群聊 {chat_id}: {history_len} 条消息")
                all_messages.extend([
                    {
                        "chat_type": "group",
                        "chat_id": chat_id,
                        "message": msg
                    }
                    for msg in history
                ])
            
            for chat_id, history in private_chat_histories.items():
                history_len = len(history)
                memory_count += history_len
                _log.info(f"   私聊 {chat_id}: {history_len} 条消息")
                all_messages.extend([
                    {
                        "chat_type": "private",
                        "chat_id": chat_id,
                        "message": msg
                    }
                    for msg in history
                ])
        else:
            _log.warning("无法访问api_server_qwen3vl模块，跳过内存记录加载")
        
        _log.info(f"从内存加载 {memory_count} 条消息")
        
        # 2. 加载历史JSON文件
        json_count = 0
        _log.info(f"检查聊天记录存储目录: {self.chat_history_storage_dir}")
        if os.path.exists(self.chat_history_storage_dir):
            json_files = list(Path(self.chat_history_storage_dir).glob("*.json"))
            _log.info(f"找到 {len(json_files)} 个JSON文件")
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            # 如果data是列表，直接使用
                            json_count += len(data)
                            all_messages.extend(data)
                            _log.info(f"从 {json_file.name} 加载 {len(data)} 条消息（列表格式）")
                        elif isinstance(data, dict) and "messages" in data:
                            # 如果data是字典且包含messages字段
                            messages = data.get("messages", [])
                            json_count += len(messages)
                            # 需要将消息转换为统一格式
                            chat_type = data.get("chat_type", "unknown")
                            chat_id = data.get("chat_id", "unknown")
                            for msg in messages:
                                all_messages.append({
                                    "chat_type": chat_type,
                                    "chat_id": chat_id,
                                    "message": msg
                                })
                            _log.info(f"从 {json_file.name} 加载 {len(messages)} 条消息（字典格式，chat_type={chat_type}, chat_id={chat_id}）")
                        else:
                            _log.warning(f"JSON文件 {json_file.name} 格式不正确，跳过")
                except Exception as e:
                    _log.warning(f"加载 {json_file} 失败: {e}", exc_info=True)
        else:
            _log.warning(f"聊天记录存储目录不存在: {self.chat_history_storage_dir}")
        
        _log.info(f"总共加载 {len(all_messages)} 条消息（内存: {memory_count}, JSON: {json_count}）")
        return all_messages
    
    def _save_chat_history_directly(self, chat_type: str, chat_id: str, messages: List[Dict[str, Any]]):
        """
        直接保存聊天记录到JSON文件（当无法使用api_server的函数时）
        
        Args:
            chat_type: "group" 或 "private"
            chat_id: 群ID或用户ID
            messages: 要保存的消息列表
        """
        if not self.chat_history_storage_dir:
            raise ValueError("聊天记录存储目录未配置")
        
        # 确保目录存在
        os.makedirs(self.chat_history_storage_dir, exist_ok=True)
        
        # 创建存储文件路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{chat_type}_{chat_id}_{timestamp}.json"
        filepath = os.path.join(self.chat_history_storage_dir, filename)
        
        # 保存到JSON文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "chat_type": chat_type,
                "chat_id": chat_id,
                "timestamp": timestamp,
                "messages": messages
            }, f, ensure_ascii=False, indent=2)
        
        _log.info(f"✅ 直接保存 {len(messages)} 条聊天记录到: {filename}")
    
    def extract_memory_entries(self, chat_messages: List[Dict[str, Any]], model=None, processor=None) -> str:
        """
        提取记忆条目并生成监督向量，直接保存到临时文件

        Args:
            chat_messages: 聊天消息列表

        Returns:
            临时训练数据文件路径
        """
        _log.info("开始提取记忆条目...")

        # 按聊天分组
        chat_groups = {}
        for msg_data in chat_messages:
            chat_type = msg_data.get("chat_type", "unknown")
            chat_id = msg_data.get("chat_id", "unknown")
            message = msg_data.get("message", {})

            key = f"{chat_type}_{chat_id}"
            if key not in chat_groups:
                chat_groups[key] = []
            chat_groups[key].append(message)

        _log.info(f"共 {len(chat_groups)} 个聊天组")

        # 检查是否提供了模型和processor
        if model is None or processor is None:
            _log.error("❌ extract_memory_entries需要提供model和processor参数")
            return None

        _log.info("使用统一的训练模型进行记忆提取")

        # 从配置中获取最大token限制（用于批量提取向量时的截断）
        # 如果配置中没有，使用默认值 35000
        max_tokens = self.training_config.get("max_tokens_for_embedding", 35000)
        _log.debug(f"使用最大token限制: {max_tokens}（用于批量提取向量时的截断）")

        # 角色设定（用于记忆提取时提醒模型自己的身份）
        role_playing_prompt = ""
        try:
            role_playing_prompt = self.config.get("prompt", {}).get("role_playing", "")
            if role_playing_prompt:
                role_playing_prompt = role_playing_prompt.strip()
        except Exception:
            role_playing_prompt = ""

        # 临时文件路径（只包含记忆条目文本）
        temp_texts_path = os.path.join(self.memory_db_dir, "temp_memory_texts.pt")
        # 注意：保留已有的临时文件，新的记忆条目将追加到现有文件
        # 这允许分批处理聊天记录而不丢失之前的结果
        _log.debug(f"记忆条目将保存到临时文件: {temp_texts_path}")

        try:
            
            # 对每个聊天组进行总结（递归处理，支持对半分）
            def process_chat_group(messages: List[Dict[str, Any]], chat_key: str, depth: int = 0):
                """
                处理单个聊天组（递归函数，支持对半分）
                
                Args:
                    messages: 聊天消息列表
                    chat_key: 聊天组标识
                    depth: 递归深度（用于日志）
                """
                if not messages:
                    return
                
                # 构建标准格式的聊天历史（保留多模态信息）
                # 使用与generate_reply相同的格式，这样模型可以读取图片信息
                chat_messages_for_extraction = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")  # 默认值为空字符串，与旧版本保持一致
                    
                    # 保持原始content格式（可能是list，包含图片信息）
                    if isinstance(content, list):
                        _log.debug(f"🔍 聊天组 {chat_key} 消息 {role} 的content是列表，包含 {len(content)} 项")
                        # 多模态内容，需要验证图片URL是否有效
                        # 如果图片URL失效，只保留文本部分
                        filtered_content = []
                        image_count = 0
                        valid_image_count = 0
                        for item in content:
                            if item.get("type") == "text":
                                # 文本内容直接保留
                                filtered_content.append(item)
                            elif item.get("type") == "image":
                                # 图片内容，简化验证（因为图片已经在聊天时被验证过了）
                                image_url = item.get("image", "")
                                image_count += 1
                                if image_url:
                                    # 训练时简化验证：只检查URL格式，不进行实际网络访问
                                    # 因为QQ图片URL有时效性，在聊天时有效但训练时可能已过期
                                    if image_url.startswith('http://') or image_url.startswith('https://'):
                                        # URL格式正确，保留图片（信任聊天时的验证结果）
                                        filtered_content.append(item)
                                        valid_image_count += 1
                                    else:
                                        _log.warning(f"⚠️ 聊天组 {chat_key} 图片URL格式无效，跳过")
                                else:
                                    _log.warning(f"⚠️ 聊天组 {chat_key} 发现无效的图片项（无URL），跳过")
                            elif item.get("type") == "video":
                                video_url = item.get("video") or item.get("url")
                                # 视频内容，保留所有有效的URL（视频不像图片那样有URL过期问题）
                                if not video_url:
                                    _log.warning(f"⚠️ 聊天组 {chat_key} 发现无效的视频项（无URL），跳过")
                                    continue

                                # 训练时简化验证：保留本地服务器URL和本地文件路径（信任聊天时的验证结果）
                                from server.api_server_qwen3vl import server_base_url
                                is_local_server_url = (video_url.startswith('http://127.0.0.1:9999/static/videos/') or \
                                                       video_url.startswith('http://localhost:9999/static/videos/') or \
                                                       (server_base_url and video_url.startswith(f"{server_base_url.rstrip('/')}/static/videos/")))
                                is_local_file = os.path.exists(video_url) and os.path.isfile(video_url)
                                is_file_url = video_url.startswith('file://') and os.path.exists(video_url[7:])

                                _log.debug(f"🔍 视频URL检查: {video_url}")
                                _log.debug(f"  is_local_server_url: {is_local_server_url}")
                                _log.debug(f"  is_local_file: {is_local_file} (文件存在: {os.path.exists(video_url) if video_url else False})")
                                _log.debug(f"  is_file_url: {is_file_url}")
                                _log.debug(f"  is_http: {video_url.startswith('http://') or video_url.startswith('https://') if video_url else False}")

                                if is_local_server_url or is_local_file or is_file_url or video_url.startswith('http://') or video_url.startswith('https://'):
                                    # 保留视频（信任聊天时的验证结果）
                                    filtered_content.append({
                                        "type": "video",
                                        "video": video_url
                                    })
                                    _log.info(f"✅ 保留视频: {video_url}")
                                else:
                                    # 无效URL格式，移除
                                    _log.warning(f"⚠️ 移除无效视频URL: {video_url}")

                        # 如果过滤后还有内容，添加消息
                        if filtered_content:
                            # 统计图片和视频数量
                            img_count = sum(1 for item in filtered_content if item.get("type") == "image")
                            vid_count = sum(1 for item in filtered_content if item.get("type") == "video")
                            if img_count > 0 or vid_count > 0:
                                _log.info(f"📊 聊天组 {chat_key} 消息包含 {img_count} 张图片和 {vid_count} 个视频")
                                for item in filtered_content:
                                    if item.get("type") == "image":
                                        _log.info(f"   📷 图片URL: {item.get('image', '')}")
                                    elif item.get("type") == "video":
                                        _log.info(f"   🎥 视频URL: {item.get('video', '')}")
                            chat_messages_for_extraction.append({
                                "role": role,
                                "content": filtered_content
                            })
                        else:
                            _log.warning(f"⚠️ 聊天组 {chat_key} 消息过滤后无内容，跳过该消息")
                    elif isinstance(content, str):
                        # 纯文本内容，转换为标准格式
                        chat_messages_for_extraction.append({
                            "role": role,
                            "content": [{"type": "text", "text": content}]
                        })
                    else:
                        # 未知格式，跳过
                        _log.warning(f"⚠️ 聊天组 {chat_key} 消息content格式未知: {type(content)}，跳过")
                
                # 添加系统提示，要求模型提取记忆条目
                extraction_system_prompt = """请分析以下对话，提取出值得记忆的独立信息条目。

注意：对话中的"助手"就是你自己（AI助手），在总结记忆条目时，如果涉及助手的行为、回复或信息，请使用第一人称（我、我的）来描述。

值得记忆的信息类型包括但不限于：
1. 关于人物的事实性知识：姓名、身份、职业、关系、性格特点、兴趣爱好、习惯等
2. 关于世界的事实性知识：地点、事件、历史、文化背景等
3. 时事新闻：当前发生的重要事件、社会动态等
4. 科学知识：科学原理、技术信息、专业知识等
5. 用户的偏好和习惯：喜欢什么、不喜欢什么、常用表达方式等
6. 重要的约定和承诺：用户提到的重要事项、约定等
7. 关于我（助手）的信息：用户对我的称呼、我与用户的关系、我告诉用户的信息等

每个记忆条目应该是一个具体、独立的事实或偏好，格式必须严格按照：
条目1: [具体记忆内容]
条目2: [具体记忆内容]
...

重要要求：
1. 每条记忆条目应该尽可能包含完整信息，包括人物、时间、地点、事件等所有相关细节，不要将这些信息分散在多个条目中
2. 条目与条目之间应该是完全独立的，不存在关联或依赖关系
3. 如果一条记忆涉及多个要素（如人物、时间、地点、事件），应该将它们整合在一条记忆条目中
4. 如果记忆涉及助手（我）的信息，请使用第一人称描述，例如"我告诉用户..."、"用户称呼我为..."等
5. 记忆条目不一定要非常简略，必要的信息要充分记录。如果某个细节对于理解或回忆这条记忆很重要，应该包含在内

如果对话中没有值得记忆的信息，请输出"无记忆条目"。

请先进行思考（使用<think>标签），然后在</think>标签后的正式回答中，严格按照上述格式输出记忆条目。"""

                if role_playing_prompt:
                    extraction_system_prompt += (
                        "\n\n======\n角色设定提示（帮助你理解对话中自己的身份）：\n"
                        f"{role_playing_prompt}\n"
                        "------\n"
                        "注意：以上内容仅用于提醒你在原始聊天中的身份和关系。"
                        " 在提取记忆条目时，请用客观、第一人称的方式描述事实，不需要模仿口癖或聊天语气。"
                        "\n======"
                    )
                
                # 构建完整的消息列表（系统提示 + 聊天历史 + 用户提示）
                full_messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": extraction_system_prompt}]
                    }
                ]
                full_messages.extend(chat_messages_for_extraction)
                full_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": "请开始提取记忆条目。"}]
                })
                
                # 统计图片和视频数量（与v1.0版本保持一致，不做URL转换，直接使用原始URL）
                # v1.0版本直接使用HTTP URL，processor能够访问（因为服务器正在运行）
                total_images = 0
                total_videos = 0
                for msg in full_messages:
                    msg_content = msg.get("content", [])
                    if isinstance(msg_content, list):
                        for item in msg_content:
                            if item.get("type") == "image":
                                total_images += 1
                                _log.debug(f"🔍 图片URL: {item.get('image', '')}")
                            elif item.get("type") == "video":
                                total_videos += 1
                                _log.debug(f"🔍 视频URL: {item.get('video', '')}")
                _log.info(f"📊 准备处理的消息包含 {total_images} 张图片和 {total_videos} 个视频")
                
                try:
                    # 使用processor.apply_chat_template处理消息（与generate_reply一致）
                    # 这样可以保留图片信息
                    # 如果图片URL失效，会抛出异常，需要捕获并处理
                    try:
                        inputs = processor.apply_chat_template(
                            full_messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_dict=True,
                            return_tensors="pt"
                        )

                        # 打印完整的输入（包括特殊token），与api_server_qwen3vl.py保持一致
                        input_ids_text = processor.batch_decode(
                            inputs['input_ids'],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False
                        )
                        _log.info("=" * 80)
                        _log.info("🔤 模型完整输入（包括特殊token）：")
                        _log.info(input_ids_text[0])
                        _log.info("=" * 80)
                    except (PIL.UnidentifiedImageError, OSError, requests.RequestException, Exception) as img_error:
                        # 图片/视频加载失败，尝试移除所有图片和视频，只保留文本
                        _log.warning(f"⚠️ 聊天组 {chat_key} 图片/视频处理失败: {str(img_error)}")
                        _log.warning(f"   错误类型: {type(img_error).__name__}")
                        _log.warning(f"   错误详情: {img_error}", exc_info=True)
                        if isinstance(img_error, requests.RequestException):
                            _log.warning("   📷 图片/视频URL失效或网络连接失败，可能是QQ临时URL已过期")
                        elif isinstance(img_error, PIL.UnidentifiedImageError):
                            _log.warning("   📷 图片格式无法识别，可能是图片文件损坏")
                        else:
                            _log.warning("   📷 图片/视频加载出错，将移除所有图片和视频继续处理")
                        _log.info(f"   🔄 自动降级：移除所有图片和视频，只使用文本内容进行记忆提取...")
                        
                        # 重新构建消息，移除所有图片
                        text_only_messages = []
                        for msg in full_messages:
                            msg_content = msg.get("content", [])
                            if isinstance(msg_content, list):
                                # 只保留文本项
                                text_items = [item for item in msg_content if item.get("type") == "text"]
                                if text_items:
                                    text_only_messages.append({
                                        "role": msg.get("role"),
                                        "content": text_items
                                    })
                            else:
                                # 已经是文本，直接保留
                                text_only_messages.append(msg)
                        
                        if not text_only_messages:
                            _log.warning(f"⚠️ 聊天组 {chat_key} 移除图片后无有效消息，跳过处理")
                            return
                        
                        # 使用纯文本消息重新处理
                        inputs = processor.apply_chat_template(
                            text_only_messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_dict=True,
                            return_tensors="pt"
                        )
                    
                    # 检查输入token长度（使用函数级别的max_tokens变量）
                    input_length = inputs["input_ids"].shape[-1]
                    _log.info(f"📊 聊天组 {chat_key} (深度 {depth}) 输入token长度: {input_length}, 最大限制: {max_tokens}")
                    
                    # 如果超过限制，将聊天记录对半分
                    if input_length > max_tokens:
                        if len(messages) <= 1:
                            # 即使只有1条消息也超过限制，只能跳过
                            _log.warning(f"⚠️ 聊天组 {chat_key} 即使只有1条消息也超过限制 ({input_length} > {max_tokens})，跳过处理")
                            return
                        else:
                            # 对半分
                            _log.warning(f"⚠️ 聊天组 {chat_key} (深度 {depth}) 输入token长度 ({input_length}) 超过限制 ({max_tokens})，对半分处理")
                            half_point = len(messages) // 2
                            first_half = messages[:half_point]
                            second_half = messages[half_point:]
                            
                            # 递归处理两半
                            process_chat_group(first_half, chat_key, depth + 1)
                            process_chat_group(second_half, chat_key, depth + 1)
                            return
                    
                    # 移动到正确设备
                    # 在多GPU模式下，需要将输入移动到模型所在的设备（通常是第一个参数所在的设备）
                    if isinstance(self.device, list):
                        # 多GPU模式：获取模型实际所在的设备（第一个参数所在的设备）
                        model_device = next(model.parameters()).device
                        inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                                 for k, v in inputs.items()}
                    else:
                        # 单GPU模式：检查CUDA_VISIBLE_DEVICES设置状态
                        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                        if cuda_visible and cuda_visible.strip():
                            # CUDA_VISIBLE_DEVICES已设置，使用重新映射后的设备
                            device_for_inputs = "cuda:0"
                            _log.debug(f"🔧 记忆提取: CUDA_VISIBLE_DEVICES={cuda_visible}，使用重新映射设备 {device_for_inputs}（对应物理GPU {self.device}）")
                        else:
                            # 未设置CUDA_VISIBLE_DEVICES，使用原始设备配置
                            device_for_inputs = self.device
                            _log.debug(f"🔧 记忆提取: 使用设备 {device_for_inputs}")
                        inputs = {k: v.to(device_for_inputs) if isinstance(v, torch.Tensor) else v
                                 for k, v in inputs.items()}
                    
                    # 1. 让模型生成记忆条目列表
                    # 使用与模型运行时相同的生成参数（从config中读取）
                    gen_config = self.config.get("generation", {})
                    max_new_tokens = gen_config.get("max_new_tokens", 1000)
                    temperature = gen_config.get("temperature", 1.0)
                    do_sample = gen_config.get("do_sample", True)
                    top_p = gen_config.get("top_p", 0.95)
                    top_k = gen_config.get("top_k", 20)
                    repetition_penalty = gen_config.get("repetition_penalty", 1.0)
                    
                    # 使用transformers官方的model.generate()方法
                    with torch.no_grad():
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature if do_sample else None,
                            do_sample=do_sample,
                            top_p=top_p if do_sample else None,
                            top_k=top_k if do_sample else None,
                            repetition_penalty=repetition_penalty if repetition_penalty != 1.0 else None
                        )
                    
                    # 解码生成的文本
                    generated_text = processor.batch_decode(
                        generated[:, inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )[0].strip()
                    
                    # 2. 提取正式回答（从最后一个</think>标签后）
                    # 使用与api_server相同的提取逻辑（extract_final_reply函数）
                    import re
                    thinking_patterns = [
                        r'</think>\s*',
                        r'</thinking>\s*'
                    ]
                    
                    last_match = None
                    for pattern in thinking_patterns:
                        matches = list(re.finditer(pattern, generated_text, re.IGNORECASE))
                        if matches:
                            current_match = matches[-1]
                            if last_match is None or current_match.end() > last_match.end():
                                last_match = current_match
                    
                    if last_match:
                        # 提取最后一个标签后的内容（这是正式回答部分）
                        final_reply = generated_text[last_match.end():].strip()
                        _log.info(f"✅ 从模型输出中提取到正式回答（从最后一个{last_match.group(0).strip()}标签开始）")
                        generated_text = final_reply
                    else:
                        _log.warning("⚠️ 未找到</think>或</thinking>标签，使用完整输出")
                    
                    # 记录生成的原始文本（用于调试）
                    _log.info(f"📝 聊天组 {chat_key} (深度 {depth}) 模型生成的原始文本:")
                    _log.info("=" * 60)
                    _log.info(f"生成的文本长度: {len(generated_text)} 字符")
                    if len(generated_text) > 500:
                        _log.info(f"生成的文本（前500字符）: {generated_text[:500]}")
                    else:
                        _log.info(f"生成的完整文本: {generated_text}")
                    
                    # 2. 解析生成的文本，提取多个记忆条目
                    memory_texts = self._parse_memory_entries(generated_text)
                    
                    _log.info(f"📊 解析后提取到 {len(memory_texts)} 个记忆条目")
                    if memory_texts:
                        for i, mem_text in enumerate(memory_texts, 1):
                            _log.info(f"   记忆条目 {i}: {mem_text[:100]}...")
                    
                    # 3. 临时保存记忆条目文本到文件（不提取向量，等待批量处理）
                    for memory_text in memory_texts:
                        self._append_memory_text_to_file(memory_text, temp_texts_path)
                        _log.info(f"✅ 提取记忆条目文本 (深度 {depth}): {memory_text[:80]}...")
                
                except Exception as e:
                    _log.warning(f"处理聊天组 {chat_key} (深度 {depth}) 时出错: {e}", exc_info=True)
                    return
        
            # 对每个聊天组调用处理函数（第一阶段：只提取文本）
            for chat_key, messages in chat_groups.items():
                process_chat_group(messages, chat_key)

            # 检查是否提取到记忆条目
            if not os.path.exists(temp_texts_path):
                _log.warning("⚠️ 没有提取到任何记忆条目")
                return None

            # 从临时文件加载所有记忆条目文本
            all_memory_texts = self._load_memory_texts_from_file(temp_texts_path)
            if not all_memory_texts:
                _log.warning("⚠️ 临时文件中没有记忆条目")
                return None

            _log.info(f"📊 第一阶段完成：共提取 {len(all_memory_texts)} 个记忆条目文本")
            _log.info("=" * 60)
            _log.info("开始第二阶段：批量提取记忆条目向量")
            _log.info("=" * 60)

            # 第二阶段：批量提取所有记忆条目的向量
            all_texts, all_embeddings = self._batch_extract_embeddings(
                all_memory_texts, model, processor, max_tokens
            )

            # 保存所有条目
            if all_texts and all_embeddings:
                self._save_training_data_batch(all_texts, all_embeddings)
                _log.info(f"✅ 成功保存 {len(all_texts)} 个记忆条目及其向量到临时文件")
                
                # 删除只包含记忆条目文本的临时文件（已经获得向量，不再需要）
                try:
                    if os.path.exists(temp_texts_path):
                        os.remove(temp_texts_path)
                        _log.info(f"✅ 已删除临时文本文件: temp_memory_texts.pt")
                except Exception as e:
                    _log.warning(f"⚠️ 删除临时文本文件失败: {e}")
            else:
                _log.warning("❌ 没有成功提取到向量")
                return None

            # 获取最终的训练数据文件路径
            temp_data_path = os.path.join(self.memory_db_dir, "temp_training_data.pt")
            if os.path.exists(temp_data_path):
                data = torch.load(temp_data_path, map_location='cpu')
                total_entries = len(data.get('texts', []))
                _log.info(f"✅ 成功提取并保存 {total_entries} 个记忆条目到临时文件")
            else:
                _log.warning("❌ 没有生成训练数据文件")
                return None

        finally:
            # 清理临时变量
            if 'all_texts' in locals():
                del all_texts
            if 'all_embeddings' in locals():
                del all_embeddings
            
            # 如果程序异常退出，尝试清理临时文本文件
            temp_texts_path = os.path.join(self.memory_db_dir, "temp_memory_texts.pt")
            if os.path.exists(temp_texts_path):
                try:
                    # 检查是否已经有完整的训练数据文件
                    temp_data_path = os.path.join(self.memory_db_dir, "temp_training_data.pt")
                    if os.path.exists(temp_data_path):
                        # 如果有完整文件，删除文本临时文件
                        os.remove(temp_texts_path)
                        _log.debug(f"清理临时文本文件: temp_memory_texts.pt")
                except Exception as e:
                    _log.debug(f"清理临时文本文件失败（可能已被删除）: {e}")

            # 使用统一的训练模型，不在这里卸载
            # 模型将在训练流程结束后统一卸载
            _log.info("✅ 记忆提取完成（使用统一的训练模型）")

        return temp_data_path

    def _batch_extract_embeddings(
        self, 
        memory_texts: List[str], 
        model, 
        processor, 
        max_tokens: int
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        批量提取记忆条目的向量（第二阶段）
        
        Args:
            memory_texts: 记忆条目文本列表
            model: 模型实例
            processor: 处理器实例
            max_tokens: 最大token数
            
        Returns:
            (texts, embeddings) 元组，一一对应
        """
        all_texts = []
        all_embeddings = []
        
        # 确定设备
        if isinstance(self.device, list):
            model_device = next(model.parameters()).device
        else:
            # 单GPU模式：检查CUDA_VISIBLE_DEVICES设置状态
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible and cuda_visible.strip():
                # CUDA_VISIBLE_DEVICES已设置，使用重新映射后的设备
                model_device = "cuda:0"
                _log.debug(f"🔧 批量向量提取: CUDA_VISIBLE_DEVICES={cuda_visible}，使用重新映射设备 {model_device}（对应物理GPU {self.device}）")
            else:
                # 未设置CUDA_VISIBLE_DEVICES，使用原始设备配置
                model_device = self.device
                _log.debug(f"🔧 批量向量提取: 使用设备 {model_device}")
        
        # Batch大小（可以根据GPU显存调整）
        batch_size = self.training_config.get("embedding_batch_size", 8)
        _log.info(f"📦 使用batch_size={batch_size}进行批量向量提取")
        
        # 构建所有prompts
        prompts = []
        valid_indices = []  # 记录有效的索引（用于处理截断失败的情况）
        
        for idx, memory_text in enumerate(memory_texts):
            prompt = f"请用一个Token总结以下文本\"{memory_text}\"："
            prompts.append(prompt)
            valid_indices.append(idx)
        
        # 分批处理
        total_batches = (len(prompts) + batch_size - 1) // batch_size
        _log.info(f"📊 共 {len(prompts)} 个记忆条目，分为 {total_batches} 个batch处理")
        
        for batch_idx in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_idx:batch_idx + batch_size]
            batch_texts = [memory_texts[valid_indices[batch_idx + i]] for i in range(len(batch_prompts))]
            
            try:
                # Batch tokenize（自动处理padding）
                # 注意：对于纯文本，应该使用processor.tokenizer，而不是processor
                # processor是多模态处理器，会将文本误当作图片处理
                # 明确设置参数顺序，确保truncation在max_length之前
                batch_inputs = processor.tokenizer(
                    batch_prompts,
                    truncation=True,  # 明确设置截断
                    max_length=max_tokens,  # 设置最大长度
                    padding=True,  # 填充到相同长度
                    return_tensors="pt"
                )
                
                # 移动到设备
                batch_inputs = {
                    k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch_inputs.items()
                }
                
                # Batch推理
                with torch.no_grad():
                    backbone_outputs = forward_backbone(
                        model,
                        input_ids=batch_inputs["input_ids"],
                        attention_mask=batch_inputs["attention_mask"],
                        use_cache=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                
                # 提取每个样本的最后一个有效token的hidden state
                last_hidden_states = ensure_last_hidden_state(backbone_outputs)
                attention_mask = batch_inputs["attention_mask"]  # [batch_size, seq_len]
                
                for i in range(len(batch_prompts)):
                    # 找到最后一个有效token的位置
                    last_token_idx = attention_mask[i].sum().item() - 1
                    if last_token_idx < 0:
                        _log.warning(f"⚠️ Batch {batch_idx//batch_size + 1} 样本 {i} 的attention_mask无效，跳过")
                        continue
                    
                    # 提取最后一个token的embedding
                    embedding = last_hidden_states[i, last_token_idx, :].detach().cpu()
                    
                    all_texts.append(batch_texts[i])
                    all_embeddings.append(embedding)
                
                # 进度日志
                processed = min(batch_idx + batch_size, len(prompts))
                _log.info(f"✅ Batch {batch_idx//batch_size + 1}/{total_batches}: 已处理 {processed}/{len(prompts)} 个条目")
                
                # 定期清理显存
                if (batch_idx // batch_size + 1) % 10 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        _log.debug(f"🧹 已清理GPU显存（处理了 {processed} 个条目）")
                
            except Exception as e:
                _log.error(f"❌ Batch {batch_idx//batch_size + 1} 处理失败: {e}", exc_info=True)
                # 如果batch失败，尝试逐个处理这个batch中的条目
                _log.warning(f"🔄 尝试逐个处理该batch中的条目...")
                for i, memory_text in enumerate(batch_texts):
                    try:
                        prompt = batch_prompts[i]
                        # 注意：对于纯文本，应该使用processor.tokenizer，而不是processor
                        inputs = processor.tokenizer(
                            prompt,
                            truncation=True,
                            max_length=max_tokens,
                            return_tensors="pt"
                        )
                        inputs = {
                            k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                            for k, v in inputs.items()
                        }
                        
                        with torch.no_grad():
                            backbone_outputs = forward_backbone(
                                model,
                                input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                use_cache=False,
                                output_hidden_states=False,
                                return_dict=True,
                            )
                        
                        last_token_idx = inputs["attention_mask"].sum().item() - 1
                        if last_token_idx >= 0:
                            last_hidden = ensure_last_hidden_state(backbone_outputs)
                            embedding = last_hidden[0, last_token_idx, :].detach().cpu()
                            all_texts.append(memory_text)
                            all_embeddings.append(embedding)
                    except Exception as single_e:
                        _log.warning(f"⚠️ 单个条目处理也失败: {memory_text[:50]}... 错误: {single_e}")
                        continue
        
        _log.info(f"✅ 批量向量提取完成：成功提取 {len(all_embeddings)}/{len(memory_texts)} 个向量")
        
        return all_texts, all_embeddings

    def _append_memory_text_to_file(self, memory_text: str, file_path: str):
        """
        追加记忆条目文本到临时文件（追加模式）
        
        Args:
            memory_text: 记忆条目文本
            file_path: 临时文件路径
        """
        try:
            if os.path.exists(file_path):
                # 加载现有数据并追加
                existing_data = torch.load(file_path, map_location='cpu')
                existing_texts = existing_data.get('texts', [])
                existing_texts.append(memory_text)
            else:
                # 创建新文件
                existing_texts = [memory_text]
            
            # 保存到文件
            torch.save({"texts": existing_texts}, file_path)
        except Exception as e:
            _log.warning(f"追加记忆条目文本到文件失败: {e}")

    def _extract_sft_vectors_for_recall_training(
        self,
        num_memory_entries: int,
        model,
        processor
    ) -> Optional[str]:
        """
        提取等量的SFT向量用于第一步训练，防止<recall>token过拟合

        Args:
            num_memory_entries: 记忆条目数量，需要提取等量的SFT向量
            model: 基础模型
            processor: 基础处理器

        Returns:
            SFT向量文件路径，如果提取失败则返回None
        """
        try:
            if not self.sft_enabled or not self.sft_path:
                _log.info("ℹ️ SFT未启用或未配置，跳过SFT向量提取")
                return None

            # 计算需要的SFT向量数量：1.5倍于记忆条目数量
            required_sft_count = int(num_memory_entries * 1.5)
            _log.info(f"🧪 开始提取 {required_sft_count} 个SFT向量用于第一步训练（记忆条目数: {num_memory_entries}）")

            # 加载SFT数据集
            sft_samples = self._load_sft_dataset()
            if not sft_samples:
                _log.warning("⚠️ 无法加载SFT数据集，跳过SFT向量提取")
                return None

            # 随机抽取1.5倍数量的SFT样本
            if len(sft_samples) >= required_sft_count:
                selected_samples = random.sample(sft_samples, required_sft_count)
            else:
                _log.warning(f"⚠️ SFT数据集样本数 {len(sft_samples)} 少于所需数量 {required_sft_count}，使用全部样本")
                selected_samples = sft_samples

            # 提取思考部分内容（使用第二步验证过的方法）
            sft_thinking_texts = []
            for sample in selected_samples:
                # 使用第二步相同的标准化方法处理SFT数据
                messages = self._standardize_sft_messages(sample)
                if messages:
                    # 使用processor将messages转换为完整文本（包括所有消息）
                    try:
                        # 使用apply_chat_template转换为文本格式
                        full_text = processor.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False
                        )

                        # 检查是否包含思考部分
                        start_tag = "<think>"
                        end_tag = "</think>"
                        start_idx = full_text.find(start_tag)
                        end_idx = full_text.find(end_tag)

                        if start_idx != -1 and end_idx != -1:
                            # 提取思考部分的内容
                            thinking_content = full_text[start_idx + len(start_tag):end_idx].strip()
                            if thinking_content:
                                sft_thinking_texts.append(thinking_content)
                    except Exception as e:
                        _log.debug(f"处理SFT样本失败: {e}")
                        continue

            if not sft_thinking_texts:
                _log.warning("⚠️ 没有找到有效的SFT思考内容，跳过SFT向量提取")
                return None

            _log.info(f"✅ 提取到 {len(sft_thinking_texts)} 个SFT思考内容")

            # 使用_batch_extract_embeddings提取SFT向量
            sft_texts, sft_embeddings = self._batch_extract_embeddings(
                sft_thinking_texts, model, processor, max_tokens=35000
            )

            if not sft_embeddings:
                _log.warning("⚠️ SFT向量提取失败")
                return None

            # 保存SFT向量到临时文件
            sft_vectors_path = os.path.join(self.memory_db_dir, "temp_sft_vectors.pt")
            torch.save({
                "texts": sft_texts,
                "embeddings": torch.stack(sft_embeddings)
            }, sft_vectors_path)

            _log.info(f"✅ 已保存 {len(sft_embeddings)} 个SFT向量到临时文件: {sft_vectors_path}")
            return sft_vectors_path

        except Exception as e:
            _log.error(f"❌ SFT向量提取失败: {e}", exc_info=True)
            return None

    def _load_memory_texts_from_file(self, file_path: str) -> List[str]:
        """
        从临时文件加载所有记忆条目文本
        
        Args:
            file_path: 临时文件路径
            
        Returns:
            记忆条目文本列表
        """
        try:
            if not os.path.exists(file_path):
                return []
            
            data = torch.load(file_path, map_location='cpu')
            texts = data.get('texts', [])
            return texts
        except Exception as e:
            _log.error(f"从文件加载记忆条目文本失败: {e}")
            return []

    def _save_training_data_batch(self, texts: List[str], embeddings: List[torch.Tensor]):
        """
        分批保存训练数据到临时文件（追加模式）

        Args:
            texts: 记忆文本列表
            embeddings: 对应的向量列表
        """
        temp_data_path = os.path.join(self.memory_db_dir, "temp_training_data.pt")

        if not texts:
            return

        try:
            # 合并embeddings为张量
            embeddings_tensor = torch.stack(embeddings)

            if os.path.exists(temp_data_path):
                # 加载现有数据并追加
                existing_data = torch.load(temp_data_path, map_location='cpu')
                existing_texts = existing_data.get('texts', [])
                existing_embeddings = existing_data.get('embeddings')

                # 追加数据
                all_texts = existing_texts + texts
                all_embeddings = torch.cat([existing_embeddings, embeddings_tensor], dim=0)
            else:
                # 创建新文件
                all_texts = texts
                all_embeddings = embeddings_tensor

            # 保存到文件
            torch.save({
                "texts": all_texts,
                "embeddings": all_embeddings
            }, temp_data_path)

            _log.info(f"保存了 {len(texts)} 个条目的训练数据到临时文件（总计 {len(all_texts)} 个条目）")

        except Exception as e:
            _log.error(f"保存训练数据批次失败: {e}")
            raise

    def save_memory_embeddings_from_file(self, training_data_path: str):
        """
        从训练数据文件读取监督向量并保存到MemoryVectorDB

        Args:
            training_data_path: 训练数据文件路径
        """
        _log.info("从训练数据文件保存监督向量到MemoryVectorDB...")

        try:
            # 加载训练数据
            training_data = torch.load(training_data_path, map_location='cpu')
            embeddings = training_data.get('embeddings')

            if embeddings is None or len(embeddings) == 0:
                _log.warning("⚠️ 训练数据文件中没有向量数据")
                return

            # 记忆数据库文件路径
            memory_db_path = os.path.join(self.memory_db_dir, "memory_embeddings.pt")

            # 创建MemoryVectorDB并加载现有数据（如果存在）
            embedding_dim = embeddings.shape[-1]
            storage_device = "cpu"
            memory_db = MemoryVectorDB(embedding_dim=embedding_dim, device=storage_device)
            _log.info(f"MemoryVectorDB将在 {storage_device} 上执行保存操作，以避免GPU设备编号不一致问题")

            # 如果文件已存在，先加载现有数据
            if os.path.exists(memory_db_path):
                try:
                    memory_db.load_from_pt(memory_db_path)
                    _log.info(f"加载现有记忆数据库，已有 {memory_db.embeddings.shape[0]} 个向量")
                except Exception as e:
                    _log.warning(f"加载现有记忆数据库失败: {e}，将创建新的数据库")

            # 追加新的向量
            memory_db.add_vectors(embeddings)

            # 保存到文件
            memory_db.save_to_pt(memory_db_path)

            _log.info(f"✅ 成功保存 {len(embeddings)} 个新的监督向量到 {memory_db_path}（总计 {memory_db.embeddings.shape[0]} 个向量）")

        except Exception as e:
            _log.error(f"从文件保存记忆向量失败: {e}")
            raise

    def _parse_memory_entries(self, generated_text: str) -> List[str]:
        """
        解析模型生成的文本，提取多个独立的记忆条目
        
        Args:
            generated_text: 模型生成的文本，可能包含多个记忆条目
        
        Returns:
            记忆条目文本列表
        """
        if not generated_text or not generated_text.strip():
            return []
        
        # 检查是否是无记忆条目
        if "无记忆条目" in generated_text or "无" in generated_text[:20]:
            return []
        
        entries = []
        
        # 方法1: 按行解析，查找"条目X:"格式
        lines = generated_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 匹配"条目1:", "条目2:", "1.", "2."等格式
            import re
            # 匹配 "条目1:", "1.", "-" 等开头
            match = re.match(r'^(?:条目\s*\d+|[\d一二三四五六七八九十]+[\.、]|[-*])\s*[:：]?\s*(.+)', line)
            if match:
                entry_text = match.group(1).strip()
                if entry_text and len(entry_text) > 3:  # 至少3个字符
                    entries.append(entry_text)
            elif line and not line.startswith('条目') and len(line) > 10:
                # 如果没有匹配到格式，但内容较长，也可能是记忆条目
                # 检查是否包含常见的关键词（如"喜欢"、"是"、"在"等）
                if any(keyword in line for keyword in ['喜欢', '是', '在', '有', '的', '了', '会', '要', '去', '来']):
                    entries.append(line)
        
        # 方法2: 如果没有找到格式化的条目，尝试按句号、换行等分割
        if not entries:
            # 按句号、问号、感叹号分割
            sentences = re.split(r'[。！？\n]', generated_text)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 5:  # 至少5个字符
                    # 过滤掉明显不是记忆条目的内容（如"请分析"、"对话内容"等）
                    if not any(keyword in sentence for keyword in ['请', '分析', '对话', '内容', '记忆条目', '以下']):
                        entries.append(sentence)
        
        # 去重并过滤
        seen = set()
        unique_entries = []
        for entry in entries:
            # 归一化（去除多余空格）
            normalized = ' '.join(entry.split())
            if normalized not in seen and len(normalized) > 3:
                seen.add(normalized)
                unique_entries.append(normalized)
        
        return unique_entries
    
    def save_memory_embeddings(self, memory_entries: List[Tuple[str, torch.Tensor]]):
        """
        保存监督向量到MemoryVectorDB（追加模式）
        
        Args:
            memory_entries: (记忆文本, 监督向量) 的列表
        """
        _log.info("保存监督向量到MemoryVectorDB...")
        
        # 提取所有监督向量
        embeddings = torch.stack([entry[1] for entry in memory_entries])
        
        # 记忆数据库文件路径
        memory_db_path = os.path.join(self.memory_db_dir, "memory_embeddings.pt")
        
        # 创建MemoryVectorDB并加载现有数据（如果存在）
        # 注意：MemoryVectorDB主要用于存储，应该使用CPU以避免GPU设备问题
        embedding_dim = embeddings.shape[-1]
        storage_device = "cpu"
        memory_db = MemoryVectorDB(embedding_dim=embedding_dim, device=storage_device)
        _log.debug(f"MemoryVectorDB将在 {storage_device} 上执行保存操作")
        
        # 如果文件已存在，先加载现有数据
        if os.path.exists(memory_db_path):
            try:
                memory_db.load_from_pt(memory_db_path)
                _log.info(f"加载现有记忆数据库，已有 {memory_db.embeddings.shape[0]} 个向量")
            except Exception as e:
                _log.warning(f"加载现有记忆数据库失败: {e}，将创建新的数据库")
        
        # 追加新的向量
        memory_db.add_vectors(embeddings)
        
        # 保存到文件
        memory_db.save_to_pt(memory_db_path)

        _log.info(f"✅ 成功保存 {len(memory_entries)} 个新的监督向量到 {memory_db_path}（总计 {memory_db.embeddings.shape[0]} 个向量）")

        # 注意：memory_entries暂时保留在内存中，用于后续训练
        # 训练完成后在cleanup_after_training中统一清理

    def load_training_model(self):
        """加载统一的训练模型（用于记忆提取和训练）"""
        _log.info(f"加载训练模型: {self.base_model_path}")

        # 使用与initialize_model相同的加载逻辑
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        # 将相对路径转换为绝对路径
        model_path = self.base_model_path
        if not os.path.isabs(model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            model_path = os.path.abspath(os.path.join(project_root, model_path))

        # 检查是否为本地路径
        is_local_path = os.path.exists(model_path) and os.path.isdir(model_path)

        try:
            # 加载processor（使用AutoProcessor而不是AutoTokenizer，因为需要处理图片和视频）
            # 正常推理时使用AutoProcessor，训练时也应该使用AutoProcessor
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=is_local_path
            )

            # 准备加载参数
            load_kwargs = {
                "torch_dtype": torch.bfloat16,
                "trust_remote_code": True,
                "local_files_only": is_local_path
            }
            
            # 根据设备配置决定device_map（使用与TrainingModelContext相同的逻辑）
            multi_gpu_config = self.config.get("model", {}).get("multi_gpu", {})
            multi_gpu_enabled = multi_gpu_config.get("enabled", False)
            
            if isinstance(self.device, list) and multi_gpu_enabled:
                # 多GPU配置：使用优化的分配策略
                cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                cuda_visible_set = bool(cuda_visible)
                max_memory_config = multi_gpu_config.get("max_memory", {})
                allocation = _optimize_multi_gpu_allocation(self.device, max_memory_config, cuda_visible_set=cuda_visible_set)
                load_kwargs["device_map"] = allocation["device_map"]
                if allocation["max_memory"]:
                    load_kwargs["max_memory"] = allocation["max_memory"]
                _log.info(f"🔧 训练模型: 指定设备{self.device}，使用优化的分配策略")
            elif isinstance(self.device, str) and self.device.startswith("cuda"):
                load_kwargs["device_map"] = {"": self.device}
                _log.info(f"🔧 训练模型: 单GPU模式，设备映射到 {self.device}")
            else:
                load_kwargs["device_map"] = "auto"
                _log.info("🔧 训练模型: 使用自动设备分配")

            # 加载模型 - 注意这里使用Qwen3VLForConditionalGeneration，不是AutoModelForCausalLM
            from transformers import Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                **load_kwargs
            )

            # 添加特殊token（如果没有的话）
            # 使用MemoryTokenManager，与正常推理时保持一致
            from memory.token_manager import MemoryTokenManager
            token_manager = MemoryTokenManager(model, processor.tokenizer)
            recall_token_ids = token_manager.check_and_add_tokens(perturbation_std=0.02)
            _log.info(f"✅ 特殊token处理完成: {recall_token_ids}")

            _log.info("✅ 训练模型加载成功")
            return model, processor

        except Exception as e:
            _log.error(f"❌ 加载训练模型失败: {e}")
            raise
    
    def train_recall_token(self, training_data_path: str, model=None, processor=None, sft_vectors_path: Optional[str] = None) -> str:
        """
        第一步训练：训练<recall> token的embedding

        Args:
            training_data_path: 训练数据文件路径

        Returns:
            训练后的模型路径
        """
        # 尝试确保训练模块已加载
        if not _ensure_training_modules_loaded():
            raise ImportError("训练模块不可用，无法执行训练。请检查 recall/ 目录是否存在且包含必要的训练脚本。")

        _log.info("开始第一步训练：<recall> token embedding训练...")

        trainer = None
        try:
            # 从文件加载训练数据
            training_data = torch.load(training_data_path, map_location='cpu')
            texts = training_data.get('texts', [])
            embeddings = training_data.get('embeddings')

            if not texts or embeddings is None:
                raise ValueError("训练数据文件无效或为空")

            # 创建训练器（传入预加载的模型）
            lora_r = self.lora_config.get("r", 8)
            lora_alpha = self.lora_config.get("lora_alpha", 32)
            lora_dropout = self.lora_config.get("lora_dropout", 0.1)
            # 获取第一步训练的LoRA目标模块（如果配置了，只使用Q和V以减少显存）
            step1_lora_target_modules = self.lora_config.get("step1_lora_target_modules", None)
            # 获取梯度累积步数
            gradient_accumulation_steps = self.config.get("model", {}).get("multi_gpu", {}).get("gradient_accumulation_steps", 1)
            # 获取max_memory配置
            max_memory = self.config.get("model", {}).get("multi_gpu", {}).get("max_memory")

            # 获取第一步训练的最大序列长度（None表示不限制）
            max_length_recall_training = self.config.get("model", {}).get("training", {}).get("training_config", {}).get("max_length_recall_training")
            if max_length_recall_training is None:
                max_length_recall_training = None  # 明确设置为None

            trainer = RecallMemoryTrainer(
                self.base_model_path,
                device=self.device,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                original_device=self.original_device,
                preloaded_model=model,
                preloaded_tokenizer=processor,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_memory=max_memory,
                max_length=max_length_recall_training,
                lora_target_modules=step1_lora_target_modules
                # 第一步训练不设置epoch_end_hook，不插入SFT
            )

            # 准备训练数据：分别加载记忆条目和SFT向量
            memory_data = None
            sft_data = None

            # 加载记忆条目数据
            if os.path.exists(training_data_path):
                memory_data = torch.load(training_data_path, map_location='cpu')
                memory_count = len(memory_data.get('texts', []))
                _log.info(f"📖 加载记忆条目数据: {memory_count} 条")

            # 加载SFT向量数据
            if sft_vectors_path and os.path.exists(sft_vectors_path):
                sft_data = torch.load(sft_vectors_path, map_location='cpu')
                sft_count = len(sft_data.get('texts', []))
                _log.info(f"📖 加载SFT向量数据: {sft_count} 条")

            # 创建训练数据：记忆条目 + 随机抽取的SFT向量
            if memory_data and sft_data:
                memory_texts = memory_data.get('texts', [])
                memory_embeddings = memory_data.get('embeddings', torch.empty(0))
                sft_texts = sft_data.get('texts', [])
                sft_embeddings = sft_data.get('embeddings', torch.empty(0))

                memory_count = len(memory_texts)
                sft_total_count = len(sft_texts)

                # 计算需要的SFT向量数量：1.5倍于记忆条目数量
                required_sft_count = int(memory_count * 1.5)
                actual_sft_count = min(required_sft_count, sft_total_count)

                # 随机抽取SFT向量
                if actual_sft_count < sft_total_count:
                    import random
                    random.seed(42)  # 确保可重现
                    selected_indices = random.sample(range(sft_total_count), actual_sft_count)
                    selected_sft_texts = [sft_texts[i] for i in selected_indices]
                    selected_sft_embeddings = sft_embeddings[selected_indices]
                else:
                    selected_sft_texts = sft_texts
                    selected_sft_embeddings = sft_embeddings

                # 合并数据
                combined_texts = memory_texts + selected_sft_texts
                combined_embeddings = torch.cat([memory_embeddings, selected_sft_embeddings], dim=0)

                # 创建包含元信息的训练数据
                training_data = {
                    'texts': combined_texts,
                    'embeddings': combined_embeddings,
                    'memory_count': memory_count,  # 记忆条目数量
                    'sft_count': actual_sft_count  # SFT向量数量
                }

                temp_data_path = os.path.join(self.memory_db_dir, "temp_recall_training_data.pt")
                torch.save(training_data, temp_data_path)
                _log.info(f"✅ 已准备训练数据: {memory_count} 条记忆条目 + {actual_sft_count} 条SFT向量")

                # 删除SFT向量文件
                try:
                    os.remove(sft_vectors_path)
                    _log.info("🗑️ 已删除临时SFT向量文件")
                except Exception as e:
                    _log.warning(f"⚠️ 删除SFT向量文件失败: {e}")

            elif memory_data:
                # 只有记忆条目数据
                temp_data_path = training_data_path
                _log.info("ℹ️ 只有记忆条目数据，将直接使用")
            else:
                raise ValueError("❌ 没有找到有效的训练数据")

            # 训练
            embedding_epochs = self.training_config.get("embedding_epochs", 10)
            batch_size = self.training_config.get("batch_size", 4)
            learning_rate = float(self.training_config.get("learning_rate", 1e-4))

            step1_save_path = os.path.join(self.trained_model_dir, "step1_recall_token_trained")
            self._prepare_output_dir(step1_save_path)

            # Step1 只训练特殊token，此阶段不插入SFT
            self._current_epoch_sample_n = None
            res = trainer.train(
                pt_file_path=temp_data_path,
                num_epochs=embedding_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                save_path=step1_save_path
            )
            _ = res

            # 合并LoRA并保存
            model_path = trainer.merge_and_save_model(step1_save_path)
            if self.export_save_full_vl_assets:
                self._ensure_full_vl_assets(model_path)

            # 保存Processor配置
            self._save_processor_to_path(model_path)

            _log.info(f"第一步训练完成，模型保存在: {model_path}")
            return model_path

        except Exception as e:
            _log.error(f"第一步训练失败: {e}")
            raise
        finally:
            # 清理训练器创建的所有模型实例
            if trainer is not None:
                trainer.cleanup()
                del trainer

            # 删除临时合并的训练数据文件（如果存在）
            try:
                temp_merge_path = os.path.join(self.memory_db_dir, "temp_recall_training_data.pt")
                if os.path.exists(temp_merge_path):
                    os.remove(temp_merge_path)
                    _log.info("🗑️ 已删除临时合并的训练数据文件")
            except Exception as e:
                _log.warning(f"⚠️ 删除临时训练数据文件失败: {e}")
    
    def train_memory_decoding(self, training_data_path: str, model_path: str) -> str:
        """
        第二步训练：训练记忆解码能力

        Args:
            training_data_path: 训练数据文件路径
            model_path: 第一步训练后的模型路径

        Returns:
            训练后的模型路径
        """
        # 尝试确保训练模块已加载
        if not _ensure_training_modules_loaded():
            raise ImportError("训练模块不可用，无法执行训练。请检查 recall/ 目录是否存在且包含必要的训练脚本。")

        _log.info("开始第二步训练：记忆解码训练...")

        trainer = None
        try:
            # 从文件加载训练数据
            training_data = torch.load(training_data_path, map_location='cpu')
            texts = training_data.get('texts', [])
            embeddings = training_data.get('embeddings')

            if not texts or embeddings is None:
                raise ValueError("训练数据文件无效或为空")

            # 直接使用传入的训练数据文件路径
            temp_data_path = training_data_path

            # 在创建训练器之前，先确保模型中的特殊token存在
            # 使用MemoryTokenManager加载并检查token，然后将处理过的模型传递给训练器
            _log.info(f"🔧 预处理模型token: {model_path}")
            preloaded_model, preloaded_processor = TrainingModelContext.load_training_model(
                model_path, self.device, self.config.get("model", {}).get("multi_gpu", {})
            )
            _log.info("✅ 模型token预处理完成，已添加特殊token")

            # 创建训练器（传入预处理过的模型和tokenizer）
            lora_r = self.lora_config.get("r", 8)
            lora_alpha = self.lora_config.get("lora_alpha", 32)
            lora_dropout = self.lora_config.get("lora_dropout", 0.1)
            # 获取第二步训练的LoRA目标模块（如果配置了，使用完整配置）
            step2_lora_target_modules = self.lora_config.get("step2_lora_target_modules", None)
            # 获取梯度累积步数
            gradient_accumulation_steps = self.config.get("model", {}).get("multi_gpu", {}).get("gradient_accumulation_steps", 1)
            # 获取max_memory配置
            max_memory = self.config.get("model", {}).get("multi_gpu", {}).get("max_memory")

            dataset_max_length = int(self.training_config.get("memory_dataset_max_length", 3000) or 3000)
            test_sample_count = int(self.training_config.get("memory_test_sample_count", 2) or 2)
            test_max_new_tokens = int(self.training_config.get("memory_test_max_new_tokens", 300) or 300)
            test_use_cache = bool(self.training_config.get("memory_test_use_cache", False))
            activation_prompts = self.guides_config.get("activation_prompts")
            end_prompts = self.guides_config.get("end_prompts")

            trainer = EnhancedTextMemoryTrainer(
                model_path,
                device=self.device,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                original_device=self.original_device,
                preloaded_model=preloaded_model,  # 传入预处理过的模型
                preloaded_tokenizer=preloaded_processor,  # 传入预处理过的tokenizer
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_memory=max_memory,
                generation_config=self.config.get("generation", {}),
                epoch_end_hook=(lambda ep, tr: self._run_sft_one_epoch(tr, epoch=ep, epoch_sample_n=self._current_epoch_sample_n)),
                lora_target_modules=step2_lora_target_modules,
                dataset_max_length=dataset_max_length,
                test_sample_count=test_sample_count,
                test_max_new_tokens=test_max_new_tokens,
                test_use_cache=test_use_cache,
                activation_prompts=activation_prompts,
                end_prompts=end_prompts,
            )

            # 加载SFT数据并提取完整内容（截断点将在思考部分内部）
            sft_full_texts = []
            if self.sft_enabled and self.sft_path:
                try:
                    # 需要processor来将messages转换为文本
                    from transformers import AutoProcessor
                    processor = AutoProcessor.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        local_files_only=True
                    )
                    
                    sft_samples = self._load_sft_dataset()
                    for sample in sft_samples:
                        messages = self._standardize_sft_messages(sample)
                        if messages:
                            # 使用processor将messages转换为完整文本（包括所有消息）
                            try:
                                # 使用apply_chat_template转换为文本格式
                                full_text = processor.apply_chat_template(
                                    messages,
                                    tokenize=False,
                                    add_generation_prompt=False
                                )
                                
                                # 检查是否包含思考部分
                                start_tag = "<think>"
                                end_tag = "</think>"
                                start_idx = full_text.find(start_tag)
                                end_idx = full_text.find(end_tag)
                                
                                if start_idx != -1 and end_idx != -1:
                                    # 找到思考部分，保存完整文本和思考部分的起止位置
                                    # 注意：这里保存的是完整文本，截断会在训练时进行
                                    sft_full_texts.append({
                                        "full_text": full_text,
                                        "thinking_start": start_idx,
                                        "thinking_end": end_idx + len(end_tag)
                                    })
                            except Exception as e:
                                _log.debug(f"处理SFT样本失败: {e}")
                                continue
                    
                    _log.info(f"✅ 从SFT数据中提取了 {len(sft_full_texts)} 条完整文本，截断点将控制在思考部分内部")
                except Exception as e:
                    _log.warning(f"⚠️ 加载SFT数据失败，将使用记忆条目作为上下文: {e}")

            # 训练
            memory_epochs = self.training_config.get("memory_epochs", 20)
            batch_size = self.training_config.get("batch_size", 4)
            learning_rate = float(self.training_config.get("learning_rate", 1e-4))

            step2_save_path = os.path.join(self.trained_model_dir, "step2_memory_decoding_trained")
            self._prepare_output_dir(step2_save_path)

            # 设置SFT每epoch采样参考数（与记忆条目数量相同）
            training_data = torch.load(temp_data_path, map_location='cpu')
            memory_texts = training_data.get('texts', [])
            self._current_epoch_sample_n = len(memory_texts)
            res2 = trainer.train(
                pt_file_path=temp_data_path,
                num_epochs=memory_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                noise_std=0.01,
                save_path=step2_save_path,
                sft_full_texts=sft_full_texts if sft_full_texts else None
            )
            _ = res2

            # 合并LoRA并保存
            final_model_path = trainer.merge_and_save_model(step2_save_path)
            if self.export_save_full_vl_assets:
                self._ensure_full_vl_assets(final_model_path)

            # 保存Processor配置
            self._save_processor_to_path(final_model_path)

            _log.info(f"第二步训练完成，模型保存在: {final_model_path}")
            return final_model_path

        except Exception as e:
            _log.error(f"第二步训练失败: {e}")
            raise
        finally:
            # 清理训练器创建的所有模型实例
            if trainer is not None:
                trainer.cleanup()
                del trainer

    def cleanup_after_training(self):
        """
        训练完成后清理临时文件
        
        注意：
        - JSON聊天记录文件会被删除（已用于训练，不再需要）
        - 临时训练数据文件会被删除（训练完成后不再需要）
        - 内存中的聊天缓存会被清空（训练完成后不再需要）
        - 记忆向量数据库（memory_embeddings.pt）会被保留（这是训练好的记忆，需要保留）
        """
        _log.info("清理训练后的临时文件和缓存...")
        
        # 1. 清空JSON聊天记录文件（训练完成后不再需要）
        if os.path.exists(self.chat_history_storage_dir):
            json_files = list(Path(self.chat_history_storage_dir).glob("*.json"))
            deleted_count = 0
            for json_file in json_files:
                try:
                    os.remove(json_file)
                    deleted_count += 1
                    _log.info(f"删除JSON文件: {json_file.name}")
                except Exception as e:
                    _log.warning(f"删除JSON文件失败 {json_file}: {e}")
            if deleted_count > 0:
                _log.info(f"✅ 共删除 {deleted_count} 个JSON聊天记录文件")
        
        # 2. 删除临时训练数据文件
        temp_data_path = os.path.join(self.memory_db_dir, "temp_training_data.pt")
        if os.path.exists(temp_data_path):
            try:
                os.remove(temp_data_path)
                _log.info(f"✅ 删除临时训练数据文件: temp_training_data.pt")
            except Exception as e:
                _log.warning(f"删除临时训练数据文件失败: {e}")
        
        # 3. 清空内存中的聊天缓存
        # 使用延迟导入避免循环导入
        import importlib
        try:
            api_server = importlib.import_module('api_server_qwen3vl')
            group_chat_histories = getattr(api_server, 'group_chat_histories', {})
            private_chat_histories = getattr(api_server, 'private_chat_histories', {})
            
            group_count = len(group_chat_histories)
            private_count = len(private_chat_histories)
            
            group_chat_histories.clear()
            private_chat_histories.clear()
            
            _log.info(f"✅ 清空内存中的聊天缓存（群聊: {group_count}, 私聊: {private_count}）")
        except Exception as e:
            _log.warning(f"清空内存缓存失败: {e}")
        
        # 4. 记忆向量数据库（memory_embeddings.pt）会被保留，不删除
        memory_db_path = os.path.join(self.memory_db_dir, "memory_embeddings.pt")
        if os.path.exists(memory_db_path):
            _log.info(f"📌 记忆向量数据库已保留: {memory_db_path}（这是训练好的记忆，不会被删除）")
        
        _log.info("✅ 训练后的清理完成")

__all__ = ["MemoryTrainingService", "TrainingModelContext"]
