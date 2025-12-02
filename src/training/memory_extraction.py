# -*- coding: utf-8 -*-
"""
记忆条目/向量提取相关的辅助函数。
从原 MemoryTrainingService 中拆分，方便独立维护与测试。
"""

from __future__ import annotations

import logging
import os
import json
import random
import re
import gc
from typing import List, Dict, Any, Tuple, Optional

import requests
from PIL import UnidentifiedImageError
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from training.model_utils import forward_backbone, ensure_last_hidden_state

_log = logging.getLogger(__name__)


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    从last_hidden_states中提取最后一个token的hidden state
    参考Qwen3-Embedding的官方实现
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class EmbeddingModelContext:
    """Embedding模型上下文管理器 - 管理embedding模型的生命周期"""
    
    def __init__(self, model_path: str, device, multi_gpu_config: Dict[str, Any] = None):
        """
        初始化embedding模型上下文管理器
        
        Args:
            model_path: embedding模型路径
            device: 设备（如 "cuda:0" 或 ["cuda:0", "cuda:1"]）
            multi_gpu_config: 多GPU配置（可选）
        """
        self.model_path = model_path
        self.device = device
        self.multi_gpu_config = multi_gpu_config or {}
        self.model = None
        self.tokenizer = None
    
    def __enter__(self):
        """进入上下文，加载embedding模型"""
        _log.info(f"加载embedding模型: {self.model_path}")
        self.model, self.tokenizer = self._load_embedding_model()
        return self.model, self.tokenizer
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，彻底清理模型"""
        _log.info("清理embedding模型上下文...")
        
        try:
            # 清理模型
            if self.model is not None:
                try:
                    self.model.cpu()
                except Exception as e:
                    _log.warning(f"移动embedding模型到CPU时出现警告: {e}")
                del self.model
                self.model = None
            
            # 清理tokenizer
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # 强制垃圾回收和显存清理
            for _ in range(3):
                gc.collect()
            
            # 清理GPU显存
            if torch.cuda.is_available():
                if isinstance(self.device, str) and self.device.startswith("cuda:"):
                    try:
                        device_idx = int(self.device.split(":")[1])
                        with torch.cuda.device(device_idx):
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                torch.cuda.empty_cache()
            
            _log.info("✅ embedding模型上下文清理完成")
        
        except Exception as cleanup_error:
            _log.warning(f"embedding模型上下文清理时出现错误: {cleanup_error}")
        
        return False  # 不抑制异常
    
    def _load_embedding_model(self):
        """加载embedding模型（内部方法）- 参考训练模型的加载方式"""
        # 将相对路径转换为绝对路径
        if not os.path.isabs(self.model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            model_path = os.path.abspath(os.path.join(project_root, self.model_path))
        else:
            model_path = self.model_path
        
        # 检查是否为本地路径
        is_local_path = os.path.exists(model_path) and os.path.isdir(model_path)
        
        _log.info(f"从路径加载embedding模型: {model_path} (本地: {is_local_path})")
        
        # 加载tokenizer（需要设置padding_side='left'）
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=is_local_path,
            padding_side='left'
        )
        
        # 准备加载参数（参考训练模型的加载方式）
        load_kwargs = {
            "trust_remote_code": True,
            "local_files_only": is_local_path,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        
        # 检查CUDA_VISIBLE_DEVICES设置状态
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        cuda_visible_set = bool(cuda_visible)
        
        # 根据设备配置决定device_map（参考训练模型的逻辑）
        multi_gpu_enabled = self.multi_gpu_config.get("enabled", False)
        
        if isinstance(self.device, list) and multi_gpu_enabled:
            # 多GPU配置（embedding模型通常不需要多GPU，但为了兼容性保留）
            from training.training_service import _optimize_multi_gpu_allocation
            max_memory_config = self.multi_gpu_config.get("max_memory", {})
            allocation = _optimize_multi_gpu_allocation(self.device, max_memory_config, cuda_visible_set=cuda_visible_set)
            load_kwargs["device_map"] = allocation["device_map"]
            if allocation["max_memory"]:
                load_kwargs["max_memory"] = allocation["max_memory"]
            _log.info(f"🔧 embedding模型: 多GPU模式，设备 {self.device}")
        elif isinstance(self.device, str) and self.device.startswith("cuda"):
            # 单GPU配置（参考训练模型的逻辑）
            if cuda_visible_set and cuda_visible:
                # CUDA_VISIBLE_DEVICES已设置，使用重新映射后的索引
                device_map_device = "cuda:0"
                _log.info(f"🔧 embedding模型: 单GPU模式，CUDA_VISIBLE_DEVICES={cuda_visible}，使用重新映射设备 {device_map_device}（对应物理GPU {self.device}）")
            else:
                # 未设置CUDA_VISIBLE_DEVICES，直接使用物理设备
                device_map_device = self.device
                _log.info(f"🔧 embedding模型: 单GPU模式，设备映射到 {self.device}")
            load_kwargs["device_map"] = {"": device_map_device}
        else:
            # 默认使用auto
            load_kwargs["device_map"] = "auto"
            _log.info("🔧 embedding模型: 使用自动设备分配")
        
        # 加载模型
        model = AutoModel.from_pretrained(
            model_path,
            **load_kwargs
        )
        
        model.eval()
        actual_device = next(model.parameters()).device
        _log.info(f"✅ embedding模型加载完成，实际设备: {actual_device}")
        
        return model, tokenizer


def _strip_formal_reply(generated_text: str) -> Tuple[str, bool, Optional[str]]:
    """
    从模型输出中截取最后一个 </think>/<thinking> 标签之后的正式回答。
    返回 (trimmed_text, trimmed_flag, matched_tag)。
    """
    if not generated_text:
        return generated_text, False, None

    thinking_patterns = [
        r"</think\s*>",
        r"</thinking\s*>",
    ]
    last_match = None
    for pattern in thinking_patterns:
        matches = list(re.finditer(pattern, generated_text, flags=re.IGNORECASE))
        if matches:
            candidate = matches[-1]
            if last_match is None or candidate.end() > last_match.end():
                last_match = candidate

    if last_match:
        trimmed = generated_text[last_match.end():].strip()
        return trimmed, True, last_match.group(0).strip()

    return generated_text, False, None


def extract_memory_entries(
    service,
    chat_messages: List[Dict[str, Any]],
    model=None,
    processor=None,
) -> Optional[str]:
    """
    提取记忆条目并生成监督向量，直接保存到临时文件
    
    注意：第一阶段（提取记忆文本）仍使用普通模型，第二阶段（提取向量）使用embedding模型

    Args:
        service: MemoryTrainingService 实例
        chat_messages: 聊天消息列表
        model: 普通模型（用于第一阶段提取记忆文本）
        processor: processor（用于第一阶段提取记忆文本）

    Returns:
        临时训练数据文件路径
    """
    self = service
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

    # 临时文件路径（只包含记忆条目文本）
    temp_texts_path = os.path.join(self.memory_db_dir, "temp_memory_texts.pt")
    _log.debug(f"记忆条目将保存到临时文件: {temp_texts_path}")

    # 检查是否已有提取好的记忆条目文件
    skip_stage1 = False
    if os.path.exists(temp_texts_path):
        all_memory_texts = _load_memory_texts_from_file(self, temp_texts_path)
        if all_memory_texts:
            _log.info(f"✅ 检测到已有的记忆条目文件，包含 {len(all_memory_texts)} 个记忆条目")
            _log.info("⏭️  跳过第一阶段（记忆条目文本提取），直接进入第二阶段（向量提取）")
            skip_stage1 = True
        else:
            _log.info("⚠️ 检测到记忆条目文件但内容为空，将重新提取")
            skip_stage1 = False

    # 从配置中获取最大token限制（用于批量提取向量时的截断）
    # 如果配置中没有，使用默认值 35000
    max_tokens = self.training_config.get("max_tokens_for_embedding", 35000)
    _log.debug(f"使用最大token限制: {max_tokens}（用于批量提取向量时的截断）")

    if not skip_stage1:
        # 检查是否提供了模型和processor（用于第一阶段）
        if model is None or processor is None:
            _log.error("❌ extract_memory_entries需要提供model和processor参数（用于第一阶段提取记忆文本）")
            return None

        _log.info("第一阶段：使用统一的训练模型进行记忆文本提取")

        # 角色设定（用于记忆提取时提醒模型自己的身份）
        role_playing_prompt = ""
        extraction_prompts = getattr(self, "memory_extraction_prompts", {}) or {}
        try:
            role_playing_prompt = self.config.get("prompt", {}).get("role_playing", "")
            if role_playing_prompt:
                role_playing_prompt = role_playing_prompt.strip()
        except Exception:
            role_playing_prompt = ""

    try:
        if not skip_stage1:
            # 对每个聊天组进行总结（递归处理，支持对半分）
            def process_chat_group(messages: List[Dict[str, Any]], chat_key: str, depth: int = 0):
                """
                处理单个聊天组（递归函数，支持对半分）
                """
                if not messages:
                    return

                # 构建标准格式的聊天历史（保留多模态信息）
                chat_messages_for_extraction = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")  # 默认值为空字符串，与旧版本保持一致

                    # 保持原始content格式（可能是list，包含图片信息）
                    if isinstance(content, list):
                        _log.debug(f"🔍 聊天组 {chat_key} 消息 {role} 的content是列表，包含 {len(content)} 项")
                        # 多模态内容，需要验证图片URL是否有效
                        filtered_content = []
                        image_count = 0
                        valid_image_count = 0
                        for item in content:
                            if item.get("type") == "text":
                                filtered_content.append(item)
                            elif item.get("type") == "image":
                                image_url = item.get("image", "")
                                image_count += 1
                                if image_url:
                                    if image_url.startswith('http://') or image_url.startswith('https://'):
                                        filtered_content.append(item)
                                        valid_image_count += 1
                                    else:
                                        _log.warning(f"⚠️ 聊天组 {chat_key} 图片URL格式无效，跳过")
                                else:
                                    _log.warning(f"⚠️ 聊天组 {chat_key} 发现无效的图片项（无URL），跳过")
                            elif item.get("type") == "video":
                                video_url = item.get("video") or item.get("url")
                                if not video_url:
                                    _log.warning(f"⚠️ 聊天组 {chat_key} 发现无效的视频项（无URL），跳过")
                                    continue

                                is_local_server_url = (
                                    video_url.startswith('http://127.0.0.1:9999/static/videos/') or
                                    video_url.startswith('http://localhost:9999/static/videos/') or
                                    (self.server_base_url and video_url.startswith(f"{self.server_base_url.rstrip('/')}/static/videos/"))
                                )
                                is_local_file = os.path.exists(video_url) and os.path.isfile(video_url)
                                is_file_url = video_url.startswith('file://') and os.path.exists(video_url[7:])

                                _log.debug(f"🔍 视频URL检查: {video_url}")
                                _log.debug(f"  is_local_server_url: {is_local_server_url}")
                                _log.debug(f"  is_local_file: {is_local_file} (文件存在: {os.path.exists(video_url) if video_url else False})")
                                _log.debug(f"  is_file_url: {is_file_url}")
                                _log.debug(f"  is_http: {video_url.startswith('http://') or video_url.startswith('https://') if video_url else False}")

                                if is_local_server_url or is_local_file or is_file_url or video_url.startswith('http://') or video_url.startswith('https://'):
                                    filtered_content.append({
                                        "type": "video",
                                        "video": video_url
                                    })
                                    _log.info(f"✅ 保留视频: {video_url}")
                                else:
                                    _log.warning(f"⚠️ 移除无效视频URL: {video_url}")

                        if filtered_content:
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
                        chat_messages_for_extraction.append({
                            "role": role,
                            "content": [{"type": "text", "text": content}]
                        })
                    else:
                        _log.warning(f"⚠️ 聊天组 {chat_key} 消息content格式未知: {type(content)}，跳过")

                extraction_system_prompt = _build_extraction_prompt(role_playing_prompt, extraction_prompts)

                if not chat_messages_for_extraction:
                    _log.warning(f"⚠️ 聊天组 {chat_key} 处理后无有效内容，跳过")
                    return

                if len(chat_messages_for_extraction) > self.training_config.get("max_messages_per_group", 80):
                    mid = len(messages) // 2
                    _log.debug(f"✂️ 聊天组 {chat_key} 消息过长，拆分为两个子组（深度 {depth + 1}）")
                    process_chat_group(messages[:mid], f"{chat_key}_part1", depth + 1)
                    process_chat_group(messages[mid:], f"{chat_key}_part2", depth + 1)
                    return

                try:
                    _log.info(f"🧠 开始处理聊天组 {chat_key}（深度 {depth}）...")
                    _log.info(f"   消息条数: {len(chat_messages_for_extraction)}")

                    if depth > 0:
                        child_prompt = self.training_config.get("child_depth_prompt")
                        if not child_prompt:
                            child_prompt = extraction_prompts.get("child_depth_prompt")
                        if child_prompt:
                            extraction_system_prompt += "\n\n" + child_prompt
                    
                    media_instruction = extraction_prompts.get("media_instruction")
                    if media_instruction:
                        extraction_system_prompt += f"\n\n{media_instruction}"
                    
                    activation_instruction = self.training_config.get("memory_activation_prompt")
                    if not activation_instruction:
                        activation_instruction = extraction_prompts.get("memory_activation_prompt")
                    if activation_instruction:
                        extraction_system_prompt += f"\n\n记忆激活提示：{activation_instruction}"
                    
                    user_prompt = extraction_prompts.get("user_prompt", "请开始提取记忆条目。")

                    full_messages = [
                        {"role": "system", "content": [{"type": "text", "text": extraction_system_prompt}]}
                    ]
                    full_messages.extend(chat_messages_for_extraction)
                    full_messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": user_prompt}]
                    })

                    try:
                        inputs = processor.apply_chat_template(
                            full_messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_dict=True,
                            return_tensors="pt"
                        )

                        input_ids_text = processor.batch_decode(
                            inputs["input_ids"],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False
                        )
                        _log.info("=" * 80)
                        _log.info("🔤 模型完整输入（包括特殊token）：")
                        _log.info(input_ids_text[0])
                        _log.info("=" * 80)
                    except (UnidentifiedImageError, OSError, requests.RequestException, Exception) as media_error:
                        _log.warning(f"⚠️ 聊天组 {chat_key} 图片/视频处理失败: {media_error}")
                        _log.warning(f"   错误类型: {type(media_error).__name__}", exc_info=True)
                        _log.info("   🔄 自动降级：移除所有图片和视频，只使用文本内容进行记忆提取...")

                        text_only_messages = []
                        for msg in full_messages:
                            msg_content = msg.get("content", [])
                            if isinstance(msg_content, list):
                                text_items = [item for item in msg_content if item.get("type") == "text"]
                                if text_items:
                                    text_only_messages.append({
                                        "role": msg.get("role", "user"),
                                        "content": text_items
                                    })
                            else:
                                text_only_messages.append(msg)

                        if not text_only_messages:
                            _log.warning(f"⚠️ 聊天组 {chat_key} 移除多模态内容后无有效消息，跳过处理")
                            return

                        inputs = processor.apply_chat_template(
                            text_only_messages,
                            tokenize=True,
                            add_generation_prompt=True,
                            return_dict=True,
                            return_tensors="pt"
                        )

                    input_length = inputs["input_ids"].shape[-1]
                    _log.info(f"📊 聊天组 {chat_key} (深度 {depth}) 输入token长度: {input_length}, 最大限制: {max_tokens}")

                    if input_length > max_tokens:
                        if len(messages) <= 1:
                            _log.warning(f"⚠️ 聊天组 {chat_key} 仅一条消息但token长度 {input_length} 超过限制 {max_tokens}，跳过处理")
                            return

                        _log.warning(f"⚠️ 聊天组 {chat_key} (深度 {depth}) 输入token长度超限，拆分原始消息重新处理")
                        half_point = len(messages) // 2
                        process_chat_group(messages[:half_point], f"{chat_key}_part1", depth + 1)
                        process_chat_group(messages[half_point:], f"{chat_key}_part2", depth + 1)
                        return

                    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

                    gen_config = self.config.get("generation", {})
                    max_new_tokens = gen_config.get("max_new_tokens", 1000)
                    temperature = gen_config.get("temperature", 1.0)
                    top_p = gen_config.get("top_p", 0.95)
                    top_k = gen_config.get("top_k", 20)
                    repetition_penalty = gen_config.get("repetition_penalty", 1.0)

                    _log.info(
                        "🎯 记忆提取生成参数: max_new_tokens=%s, temperature=%s, top_p=%s, top_k=%s, repetition_penalty=%s",
                        max_new_tokens,
                        temperature,
                        top_p,
                        top_k,
                        repetition_penalty,
                    )

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            repetition_penalty=repetition_penalty,
                            do_sample=True,
                        )

                    generated_text = processor.batch_decode(
                        outputs[:, inputs["input_ids"].shape[1]:],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False
                    )[0]
                    generated_text = generated_text.strip()

                    generated_text, trimmed, matched_tag = _strip_formal_reply(generated_text)
                    if trimmed:
                        tag_text = matched_tag or "</think>"
                        _log.info(f"✅ 从模型输出中提取到正式回答（截取最后一个{tag_text}之后的内容）")
                    else:
                        _log.warning("⚠️ 未找到</think>或</thinking>标签，使用完整输出")

                    _log.info(f"📝 聊天组 {chat_key} (深度 {depth}) 模型正式回答（长度 {len(generated_text)}）:")
                    _log.info(generated_text)

                    memory_texts = _parse_memory_entries(self, generated_text)

                    _log.info(f"📊 解析后提取到 {len(memory_texts)} 个记忆条目")
                    if memory_texts:
                        for i, mem_text in enumerate(memory_texts, 1):
                            _log.info(f"   记忆条目 {i}: {mem_text[:100]}...")

                    for memory_text in memory_texts:
                        _append_memory_text_to_file(self, memory_text, temp_texts_path)
                        _log.info(f"✅ 提取记忆条目文本 (深度 {depth}): {memory_text[:80]}...")

                except Exception as e:
                    _log.warning(f"处理聊天组 {chat_key} (深度 {depth}) 时出错: {e}", exc_info=True)
                    return

            for chat_key, messages in chat_groups.items():
                process_chat_group(messages, chat_key)

            if not os.path.exists(temp_texts_path):
                _log.warning("⚠️ 没有提取到任何记忆条目")
                return None

            all_memory_texts = _load_memory_texts_from_file(self, temp_texts_path)
            if not all_memory_texts:
                _log.warning("⚠️ 临时文件中没有记忆条目")
                return None
        else:
            # 使用已有的记忆条目文件
            all_memory_texts = _load_memory_texts_from_file(self, temp_texts_path)

        _log.info(f"📊 第一阶段完成：共提取 {len(all_memory_texts)} 个记忆条目文本")
        _log.info("=" * 60)
        _log.info("开始第二阶段：批量提取记忆条目向量（使用Qwen3-Embedding模型）")
        _log.info("=" * 60)

        # 获取embedding模型路径和设备
        embedding_model_path = self.memory_config.get("embedding_model_path")
        if not embedding_model_path:
            _log.error("❌ 未配置embedding_model_path，无法提取向量")
            return None
        
        # 获取多GPU配置（用于embedding模型加载）
        multi_gpu_config = self.config.get("model", {}).get("multi_gpu", {})
        
        # 使用embedding模型上下文提取向量（参考训练模型的加载方式）
        with EmbeddingModelContext(embedding_model_path, self.device, multi_gpu_config) as (embedding_model, embedding_tokenizer):
            all_texts, all_embeddings = _batch_extract_embeddings(
                self, all_memory_texts, embedding_model, embedding_tokenizer, max_tokens
            )

        if all_texts and all_embeddings:
            _save_training_data_batch(self, all_texts, all_embeddings)
            _log.info(f"✅ 成功保存 {len(all_texts)} 个记忆条目及其向量到临时文件")

            try:
                if os.path.exists(temp_texts_path):
                    os.remove(temp_texts_path)
                    _log.info(f"✅ 已删除临时文本文件: temp_memory_texts.pt")
            except Exception as e:
                _log.warning(f"⚠️ 删除临时文本文件失败: {e}")
        else:
            _log.warning("❌ 没有成功提取到向量")
            return None

        temp_data_path = os.path.join(self.memory_db_dir, "temp_training_data.pt")
        if os.path.exists(temp_data_path):
            data = torch.load(temp_data_path, map_location='cpu')
            total_entries = len(data.get('texts', []))
            _log.info(f"✅ 成功提取并保存 {total_entries} 个记忆条目到临时文件")
        else:
            _log.warning("❌ 没有生成训练数据文件")
            return None

        return temp_data_path

    finally:
        if 'all_texts' in locals():
            del all_texts
        if 'all_embeddings' in locals():
            del all_embeddings

        temp_texts_path = os.path.join(self.memory_db_dir, "temp_memory_texts.pt")
        if os.path.exists(temp_texts_path):
            try:
                temp_data_path = os.path.join(self.memory_db_dir, "temp_training_data.pt")
                if os.path.exists(temp_data_path):
                    os.remove(temp_texts_path)
                    _log.debug(f"清理临时文本文件: temp_memory_texts.pt")
            except Exception as e:
                _log.debug(f"清理临时文本文件失败（可能已被删除）: {e}")

        _log.info("✅ 记忆提取完成（使用统一的训练模型）")


def extract_sft_vectors_for_recall_training(
    service,
    num_memory_entries: int,
    model=None,
    processor=None
) -> Optional[str]:
    """
    提取等量的SFT向量用于第一步训练
    
    注意：现在使用Qwen3-Embedding模型提取向量，但提取SFT思考内容时仍需要processor来格式化消息
    """
    self = service
    try:
        if not self.sft_enabled or not self.sft_path:
            _log.info("ℹ️ SFT未启用或未配置，跳过SFT向量提取")
            return None

        required_sft_count = int(num_memory_entries * 1.5)
        _log.info(f"🧪 开始提取 {required_sft_count} 个SFT向量用于第一步训练（记忆条目数: {num_memory_entries}）")

        sft_samples = _load_sft_dataset(self)
        if not sft_samples:
            _log.warning("⚠️ 无法加载SFT数据集，跳过SFT向量提取")
            return None

        # 如果没有传递processor，从base_model_path加载processor（只加载processor，不加载模型）
        if processor is None:
            from transformers import AutoProcessor
            _log.info("📦 加载processor用于SFT消息格式化...")
            base_model_path = self.base_model_path
            if not os.path.isabs(base_model_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(script_dir))
                base_model_path = os.path.abspath(os.path.join(project_root, base_model_path))
            
            is_local_path = os.path.exists(base_model_path) and os.path.isdir(base_model_path)
            processor = AutoProcessor.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                local_files_only=is_local_path
            )
            _log.info("✅ processor加载完成")

        max_tokens = int(service.training_config.get("sft_max_tokens") or 0)
        tokenizer = service._get_base_tokenizer(processor)
        sft_thinking_texts = []
        collected = 0
        random.shuffle(sft_samples)
        processed = 0
        no_messages_count = 0
        no_tag_count = 0
        empty_content_count = 0
        token_exceeded_count = 0
        
        for sample in sft_samples:
            messages = _standardize_sft_messages(self, sample)
            if not messages:
                no_messages_count += 1
                continue
            processed += 1
            try:
                full_text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )

                start_tag = "<think>"
                end_tag = "</think>"
                start_idx = full_text.find(start_tag)
                end_idx = full_text.find(end_tag)

                if start_idx == -1 or end_idx == -1:
                    no_tag_count += 1
                    if processed <= 5:  # 只记录前5个样本的详细信息
                        _log.debug(f"样本 {processed} 未找到标签: start_idx={start_idx}, end_idx={end_idx}")
                    continue
                
                thinking_content = full_text[start_idx + len(start_tag):end_idx].strip()
                if not thinking_content:
                    empty_content_count += 1
                    continue
                    
                if max_tokens:
                    encoded = tokenizer(
                        thinking_content,
                        return_tensors="pt",
                        add_special_tokens=True,
                        padding=False,
                        truncation=False
                    )
                    if encoded["input_ids"].shape[1] > max_tokens:
                        token_exceeded_count += 1
                        continue
                        
                sft_thinking_texts.append(thinking_content)
                collected += 1
            except Exception as e:
                _log.debug(f"处理SFT样本失败: {e}", exc_info=True)
                continue
            if collected >= required_sft_count:
                break

        _log.info(f"📊 SFT样本处理统计: 处理了 {processed} 个样本，收集到 {collected} 个有效思考内容")
        _log.info(f"   无消息: {no_messages_count}, 无标签: {no_tag_count}, 空内容: {empty_content_count}, token超限: {token_exceeded_count}")

        if not sft_thinking_texts:
            _log.warning("⚠️ 没有找到有效的SFT思考内容，跳过SFT向量提取")
            _log.warning(f"   处理了 {processed} 个样本，但未找到包含 <think> 标签的有效内容")
            return None

        if len(sft_thinking_texts) < required_sft_count:
            raise ValueError(
                f"SFT思考样本不足：需要 {required_sft_count} 条满足token限制的thinking段，"
                f"但仅收集到 {len(sft_thinking_texts)} 条。"
            )

        _log.info(f"✅ 提取到 {len(sft_thinking_texts)} 个SFT思考内容")

        # 获取embedding模型路径和设备
        embedding_model_path = self.memory_config.get("embedding_model_path")
        if not embedding_model_path:
            _log.error("❌ 未配置embedding_model_path，无法提取SFT向量")
            return None
        
        # 获取多GPU配置（用于embedding模型加载）
        multi_gpu_config = self.config.get("model", {}).get("multi_gpu", {})
        
        # 使用embedding模型上下文提取向量（参考训练模型的加载方式）
        with EmbeddingModelContext(embedding_model_path, self.device, multi_gpu_config) as (embedding_model, embedding_tokenizer):
            sft_texts, sft_embeddings = _batch_extract_embeddings(
                self,
                sft_thinking_texts,
                embedding_model,
                embedding_tokenizer,
                self.training_config.get("max_tokens_for_embedding", 35000)
            )

        if not sft_embeddings or len(sft_embeddings) < required_sft_count:
            _log.warning("⚠️ SFT向量提取失败")
            return None

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


# ----------------------- Helper Functions -----------------------

def get_detailed_instruct(task_description: str, text: str) -> str:
    """
    构建带指令的文本，用于Qwen3-Embedding模型
    参考样例脚本：Instruct: {task_description}\nQuery:{text}
    
    Args:
        task_description: 任务描述，指导模型提取什么类型的嵌入
        text: 原始文本
    
    Returns:
        格式化的文本：Instruct: {task_description}\nQuery:{text}
    """
    return f'Instruct: {task_description}\nQuery:{text}'


def _batch_extract_embeddings(self, memory_texts, embedding_model, embedding_tokenizer, max_tokens):
    """
    使用Qwen3-Embedding模型批量提取向量
    
    根据Qwen3-Embedding的设计，可以通过Instruct格式指导提取的嵌入类型：
    - 对于需要指导的文本，使用格式：Instruct: {task_description}\nQuery:{text}
    - 对于普通文档，直接使用文本
    
    Args:
        memory_texts: 记忆文本列表
        embedding_model: Qwen3-Embedding模型
        embedding_tokenizer: Qwen3-Embedding tokenizer
        max_tokens: 最大token长度
    
    Returns:
        (all_texts, all_embeddings): 文本列表和对应的embedding列表
    """
    all_texts = []
    all_embeddings = []

    # 获取模型设备
    model_device = next(embedding_model.parameters()).device
    _log.debug(f"🔧 批量向量提取: 使用设备 {model_device}")

    # 不使用Instruct格式，直接使用文本提取表征（类似样例脚本中的documents）
    # 因为我们的记忆条目和SFT文本都是完整的语义单元，不需要指导
    use_instruct = False
    _log.info("📝 直接使用文本提取嵌入（不使用Instruct格式，类似样例脚本中的documents）")

    batch_size = self.training_config.get("embedding_batch_size", 8)
    _log.info(f"📦 使用batch_size={batch_size}进行批量向量提取（使用Qwen3-Embedding模型）")

    total_batches = (len(memory_texts) + batch_size - 1) // batch_size
    _log.info(f"📊 共 {len(memory_texts)} 个记忆条目，分为 {total_batches} 个batch处理")
    # 立即刷新日志，确保用户能看到进度信息
    for handler in _log.handlers:
        if hasattr(handler, 'flush'):
            handler.flush()

    for batch_idx in range(0, len(memory_texts), batch_size):
        batch_num = batch_idx // batch_size + 1
        batch_texts = memory_texts[batch_idx:batch_idx + batch_size]

        _log.info(f"🔄 开始处理 Batch {batch_num}/{total_batches} (条目 {batch_idx + 1}-{min(batch_idx + batch_size, len(memory_texts))}/{len(memory_texts)})")
        # 立即刷新日志
        for handler in _log.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()

        try:
            # 直接使用文本提取表征（类似样例脚本中的documents）
            # 不使用Instruct格式，因为记忆条目和SFT文本都是完整的语义单元
            batch_input_texts = batch_texts
            
            # 使用Qwen3-Embedding的方式：tokenize文本
            batch_dict = embedding_tokenizer(
                batch_input_texts,
                padding=True,
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt",
            )
            batch_dict = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}
            
            with torch.no_grad():
                outputs = embedding_model(**batch_dict)
                # 使用last_token_pool提取最后一个token的hidden state
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                # 归一化embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # 保存结果
            for i in range(len(batch_texts)):
                all_texts.append(batch_texts[i])
                all_embeddings.append(embeddings[i].detach().cpu())

            processed = min(batch_idx + batch_size, len(memory_texts))
            _log.info(f"✅ Batch {batch_num}/{total_batches} 完成: 已处理 {processed}/{len(memory_texts)} 个条目")
            # 立即刷新日志
            for handler in _log.handlers:
                if hasattr(handler, 'flush'):
                    handler.flush()

            if (batch_idx // batch_size + 1) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                _log.debug(f"🧹 已清理GPU显存（处理了 {processed} 个条目）")

        except Exception as e:
            _log.error(f"❌ Batch {batch_num} 处理失败: {e}", exc_info=True)
            _log.warning(f"🔄 尝试逐个处理该batch中的条目...")
            for i, memory_text in enumerate(batch_texts):
                try:
                    # 单个处理：直接使用文本（不使用Instruct格式）
                    single_dict = embedding_tokenizer(
                        memory_text,
                        padding=True,
                        truncation=True,
                        max_length=max_tokens,
                        return_tensors="pt",
                    )
                    single_dict = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in single_dict.items()}
                    
                    with torch.no_grad():
                        outputs = embedding_model(**single_dict)
                        embedding = last_token_pool(outputs.last_hidden_state, single_dict['attention_mask'])
                        embedding = F.normalize(embedding, p=2, dim=1)
                    
                    all_texts.append(memory_text)
                    all_embeddings.append(embedding[0].detach().cpu())
                except Exception as single_e:
                    _log.warning(f"⚠️ 单个条目处理也失败: {memory_text[:50]}... 错误: {single_e}")
                    continue

    _log.info(f"✅ 批量向量提取完成：成功提取 {len(all_embeddings)}/{len(memory_texts)} 个向量")
    return all_texts, all_embeddings


def _append_memory_text_to_file(self, memory_text: str, file_path: str):
    try:
        if os.path.exists(file_path):
            existing_data = torch.load(file_path, map_location='cpu')
            existing_texts = existing_data.get('texts', [])
            existing_texts.append(memory_text)
        else:
            existing_texts = [memory_text]

        torch.save({"texts": existing_texts}, file_path)
    except Exception as e:
        _log.warning(f"追加记忆条目文本到文件失败: {e}")


def _load_memory_texts_from_file(self, file_path: str) -> List[str]:
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
    temp_data_path = os.path.join(self.memory_db_dir, "temp_training_data.pt")

    if not texts:
        return

    try:
        embeddings_tensor = torch.stack(embeddings)

        if os.path.exists(temp_data_path):
            existing_data = torch.load(temp_data_path, map_location='cpu')
            existing_texts = existing_data.get('texts', [])
            existing_embeddings = existing_data.get('embeddings')
            all_texts = existing_texts + texts
            all_embeddings = torch.cat([existing_embeddings, embeddings_tensor], dim=0)
        else:
            all_texts = texts
            all_embeddings = embeddings_tensor

        torch.save({
            "texts": all_texts,
            "embeddings": all_embeddings
        }, temp_data_path)

        _log.info(f"保存了 {len(texts)} 个条目的训练数据到临时文件（总计 {len(all_texts)} 个条目）")

    except Exception as e:
        _log.error(f"保存训练数据批次失败: {e}")
        raise


def _load_sft_dataset(self) -> List[Dict[str, Any]]:
    dataset_path = self.sft_path
    if not dataset_path:
        _log.warning("⚠️ 未配置SFT数据集路径")
        return []
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.abspath(os.path.join(self._project_root, dataset_path))
    if not os.path.exists(dataset_path):
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
    msgs = sample.get("messages")
    if isinstance(msgs, list) and msgs:
        std = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content") or m.get("text") or ""
            if isinstance(content, str):
                std.append({"role": role, "content": [{"type": "text", "text": content}]})
            elif isinstance(content, list):
                text_join = ""
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_join += item.get("text", "")
                    elif isinstance(item, str):
                        text_join += item
                std.append({"role": role, "content": [{"type": "text", "text": text_join}]})
        if std:
            return std
    inst = sample.get("instruction") or sample.get("input")
    out = sample.get("output") or sample.get("answer")
    if isinstance(inst, str) and isinstance(out, str):
        return [
            {"role": "user", "content": [{"type": "text", "text": inst}]},
            {"role": "assistant", "content": [{"type": "text", "text": out}]},
        ]
    q = sample.get("query") or sample.get("question")
    a = sample.get("response") or sample.get("answer")
    if isinstance(q, str) and isinstance(a, str):
        return [
            {"role": "user", "content": [{"type": "text", "text": q}]},
            {"role": "assistant", "content": [{"type": "text", "text": a}]},
        ]
    return []


def _parse_memory_entries(self, generated_text: str) -> List[str]:
    """
    解析模型输出，提取格式化的记忆条目。严格按照 v1.2 的解析逻辑，避免把思考内容或指令原文当成记忆。
    """
    if not generated_text or not generated_text.strip():
        return []

    text = generated_text.strip()
    text = re.sub(r"<\|[^>]+?\|>", "", text)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?thinking>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?analysis>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?reflect>", "", text, flags=re.IGNORECASE)
    text = text.replace("<no_reply>", "").replace("</no_reply>", "")
    # 显式拒绝“无记忆”提示
    if "无记忆条目" in text or "无记忆" in text[:20]:
        return []

    entries: List[str] = []

    # 方法1：逐行匹配“条目/编号”格式
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r'^(?:条目\s*\d+|[\d一二三四五六七八九十]+\s*[\.、]|[-*])\s*[:：]?\s*(.+)', stripped)
        if match:
            candidate = match.group(1).strip()
            if candidate and len(candidate) > 3:
                entries.append(candidate)
        elif len(stripped) > 10:
            # 备用：没有明显编号但看起来像事实陈述
            if any(keyword in stripped for keyword in ["喜欢", "是", "在", "有", "的", "了", "会", "要", "去", "来"]):
                entries.append(stripped)

    # 方法2：若仍无条目，按句子切分并过滤掉指令/思考描述
    if not entries:
        sentences = re.split(r'[。！？\n]', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) <= 5:
                continue
            if any(keyword in sentence for keyword in ["请", "分析", "对话", "内容", "记忆条目", "以下", "思考", "步骤"]):
                continue
            entries.append(sentence)

    # 去重、去噪
    seen = set()
    cleaned: List[str] = []
    for entry in entries:
        normalized = " ".join(entry.split())
        if len(normalized) < 3:
            continue
        if normalized in seen:
            continue
        if any(token in normalized for token in ["对话内容", "提取", "请开始", "思考内容"]):
            continue
        seen.add(normalized)
        cleaned.append(normalized)

    _log.debug(f"解析后的记忆条目: {cleaned}")
    return cleaned


def _build_extraction_prompt(role_prompt: str, extraction_prompts: Dict[str, Any]) -> str:
    prompts_cfg = extraction_prompts or {}
    base_prompt = prompts_cfg.get("system_prompt")
    if not base_prompt:
        raise ValueError("memory_extraction.system_prompt 未配置，请在 prompts.yaml 中设置")
    wrapper = prompts_cfg.get("role_prompt_wrapper")
    if role_prompt:
        if wrapper and "{role_prompt}" in wrapper:
            base_prompt += "\n\n" + wrapper.replace("{role_prompt}", role_prompt)
        else:
            base_prompt += f"\n\n{role_prompt}"
    return base_prompt.strip()

