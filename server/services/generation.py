# -*- coding: utf-8 -*-
import torch
from transformers import StoppingCriteria

class InterruptStoppingCriteria(StoppingCriteria):
    """支持中断的StoppingCriteria（独立于服务器主模块）"""
    def __init__(self, interrupt_event):
        self.interrupt_event = interrupt_event
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.interrupt_event and self.interrupt_event.is_set():
            return True
        return False


