# -*- coding: utf-8 -*-
from typing import Optional

def initialize_model(model_path: Optional[str] = None, device_id: str = "cuda:0"):
    """
    薄封装：委托到现有 api_server_qwen3vl.initialize_model
    """
    from server import api_server_qwen3vl as api
    return api.initialize_model(model_path, device_id)


