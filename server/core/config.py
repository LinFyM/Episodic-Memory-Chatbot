# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    薄封装：委托到现有 api_server_qwen3vl.load_config，避免行为变化
    """
    from server import api_server_qwen3vl as api
    return api.load_config(config_path)

def get_default_config() -> Dict[str, Any]:
    from server import api_server_qwen3vl as api
    return api.get_default_config()


