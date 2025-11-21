# -*- coding: utf-8 -*-
from typing import Any, Dict, List

def get_metrics() -> Dict[str, Any]:
    """
    薄封装：读现有 api_server_qwen3vl.metrics
    """
    from server import api_server_qwen3vl as api
    return api.metrics

def add_metric(key: str, value: int = 1) -> None:
    from server import api_server_qwen3vl as api
    api._metrics_add(key, value)

def add_latency(ms: float) -> None:
    from server import api_server_qwen3vl as api
    api._metrics_add_latency(ms)


