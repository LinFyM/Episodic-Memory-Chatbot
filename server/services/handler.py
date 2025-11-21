# -*- coding: utf-8 -*-
from typing import Any

def run_process_message_task(task: Any) -> None:
    """
    动态导入并委托到现有 api_server_qwen3vl.process_message_task
    便于后续将实现迁移进 services 而不改调用方。
    """
    try:
        # 延迟导入，避免模块级循环依赖
        from server import api_server_qwen3vl as api
        api.process_message_task(task)
    except Exception as e:
        # 这里不引入日志依赖，保持最小耦合
        raise


