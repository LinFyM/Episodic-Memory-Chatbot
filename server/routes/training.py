# -*- coding: utf-8 -*-
from flask import Blueprint, jsonify, request
import threading
import os
import sys
from datetime import datetime
try:
    from .. import api_server_qwen3vl as api  # package mode
except Exception:
    import sys as _sys, os as _os
    _sys.path.append(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import api_server_qwen3vl as api

bp_training = Blueprint("training", __name__)

@bp_training.route("/api/training/trigger", methods=["POST"])
def training_trigger():
    try:
        api._log.info("收到手动训练触发请求（蓝图）")
        if api.training_scheduler is None:
            try:
                api._log.warning("检测到training_scheduler为None，尝试重新初始化...")
                from ..memory_training_scheduler import MemoryTrainingScheduler
                script_path = os.path.abspath(api.__file__)
                script_args = sys.argv[1:]
                api.training_scheduler = MemoryTrainingScheduler(api.config, script_path, script_args)
                if not hasattr(api.training_scheduler, 'scheduler') or not api.training_scheduler.scheduler.running:
                    api.training_scheduler.start()
            except Exception as init_error:
                api._log.error(f"训练调度器重新初始化失败: {init_error}")
                return jsonify({"success": False, "error": "训练调度器重新初始化失败"}), 500
        training_result = {"status": "running", "details": {}}
        def run_training_async():
            api._log.info("蓝图训练线程启动")
            api.run_training_async = True  # 标记仅用于调试
            # 复用 api_server_qwen3vl 中的完整实现
            from ..api_server_qwen3vl import trigger_training as original_trigger
            try:
                # 直接调用原实现以保持完全一致的行为
                pass
            finally:
                pass
        # 直接调用原端点逻辑，避免偏差
        return api.trigger_training()
    except Exception as e:
        api._log.error(f"[蓝图] 触发训练时出错: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@bp_training.route("/api/training/status", methods=["GET"])
def training_status():
    return api.get_training_status()

@bp_training.route("/api/training/debug", methods=["GET"])
def training_debug():
    return api.debug_training_scheduler()

@bp_training.route("/api/training/save-chat-history", methods=["POST"])
def training_save_chat_history():
    return api.save_chat_history_manually()


