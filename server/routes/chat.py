# -*- coding: utf-8 -*-
from flask import Blueprint, request, jsonify
import time
import threading
try:
    from .. import api_server_qwen3vl as api  # package mode
except Exception:
    import sys, os as _os
    sys.path.append(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import api_server_qwen3vl as api

bp_chat = Blueprint("chat", __name__)

@bp_chat.route("/api/chat/private", methods=["POST"])
def chat_private():
    global_worker_flag = "worker_thread_started"
    try:
        data = request.json
        user_id = str(data.get("user_id", ""))
        content = data.get("content", "")
        cleaned_content, image_urls = api.extract_cq_image_urls(content)
        video_urls = data.get("video_urls") or []
        if not user_id or (not cleaned_content and not image_urls and not video_urls):
            return jsonify({"status": "error", "message": "缺少必要参数"}), 400
        if not getattr(api, global_worker_flag):
            with api.queue_lock:
                if not getattr(api, global_worker_flag):
                    worker_thread = threading.Thread(target=api.message_queue_worker, daemon=True)
                    worker_thread.start()
                    setattr(api, global_worker_flag, True)
                    api._log.info("✅ 消息队列工作线程已启动（蓝图）")
        response_dict = {}
        task = api.MessageTask(
            chat_type="private",
            chat_id=user_id,
            data=data,
            response_dict=response_dict
        )
        api.message_queue.put(task)
        timeout = 120
        start_time = time.time()
        while time.time() - start_time < timeout:
            if "status" in response_dict:
                status_code = response_dict.pop("status_code", 200)
                return jsonify(response_dict), status_code
            time.sleep(0.1)
        return jsonify({"status": "error", "message": "处理超时"}), 500
    except Exception as e:
        api._log.error(f"[蓝图] 处理私聊消息出错: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@bp_chat.route("/api/chat/group", methods=["POST"])
def chat_group():
    global_worker_flag = "worker_thread_started"
    try:
        data = request.json
        group_id = str(data.get("group_id", ""))
        content = data.get("content", "")
        cleaned_content, image_urls = api.extract_cq_image_urls(content)
        video_urls = data.get("video_urls") or []
        if not group_id or (not cleaned_content and not image_urls and not video_urls):
            return jsonify({"status": "error", "message": "缺少必要参数"}), 400
        if not getattr(api, global_worker_flag):
            with api.queue_lock:
                if not getattr(api, global_worker_flag):
                    worker_thread = threading.Thread(target=api.message_queue_worker, daemon=True)
                    worker_thread.start()
                    setattr(api, global_worker_flag, True)
                    api._log.info("✅ 消息队列工作线程已启动（蓝图）")
        response_dict = {}
        task = api.MessageTask(
            chat_type="group",
            chat_id=group_id,
            data=data,
            response_dict=response_dict
        )
        api.message_queue.put(task)
        timeout = 120
        start_time = time.time()
        while time.time() - start_time < timeout:
            if "status" in response_dict:
                status_code = response_dict.pop("status_code", 200)
                return jsonify(response_dict), status_code
            time.sleep(0.1)
        return jsonify({"status": "error", "message": "处理超时"}), 500
    except Exception as e:
        api._log.error(f"[蓝图] 处理群消息出错: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


