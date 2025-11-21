# -*- coding: utf-8 -*-
from flask import Blueprint, jsonify, request, url_for
import base64
from datetime import datetime
from uuid import uuid4
import os
try:
    from .. import api_server_qwen3vl as api  # package mode
except Exception:
    import sys, os as _os
    sys.path.append(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import api_server_qwen3vl as api

bp_upload = Blueprint("upload", __name__)

@bp_upload.route("/api/upload/image", methods=["POST"])
def upload_image():
    try:
        payload = request.get_json(force=True) or {}
        image_data = payload.get("data")
        image_format = str(payload.get("format", "jpeg")).lower().strip()
        if not image_data:
            return jsonify({"status": "error", "message": "缺少图片数据"}), 400
        format_map = {
            "jpg": "jpg",
            "jpeg": "jpg",
            "png": "png",
            "webp": "webp",
            "gif": "gif",
        }
        file_ext = format_map.get(image_format, "jpg")
        try:
            image_bytes = base64.b64decode(image_data, validate=True)
        except Exception as decode_err:
            api._log.warning(f"图片Base64解码失败: {decode_err}")
            return jsonify({"status": "error", "message": "图片数据无效"}), 400
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        filename = f"{timestamp}_{uuid4().hex}.{file_ext}"
        file_path = os.path.join(api.IMAGE_UPLOAD_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        file_url = url_for('serve_uploaded_image', filename=filename, _external=True)
        api._log.info(f"✅ 图片已保存(蓝图): {file_path} -> {file_url}")
        return jsonify({"status": "success", "url": file_url, "filename": filename}), 200
    except Exception as e:
        api._log.error(f"[蓝图] 图片上传失败: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@bp_upload.route("/api/upload/video", methods=["POST"])
def upload_video():
    try:
        # 支持两种上传方式：Base64数据 或 文件直接上传
        if request.content_type and 'multipart/form-data' in request.content_type:
            # 文件直接上传方式
            if 'file' not in request.files:
                return jsonify({"status": "error", "message": "缺少文件"}), 400
            file = request.files['file']
            if not file or not file.filename:
                return jsonify({"status": "error", "message": "无效的文件"}), 400

            # 获取文件扩展名
            filename = file.filename
            file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'mp4'
            if file_ext not in ['mp4', 'mov', 'webm', 'mkv', 'avi', 'm4v']:
                file_ext = 'mp4'

            # 保存文件
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
            new_filename = f"{timestamp}_{uuid4().hex}.{file_ext}"
            file_path = os.path.join(api.VIDEO_UPLOAD_DIR, new_filename)
            file.save(file_path)

            file_url = url_for('serve_uploaded_video', filename=new_filename, _external=True)
            api._log.info(f"✅ 视频已保存(文件上传): {file_path} -> {file_url}")
            return jsonify({"status": "success", "url": file_url, "filename": new_filename}), 200

        else:
            # Base64数据上传方式（原有逻辑）
            payload = request.get_json(force=True) or {}
            video_data = payload.get("data")
            video_format = str(payload.get("format", "mp4")).lower().strip()
            if not video_data:
                return jsonify({"status": "error", "message": "缺少视频数据"}), 400
            format_map = {
                "mp4": "mp4",
                "mov": "mov",
                "webm": "webm",
                "mkv": "mkv",
                "avi": "avi",
            }
            file_ext = format_map.get(video_format, "mp4")
            try:
                video_bytes = base64.b64decode(video_data, validate=True)
            except Exception as decode_err:
                api._log.warning(f"视频Base64解码失败: {decode_err}")
                return jsonify({"status": "error", "message": "视频数据无效"}), 400
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
            filename = f"{timestamp}_{uuid4().hex}.{file_ext}"
            file_path = os.path.join(api.VIDEO_UPLOAD_DIR, filename)
            with open(file_path, "wb") as f:
                f.write(video_bytes)
            file_url = url_for('serve_uploaded_video', filename=filename, _external=True)
            api._log.info(f"✅ 视频已保存(Base64上传): {file_path} -> {file_url}")
            return jsonify({"status": "success", "url": file_url, "filename": filename}), 200
    except Exception as e:
        api._log.error(f"[蓝图] 视频上传失败: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

