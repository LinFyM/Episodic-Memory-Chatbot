# -*- coding: utf-8 -*-
"""
应用装配入口（蓝图优先）
- 创建全新 Flask 应用并注册蓝图与静态路由
- 复用 api_server_qwen3vl 的全局对象/目录/逻辑，但不复用其中的 app
"""
from flask import Flask, send_from_directory
# from flask_cors import CORS  # 临时注释掉，网络问题无法安装
import os
import yaml

# ⚠️ 关键：在导入torch之前设置CUDA_VISIBLE_DEVICES
# 先加载配置，检查是否需要设置CUDA_VISIBLE_DEVICES
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config_qwen3vl.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            early_config = yaml.safe_load(f)
        device_config = early_config.get("model", {}).get("device", "cuda:0")
        if isinstance(device_config, list):
            # 多GPU配置，提取GPU索引并设置CUDA_VISIBLE_DEVICES
            gpu_indices = []
            for device in device_config:
                if device.startswith("cuda:"):
                    try:
                        gpu_idx = int(device.split(":")[1])
                        gpu_indices.append(str(gpu_idx))
                    except (ValueError, IndexError):
                        pass
            if gpu_indices:
                cuda_visible_devices = ",".join(gpu_indices)
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
                print(f"🔧 在导入torch之前设置CUDA_VISIBLE_DEVICES={cuda_visible_devices}（对应实际GPU {device_config}）")
        elif isinstance(device_config, str) and device_config.startswith("cuda:"):
            # 单GPU配置，也需要设置CUDA_VISIBLE_DEVICES
            try:
                gpu_idx = int(device_config.split(":")[1])
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                print(f"🔧 在导入torch之前设置CUDA_VISIBLE_DEVICES={gpu_idx}（对应实际GPU {device_config}）")
            except (ValueError, IndexError):
                print(f"⚠️ 无法解析单GPU配置: {device_config}")
except Exception as e:
    print(f"⚠️ 预加载配置失败，将在模型初始化时设置CUDA_VISIBLE_DEVICES: {e}")

try:
    # 支持以包方式运行：python -m server.app
    from . import api_server_qwen3vl as api
    from .routes.health import bp_health
    from .routes.chat import bp_chat
    from .routes.upload import bp_upload
    from .routes.training import bp_training
except Exception:
    # 支持直接脚本运行：python server/app.py
    import sys
    import os as _os
    sys.path.append(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import api_server_qwen3vl as api
    from routes.health import bp_health
    from routes.chat import bp_chat
    from routes.upload import bp_upload
    from routes.training import bp_training
finally:
    # 统一模块实例，避免同时存在 'api_server_qwen3vl' 与 'server.api_server_qwen3vl' 导致的全局状态分裂
    import sys as _sys
    _sys.modules['api_server_qwen3vl'] = api
    _sys.modules['server.api_server_qwen3vl'] = api

def create_app():
    app = Flask(__name__)
    # CORS(app)  # 临时注释掉，网络问题无法安装

    # 静态文件路由，委托到 api 模块中的上传目录
    @app.route("/static/images/<path:filename>")
    def serve_uploaded_image(filename: str):
        return send_from_directory(api.IMAGE_UPLOAD_DIR, filename)

    @app.route("/static/videos/<path:filename>")
    def serve_uploaded_video(filename: str):
        return send_from_directory(api.VIDEO_UPLOAD_DIR, filename)

    @app.route("/static/audios/<path:filename>")
    def serve_uploaded_audio(filename: str):
        return send_from_directory(api.AUDIO_UPLOAD_DIR, filename)

    @app.route("/static/files/<path:filename>")
    def serve_uploaded_file(filename: str):
        return send_from_directory(api.FILE_UPLOAD_DIR, filename)

    # 注册蓝图（真实实现）
    app.register_blueprint(bp_health)
    app.register_blueprint(bp_chat)
    app.register_blueprint(bp_upload)
    app.register_blueprint(bp_training)

    # 初始化必要的配置（设定 server_base_url，避免下载到本地时URL拼接失败）
    try:
        api.config = api.load_config(None)
        host_for_url = api.config["server"].get("public_host") or api.config["server"].get("host", "127.0.0.1")
        if host_for_url in ("0.0.0.0", "::"):
            host_for_url = "127.0.0.1"
        api.server_base_url = f"http://{host_for_url}:{api.config['server']['port']}"
    except Exception:
        # 安全兜底
        api.server_base_url = "http://127.0.0.1:9999"

    # 初始化模型（修复未加载模型导致无CUDA占用的问题）
    try:
        device = api.config.get("model", {}).get("device", "cuda:0")
        
        # 优先尝试查找最新训练模型
        api._log.info("=" * 60)
        api._log.info("🔍 开始查找模型路径（优先查找最新训练模型）")
        api._log.info("=" * 60)
        
        memory_config = api.config.get("memory", {}).get("training", {})
        trained_model_dir = memory_config.get("trained_model_dir", "./server/models/trained")
        token_added_model_dir = memory_config.get("token_added_model_dir", "./server/models/token_added")
        
        # 转换为绝对路径（相对于项目根目录）
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))  # server目录
        project_root = os.path.dirname(script_dir)  # 项目根目录
        if not os.path.isabs(trained_model_dir):
            # 路径相对于项目根目录，直接拼接
            trained_model_dir = os.path.abspath(os.path.join(project_root, trained_model_dir))
        if not os.path.isabs(token_added_model_dir):
            token_added_model_dir = os.path.abspath(os.path.join(project_root, token_added_model_dir))
        
        # 确保目录存在（避免os.listdir报错）
        os.makedirs(trained_model_dir, exist_ok=True)
        os.makedirs(token_added_model_dir, exist_ok=True)
        
        api._log.info(f"📁 训练模型目录: {trained_model_dir}")
        
        model_path = None
        # 查找所有按时间戳命名的模型目录
        if os.path.exists(trained_model_dir):
            api._log.info(f"✅ 训练模型目录存在，开始扫描...")
            model_dirs = [
                d for d in os.listdir(trained_model_dir)
                if os.path.isdir(os.path.join(trained_model_dir, d)) and d.startswith("model_")
            ]
            
            if model_dirs:
                api._log.info(f"📊 找到 {len(model_dirs)} 个训练模型目录")
                # 按时间戳排序，选择最新的
                model_dirs.sort(reverse=True)
                api._log.info(f"📋 训练模型列表（按时间排序）:")
                for i, d in enumerate(model_dirs[:5], 1):  # 只显示前5个
                    api._log.info(f"   {i}. {d}")
                if len(model_dirs) > 5:
                    api._log.info(f"   ... 还有 {len(model_dirs) - 5} 个模型")
                
                latest_model = os.path.join(trained_model_dir, model_dirs[0])
                model_path = latest_model
                api._log.info("=" * 60)
                api._log.info(f"✅ 找到最新训练模型，将优先加载")
                api._log.info(f"📦 模型路径: {model_path}")
                api._log.info(f"📅 模型时间戳: {model_dirs[0]}")
                api._log.info("=" * 60)
            else:
                api._log.warning(f"⚠️ 训练模型目录存在但为空，未找到任何训练模型")
        else:
            api._log.warning(f"⚠️ 训练模型目录不存在: {trained_model_dir}")
        
        # 如果没有训练模型，查找添加了token的模型
        if model_path is None:
            api._log.info("=" * 60)
            api._log.info("🔍 未找到训练模型，检查已添加token的模型")
            api._log.info("=" * 60)
            api._log.info(f"📁 token模型目录: {token_added_model_dir}")
            if os.path.exists(token_added_model_dir):
                model_dirs = [
                    d for d in os.listdir(token_added_model_dir)
                    if os.path.isdir(os.path.join(token_added_model_dir, d)) and d.startswith("model_")
                ]
                if model_dirs:
                    model_dirs.sort(reverse=True)
                    latest_token_model = os.path.join(token_added_model_dir, model_dirs[0])
                    model_path = latest_token_model
                    api._log.info(f"✅ 找到添加了token的模型: {model_path}")
                    api._log.info(f"📅 模型时间戳: {model_dirs[0]}")
                else:
                    api._log.warning("⚠️ token模型目录为空")
            else:
                api._log.warning("⚠️ token模型目录不存在")
        
        # 如果没有找到训练模型，使用配置中的路径
        if model_path is None:
            api._log.info("=" * 60)
            api._log.info("ℹ️ 未找到训练模型，使用配置中的基础模型路径")
            api._log.info("=" * 60)
            model_path = api.config.get("model", {}).get("path")
            if model_path:
                api._log.info(f"📦 使用配置中的模型路径: {model_path}")
            else:
                # 如果配置中也没有，使用基础模型路径
                model_path = memory_config.get("base_model_path", "./models/Qwen3-VL-4B-Thinking")
                api._log.info(f"📦 使用默认基础模型路径: {model_path}")
        
        api.initialize_model(model_path, device)
        api._log.info("✅ 模型已在应用启动时初始化完成")
    except Exception as e:
        api._log.error(f"❌ 模型初始化失败: {e}", exc_info=True)

    # 初始化训练调度器（如果启用）
    try:
        memory_config = api.config.get("memory", {}).get("training", {})
        training_enabled = memory_config.get("enabled", False)
        if training_enabled:
            from memory.training_scheduler import MemoryTrainingScheduler
            import sys
            script_path = os.path.abspath(__file__)
            script_args = sys.argv[1:] if hasattr(sys, 'argv') else []
            api.training_scheduler = MemoryTrainingScheduler(api.config, script_path, script_args)
            api.training_scheduler.start()
            api._log.info("✅ 训练调度器已启动，将在指定时间自动执行训练")
        else:
            api._log.info("ℹ️ 记忆训练未启用，跳过训练调度器启动")
    except Exception as e:
        api._log.error(f"❌ 训练调度器初始化失败: {e}", exc_info=True)
        # 不阻止应用启动，只是记录错误

    return app

app = create_app()

__all__ = ["app", "create_app"]

if __name__ == "__main__":
    # 统一入口：直接运行本文件即可启动（蓝图模式）
    host = api.config.get("server", {}).get("host", "0.0.0.0") if isinstance(getattr(api, "config", {}), dict) else "0.0.0.0"
    port = api.config.get("server", {}).get("port", 9999) if isinstance(getattr(api, "config", {}), dict) else 9999
    app.run(host=host, port=port, debug=False, threaded=True)
