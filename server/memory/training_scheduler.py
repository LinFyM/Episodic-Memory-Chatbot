# -*- coding: utf-8 -*-
"""
记忆训练定时任务调度器
使用APScheduler在指定时间自动执行训练
"""

import logging
from datetime import datetime
import threading
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# 从新路径导入训练服务（主实现）
from .training_service import MemoryTrainingService

_log = logging.getLogger(__name__)


class MemoryTrainingScheduler:
    """记忆训练定时任务调度器"""
    
    def __init__(self, config: dict, script_path: str = None, script_args: list = None):
        self.config = config
        self.scheduler = BackgroundScheduler()
        self.training_service = None
        self.is_running = False
        self._lock = threading.Lock()
        self._restart_lock = threading.Lock()  # 重启操作的互斥锁
        self._restarting = False  # 是否正在重启的标志
        self.script_path = script_path
        self.script_args = script_args or []
        _log.info("记忆训练调度器初始化完成")
    
    def _setup_training_service(self):
        if self.training_service is None:
            self.training_service = MemoryTrainingService(self.config)
    
    def train_job(self):
        _log.info("=" * 60)
        _log.info("定时训练任务触发")
        _log.info(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        _log.info("=" * 60)
        with self._lock:
            if self.is_running:
                _log.warning("训练任务已在运行，跳过本次调度")
                return
            self.is_running = True
        try:
            self._setup_training_service()
            model_path = self.training_service.run_training()
            if model_path:
                _log.info(f"✅ 训练完成，模型保存在: {model_path}")
                memory_config = self.config.get("memory", {}).get("training", {})
                auto_restart = memory_config.get("auto_restart_after_training", False)
                restart_mode = memory_config.get("restart_mode", "reload_model")
                
                # 训练完成后，总是重新加载模型（因为训练时可能卸载了主模型）
                # 如果配置了自动重启，则根据配置决定是重新加载还是重启服务器
                if auto_restart:
                    if restart_mode == "restart_server":
                        _log.info("配置了自动重启服务器，将在3秒后重启...")
                        self.restart_server()  # 这会终止当前进程，不会返回
                    elif restart_mode == "reload_model":
                        _log.info("配置了自动重新加载模型，开始重新加载...")
                        self.reload_model()
                else:
                    # 即使没有配置自动重启，也要重新加载模型（因为训练时卸载了主模型）
                    _log.info("训练完成，重新加载主模型（训练时可能卸载了主模型）...")
                    self.reload_model()
            else:
                _log.warning("⚠️ 训练未执行（可能没有聊天记录或没有提取到记忆条目）")
                # 即使训练未执行，如果训练过程中卸载了主模型，也需要重新加载
                # 但这里假设训练未执行时主模型没有被卸载，所以不重新加载
        except Exception as e:
            _log.error(f"❌ 训练任务执行失败: {e}", exc_info=True)
            # 训练失败时，如果主模型被卸载，也需要重新加载
            # 但这里假设训练失败时主模型可能还在，所以不强制重新加载
        finally:
            with self._lock:
                self.is_running = False
    
    def start(self):
        memory_config = self.config.get("memory", {}).get("training", {})
        training_enabled = memory_config.get("enabled", False)
        if not training_enabled:
            _log.info("记忆训练未启用，跳过调度器启动")
            return
        schedule = memory_config.get("schedule", "3")
        try:
            try:
                train_hour = int(schedule)
            except ValueError:
                if "-" in schedule:
                    _log.warning(f"⚠️ 检测到旧格式的时间配置 '{schedule}'，将只使用开始时间")
                    train_hour = int(schedule.split("-")[0])
                else:
                    _log.warning(f"⚠️ 无法解析时间配置 '{schedule}'，使用默认值 3")
                    train_hour = 3
            _log.info(f"设置训练时间：每两天 {train_hour}:00 执行一次训练")
            self.scheduler.add_job(
                func=self.train_job,
                trigger=CronTrigger(hour=train_hour, minute=0, day='*/2'),
                id='memory_training',
                name=f'记忆训练任务-{train_hour}点',
                replace_existing=True
            )
            self.scheduler.start()
            _log.info("记忆训练调度器已启动")
        except Exception as e:
            _log.error(f"启动调度器失败: {e}", exc_info=True)
            raise
    
    def stop(self):
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            _log.info("记忆训练调度器已停止")
    
    def reload_model(self):
        try:
            _log.info("开始重新加载模型...")
            import api_server_qwen3vl
            # 传递 None 让 initialize_model 自动查找最新训练模型
            model_path = None
            device = self.config.get("model", {}).get("device", "cuda:0")
            _log.info(f"重新加载模型，设备: {device}，将自动查找最新训练模型")
            api_server_qwen3vl.initialize_model(model_path, device)
            _log.info("✅ 模型重新加载完成（已加载最新训练模型）")
        except Exception as e:
            _log.error(f"❌ 重新加载模型失败: {e}", exc_info=True)
    
    def restart_server(self):
        """
        重启服务器进程
        
        使用 subprocess.Popen 在后台启动新进程，然后使用 os._exit() 强制退出当前进程。
        这样可以确保：
        1. 新进程在旧进程退出前启动，避免服务中断
        2. 旧进程强制退出，确保端口和资源被释放
        3. 新进程继承正确的环境和工作目录
        
        注意：此方法使用互斥锁确保只执行一次，防止重复重启导致端口冲突
        """
        # 使用互斥锁确保重启只执行一次
        with self._restart_lock:
            # 检查是否已经在重启中
            if self._restarting:
                _log.warning("⚠️ 服务器重启已在进行中，跳过重复的重启请求")
                return
            
            # 设置重启标志
            self._restarting = True
        
        try:
            import os
            import sys
            import subprocess
            import time
            
            _log.info("=" * 60)
            _log.info("准备重启服务器进程...")
            _log.info("=" * 60)
            
            if not self.script_path:
                _log.error("❌ 未设置script_path，无法重启服务器")
                # 重置标志
                with self._restart_lock:
                    self._restarting = False
                return
            
            python_exe = sys.executable
            args = [python_exe, self.script_path] + (self.script_args or [])
            project_root = os.path.dirname(os.path.dirname(self.script_path))
            
            # 停止调度器，避免在新进程中重复启动旧的scheduler线程
            _log.info("停止训练调度器...")
            self.stop()
            
            # 等待一小段时间，确保调度器完全停止
            time.sleep(0.5)
            
            # 刷新所有输出，确保日志被写入
            sys.stdout.flush()
            sys.stderr.flush()
            
            # 切换到项目根目录（新进程会继承）
            if project_root and os.path.exists(project_root):
                os.chdir(project_root)
                _log.info(f"切换到项目根目录: {project_root}")
            
            _log.info(f"启动新进程: {' '.join(args)}")
            _log.info("旧进程将在新进程启动后退出...")
            
            # 在后台启动新进程
            # 使用 subprocess.Popen 启动新进程，detach=True 让新进程独立运行
            try:
                # 设置环境变量，确保新进程使用正确的环境
                env = os.environ.copy()
                
                # 启动新进程（不等待完成）
                new_process = subprocess.Popen(
                    args,
                    cwd=project_root if project_root and os.path.exists(project_root) else None,
                    env=env,
                    stdout=sys.stdout,  # 继承标准输出
                    stderr=sys.stderr,  # 继承标准错误
                    start_new_session=True  # 创建新的会话，让新进程独立
                )
                _log.info(f"✅ 新进程已启动 (PID: {new_process.pid})")
            except Exception as start_error:
                _log.error(f"❌ 启动新进程失败: {start_error}", exc_info=True)
                _log.error("将尝试使用 os.execv 原地替换进程...")
                # 如果 subprocess 失败，回退到 os.execv
                os.execv(python_exe, args)
                return
            
            # 等待足够的时间，确保新进程已经开始启动
            # 同时给旧进程一些时间释放资源（虽然 os._exit 会立即释放）
            _log.info("等待新进程启动...")
            time.sleep(2.0)
            
            # 强制退出当前进程（不执行清理代码，确保立即退出）
            _log.info("旧进程即将退出，释放端口和资源...")
            sys.stdout.flush()
            sys.stderr.flush()
            
            # 使用 os._exit() 强制退出，不执行任何清理代码
            # 这样可以确保端口立即释放，新进程可以绑定端口
            # os._exit() 会立即终止进程，包括所有线程和 Flask 服务器
            os._exit(0)
            
        except Exception as e:
            _log.error(f"❌ 重启服务器失败: {e}", exc_info=True)
            _log.error("服务器将继续运行，但可能使用的是旧模型")
            # 如果所有方法都失败，尝试使用 os.execv 作为最后的回退
            try:
                import os
                import sys
                if self.script_path:
                    python_exe = sys.executable
                    args = [python_exe, self.script_path] + (self.script_args or [])
                    _log.warning("尝试使用 os.execv 作为最后的回退...")
                    # os.execv 会替换进程，不会返回，所以不需要重置标志
                    os.execv(python_exe, args)
            except Exception as fallback_error:
                _log.error(f"❌ 回退重启方法也失败: {fallback_error}", exc_info=True)
                # 所有重启方法都失败，重置标志以允许后续重试
                with self._restart_lock:
                    self._restarting = False
    
    def run_training_now(self):
        """立即执行一次训练（用于测试或手动触发）"""
        _log.info("手动触发训练任务...")
        self.train_job()

