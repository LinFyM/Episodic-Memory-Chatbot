# -*- coding: utf-8 -*-
from typing import Any, Optional, Callable
import threading
import queue

_queue: Optional[queue.Queue] = None
_worker_started: bool = False
_lock = threading.Lock()

def bind_queue(q: queue.Queue) -> None:
    global _queue
    _queue = q

def put(item: Any) -> None:
    if _queue is None:
        return
    _queue.put(item)

def get(block: bool = True, timeout: Optional[float] = None) -> Any:
    if _queue is None:
        raise RuntimeError("queue not bound")
    return _queue.get(block=block, timeout=timeout)  # type: ignore

def task_done() -> None:
    if _queue is None:
        return
    _queue.task_done()


