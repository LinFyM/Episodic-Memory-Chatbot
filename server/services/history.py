# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Optional
import threading

_group_histories: Optional[Dict[str, list]] = None
_private_histories: Optional[Dict[str, list]] = None
_lock: Optional[threading.Lock] = None

def bind_backing_stores(group_histories: Dict[str, list],
                        private_histories: Dict[str, list],
                        lock: threading.Lock) -> None:
    global _group_histories, _private_histories, _lock
    _group_histories = group_histories
    _private_histories = private_histories
    _lock = lock

def append_group(chat_id: str, item: Any, max_len: int = 30) -> None:
    if _group_histories is None or _lock is None:
        return
    with _lock:
        hist = _group_histories.setdefault(chat_id, [])
        hist.append(item)
        if len(hist) > max_len:
            del hist[: len(hist) - max_len]

def get_group(chat_id: str) -> List[Any]:
    if _group_histories is None or _lock is None:
        return []
    with _lock:
        return list(_group_histories.get(chat_id, []))

def append_private(user_id: str, item: Any, max_len: int = 30) -> None:
    if _private_histories is None or _lock is None:
        return
    with _lock:
        hist = _private_histories.setdefault(user_id, [])
        hist.append(item)
        if len(hist) > max_len:
            del hist[: len(hist) - max_len]

def get_private(user_id: str) -> List[Any]:
    if _private_histories is None or _lock is None:
        return []
    with _lock:
        return list(_private_histories.get(user_id, []))


