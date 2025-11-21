# -*- coding: utf-8 -*-
from flask import Blueprint, jsonify
try:
    from .. import api_server_qwen3vl as api  # package mode
except Exception:
    # script mode fallback
    import sys, os as _os
    sys.path.append(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import api_server_qwen3vl as api

bp_health = Blueprint("health", __name__)

@bp_health.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": api.model is not None,
        "processor_loaded": api.processor is not None,
        "device": str(api.device) if api.device else None
    })

@bp_health.route("/metrics", methods=["GET"])
def metrics():
    with api.metrics_lock:
        m = dict(api.metrics)
        lat = m.get("latency_ms") or []
        if isinstance(lat, list) and lat:
            import statistics as _stat
            m["latency_p50_ms"] = _stat.median(lat)
            m["latency_avg_ms"] = sum(lat) / len(lat)
            m["latency_count"] = len(lat)
        else:
            m["latency_p50_ms"] = 0
            m["latency_avg_ms"] = 0
            m["latency_count"] = 0
        if "latency_ms" in m:
            del m["latency_ms"]
        return jsonify(m), 200


