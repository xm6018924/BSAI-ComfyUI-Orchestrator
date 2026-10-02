# -*- coding: utf-8 -*-
"""
BSAI-ComfyUI-Orchestrator —— 多硬件编排插件
============================================
- 跨进程资源租约（reslock：文件锁 + PID 存活 + TTL 回收）
- 全局优先级队列（主采样 > 用户实时 > 后台批量）
- 水位分流决策（GPU1 显存紧张 → VAE 解码分流 XPU/CPU）
- HUD 双水位信号（xpu_offload_min_vram / cuda_reserve_mb）

核心库 bsai_orchestrator 注入 sys.modules，任何 BSAI 插件可直接
    import bsai_orchestrator
即用（同 bsai_npu_client 注入模式）。
"""

import importlib.util
import os
import sys


def _inject_core():
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bsai_orchestrator.py")
    if not os.path.exists(src):
        print("[BSAI-Orchestrator] 核心库缺失: %s" % src)
        return
    try:
        spec = importlib.util.spec_from_file_location("bsai_orchestrator", src)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["bsai_orchestrator"] = mod
        spec.loader.exec_module(mod)
        print("[BSAI-Orchestrator] 核心库已注入（reslock + 优先级队列 + 水位决策）")
    except Exception as e:
        print("[BSAI-Orchestrator] 核心库注入跳过: %s" % e)


_inject_core()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


# ---------------- 挂载编排端点（自动判断本实例是否 XPU worker） ----------------
def _mount_routes():
    try:
        import bsai_orchestrator as _bo
        _bo.register_routes()
    except Exception as e:
        print("[BSAI-Orchestrator] 端点挂载跳过: %s" % e)


_mount_routes()
