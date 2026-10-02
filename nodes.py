# -*- coding: utf-8 -*-
"""
BSAI 编排节点
=============
- BSAIOrchestratorGate   ：水位策略设置 + 设备决策门（执行入口登记任务 + 决策）
- BSAIOrchestratorStatus ：画布内展示编排状态（租约 / 队列 / 信号 JSON）
"""

import json
import os

from bsai_orchestrator import (
    ResourceLease,
    TaskRegistry,
    decide_vae_target,
    get_signals,
    set_policy,
    RES_GPU1_SAMPLING,
    PRIO_SAMPLING,
)


class BSAIOrchestratorGate:
    """编排门：执行前登记任务 + 按水位决策 VAE 分流目标 + 写入策略。

    用法：放在采样/VAE 流程前；latent/images 可透传（可选）。
    vae_target 输出供人工参考（或接入自动分流逻辑）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "xpu_offload_min_vram": (
                    "INT", {"default": 2500, "min": 0, "max": 65536, "step": 100,
                            "tooltip": "GPU1 显存余量低于此值(MB) → VAE 解码分流 XPU/CPU"}),
                "cuda_reserve_mb": (
                    "INT", {"default": 750, "min": 0, "max": 65536, "step": 100,
                            "tooltip": "GPU1 常驻保留显存(MB)，采样不越过"}),
            },
            "optional": {
                "latent": ("LATENT",),
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "LATENT", "IMAGE")
    RETURN_NAMES = ("vae_target", "latent", "images")
    FUNCTION = "route"
    CATEGORY = "BSAI/编排"

    def route(self, xpu_offload_min_vram, cuda_reserve_mb,
              latent=None, images=None):
        # 1) 写入水位策略（跨进程生效，HUD 可见）
        policy = set_policy(xpu_offload_min_vram=xpu_offload_min_vram,
                            cuda_reserve_mb=cuda_reserve_mb)
        # 2) 登记主采样任务（取任务前判定）
        TaskRegistry.register("sampling:%d" % os.getpid(),
                              priority=PRIO_SAMPLING, kind="sampling")
        # 3) 取 GPU1 采样租约（防止与后台批量 VAE 抢活）。
        #    acquire_with_keepalive：后台 watchdog 续租，采样长任务不再因
        #    TTL 到期被误回收；on_lost 在租约被抢占时告警（双写防线）。
        lease = ResourceLease()
        _lease_ok, _lease_gen, _lease_stop = lease.acquire_with_keepalive(
            RES_GPU1_SAMPLING,
            holder="sampling:%d" % os.getpid(),
            ttl=300, keepalive=40,
            on_lost=lambda r, g: print(
                "[BSAI-Orchestrator] 采样租约被抢占 gen=%s → 立即停止写 GPU1" % g))
        # 4) 水位决策：VAE 解码目标
        target = decide_vae_target(policy=policy)
        decision = {
            "vae_target": target,
            "gpu1_lease": _lease_ok,
            "gpu1_gen": _lease_gen,
            "policy": policy,
            "xpu_online": _xpu_online_flag(),
        }
        print("[BSAI-Orchestrator] 决策: vae_target=%s gpu1_lease=%s gen=%s 水位=%sMB 保留=%sMB" % (
            target, _lease_ok, _lease_gen, xpu_offload_min_vram, cuda_reserve_mb))
        return (json.dumps(decision, ensure_ascii=False), latent, images)


def _xpu_online_flag():
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:8190/health/ready", timeout=1.5) as r:
            return bool(json.loads(r.read()).get("ready", False))
    except Exception:
        return False


class BSAIOrchestratorStatus:
    """编排状态展示：租约 / 优先级队列 / 水位信号（输出 JSON 字符串）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status_json",)
    FUNCTION = "status"
    CATEGORY = "BSAI/编排"

    def status(self):
        sig = get_signals()
        print("[BSAI-Orchestrator] %s" % json.dumps(sig, ensure_ascii=False)[:400])
        return (json.dumps(sig, ensure_ascii=False, indent=1),)


NODE_CLASS_MAPPINGS = {
    "BSAIOrchestratorGate": BSAIOrchestratorGate,
    "BSAIOrchestratorStatus": BSAIOrchestratorStatus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAIOrchestratorGate": "BSAI 编排门 (水位决策 + 资源租约)",
    "BSAIOrchestratorStatus": "BSAI 编排状态 (租约/队列/信号)",
}
