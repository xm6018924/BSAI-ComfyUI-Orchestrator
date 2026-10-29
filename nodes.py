# -*- coding: utf-8 -*-
"""
BSAI 编排节点
=============
- BSAIOrchestratorGate      ：水位策略 + 设备决策门（执行入口登记任务 + 决策）
- BSAIOrchestratorStatus    ：画布内展示编排状态（租约 / 队列 / 信号 JSON）
- BSAIOrchestratorBackendGate：双轨策略门（prefer_tier + 水位 + 各 kind 决策预览）
- BSAITTSSwitch             ：声音统一入口（本地 IndexTTS ↔ 云端 HeyGen/ElevenLabs）
- BSAIUpscaleSwitch         ：超分统一入口（本地 SeedVR2/H3 ↔ 云端 FLUX/Crystal）
"""

import io
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
from bsai_orch_client import (
    BSAIOrch,
    cloud_key_available,
    register_builtin_providers,
)

# 模块加载即注册内置档位（本地 IndexTTS/SeedVR2/H3 + 云端 HeyGen/ElevenLabs/
# FLUX Upscale/Crystal），保证 decide_tier 开箱可见。
register_builtin_providers()


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
                            "tooltip": "When GPU1 free VRAM drops below this (MB), offload VAE decode to XPU/CPU"}),
                "cuda_reserve_mb": (
                    "INT", {"default": 750, "min": 0, "max": 65536, "step": 100,
                            "tooltip": "Resident VRAM kept free on GPU1 (MB); sampling must not exceed this"}),
            },
            "optional": {
                "latent": ("LATENT",),
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING", "LATENT", "IMAGE")
    RETURN_NAMES = ("vae_target", "latent", "images")
    FUNCTION = "route"
    CATEGORY = "BSAI/Orchestrator"

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
    """编排状态展示：租约 / 优先级队列 / 水位信号 / 云端遥测（输出 JSON 字符串）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status_json",)
    FUNCTION = "status"
    CATEGORY = "BSAI/Orchestrator"

    def status(self):
        sig = get_signals()
        print("[BSAI-Orchestrator] %s" % json.dumps(sig, ensure_ascii=False)[:400])
        return (json.dumps(sig, ensure_ascii=False, indent=1),)


# ---------------- 双轨：后端策略门 ----------------
class BSAIOrchestratorBackendGate:
    """双轨策略门：设置 prefer_tier / local_min_vram_mb，并即时预览
    各 kind（声音 / 超分）当前会落到哪一档、哪个引擎。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prefer_tier": (
                    ["local-first", "cloud-first"],
                    {"default": "local-first",
                     "tooltip": "local-first: 本地优先，水位不足/无本地档时自动切云端"}),
                "local_min_vram_mb": (
                    "INT", {"default": 2000, "min": 0, "max": 65536, "step": 100,
                            "tooltip": "本地档所需最低空闲显存(MB)；低于此值自动转云端"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("backend_json",)
    FUNCTION = "gate"
    CATEGORY = "BSAI/Orchestrator"

    def gate(self, prefer_tier, local_min_vram_mb):
        # 跨进程写策略
        BSAIOrch.set_policy(prefer_tier=prefer_tier,
                            local_min_vram_mb=local_min_vram_mb)
        tts_d = BSAIOrch.decide_tier("tts_synth", prefer_tier=prefer_tier)
        up_d = BSAIOrch.decide_tier("video_upscale", prefer_tier=prefer_tier)
        out = {
            "policy": {"prefer_tier": prefer_tier,
                       "local_min_vram_mb": local_min_vram_mb},
            "tts_synth": tts_d,
            "video_upscale": up_d,
        }
        print("[BSAI-Orchestrator] 双轨策略=%s 本地档=%sMB → TTS:%s/%s 超分:%s/%s" % (
            prefer_tier, local_min_vram_mb,
            tts_d.get("tier"), tts_d.get("engine"),
            up_d.get("tier"), up_d.get("engine")))
        return (json.dumps(out, ensure_ascii=False, indent=1),)


# ---------------- 双轨：声音统一入口 ----------------
class BSAITTSSwitch:
    """声音统一入口：同一 text 按策略落到本地 IndexTTS 或云端
    HeyGen/ElevenLabs。本地档由画布上原 IndexTTS 节点执行（本节点不重载模型），
    云端档由本节点直接 HTTP 执行并输出 AUDIO。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True,
                                    "default": "Hello from BSAI dual-track."}),
                "engine": (
                    ["auto", "heygen", "elevenlabs", "indextts"],
                    {"default": "auto",
                     "tooltip": "auto = 按 BackendGate 策略决策"}),
            },
            "optional": {
                "voice_id": ("STRING", {"default": "",
                                         "tooltip": "云端音色 ID（留空用默认）"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0,
                                    "step": 0.05}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "backend")
    FUNCTION = "tts"
    CATEGORY = "BSAI/Orchestrator"

    def tts(self, text, engine, voice_id="", speed=1.0):
        backend = engine
        if engine == "auto":
            d = BSAIOrch.decide_tier("tts_synth")
            backend = d.get("engine") or "indextts"
            if d.get("reason"):
                print("[BSAI-Orchestrator] TTS auto: %s" % d["reason"])

        if backend in ("heygen", "elevenlabs"):
            if backend == "heygen":
                # 核心 HeyGen Voice 1.0 节点（comfy_api_nodes/nodes_heygen.py）现已可用：
                # 走平台工作区登录令牌 + 自动续期；本节点直连 REST 用静态 COMFY_API_KEY。
                print("[BSAI-Orchestrator] TTS heygen: 也可直接使用核心 HeyGen Voice 节点"
                      "（工作区登录令牌自动续期）；本节点走静态 COMFY_API_KEY 直连")
            try:
                import bsai_cloud_gate as gate
                audio_bytes, mime = gate.tts(
                    backend, text,
                    voice_id=voice_id or None, speed=speed)
                return (_audio_bytes_to_waveform(audio_bytes), backend)
            except Exception as e:
                status = getattr(e, "status", None)
                # 云端失败 → 明确报错（不静默返回空），按状态码给出建议
                if status in (401, 403):
                    hint = ("鉴权失败(HTTP %s)：1) 检查/更换 COMFY_API_KEY（静态密钥不会自动续期）；"
                            "2) 或改用画布上的核心 HeyGen Voice 节点（工作区登录令牌 + 自动续期）；"
                            "3) 本地执行请把 engine 切为 indextts" % status)
                else:
                    hint = ("排查: 1) 环境变量已配置? 2) 网络可达? "
                            "3) 如需本地执行请把 engine 切为 indextts")
                raise RuntimeError(
                    "[BSAI-Orchestrator] 云端 TTS(%s) 失败: %s\n%s" % (backend, e, hint))
        # 本地档：IndexTTS 由画布原节点执行，这里返回 1s 静音占位并提示
        print("[BSAI-Orchestrator] TTS 本地档(indextts): 请把 text 接到 "
              "BSAI IndexTTS 节点执行；本节点不重载模型")
        return (_silent_audio(), backend or "indextts")


# ---------------- 双轨：超分统一入口（首版 image 档） ----------------
class BSAIUpscaleSwitch:
    """超分统一入口（首版 image 档）：IMAGE 进、IMAGE 出。
    云端档由本节点直接 HTTP 执行；本地档由画布原 SeedVR2/H3 节点执行。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "engine": (
                    ["auto", "flux_upscale", "crystal", "seedvr2", "h3"],
                    {"default": "auto"}),
            },
            "optional": {
                "target_res": (["1080p", "2K", "4K"], {"default": "2K"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "backend")
    FUNCTION = "upscale"
    CATEGORY = "BSAI/Orchestrator"

    def upscale(self, images, engine, target_res="2K"):
        backend = engine
        if engine == "auto":
            d = BSAIOrch.decide_tier("video_upscale")
            backend = d.get("engine") or "seedvr2"
        if backend in ("flux_upscale", "crystal"):
            raise RuntimeError(
                "[BSAI-Orchestrator] 云端超分(%s)需要可访问的图床 URL，"
                "首版节点仅做决策预览；请把 engine 切到 seedvr2/h3 走本地档，"
                "或等待 image_bytes 直传联调完成" % backend)
        print("[BSAI-Orchestrator] 超分本地档(%s): 请把 images 接到 "
              "BSAI SeedVR2/H3-upscale 节点执行" % backend)
        return (images, backend)


# ---------------- 双轨：视频配音统一入口（官方 Partner Node #17388） ----------------
class BSAIDubSwitch:
    """视频配音统一入口：VIDEO URL 进、VIDEO 出（VEED Dubbing）。

    对齐官方 Partner Node VeedDubbingNode（#17388）：输入可访问的视频 URL +
    目标语言，输出配音+对口型同步的新视频。走 COMFY_API_KEY。
    本地无免费配音档，engine 仅 veed（云端）。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"default": "",
                                         "tooltip": "可访问的视频 URL（含人脸+语音，≤599s）"}),
                "engine": (
                    ["auto", "veed"],
                    {"default": "auto",
                     "tooltip": "auto = 按 BackendGate 策略决策（当前仅云端档）"}),
            },
            "optional": {
                "model": (["standard", "pro"], {"default": "standard",
                          "tooltip": "standard=单声道低价；pro=地区口音+立体声"}),
                "target_language": ("STRING", {"default": "English",
                                               "tooltip": "目标配音语言"}),
                "source_language": ("STRING", {"default": "auto",
                                               "tooltip": "视频原语言（auto 自动检测）"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_url_out", "backend")
    FUNCTION = "dub"
    CATEGORY = "BSAI/Orchestrator"

    def dub(self, video_url, engine, model="standard",
            target_language="English", source_language="auto"):
        backend = engine
        if engine == "auto":
            d = BSAIOrch.decide_tier("video_dub")
            backend = d.get("engine") or "veed"
            if d.get("reason"):
                print("[BSAI-Orchestrator] Dub auto: %s" % d["reason"])
        if backend != "veed":
            raise RuntimeError(
                "[BSAI-Orchestrator] 视频配音暂无本地档，engine 请使用 veed（云端）")
        if not video_url or not video_url.strip():
            raise RuntimeError(
                "[BSAI-Orchestrator] VEED 配音需要可访问的视频 URL（video_url）")
        import bsai_cloud_gate as gate
        media_bytes, mime = gate.dub(
            "veed", video_url, target_language=target_language,
            source_language=source_language, model=model)
        # 把配音视频落盘到 output 目录并返回可访问 URL（可对接后续 H3 等流程）
        out_path = _save_media(media_bytes, "bsai_veed_dub")
        print("[BSAI-Orchestrator] VEED 配音完成: %s (%s)" % (out_path, mime))
        return (out_path, backend)


# ---------------- 工具 ----------------
def _save_media(media_bytes, prefix):
    """把云端返回的媒体字节落盘到 output 目录，返回绝对路径。"""
    import folder_paths
    out_dir = folder_paths.get_output_directory()
    import hashlib, os
    fn = "%s_%s.mp4" % (prefix, hashlib.md5(media_bytes).hexdigest()[:10])
    path = os.path.join(out_dir, fn)
    with open(path, "wb") as f:
        f.write(media_bytes)
    return path


def _silent_audio(seconds=1.0, sr=24000):
    import torch
    return {"waveform": torch.zeros(1, 1, int(sr * seconds))}


def _audio_bytes_to_waveform(audio_bytes):
    """mp3/wav bytes → ComfyUI AUDIO dict（torchaudio 解码）。"""
    import torchaudio
    wav, sr = torchaudio.load(io.BytesIO(audio_bytes))  # (channels, samples)
    return {"waveform": wav.unsqueeze(0)}  # (1, channels, samples)


NODE_CLASS_MAPPINGS = {
    "BSAIOrchestratorGate": BSAIOrchestratorGate,
    "BSAIOrchestratorStatus": BSAIOrchestratorStatus,
    "BSAIOrchestratorBackendGate": BSAIOrchestratorBackendGate,
    "BSAITTSSwitch": BSAITTSSwitch,
    "BSAIUpscaleSwitch": BSAIUpscaleSwitch,
    "BSAIDubSwitch": BSAIDubSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAIOrchestratorGate": "BSAI Orchestrator Gate (VRAM policy + resource lease)",
    "BSAIOrchestratorStatus": "BSAI Orchestrator Status (leases/queue/signals/cloud)",
    "BSAIOrchestratorBackendGate": "BSAI Backend Gate (local/cloud dual-track policy)",
    "BSAITTSSwitch": "BSAI TTS Switch (IndexTTS local ↔ HeyGen/ElevenLabs cloud)",
    "BSAIUpscaleSwitch": "BSAI Upscale Switch (SeedVR2/H3 local ↔ FLUX/Crystal cloud)",
    "BSAIDubSwitch": "BSAI Dub Switch (VEED Dubbing cloud, #17388)",
}
