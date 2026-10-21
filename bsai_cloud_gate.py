# -*- coding: utf-8 -*-
"""
BSAI 云端统一网关（bsai_cloud_gate）
===================================
屏蔽各家云端 TTS / 超分 API 差异，对外只暴露语义化函数：
    audio_bytes, mime = tts(engine, text, voice_id=None, speed=1.0)
    media_bytes, mime = video_upscale(engine, media_bytes, target_res="2K", ...)

设计原则（吸取 IndexTTS 静默吞 None 的教训）：
- key 缺失     → 立即抛 CloudKeyError，明确告诉你配哪个环境变量
- HTTP 失败    → 抛 CloudApiError，带状态码与响应片段（不静默返回空）
- 每次调用     → 自动写遥测（BSAIOrch.record_cloud_call），Status 面板可见
- 端点        → 只实现文档确认过的路径；未联调端点默认值在函数内标注
"""

import json
import os
import time
import urllib.error
import urllib.request

# 注册内置档位（保证 decide_tier 开箱可见）
try:
    from bsai_orch_client import (BSAIOrch, register_builtin_providers)
    register_builtin_providers()
except Exception:  # pragma: no cover
    BSAIOrch = None


class CloudKeyError(RuntimeError):
    """环境变量缺失：云端档位不可用。"""


class CloudApiError(RuntimeError):
    """云端 HTTP 调用失败（状态码/响应异常）。status 为 HTTP 状态码（可解析时）。"""

    def __init__(self, msg, status=None):
        super().__init__(msg)
        self.status = status


def _record(engine, ok, latency_s, error=""):
    if BSAIOrch is None:
        return
    try:
        BSAIOrch.record_cloud_call(engine, ok, latency_ms=int(latency_s * 1000),
                                   error=error)
    except Exception:
        pass


def _need_key(env):
    v = os.environ.get(env, "").strip()
    if not v:
        raise CloudKeyError("缺少环境变量 %s（云端档位需要它，请到系统环境变量配置）"
                            % env)
    return v


def _http_json(url, payload, headers, timeout=120, method="POST"):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    for k, v in headers.items():
        req.add_header(k, v)
    req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        body = e.read()[:500].decode("utf-8", "ignore")
        raise CloudApiError("HTTP %s @ %s: %s" % (e.code, url, body), status=e.code)


def _http_get(url, headers, timeout=60):
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        body = e.read()[:300].decode("utf-8", "ignore")
        raise CloudApiError("HTTP %s @ %s: %s" % (e.code, url, body), status=e.code)


# ================= TTS =================
def tts(engine, text, voice_id=None, speed=1.0):
    """云端 TTS 统一入口。返回 (audio_bytes, mime)。

    engine:
      "elevenlabs" → ELEVENLABS_API_KEY，返回 mp3
      "heygen"     → COMFY_API_KEY（优先）或 HEYGEN_API_KEY，返回 mp3/wav
    """
    if not text or not text.strip():
        raise CloudApiError("text 为空")
    started = time.time()
    try:
        if engine == "elevenlabs":
            out = _tts_elevenlabs(text, voice_id, speed)
        elif engine == "heygen":
            out = _tts_heygen(text, voice_id, speed)
        else:
            raise CloudApiError("unknown tts engine: %r" % engine)
        _record(engine, True, time.time() - started, "")
        return out
    except Exception as e:
        _record(engine, False, time.time() - started, str(e))
        raise


def _tts_elevenlabs(text, voice_id, speed):
    key = _need_key("ELEVENLABS_API_KEY")
    vid = voice_id or "21m00Tcm4TlvDq8ikWAM"  # Rachel 公开示例音色
    url = "https://api.elevenlabs.io/v1/text-to-speech/%s" % vid
    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    status, body, headers = _http_json(
        url, payload,
        {"xi-api-key": key, "accept": "audio/mpeg"})
    return body, headers.get("content-type", "audio/mpeg")


def _tts_heygen(text, voice_id, speed):
    """HeyGen Starfish 走 Comfy Router 异步提交→轮询模式。

    端点依据 Comfy 官方文档：POST .../heygen/starfish/requests → 201 带
    response_url → 轮询至 completed，再取音频。
    """
    key = (os.environ.get("COMFY_API_KEY", "").strip()
           or _need_key("HEYGEN_API_KEY"))
    base = "https://api.comfy.org/v2/models/heygen/starfish"
    payload = {"text": text}
    if voice_id:
        payload["voice_id"] = voice_id
    status, body, _ = _http_json(
        base + "/requests", payload,
        {"X-API-KEY": key,
         "Idempotency-Key": "bsai-%d" % int(time.time())})
    info = json.loads(body.decode("utf-8", "ignore"))
    resp_url = info.get("response_url")
    if not resp_url:
        raise CloudApiError("heygen submit missing response_url: %s"
                            % body[:300])
    # 轮询最多 ~60s
    audio = None
    for _ in range(30):
        time.sleep(2)
        try:
            st, data_body, _ = _http_get(resp_url, {"X-API-KEY": key})
            data = json.loads(data_body.decode("utf-8", "ignore"))
        except CloudApiError:
            continue
        if data.get("status") == "completed":
            audio = _extract_audio(data)
            break
        if data.get("status") in ("failed", "error"):
            raise CloudApiError("heygen job failed: %s" % str(data)[:300])
    if audio is None:
        raise CloudApiError("heygen job timeout (~60s)")
    if isinstance(audio, str) and audio.startswith("http"):
        _, audio, _ = _http_get(audio, {"X-API-KEY": key}, timeout=120)
    return audio, "audio/mpeg"


def _extract_audio(data):
    """从 HeyGen 响应里防御式取音频 bytes/url（字段名待真实联调确认）。"""
    for path in (("result", "audio"), ("audio",), ("audio_url",), ("url",)):
        node = data
        ok = True
        for k in path:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                ok = False
                break
        if ok and node:
            return node
    raise CloudApiError("heygen completed but audio field not found: %s"
                        % str(data)[:300])


# ================= 视频超分 =================
def video_upscale(engine, media_bytes=None, target_res="2K",
                  source_url=None):
    """云端视频超分统一入口。返回 (media_bytes, mime)。

    engine:
      "flux_upscale" → BFL_API_KEY，BFL FLUX Video Upscale
      "crystal"      → CAI_API_KEY，ClarityAI Crystal
    注意：两家原生端点路径未联调，默认值为依据公开文档的最佳推断，
    首次真实调用前请按返回的 CloudApiError 校准 URL。
    """
    started = time.time()
    try:
        if engine == "flux_upscale":
            out = _upscale_bfl(media_bytes, target_res, source_url)
        elif engine == "crystal":
            out = _upscale_crystal(media_bytes, target_res, source_url)
        else:
            raise CloudApiError("unknown upscale engine: %r" % engine)
        _record(engine, True, time.time() - started, "")
        return out
    except Exception as e:
        _record(engine, False, time.time() - started, str(e))
        raise


def _upscale_bfl(media_bytes, target_res, source_url):
    key = _need_key("BFL_API_KEY")
    # BFL 异步：提交 → polling_url → 轮询取结果（端点路径待联调校准）
    url = "https://api.bfl.ai/v1/flux-video-upscale"
    payload = {"target_resolution": target_res}
    if source_url:
        payload["video_url"] = source_url
    status, body, _ = _http_json(url, payload,
                                  {"x-key": key}, timeout=300)
    info = json.loads(body.decode("utf-8", "ignore"))
    poll = info.get("polling_url") or info.get("status_url")
    if not poll:
        raise CloudApiError("bfl submit missing polling_url: %s" % body[:300])
    for _ in range(60):
        time.sleep(3)
        st, data_body, _ = _http_get(poll, {"x-key": key})
        data = json.loads(data_body.decode("utf-8", "ignore"))
        if data.get("status") == "Ready":
            result = data.get("result", {})
            vurl = result.get("video_url") or result.get("url")
            if vurl:
                _, media, _ = _http_get(vurl, {}, timeout=300)
                return media, "video/mp4"
            raise CloudApiError("bfl ready but no video_url: %s" % str(data)[:300])
        if data.get("status") in ("Error", "Failed"):
            raise CloudApiError("bfl job failed: %s" % str(data)[:300])
    raise CloudApiError("bfl job timeout (~180s)")


def _upscale_crystal(media_bytes, target_res, source_url):
    key = _need_key("CAI_API_KEY")
    url = "https://api.clarityai.co/v1/upscale"  # 待联调校准
    payload = {"target_resolution": target_res}
    if source_url:
        payload["image_url"] = source_url
    status, body, _ = _http_json(url, payload,
                                 {"Authorization": "Bearer " + key},
                                 timeout=300)
    info = json.loads(body.decode("utf-8", "ignore"))
    out_url = info.get("output_url") or info.get("result_url") or info.get("url")
    if not out_url:
        raise CloudApiError("crystal response missing output url: %s" % body[:300])
    _, media, _ = _http_get(out_url, {"Authorization": "Bearer " + key},
                           timeout=300)
    return media, "image/png"
