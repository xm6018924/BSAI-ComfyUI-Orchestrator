# -*- coding: utf-8 -*-
"""
BSAI 插件协同 SDK（bsai_orch_client）
=====================================
让任何 BSAI 插件"一行接入"编排层，自动完成：能力注册 → 硬件探活 → 水位
分流 → 跨进程资源租约（互斥 + 世代号 fencing + watchdog 续租）→ 分配目标
→ 释放。自包含单文件（仅标准库），与编排核心共享状态文件/租约协议，
不依赖 ComfyUI 内部模块——任何 BSAI 插件 import 即可用。

================ 接入方式 ================
插件 __init__.py / nodes.py 顶部（ComfyUI 启动时自动执行，即自动注册）：
    import sys, os
    _ORCH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "BSAI-ComfyUI-Orchestrator")
    if _ORCH not in sys.path:
        sys.path.insert(0, _ORCH)
    from bsai_orch_client import BSAIOrch

    BSAIOrch.register(
        name="BSAI-VAERouter",          # 插件名（唯一）
        kind="vae_decode",              # 能力类型（见 KIND_RESOURCE 映射）
        hardware=["xpu", "cuda", "cpu"],  # 硬件偏好顺序
        endpoint="http://127.0.0.1:8190/vae/decode",  # 服务端点（可选）
        health="http://127.0.0.1:8190/health/ready",  # 健康探针（可选）
        weight=1.0,
    )

执行时（取任务前）：
    alloc = BSAIOrch.allocate("vae_decode", requester="8191")
    if alloc.ok:
        try:
            # alloc.target / alloc.endpoint / alloc.gen / alloc.lease_ok
            ... 按 alloc.target 干活 ...
        finally:
            alloc.release()          # 释放租约 + 停 watchdog

自动发生的事：
  - 插件加载即注册（ComfyUI 启动自动触发，HUD 可读 capabilities）
  - allocate 自动探活（health 5s 缓存）+ 水位分流（GPU1 余量 < 阈值 → 不选 cuda）
  - 自动取跨进程租约（fencing gen + watchdog 续租，与编排核心同协议互斥）
  - 自动登记优先级队列（kind 深度）
  未来所有 BSAI 插件按此接入 → 自动协同，无需人工接线。
"""

import json
import os
import struct
import threading
import time

try:
    import msvcrt  # Windows 文件锁
except ImportError:  # Linux/macOS
    msvcrt = None

# ---------------- 路径与常量（与编排核心共享状态） ----------------
DEFAULT_STATE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "user"
)
STATE_FILE = os.path.join(DEFAULT_STATE_DIR, "bsai_orchestrator_state.json")
LOCK_FILE = os.path.join(DEFAULT_STATE_DIR, "bsai_orchestrator.lock")

# 能力类型 → 跨进程资源名（与编排核心 RES_* 约定一致）
KIND_RESOURCE = {
    "sampling": "gpu1_sampling",        # RTX5090 主采样
    "vae_decode": "xpu_vae_decode",     # 核显 XPU VAE 解码
    "face_detect": "npu_face_detect",   # NPU 人脸检测
    "cpu_vae_decode": "cpu_vae_decode", # CPU VAE 兜底
}

LOCK_TTL = 120          # 租约默认 TTL（秒）
LOCK_RETRY = 25         # 文件锁重试次数
LOCK_WAIT = 0.1         # 重试间隔
LOCK_KEEPALIVE = 40     # watchdog 续租间隔

_HEAD = "BSAIVAE1"      # 与 8190 服务的二进制协议 magic 一致（vae 转发用）

# ---------------- 文件锁（与编排核心同协议：写路径安全失败） ----------------
def _with_lock(callback, fallback=True):
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    for attempt in range(LOCK_RETRY):
        try:
            with open(LOCK_FILE, "a+b") as f:
                f.seek(0)
                if msvcrt is not None:
                    try:
                        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    except OSError:
                        time.sleep(LOCK_WAIT)
                        continue
                try:
                    return callback()
                finally:
                    if msvcrt is not None:
                        try:
                            f.seek(0)
                            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                        except OSError:
                            pass
        except OSError:
            time.sleep(LOCK_WAIT)
    if fallback:
        return callback()
    return None


def _load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"leases": {}, "tasks": {}, "capabilities": {}, "policy": {}}


def _save_state(state):
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


def _pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _now():
    return time.time()


def _gpu1_free_mb():
    """GPU1（CUDA）显存余量 MB。非 CUDA 进程返回 None（跳过水位判断）。"""
    try:
        import torch
        free, _ = torch.cuda.mem_get_info()
        return int(free / (1024 * 1024))
    except Exception:
        return None


def _policy():
    st = _load_state()
    p = st.get("policy") or {}
    return {
        "xpu_offload_min_vram": int(p.get("xpu_offload_min_vram", 2500)),
        "cuda_reserve_mb": int(p.get("cuda_reserve_mb", 750)),
        # 双轨（tier）策略：local-first / cloud-first
        "prefer_tier": p.get("prefer_tier", "local-first"),
        "local_min_vram_mb": int(p.get("local_min_vram_mb", 2000)),
    }


def cloud_key_available(api_key_env):
    """云端档位可用性：指定环境变量非空即视为可用（不发起网络请求）。"""
    if not api_key_env:
        return False
    v = os.environ.get(api_key_env, "")
    return bool(v and v.strip())


# ---------------- Allocation（分配结果） ----------------
class Allocation:
    """allocate 的返回：目标硬件 + 端点 + 租约控制。"""

    def __init__(self, ok, target=None, endpoint=None, gen=0, lease_ok=False,
                 stop_fn=None, resource=None, reason=""):
        self.ok = ok
        self.target = target
        self.endpoint = endpoint
        self.gen = gen
        self.lease_ok = lease_ok
        self._stop = stop_fn
        self.resource = resource
        self.reason = reason
        self._released = False

    def release(self):
        """任务结束：停 watchdog + 释放租约（幂等）"""
        if self._released:
            return
        self._released = True
        if self._stop is not None:
            try:
                self._stop()
            except Exception:
                pass
        if self.lease_ok and self.resource:
            try:
                BSAIOrch.release(self.resource, gen=self.gen)
            except Exception:
                pass


# ---------------- 核心 SDK ----------------
class BSAIOrch:
    """BSAI 插件协同入口（静态类）。"""

    # ---------- 注册 ----------
    @staticmethod
    def register(name, kind, hardware=None, endpoint="", health="",
                 weight=1.0, meta=None):
        """插件加载时注册能力（ComfyUI 启动自动触发）。
        name 唯一；kind 见 KIND_RESOURCE；hardware 偏好顺序列表。"""
        hardware = hardware or []
        cap = {
            "name": name, "kind": kind, "hardware": hardware,
            "endpoint": endpoint, "health": health,
            "weight": float(weight), "ts": _now(),
            "meta": meta or {},
        }

        def _do():
            st = _load_state()
            caps = st.setdefault("capabilities", {})
            caps[name] = cap
            _save_state(st)
            return True

        return _with_lock(_do, fallback=False) is True

    @staticmethod
    def capabilities():
        st = _load_state()
        return st.get("capabilities", {})

    @staticmethod
    def list_kinds():
        return sorted({c.get("kind") for c in
                       BSAIOrch.capabilities().values()})

    # ---------- 双轨（tier）决策 ----------
    @staticmethod
    def decide_tier(kind, prefer_tier=None, gpu_free_mb=None):
        """双轨决策：同 kind 的能力按 meta.tier(local/cloud) 分组，按策略选档。

        返回 dict：{tier, backend, engine, endpoint, reason,
                    local_caps[], cloud_caps[]}；无可用档时 tier=None。
        与本地硬件分流（allocate）正交：先选档，local 档再走 allocate 分流。
        """
        policy = _policy()
        prefer_tier = prefer_tier or policy.get("prefer_tier", "local-first")
        if gpu_free_mb is None:
            gpu_free_mb = _gpu1_free_mb()

        def _tier_of(c):
            return (c.get("meta") or {}).get("tier", "local")

        local_caps = [c for c in BSAIOrch.capabilities().values()
                      if c.get("kind") == kind and _tier_of(c) == "local"]
        cloud_caps = [c for c in BSAIOrch.capabilities().values()
                      if c.get("kind") == kind and _tier_of(c) == "cloud"]
        # 云端档必须 key 已配置才可选
        cloud_avail = [c for c in cloud_caps
                       if cloud_key_available((c.get("meta") or {}).get("api_key_env"))]

        reason = ""
        chosen = None
        if prefer_tier == "cloud-first":
            chosen = cloud_avail[0] if cloud_avail else None
            if chosen is None:
                reason = "cloud-first but no cloud provider has API key configured"
                if local_caps:
                    chosen = local_caps[0]
                    reason += "; fallback to local"
        else:  # local-first（默认）
            need = policy.get("local_min_vram_mb", 2000)
            water_ok = (gpu_free_mb is None) or (gpu_free_mb >= need)
            if water_ok and local_caps:
                chosen = local_caps[0]
                if not cloud_avail:
                    pass
            elif cloud_avail:
                chosen = cloud_avail[0]
                reason = ("local VRAM low (%sMB < %sMB); using cloud"
                           % (gpu_free_mb, need))
            elif local_caps:
                chosen = local_caps[0]
                reason = "cloud unavailable (no API key); using local anyway"

        if chosen is None:
            return {"tier": None, "backend": None, "engine": None,
                    "endpoint": "",
                    "reason": reason or ("no provider registered for kind '%s'" % kind),
                    "local_caps": [c.get("name") for c in local_caps],
                    "cloud_caps": [c.get("name") for c in cloud_caps]}
        meta = chosen.get("meta") or {}
        return {"tier": meta.get("tier", "local"),
                "backend": chosen.get("name"),
                "engine": meta.get("engine", chosen.get("name")),
                "endpoint": chosen.get("endpoint", ""),
                "reason": reason,
                "local_caps": [c.get("name") for c in local_caps],
                "cloud_caps": [c.get("name") for c in cloud_caps]}

    @staticmethod
    def set_policy(prefer_tier=None, local_min_vram_mb=None,
                   xpu_offload_min_vram=None, cuda_reserve_mb=None):
        """写策略到状态文件（跨进程生效；与编排核心同文件）。"""
        def _do():
            st = _load_state()
            p = st.setdefault("policy", {})
            if prefer_tier is not None:
                p["prefer_tier"] = str(prefer_tier)
            if local_min_vram_mb is not None:
                p["local_min_vram_mb"] = int(local_min_vram_mb)
            if xpu_offload_min_vram is not None:
                p["xpu_offload_min_vram"] = int(xpu_offload_min_vram)
            if cuda_reserve_mb is not None:
                p["cuda_reserve_mb"] = int(cuda_reserve_mb)
            _save_state(st)
            return dict(p)

        return _with_lock(_do)

    # ---------- 云端调用遥测（Status 面板 P3 展示） ----------
    @staticmethod
    def record_cloud_call(engine, ok, latency_ms=0, error=""):
        def _do():
            st = _load_state()
            tel = st.setdefault("cloud_telemetry", {})
            e = tel.setdefault(engine, {
                "calls": 0, "ok": 0, "fail": 0,
                "last_latency_ms": 0, "last_ok": None,
                "last_error": "", "last_ts": 0.0})
            e["calls"] += 1
            e["ok" if ok else "fail"] += 1
            e["last_latency_ms"] = int(latency_ms)
            e["last_ok"] = bool(ok)
            e["last_error"] = error[:200]
            e["last_ts"] = _now()
            _save_state(st)
            return True

        return _with_lock(_do, fallback=False) is True

    @staticmethod
    def cloud_telemetry():
        st = _load_state()
        return st.get("cloud_telemetry", {})

    # ---------- 探活（5s 缓存） ----------
    _probe_cache = {}
    _probe_cache_ts = {}

    @staticmethod
    def _probe(health_url, timeout=1.5):
        if not health_url:
            return True
        now = _now()
        cts = BSAIOrch._probe_cache_ts.get(health_url, 0)
        if now - cts < 5 and health_url in BSAIOrch._probe_cache:
            return BSAIOrch._probe_cache[health_url]
        ok = False
        try:
            import urllib.request
            with urllib.request.urlopen(health_url, timeout=timeout) as r:
                body = r.read().decode("utf-8", "ignore")
                ok = ("true" in body.lower() or "ok" in body.lower() or
                      r.status < 400)
        except Exception:
            ok = False
        BSAIOrch._probe_cache[health_url] = ok
        BSAIOrch._probe_cache_ts[health_url] = now
        return ok

    # ---------- 分配 ----------
    @staticmethod
    def allocate(kind, requester="", ttl=LOCK_TTL, keepalive=LOCK_KEEPALIVE,
                 on_lost=None, gpu_free_mb=None, watchdog=True):
        """自动协同分配：探活 → 水位 → 租约 → 登记队列。
        watchdog=True（默认）后台续租防长任务 TTL 过期；短任务可传 False
        关闭续租线程（仅 acquire/release）。
        返回 Allocation（ok=False 时 reason 说明原因）。"""
        if not requester:
            requester = "pid%d" % os.getpid()
        resource = KIND_RESOURCE.get(kind, "bsai_task_%s" % kind)

        # 1) 注册表：该 kind 的候选插件
        caps = [c for c in BSAIOrch.capabilities().values()
                if c.get("kind") == kind]
        if not caps:
            return Allocation(False, reason="kind '%s' not registered" % kind)

        # 2) 探活：剔除离线硬件（health 探针）
        online = []
        for c in caps:
            if BSAIOrch._probe(c.get("health")):
                online.append(c)
        if not online:
            return Allocation(False, reason="all hardware offline (health probe failed)")

        # 3) 水位分流：kind=vae_decode 且候选含 cuda 且 GPU1 余量低 → 剔除 cuda
        if gpu_free_mb is None:
            gpu_free_mb = _gpu1_free_mb()
        policy = _policy()
        thr = policy.get("xpu_offload_min_vram", 2500)
        cands = online
        if kind == "vae_decode" and gpu_free_mb is not None:
            if gpu_free_mb < thr:
                cands = [c for c in cands
                         if c.get("hardware") and "cuda" not in c["hardware"]]
            else:
                cands = [c for c in cands if c.get("hardware")]

        # 4) 目标 = hardware 偏好序第一个
        order = []
        for c in cands:
            for h in (c.get("hardware") or []):
                if h not in order:
                    order.append(h)
        target = order[0] if order else None
        if target is None:
            return Allocation(False, reason="no available hardware target")
        cap = next(c for c in cands
                   if (c.get("hardware") or []) and target in c["hardware"])

        # 5) 资源租约（fencing + watchdog 续租——与编排核心同协议互斥）
        lease = BSAIOrch._Lease()
        if watchdog:
            ok, gen, stop_fn = lease.acquire_with_keepalive(
                resource, holder=requester, ttl=ttl, keepalive=keepalive,
                on_lost=on_lost)
        else:
            ok, gen = lease.acquire(resource, holder=requester, ttl=ttl)
            stop_fn = None

        # 6) 登记优先级队列（kind 深度）
        BSAIOrch._enqueue(resource, requester, kind)

        if not ok:
            return Allocation(False, target=target, reason="resource %s is occupied" % resource)
        return Allocation(True, target=target, endpoint=cap.get("endpoint", ""),
                          gen=gen, lease_ok=True, stop_fn=stop_fn,
                          resource=resource)

    @staticmethod
    def release(resource, gen=None, holder=""):
        return BSAIOrch._Lease().release(resource, holder=holder, gen=gen)

    # ---------- 队列登记（与编排核心 TaskRegistry 同格式：tasks 为 list） ----------
    @staticmethod
    def _enqueue(resource, requester, kind):
        prio = {"sampling": 0, "face_detect": 1}.get(kind, 2)

        def _do():
            st = _load_state()
            tasks = st.setdefault("tasks", [])
            if not isinstance(tasks, list):
                tasks = []  # 兼容旧 dict 形态：重建为 list
                st["tasks"] = tasks
            tasks.append({"id": "%s:%s" % (requester, kind),
                          "priority": prio, "kind": kind,
                          "holder": requester, "status": "waiting",
                          "ts": _now(), "resource": resource})
            st["tasks"] = tasks[-200:]  # 只保留最近 200 条
            _save_state(st)
            return True

        return _with_lock(_do, fallback=False) is True

    # ---------- 内部租约实现（与编排核心 ResourceLease 同协议） ----------
    class _Lease:
        def acquire(self, resource, holder="", ttl=LOCK_TTL):
            if not holder:
                holder = "pid%d" % os.getpid()

            def _do():
                st = _load_state()
                leases = st.setdefault("leases", {})
                now = _now()
                cur = leases.get(resource)
                if cur is not None:
                    expired = (now - cur.get("ts", 0)) > cur.get("ttl", LOCK_TTL)
                    dead = not _pid_alive(cur.get("pid"))
                    if not expired and not dead:
                        return (False, 0)
                    del leases[resource]
                st["epoch"] = int(st.get("epoch", 0)) + 1
                gen = int(st["epoch"])
                leases[resource] = {
                    "holder": holder, "pid": os.getpid(), "ts": now,
                    "ttl": ttl, "resource": resource, "gen": gen,
                }
                _save_state(st)
                return (True, gen)

            res = _with_lock(_do, fallback=False)
            return res if res is not None else (False, 0)

        def release(self, resource, holder="", gen=None):
            if not holder:
                holder = "pid%d" % os.getpid()

            def _do():
                st = _load_state()
                cur = st.get("leases", {}).get(resource)
                if cur is None:
                    return False
                mine = (cur.get("holder") == holder or
                        cur.get("pid") == os.getpid())
                gen_ok = gen is None or int(cur.get("gen", 0)) == gen
                if mine and gen_ok:
                    del st["leases"][resource]
                    _save_state(st)
                    return True
                return False

            return _with_lock(_do, fallback=False) is True

        def renew(self, resource, holder="", gen=None, ttl=None):
            if not holder:
                holder = "pid%d" % os.getpid()

            def _do():
                st = _load_state()
                cur = st.get("leases", {}).get(resource)
                if cur is None:
                    return False
                mine = (cur.get("holder") == holder or
                        cur.get("pid") == os.getpid())
                gen_ok = gen is None or int(cur.get("gen", 0)) == gen
                if not mine or not gen_ok:
                    return False
                cur["ts"] = _now()
                if ttl:
                    cur["ttl"] = ttl
                _save_state(st)
                return True

            return _with_lock(_do, fallback=False) is True

        def acquire_with_keepalive(self, resource, holder="", ttl=LOCK_TTL,
                                   keepalive=LOCK_KEEPALIVE, on_lost=None):
            ok, gen = self.acquire(resource, holder=holder, ttl=ttl)
            if not ok:
                return False, 0, None
            stop = threading.Event()

            def _keep():
                while not stop.is_set():
                    time.sleep(keepalive)
                    if stop.is_set():
                        break
                    try:
                        okr = self.renew(resource, holder=holder, gen=gen,
                                         ttl=ttl)
                    except Exception:
                        okr = False
                    if not okr:
                        if on_lost is not None:
                            try:
                                on_lost(resource, gen)
                            except Exception:
                                pass
                        return

            t = threading.Thread(target=_keep, daemon=True,
                                 name="bsai-orch-keepalive")
            t.start()
            return ok, gen, stop.set


# ---------------- 便捷封装：8190 XPU VAE 二进制转发 ----------------
def vae_decode_xpu(latent_np_f32, endpoint="http://127.0.0.1:8190/vae/decode",
                   timeout=600):
    """把 (B,C,T,H,W) float32 latent 转发 8190 XPU VAE 解码，返回同形状输出。
    与 8190 服务二进制协议一致（magic BSAIVAE1 + 维数 + shape + raw）。
    """
    import urllib.request
    shape = latent_np_f32.shape
    head = _HEAD.encode() + struct.pack("<I", len(shape)) + \
        struct.pack("<%dI" % len(shape), *shape)
    payload = head + latent_np_f32.astype("float32").tobytes()
    req = urllib.request.Request(endpoint, data=payload, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    ndim = struct.unpack("<I", data[8:12])[0]
    oshape = struct.unpack("<%dI" % ndim, data[12:12 + 4 * ndim])
    import numpy as np
    return np.frombuffer(data[12 + 4 * ndim:], dtype=np.float32).reshape(oshape)


# ---------------- 内置档位注册表（声明式，双轨开箱即用） ----------------
# 本地档 = BSAI 自家插件（已存在，不重写）；云端档 = ComfyUI 官方 Partner Node
# 对应后端（BSAI 只做调度与执行，不重写官方节点）。
BUILTIN_PROVIDERS = [
    # 声音（tts_synth）
    {"name": "BSAI IndexTTS-2.5", "kind": "tts_synth", "tier": "local",
     "engine": "indextts", "cost": "free"},
    {"name": "HeyGen Starfish", "kind": "tts_synth", "tier": "cloud",
     "engine": "heygen", "api_key_env": "HEYGEN_API_KEY", "cost": "per_call"},
    {"name": "ElevenLabs", "kind": "tts_synth", "tier": "cloud",
     "engine": "elevenlabs", "api_key_env": "ELEVENLABS_API_KEY", "cost": "per_call"},
    # 视频超分（video_upscale）
    {"name": "BSAI SeedVR2", "kind": "video_upscale", "tier": "local",
     "engine": "seedvr2", "cost": "free"},
    {"name": "BSAI H3-upscale-4K", "kind": "video_upscale", "tier": "local",
     "engine": "h3", "cost": "free"},
    {"name": "FLUX Video Upscale", "kind": "video_upscale", "tier": "cloud",
     "engine": "flux_upscale", "api_key_env": "COMFY_API_KEY",
     "cost": "per_call",
     "targets": ["1080p", "2K", "4K"], "modes": ["creative", "precise"],
     "endpoint_doc": "/proxy/bfl/v1/flux-tools/video-upscale-v1"},
    {"name": "Clarity Crystal", "kind": "video_upscale", "tier": "cloud",
     "engine": "crystal", "api_key_env": "CAI_API_KEY", "cost": "per_call"},
    # 视频配音（video_dub）——官方 Partner Node：VEED Dubbing (#17388)
    {"name": "VEED Dubbing", "kind": "video_dub", "tier": "cloud",
     "engine": "veed", "api_key_env": "COMFY_API_KEY", "cost": "per_call",
     "models": ["standard", "pro"],
     "endpoint_doc": "/proxy/veed/v1/dubbing"},
]


def register_builtin_providers():
    """把全家桶已知本地/云端档位写入 capabilities。
    其他插件仍可自行 BSAIOrch.register() 覆盖同名条目。
    同名条目会被覆盖（engine 升级后 key/端点随 meta 更新）。"""
    for p in BUILTIN_PROVIDERS:
        meta = {"tier": p["tier"], "engine": p["engine"],
                "api_key_env": p.get("api_key_env", ""),
                "cost": p.get("cost", "free")}
        for k in ("targets", "modes", "models", "endpoint_doc"):
            if p.get(k):
                meta[k] = p[k]
        BSAIOrch.register(
            name=p["name"], kind=p["kind"], hardware=[],
            endpoint="", health="", weight=1.0,
            meta=meta,
        )
    return len(BUILTIN_PROVIDERS)
