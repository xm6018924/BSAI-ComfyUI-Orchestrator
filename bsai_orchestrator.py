# -*- coding: utf-8 -*-
"""
BSAI 多硬件编排核心库（bsai_orchestrator）
===========================================
跨进程资源协调：CUDA 主实例(8191) / XPU worker(8190) / NPU 服务共用一套状态。

能力：
1. ResourceLease  —— reslock 式跨进程互斥：单 JSON 状态文件 + 文件锁 + PID 存活
                    检查 + TTL 自动回收。任何 BSAI 插件在"取任务前"先取租约，
                    保证 GPU1 采样 / XPU VAE decode / NPU 检测互不抢活。
2. TaskRegistry  —— 全局优先级队列（0=主采样 > 1=用户实时 > 2=后台批量），
                    记录 waiting/running 深度与队列状态。
3. decide_vae_target —— 水位分流决策：GPU1 显存低于阈值 → VAE 解码分流 XPU，
                    不可用则 CPU；判定时机 = 节点执行入口（取任务前）。
4. get_signals / set_policy —— HUD 双水位信号（xpu_offload_min_vram +
                    cuda_reserve_mb）与四路实时信号（VRAM 余量 / GPU 利用率 /
                    DDR5 带宽 / 温度）。
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

try:
    import numpy as np
except ImportError:
    np = None

# ---------------- 路径与常量 ----------------
# custom_nodes/BSAI-ComfyUI-Orchestrator/../.. = ComfyUI/ → 状态放 ComfyUI/user/
DEFAULT_STATE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "user"
)
STATE_FILE = os.path.join(DEFAULT_STATE_DIR, "bsai_orchestrator_state.json")
LOCK_FILE = os.path.join(DEFAULT_STATE_DIR, "bsai_orchestrator.lock")

# 资源名（跨插件约定）
RES_GPU1_SAMPLING = "gpu1_sampling"   # RTX 5090 主采样（最高优先）
RES_XPU_VAE = "xpu_vae_decode"        # 核显 XPU VAE 解码
RES_NPU_DETECT = "npu_face_detect"    # NPU 人脸检测
RES_CPU_VAE = "cpu_vae_decode"        # CPU VAE 兜底

# 优先级（数值小者优先）
PRIO_SAMPLING = 0
PRIO_REALTIME = 1
PRIO_BATCH = 2

DEFAULT_POLICY = {
    "xpu_offload_min_vram": 2500,  # GPU1 显存余量低于此值(MB) → VAE 分流
    "cuda_reserve_mb": 750,        # GPU1 常驻保留显存(MB)，采样不越过
}

_LOCK_TTL = 120         # 租约默认 TTL（秒）。长任务请用 acquire_with_keepalive 续租
_LOCK_RETRY = 25        # 获取文件锁最大重试次数
_LOCK_WAIT = 0.1        # 每次重试间隔（秒）
_LOCK_KEEPALIVE = 40    # watchdog 续租间隔（秒）：TTL 内 ≥3 次续租机会


# ---------------- 文件锁 ----------------
def _with_lock(callback, fallback=True):
    """文件锁保护下执行 callback（读改写状态文件）。

    fallback=True（读路径）：锁获取极端失败时无锁执行，只读无害。
    fallback=False（写路径）：锁获取失败直接返回 None——安全失败，
    杜绝"无锁抢占导致跨进程双写"（P0 正确性）。
    """
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    for attempt in range(_LOCK_RETRY):
        try:
            with open(LOCK_FILE, "a+b") as f:
                f.seek(0)
                if msvcrt is not None:
                    try:
                        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    except OSError:
                        time.sleep(_LOCK_WAIT)
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
            time.sleep(_LOCK_WAIT)
    if fallback:
        return callback()  # 读路径：极端情况下无锁执行（只读无害）
    return None            # 写路径：安全失败，不执行写


def _load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"leases": {}, "tasks": [], "policy": dict(DEFAULT_POLICY)}


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


# ---------------- ResourceLease（reslock 租约） ----------------
class ResourceLease:
    """跨进程资源租约：单 JSON 状态 + 文件锁 + PID 存活 + TTL 回收。

    P0 正确性（防跨进程双写）：
    - 写路径（acquire/release/renew）锁获取失败即安全失败，绝不无锁写入；
    - 世代号（gen）fencing：release/renew 必须携带当初 acquire 的 gen，
      过期节点无法删除/续租新持有者的租约；
    - watchdog 续租：长任务用 acquire_with_keepalive 后台刷新 ts，
      不再因 TTL 到期被误回收。

    用法（推荐，取任务前）：
        lease = ResourceLease()
        ok, gen, stop = lease.acquire_with_keepalive(
            RES_XPU_VAE, holder="8191", ttl=120, keepalive=40,
            on_lost=lambda r, g: print("租约被抢占，立即停止写资源", r))
        if ok:
            try:
                ... 执行任务 ...
            finally:
                stop()
                lease.release(RES_XPU_VAE, holder="8191", gen=gen)
    """

    @staticmethod
    def state():
        return _with_lock(_load_state)

    def acquire(self, resource, holder="", ttl=_LOCK_TTL):
        """尝试获取资源租约。持有者已死或 TTL 过期 → 回收后重新获取。
        返回 (ok, gen)：ok=False 未取到（有效持有中，或写锁获取失败）；
        gen 为本次租约的世代号（fencing token），release/renew 必须携带。
        """
        if not holder:
            holder = "pid%d" % os.getpid()

        def _do():
            st = _load_state()
            leases = st.setdefault("leases", {})
            now = _now()
            cur = leases.get(resource)
            if cur is not None:
                # 清理过期/死亡持有者
                expired = (now - cur.get("ts", 0)) > cur.get("ttl", _LOCK_TTL)
                dead = not _pid_alive(cur.get("pid"))
                if not expired and not dead:
                    return (False, 0)  # 资源被有效持有
                del leases[resource]
            # 世代号（fencing token）：跨资源单调递增
            st["epoch"] = int(st.get("epoch", 0)) + 1
            gen = int(st["epoch"])
            leases[resource] = {
                "holder": holder, "pid": os.getpid(), "ts": now, "ttl": ttl,
                "resource": resource, "gen": gen,
            }
            _save_state(st)
            return (True, gen)

        res = _with_lock(_do, fallback=False)
        return res if res is not None else (False, 0)

    def release(self, resource, holder="", gen=None):
        """释放租约。仅当持有者是自己且世代号匹配（gen）时删除。
        gen 不匹配（租约已被新持有者回收）→ 返回 False，不做任何删除。
        """
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
        """watchdog 续租：刷新持有者 ts，防止长任务 TTL 到期被误回收。
        世代号不匹配（已被抢占）→ 返回 False，调用方必须立即停止写资源。
        """
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

    def acquire_with_keepalive(self, resource, holder="", ttl=_LOCK_TTL,
                               keepalive=_LOCK_KEEPALIVE, on_lost=None):
        """推荐入口：取租约 + 后台 watchdog 线程周期续租。
        返回 (ok, gen, stop_fn)：
          ok=False 未取到；stop_fn() 停止续租线程（任务结束 finally 中调用）；
          on_lost(resource, gen) 在续租失败（租约被抢占/丢失）时回调——
          调用方收到回调必须立即停止对该资源的写入（防止双写）。
        """
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
                    okr = self.renew(resource, holder=holder, gen=gen, ttl=ttl)
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
                             name="bsai-lease-keepalive")
        t.start()
        return ok, gen, stop.set


# ---------------- TaskRegistry（全局优先级队列） ----------------
class TaskRegistry:
    """跨进程优先级任务队列。0=主采样 > 1=用户实时 > 2=后台批量。"""

    @staticmethod
    def register(task_id, priority=PRIO_BATCH, kind="vae_decode", holder=""):
        """登记一个任务（进入 waiting）。同 id 重复登记更新优先级。"""
        if not holder:
            holder = "pid%d" % os.getpid()

        def _do():
            st = _load_state()
            tasks = st.setdefault("tasks", [])
            for t in tasks:
                if t.get("id") == task_id:
                    t.update({"priority": priority, "kind": kind, "holder": holder,
                              "status": "waiting", "ts": _now()})
                    _save_state(st)
                    return task_id
            tasks.append({"id": task_id, "priority": priority, "kind": kind,
                          "holder": holder, "status": "waiting", "ts": _now()})
            _save_state(st)
            return task_id

        return _with_lock(_do)

    @staticmethod
    def start(task_id):
        def _do():
            st = _load_state()
            for t in st.get("tasks", []):
                if t.get("id") == task_id:
                    t["status"] = "running"
                    t["started_ts"] = _now()
                    _save_state(st)
                    return True
            return False

        return _with_lock(_do)

    @staticmethod
    def finish(task_id, ok=True):
        def _do():
            st = _load_state()
            tasks = st.get("tasks", [])
            for i, t in enumerate(tasks):
                if t.get("id") == task_id:
                    t["status"] = "done" if ok else "failed"
                    t["end_ts"] = _now()
                    _save_state(st)
                    return True
            return False

        return _with_lock(_do)

    @staticmethod
    def snapshot():
        """返回队列快照：waiting/running 深度 + 明细。"""
        st = _with_lock(_load_state)
        tasks = st.get("tasks", [])
        running = [t for t in tasks if t.get("status") == "running"]
        waiting = [t for t in tasks if t.get("status") == "waiting"]
        # 只保留最近 50 条
        st["tasks"] = tasks[-50:]
        return {
            "running": len(running),
            "waiting": len(waiting),
            "running_detail": running,
            "waiting_detail": waiting[:10],
        }


# ---------------- 设备分流决策 ----------------
def _cuda_free_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.mem_get_info()[0] / (1024.0 ** 2)
    except Exception:
        pass
    return 0.0


def _xpu_online():
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:8190/health/ready", timeout=1.5) as r:
            return json.loads(r.read()).get("ready", False)
    except Exception:
        return False


def decide_vae_target(cuda_free_mb=None, xpu_online=None, policy=None):
    """水位分流决策：GPU1 显存低于阈值 → XPU（在线）否则 CPU；否则留 CUDA。

    判定时机：节点执行入口（取任务前）调用一次。
    """
    if policy is None:
        policy = _load_state().get("policy", DEFAULT_POLICY)
    threshold = policy.get("xpu_offload_min_vram", DEFAULT_POLICY["xpu_offload_min_vram"])
    if cuda_free_mb is None:
        cuda_free_mb = _cuda_free_mb()
    if xpu_online is None:
        xpu_online = _xpu_online()
    if cuda_free_mb and cuda_free_mb < threshold:
        return "xpu" if xpu_online else "cpu"
    return "cuda"


# ---------------- HUD 信号 ----------------
def get_signals():
    """HUD 双水位信号 + 四路实时信号（供硬件监视插件轮询）。"""
    st = _load_state()
    policy = st.get("policy", DEFAULT_POLICY)
    sig = {
        # 双水位信号
        "xpu_offload_min_vram": policy.get("xpu_offload_min_vram", DEFAULT_POLICY["xpu_offload_min_vram"]),
        "cuda_reserve_mb": policy.get("cuda_reserve_mb", DEFAULT_POLICY["cuda_reserve_mb"]),
        # 实时四路
        "cuda_free_mb": int(_cuda_free_mb()),
        "gpu_util": 0,
        "ddr5_bw_mbps": 0,
        "temp_c": 0,
        # 编排状态
        "xpu_online": _xpu_online(),
        "leases": st.get("leases", {}),
        "tasks": TaskRegistry.snapshot(),
    }
    # 四路实时信号优先从 BSAI 硬件监视采集器拿（避免重复实现）
    try:
        import sys
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "BSAI-ComfyUI-Hardware-Perf"))
        import perf_collector as pc
        g = pc.collect_gpu_nvidia()
        if g.get("ok"):
            sig["gpu_util"] = g.get("util", 0)
            sig["temp_c"] = g.get("temp_c", 0)
        x = pc.collect_xpu()
        if isinstance(x, dict):
            sig["ddr5_bw_mbps"] = int(x.get("mem_used_mb", 0) * 0)  # 占位：带宽改由 worker 上报
    except Exception:
        pass
    return sig


def set_policy(xpu_offload_min_vram=None, cuda_reserve_mb=None):
    """HUD 可调水位（保存到状态文件，跨进程生效）。"""
    def _do():
        st = _load_state()
        p = st.setdefault("policy", dict(DEFAULT_POLICY))
        if xpu_offload_min_vram is not None:
            p["xpu_offload_min_vram"] = int(xpu_offload_min_vram)
        if cuda_reserve_mb is not None:
            p["cuda_reserve_mb"] = int(cuda_reserve_mb)
        _save_state(st)
        return dict(p)

    return _with_lock(_do)


# ---------------- XPU VAE decode 服务（挂在 ComfyUI 进程内） ----------------
_MAGIC = b"BSAIVAE1"
DEFAULT_VAE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "models",
    "vae", "minimax_h3_video_vae_fp16.safetensors")


class XPUVAEServer:
    """Intel 核显(XPU) VAE 解码服务：懒加载 VAE，线程安全。

    二进制协议（与 VAERouter xpu 策略一致）：
      请求: 8B 魔数 + 4B 维数 + shape(uint32*N) + float32 raw latent
      响应: 同结构，payload 为解码后的 float32 IMAGE
    """

    def __init__(self, vae_path=None):
        self.vae_path = vae_path or DEFAULT_VAE_PATH
        self._vae = None
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._inflight = 0
        self._total = 0
        self._started = time.time()
        # 引擎忙碌统计：滑动窗口记录每次解码任务 [start_ts, duration]，
        # duration=None 表示仍在进行中（按 now-start 计）。窗口 10s。
        self._BUSY_WINDOW = 10.0
        self._busy_hist = []

    def _load(self):
        if self._vae is None:
            with self._load_lock:
                if self._vae is None:
                    import torch
                    if not getattr(torch, "xpu", None) or not torch.xpu.is_available():
                        raise RuntimeError("torch.xpu 不可用（本服务须在 XPU 实例运行）")
                    from comfy.sd import VAE
                    from comfy.utils import load_torch_file
                    if not os.path.exists(self.vae_path):
                        raise FileNotFoundError("VAE 模型缺失: %s" % self.vae_path)
                    print("[BSAI-XPU-VAE] 加载 VAE: %s" % os.path.basename(self.vae_path), flush=True)
                    t0 = time.time()
                    _sd = load_torch_file(self.vae_path)
                    v = VAE(sd=_sd)
                    v.first_stage_model.to("xpu")
                    # 释放 load_torch_file 的临时缓冲缓存：否则 oneDNN 后续 USM 分配
                    # 报 "could not create a memory"（XPU caching allocator 占用 UR 资源）
                    try:
                        torch.xpu.empty_cache()
                    except Exception:
                        pass
                    self._vae = v
                    print("[BSAI-XPU-VAE] VAE 就绪 device=xpu 耗时 %.1fs" % (time.time() - t0), flush=True)
        return self._vae

    def decode_bytes(self, payload: bytes) -> bytes:
        t0 = time.time()
        with self._lock:
            self._inflight += 1
            self._total += 1
            self._busy_hist.append([t0, None])  # 解码任务开始（进行中）
        try:
            if len(payload) < 12 or payload[:8] != _MAGIC:
                raise ValueError("协议头错误（魔数/长度）")
            ndim = struct.unpack("<I", payload[8:12])[0]
            off = 12
            if ndim < 1 or ndim > 8:
                raise ValueError("非法维数 %d" % ndim)
            shape = list(struct.unpack("<%dI" % ndim, payload[off:off + 4 * ndim]))
            off += 4 * ndim
            import torch
            raw = np.frombuffer(payload[off:], dtype=np.float32)
            lat = torch.from_numpy(raw.copy()).reshape(shape).to("xpu", torch.float16)
            vae = self._load()
            # 释放 torch XPU 缓存，避免 oneDNN USM 分配失败（"could not create a memory"）
            try:
                torch.xpu.empty_cache()
            except Exception:
                pass
            out = self._decode_frames(vae, lat)
            if out is None:
                raise RuntimeError("first_stage_model.decode 返回 None")
            out = out.detach().cpu().float().contiguous()
            b = out.numpy().tobytes()
            head = _MAGIC + struct.pack("<I", out.dim()) + \
                struct.pack("<%dI" % out.dim(), *out.shape)
            return head + b
        finally:
            with self._lock:
                self._inflight -= 1
                # 收尾：记录任务实际耗时（引擎忙碌时长）
                if self._busy_hist and self._busy_hist[-1][1] is None:
                    self._busy_hist[-1][1] = time.time() - t0

    def _busy_pct(self):
        """最近 10s 引擎忙碌率：窗口内解码执行时长累计 ÷ 窗口 × 100。

        进行中任务按 (now - start) 计入；已完成的按实际耗时计入；
        完全滑出窗口的旧条目剔除。单线程串行解码 → 即真实引擎负载。
        """
        now = time.time()
        keep = []
        busy_sum = 0.0
        for start, dur in self._busy_hist:
            if dur is None:
                busy = now - start
                keep.append([start, None])
            else:
                if start + dur < now - self._BUSY_WINDOW:
                    continue
                busy = dur
                keep.append([start, dur])
            busy_sum += min(busy, self._BUSY_WINDOW)
        self._busy_hist = keep
        if not self._busy_hist:
            return 0.0
        return min(100.0, busy_sum / self._BUSY_WINDOW * 100.0)

    def _decode_frames(self, vae, lat):
        """按时间维拆帧、串行解码：H3 VAE 的 tiled_decode 对整段 latent 逐 tile
        串行，长视频极慢且峰值内存高。拆帧后每帧独立 decode + 释放缓存，
        降低峰值；不并行——level_zero 驱动多线程并发提交会 DEVICE_LOST。
        """
        import torch
        if lat.dim() != 5 or lat.shape[2] <= 1:
            return vae.first_stage_model.decode(lat)
        n_frames = lat.shape[2]
        outs = []
        for i in range(n_frames):
            o = vae.first_stage_model.decode(lat[:, :, i:i + 1, :, :])
            if o is None:
                raise RuntimeError("frame decode 返回 None")
            outs.append(o.detach().cpu().float().contiguous())
            del o
            try:
                torch.xpu.empty_cache()
            except Exception:
                pass
            # 帧间喘息：level_zero 驱动连续长任务提交会 DEVICE_LOST
            time.sleep(0.2)
        return torch.cat(outs, dim=2)

    def status(self):
        st = {
            "status": "online",
            "ready": self._vae is not None,
            "device": "xpu",
            "inflight": self._inflight,
            "total_calls": self._total,
            "uptime_s": int(time.time() - self._started),
            "vae": os.path.basename(self.vae_path) if os.path.exists(self.vae_path) else "missing",
            "ts": time.time(),
        }
        return st

    def system_stats(self):
        """perf_collector 兼容：XPU 显存/内存（devices/system 结构）。
        Intel 驱动不支持 torch.xpu.mem_get_info() → 用系统内存近似（显存上限）。"""
        out = {"ok": False, "devices": [], "system": {}}
        try:
            import torch
            try:
                free, total = torch.xpu.mem_get_info()
                mem_ok = True
            except Exception:
                mem_ok = False
                free = total = 0
            name = "Intel Arc XPU"
            try:
                name = torch.xpu.get_device_name(0)
            except Exception:
                pass
            out["ok"] = True
            # 引擎忙碌率（真负载）——HUD 表盘主读数
            out["busy_pct"] = round(self._busy_pct(), 1)
            if mem_ok:
                out["devices"] = [{
                    "name": name, "vram_free": int(free),
                    "vram_total": int(total), "vram_used": int(total - free),
                }]
                out["util"] = int((total - free) / total * 100) if total else 0
        except Exception as e:
            out["error"] = str(e)
        try:
            import psutil
            vm = psutil.virtual_memory()
            out["system"] = {"used": int(vm.used), "total": int(vm.total)}
            if not out.get("devices"):
                # Intel 驱动无法查询显存 → 用系统内存近似（HUD 仍可显示占用）
                out["devices"] = [{
                    "name": name if "name" in out else "Intel Arc XPU",
                    "vram_free": int(vm.available),
                    "vram_total": int(vm.total),
                    "vram_used": int(vm.used),
                }]
                out["util"] = int(vm.used / vm.total * 100) if vm.total else 0
        except Exception:
            pass
        return out

    # ---- aiohttp handlers ----
    async def handle_ready(self, _):
        try:
            self._load()
            ready, err = True, None
        except Exception as e:
            ready, err = False, str(e)
        return _web_json({"ready": ready, "error": err, "device": "xpu"})

    async def handle_status(self, _):
        return _web_json(self.status())

    async def handle_decode(self, req):
        body = await req.read()
        try:
            return _web_bytes(self.decode_bytes(body))
        except Exception as e:
            import traceback
            traceback.print_exc()
            return _web_json({"error": str(e)}, status=500)

    async def handle_stats(self, _):
        return _web_json(self.system_stats())


def _web_json(data, status=200):
    try:
        from aiohttp import web
        return web.json_response(data, status=status)
    except Exception:
        import json as _j
        return _TextResponse(_j.dumps(data), status=status,
                             content_type="application/json")


def _web_bytes(data):
    try:
        from aiohttp import web
        return web.Response(body=data, content_type="application/octet-stream")
    except Exception:
        return _TextResponse(data, status=200)


class _TextResponse:
    """aiohttp 不可用时的极简响应（不应触发）"""

    def __init__(self, body, status=200, content_type="text/plain"):
        self.body = body
        self.status = status
        self.content_type = content_type


def register_routes(prompt_server=None, vae_service=None):
    """挂编排端点（ComfyUI prompt_server）。

    - /orch/state     ：编排状态（租约/队列/水位）—— 两实例均注册
    - /health/ready   ：XPU VAE 服务就绪
    - /vae/decode     ：核显 VAE 解码（二进制协议）
    - /system_stats   ：XPU 显存/内存（perf_collector 兼容）
    vae_service=None 时自动判断：本进程 torch.xpu 可用 → 注册 VAE 端点。
    """
    if vae_service is None:
        try:
            import torch
            vae_service = bool(getattr(torch, "xpu", None) and torch.xpu.is_available())
        except Exception:
            vae_service = False

    xvae = XPUVAEServer() if vae_service else None

    async def orch_state(_):
        return _web_json(get_signals())

    async def orch_policy(req):
        try:
            body = await req.json()
            return _web_json(set_policy(
                xpu_offload_min_vram=body.get("xpu_offload_min_vram"),
                cuda_reserve_mb=body.get("cuda_reserve_mb")))
        except Exception as e:
            return _web_json({"error": str(e)}, status=400)

    try:
        import server as _server
        _ps = prompt_server or _server.PromptServer.instance
        if _ps is not None:
            # ComfyUI 0.37：RouteTableDef 在插件加载后才动态 add_route 不会生效，
            # 必须直接挂到 app.router（aiohttp UrlDispatcher）才能动态注册。
            router = None
            app = getattr(_ps, "app", None)
            if app is not None and hasattr(app, "router"):
                router = app.router
            routes = getattr(_ps, "routes", None)
            items = [("GET", "/orch/state", orch_state),
                     ("POST", "/orch/policy", orch_policy)]
            if vae_service and xvae is not None:
                items += [
                    ("GET", "/health/ready", xvae.handle_ready),
                    ("POST", "/vae/decode", xvae.handle_decode),
                    ("GET", "/system_stats", xvae.handle_stats),
                ]
            for method, path, handler in items:
                added = False
                if router is not None:
                    try:
                        router.add_route(method, path, handler)
                        added = True
                    except Exception:
                        added = False
                if not added and routes is not None:
                    try:
                        routes.add_route(method, path, handler)
                    except Exception:
                        try:
                            res = routes.get(path)
                            if res is not None:
                                res.add_route(method, path, handler)
                        except Exception:
                            pass
            extra = " + /vae/decode /system_stats /health/ready(VAE)" if vae_service else ""
            print("[BSAI-Orchestrator] 编排端点已挂载: /orch/state /orch/policy%s" % extra)
        else:
            print("[BSAI-Orchestrator] PromptServer.instance 未就绪，路由延迟到节点触发注册")
    except Exception as e:
        print("[BSAI-Orchestrator] 路由注册容错跳过:", e)
    return xvae
