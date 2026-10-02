# BSAI-ComfyUI-Orchestrator

ComfyUI 多硬件协同编排插件 —— 在采样 / VAE 流程之前登记任务租约，按 GPU1 显存水位自动决策 VAE 解码分流目标（CUDA / XPU / CPU），并通过单文件 SDK 让其他 BSAI 插件"一行接入"自动协同。

Multi-hardware orchestration plugin for ComfyUI — registers task leases before sampling / VAE steps, auto-decides the VAE decode offload target (CUDA / XPU / CPU) based on GPU1 VRAM headroom, and lets other BSAI plugins integrate via a single-file SDK with zero manual wiring.

---

## 说明 | What & Why

### ① 跨进程资源租约 | Cross-process resource leases
**中文**：在 sampling / VAE 解码真正开始前，插件先向编排层登记一条任务租约（lease）。租约采用"互斥锁 + 世代号 fencing + watchdog 后台续租"协议，与编排核心共享同一份状态文件，因此即使多个 BSAI 插件运行在不同进程（例如主 ComfyUI、XPU VAE worker :8190、VAERouter :8191），也能保证同一时刻只有一个任务占用同一份硬件资源，长任务由 watchdog 自动续租、异常退出由 TTL 自动回收，不会出现"显存被两个进程同时踩爆"的情况。

**English**: Before sampling / VAE decode actually starts, the plugin first registers a task *lease* with the orchestrator. The lease protocol combines an exclusive lock, a fencing generation number, and a background watchdog keepalive, and shares one on-disk state file with the orchestration core. As a result, even when multiple BSAI plugins run in separate processes (main ComfyUI, XPU VAE worker on :8190, VAERouter on :8191), only one task ever occupies the same hardware at a time. Long-running tasks are kept alive by the watchdog; crashed holders are reclaimed by TTL — no more VRAM being stomped by two processes at once.

### ② 水位分流 | Water-gated offload routing
**中文**：编排层持续读取 GPU1 的空闲显存水位。当 GPU1 余量低于 `cuda_reserve_mb` / `xpu_offload_min_vram` 阈值时，`allocate("vae_decode", ...)` 会自动把候选列表里的 `cuda` 剔除，改选 `xpu` 或 `cpu`；当水位恢复后又自动回切到 CUDA。这是"按显存余量动态决策解码落点"的核心策略，不需要用户手工切换 device。

**English**: The orchestrator continuously samples GPU1 free-VRAM headroom. When headroom drops below the `cuda_reserve_mb` / `xpu_offload_min_vram` thresholds, `allocate("vae_decode", ...)` automatically drops `cuda` from the candidate list and falls back to `xpu` or `cpu`; when headroom recovers it transparently routes back to CUDA. This is the core "pick the decode target from live VRAM headroom" policy — no manual device switching required.

### ③ SDK 自动协同 | SDK-driven auto-coordination
**中文**：其他 BSAI 插件只需 `import bsai_orch_client`，即可在模块加载时自动完成**能力注册 → 硬件探活 → 水位分流 → 跨进程租约 → 目标分配 → 释放**全链路，无需人工接线、无需共享内存。SDK 是自包含单文件（仅 Python 标准库），不依赖 ComfyUI 内部模块。未来所有 BSAI 插件按同一约定接入，即自动形成一个"即插即用"的多进程硬件调度网络。

**English**: Other BSAI plugins only need to `import bsai_orch_client`; on module load the SDK automatically runs the full chain — **capability registration → health probing → water-gated routing → cross-process lease → target allocation → release** — with zero manual wiring and no shared-memory dependency. The SDK is a self-contained single file (stdlib only) and does not import any ComfyUI internal modules. Once every BSAI plugin follows the same convention, they form a plug-and-play multi-process hardware scheduling network automatically.

---

## 安装 | Install

### 方式 A：git clone（推荐）| Via git clone (recommended)
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xm6018924/BSAI-ComfyUI-Orchestrator.git
```

### 方式 B：复制目录 | Copy the folder
把本仓库整个目录复制到 `ComfyUI/custom_nodes/BSAI-ComfyUI-Orchestrator` 下即可。
Copy this repo's folder into `ComfyUI/custom_nodes/BSAI-ComfyUI-Orchestrator`.

无额外 pip 依赖（仅 Python 标准库）。重启 ComfyUI 后节点出现在 `BSAI` 分组下。
No extra pip dependencies (stdlib only). Restart ComfyUI; the nodes appear under the `BSAI` group.

### 其他插件如何接入本 SDK | How other plugins reach this SDK
本插件自身就是 SDK 的提供方。其他 BSAI 插件在自己的 `__init__.py` / `nodes.py` 顶部把 Orchestrator 目录加进 `sys.path`，再 `import bsai_orch_client`：

This plugin *is* the SDK provider. Other BSAI plugins prepend the Orchestrator folder to `sys.path` at the top of their own `__init__.py` / `nodes.py`, then `import bsai_orch_client`:

```python
# 例如 BSAI-ComfyUI-VAERouter/nodes.py 顶部
# e.g. top of BSAI-ComfyUI-VAERouter/nodes.py
import os, sys
_ORCH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "BSAI-ComfyUI-Orchestrator")
if _ORCH not in sys.path:
    sys.path.insert(0, _ORCH)
from bsai_orch_client import BSAIOrch   # noqa: E402
```

---

## 使用 | Usage

### SDK 接入三步 | Three-step SDK integration

**① 顶部引入 | Import at top of module**
```python
import os, sys
_ORCH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "BSAI-ComfyUI-Orchestrator")
if _ORCH not in sys.path:
    sys.path.insert(0, _ORCH)
from bsai_orch_client import BSAIOrch
```

**② 模块加载时自动注册 | Auto-register at module load time**
```python
# ComfyUI 启动时本模块被 import，即自动注册能力到编排层
# Runs automatically on ComfyUI startup as soon as this module is imported
BSAIOrch.register(
    name="BSAI-VAERouter",                 # 插件名（唯一）/ unique plugin name
    kind="vae_decode",                     # 能力类型 / capability kind
    hardware=["xpu", "cuda", "cpu"],       # 硬件偏好顺序 / HW preference order
    endpoint="http://127.0.0.1:8190/vae/decode",  # 服务端点（可选）/ service endpoint (optional)
    health="http://127.0.0.1:8190/health/ready",  # 健康探针（可选）/ health probe (optional)
    weight=1.0,
)
```

**③ 执行路径分配租约，finally 中释放 | Allocate on hot path, release in finally**
```python
alloc = BSAIOrch.allocate(
    "vae_decode",
    requester="8191",     # 调用方标识 / caller id
    watchdog=True,        # 默认后台续租，长任务防 TTL 过期 / keepalive for long tasks
)
if alloc.ok:
    try:
        # alloc.target  → "xpu" | "cuda" | "cpu"，编排决策出的落点
        # alloc.endpoint→ 对应的服务端点（register 时登记的那个）
        # alloc.gen     → fencing 世代号
        # alloc.lease_ok→ 是否成功拿到跨进程互斥租约
        do_decode_on(alloc.target, alloc.endpoint)
    finally:
        alloc.release()   # 释放租约 + 停 watchdog / release lease & stop watchdog
```

### 编排节点 | Orchestrator nodes

| Node 节点 | 说明 |
|---|---|
| **BSAIOrchestratorGate** | 编排门：设置**水位策略**（`xpu_offload_min_vram` / `cuda_reserve_mb`）+ 登记**任务租约** + 决策 **VAE 分流目标**（CUDA / XPU / CPU）。把它放在 sampling / VAE decode 链路最前面，把 `vae_target` 输出接到 VAERouter 即可。<br/>Orchestration gate: configures the **water-level policy** (`xpu_offload_min_vram` / `cuda_reserve_mb`), registers the **task lease**, and makes the **VAE offload decision** (CUDA / XPU / CPU). Place it at the head of your sampling / VAE decode chain and wire `vae_target` into the VAERouter. |
| **BSAIOrchestratorStatus** | 画布内实时状态面板：展示当前 capabilities、活动租约、优先级队列、信号 JSON。任意位置拖一个上来即可监控。<br/>Live on-canvas status panel: shows current capabilities, active leases, priority queue, and signal JSON. Drop it anywhere on the canvas to monitor. |

---

## Nodes | 节点（速查 / quick reference）

| Node | 说明 |
|---|---|
| **BSAIOrchestratorGate** | 编排门：水位策略 + 租约登记 + VAE 分流决策 |
| **BSAIOrchestratorStatus** | 画布内实时展示租约 / 队列 / 信号 JSON |

## Compatibility | 兼容

- ComfyUI ≥ 0.37
- Windows 10 / 11
- 与 **BSAI-ComfyUI-VAERouter**（运行时路由）及 XPU VAE worker（端口 8190）配套使用。
  Pairs with **BSAI-ComfyUI-VAERouter** (runtime routing) and the XPU VAE worker on port 8190.

## License | 许可

MIT
