# BSAI-ComfyUI-Orchestrator

ComfyUI 多硬件协同编排节点 —— 在采样/VAE 流程前登记任务租约，按 GPU1 显存水位自动决策 VAE 解码分流目标（CUDA / XPU / CPU），并在画布内展示实时编排状态。

Multi-hardware orchestrator nodes for ComfyUI — registers task leases before sampling/VAE steps, auto-decides VAE decode offload target (CUDA / XPU / CPU) based on GPU1 VRAM headroom, and shows live orchestration status on canvas.

---

## Nodes | 节点

| Node | 说明 |
|---|---|
| **BSAIOrchestratorGate** | 编排门：设置水位策略 + 登记任务租约 + 决策 VAE 分流目标 |
| **BSAIOrchestratorStatus** | 画布内实时展示租约 / 队列 / 信号 JSON |

## Install | 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xm6018924/BSAI-ComfyUI-Orchestrator.git
```

No extra pip dependencies beyond ComfyUI base.

## Usage | 使用

1. Place **BSAIOrchestratorGate** before your sampling / VAE decode chain.
2. Set `xpu_offload_min_vram` (MB) — when GPU1 free VRAM drops below this, VAE decode is routed to XPU/CPU.
3. Set `cuda_reserve_mb` — VRAM kept resident on GPU1 for sampling.
4. Connect `vae_target` output to **BSAI-ComfyUI-VAERouter** for automatic device selection.
5. Add **BSAIOrchestratorStatus** anywhere on canvas to monitor leases/queue.

## Compatibility | 兼容

- ComfyUI ≥ 0.37
- Windows 10/11
- Pairs with **BSAI-ComfyUI-VAERouter** (runtime routing) and the XPU VAE worker on port 8190.

## License | 许可

MIT
