# BSAI 插件多硬件协同示例工作流清单

生成日期：2026-10-03
适用整合包：`G:\BSAI-ComfyUI-intel-XPU-GPU-NPU-aki`
覆盖范围：ComfyUI 全部 **24 个 BSAI 插件**，每个插件新增 1 个"多硬件协同"示例工作流，共 **24 个工作流文件**，全部为 ComfyUI 前端可直接加载的 UI workflow 格式（JSON，已通过 `json.load` 解析与结构校验：节点引用、槽位、连线类型、类名存在性 24/24 OK）。

## 多硬件协同分工（所有工作流统一遵循）

| 硬件 | 承担角色 | 工作流中的节点 |
|---|---|---|
| Intel AI Boost NPU | 前置检测/理解（人脸检测、关键点、跟踪）+ 深度估计 + 决策文本 | `H3FaceTrackCrop`（BSAI-ComfyUI-FaceRefine 消费 NPU-Service 能力）、`BSAINPU_DepthEstimate`、`BSAINPU_LLMGenerate`（BSAI-NPU-Service，DepthAnything-V2-Small + Qwen3-4B 对称 INT4） |
| NVIDIA RTX 5090 (CUDA, 8191) | 主生成/重计算（采样、超分、修复、TTS） | 各插件自身核心节点 + KSampler/SamplerCustomAdvanced 等 |
| Intel XPU (8190 worker) | VAE 解码/编码加速（显存水位 auto 分流） | `BSAIVAEDecodeRouter` / `BSAIVAEEncodeRouter`（BSAI-ComfyUI-VAERouter） |
| CPU | 编排控制 + 硬件监控 | `BSAIOrchestratorGate` / `BSAIOrchestratorStatus`（Orchestrator）、`BSAIHardwarePerfMonitor`（Hardware-Perf） |

流水线：**NPU 前置检测/理解 → CUDA 主生成/处理 → XPU VAE 分流解码 → CPU 编排控制 → 输出**

## NPU 新能力演示链（v2，2026-10-03 注入）

每个工作流末尾新增一组 **NPU 深度估计 → NPU LLM 决策 → 文本输出** 的串联演示：

```
源图像 (LoadImage / 上下文帧) ──IMAGE──▶ BSAINPU_DepthEstimate ──meta_json──▶ BSAINPU_LLMGenerate ──text──▶ ShowText
```

- `BSAINPU_DepthEstimate`：Depth-Anything-V2-Small 单目深度估计（NPU，~0.65s/张），输出深度灰度图 + meta JSON
- `BSAINPU_LLMGenerate`：Qwen3-4B 对称 INT4（NPU），输入 meta_json（深度信息）→ 输出硬件协同决策文本（默认压制思考段，~17s/次）
- `ShowText`：决策文本展示

该演示链证明 NPU 在**生成链之外的纯计算卸载**能力：深度理解 + 小模型决策不再占用 CUDA/CPU。

## 工作流清单

| # | 插件名 | 示例目录路径（相对 custom_nodes） | 工作流文件名 | 核心节点 | 多硬件协同说明 |
|---|---|---|---|---|---|
| 1 | BSAI_ComfyUI_Sol-H3 | `BSAI-ComfyUI-Sol-H3\workflows\` | `BSAI_ComfyUI_SolH3_多硬件协同.json` | BSAI_SolH3_Loader、BSAI_SolH3_Info、BSAI_SolH3_LatentUpscaleAlign | NPU 前置人脸检测 → CUDA 跑 Sol-H3 一键加载 + KSampler 4 步 FastH3 极速采样 → XPU(8190) VAE 解码路由(auto 水位分流) → CPU 编排门透传 latent 并决策、状态/监控面板 |
| 2 | BSAI-NPU-Service | `BSAI-NPU-Service\example_workflows\`（新建） | `BSAI_NPU_Service_多硬件协同.json` | 无注册节点（纯 HTTP 服务）；以 H3FaceTrackCrop 消费其 NPU 能力 | NPU detect_face 检测跟踪 → CUDA 4K 超分 → XPU VAE 编解码路由往返 → CPU 编排门 + NPU 服务状态/监控面板 |
| 3 | BSAI-ComfyUI-VAERouter | `BSAI-ComfyUI-VAERouter\example_workflows\`（新建） | `BSAI_ComfyUI_VAERouter_多硬件协同.json` | BSAIVAEDecodeRouter、BSAIVAEEncodeRouter | NPU 前置检测 → CUDA 基础模型采样 → XPU(8190) DecodeRouter auto 分流解码 → CPU 编排门按显存水位决策 vae_target + 状态/监控 |
| 4 | BSAI_ComfyUI_SolarWM_H3 | `BSAI_ComfyUI_SolarWM_H3\workflows\` | `BSAI_ComfyUI_SolarWM_H3_多硬件协同.json` | BSAI_SolarWM_H3_Loader、BSAI_SolarWM_H3_CameraAttach、BSAI_SolarWM_H3_Generate | NPU 人脸检测 → CUDA 加载 FL2VA 基座 + CameraAttach 注入 PRoPE 环绕轨迹 + Generate 4 步蒸馏采样 → XPU VAE 解码 → CPU 编排/监控 |
| 5 | BSAI-ComfyUI-FastH3 | `BSAI-ComfyUI-FastH3\example_workflows\` | `BSAI_ComfyUI_FastH3_多硬件协同.json` | BSAIFastH3Loader、BSAIFastH3NativeVSA、BSAIFastH3Timesteps、BSAIFastH3EulerSampler、BSAIFastH3VSAStats | NPU 前置检测 → CUDA 加载 FastH3 蒸馏权重 + NativeVSA 稀疏注意力 + Timesteps/EulerSampler 配 BasicGuider + SamplerCustomAdvanced 采样 → XPU VAE 解码 → CPU 编排/监控 |
| 6 | BSAI-ComfyUI-MiniMax-H3-PDD-Acc | `BSAI-ComfyUI-MiniMax-H3-PDD-Acc\example_workflows\` | `BSAI_ComfyUI_MiniMaxH3_PDD_Acc_多硬件协同.json` | MiniMaxH3PDDAccApply、MiniMaxH3PDDAccScheduler、MiniMaxH3PDDAccWarmupScheduler、MiniMaxH3AVLatentUpscaleBy | NPU 人脸检测 → CUDA PDD-Acc LoRA+融合头直出块边界 SIGMAS、AVLatent 放大至 1344×768 后 SamplerCustomAdvanced 采样 → XPU VAE 解码 → CPU 编排/监控 |
| 7 | BSAI-ComfyUI-TaoMate | `BSAI-ComfyUI-TaoMate\example_workflows\` | `BSAI_ComfyUI_TaoMate_多硬件协同.json` | BSAITaoMateLoRALoader、BSAITaoMateTimesteps、BSAITaoMateEulerSampler、BSAITaoMateContinuationProbe、BSAITaoMateStreamChain | NPU 人脸检测 → CUDA 挂 TaoMate 3 步 LoRA 采样、ContinuationProbe 抽末帧/尾音频、StreamChain 续接长视频 → XPU VAE 解码 → CPU 编排/监控 |
| 8 | BSAI-ComfyUI-vdn-minimax-h3 | `BSAI-ComfyUI-vdn-minimax-h3\example_workflows\` | `BSAI_ComfyUI_vdn_minimax_h3_多硬件协同.json` | BSAIVDNH3Loader、BSAIVDNH3DualLora、BSAIVDNH3Timesteps、BSAIVDNH3EulerSampler、BSAIVDNH3Accel、BSAIVDNH3Upscale、BSAIVDNH3FaceRestore、BSAIVDNH3FaceOil、BSAIVDNH3EmptyLatentVideo | NPU 人脸检测 → CUDA VDN stage 加载 + 双 LoRA + Accel(BlockCache/CacheDiT/VSA) 加速采样，后接 4K 超分/小脸修复/去油 → XPU VAE 解码 → CPU 编排/监控 |
| 9 | BSAI-ComfyUI-H3-Film-Factory | `BSAI-ComfyUI-H3-Film-Factory\example_workflows\` | `BSAI_ComfyUI_H3_Film_Factory_多硬件协同.json` | BSAIH3FilmFactory、BSAIMiniMaxH3Extender、BSAIH3FilmFactoryFinalDecode、MiniMaxH3MotionContextRAM、MiniMaxH3MotionContextDiskJoin、MiniMaxH3MotionContextDiskFinalDecode、MiniMaxH3TailFromLatent、MiniMaxH3PromptPackBridge、BSAI_H3_3DLatentUpscale | NPU 人脸检测 → CUDA FilmFactory 主控分镜生成 + 3D latent 二采放大 + MotionContext RAM/Disk 跨镜续接、TailFromLatent 抽末帧 → XPU VAE 解码 → CPU 编排/监控 |
| 10 | BSAI_ComfyUI_IndexTTS-2.5 | `BSAI_ComfyUI_IndexTTS-2.5\workflow_example\` | `BSAI_ComfyUI_IndexTTS_2_5_多硬件协同.json` | BSAI_IndexTTS2.5Loader、BSAI_IndexTTS2.5LoadAudio、BSAI_IndexTTS2.5Synthesis、BSAI_IndexTTS2.5SaveAudio | NPU 对说话人参考肖像做人脸检测 → CUDA 跑 IndexTTS-2.5 语音合成并 SaveAudio → XPU VAE 编解码路由（伴随图像 latent 往返）→ CPU 编排登记显存水位 + 状态/监控 |
| 11 | BSAI_Premiere_Pro | `BSAI_Premiere_Pro\example_workflows\` | `BSAI_Premiere_Pro_多硬件协同.json` | BSAIPremiereProTimeline | NPU 检测输入帧人脸 → CUDA 时间轴后期（4K 超分合并出片，可选 image 输入收 crops）→ XPU VAE 路由对 crops 编解码往返预览 → CPU 编排/监控 |
| 12 | BSAI-H3-upscale-4K | `BSAI-H3-upscale-4K\workflows\` | `BSAI_H3_upscale_4K_多硬件协同.json` | BSAI_H3_Upscale4K、BSAI_H3_FaceRestore、BSAI_H3_DLSS5 | NPU 人脸检测 → CUDA RTX5090 4K 超分直出 + 送 XPU VAE 解码预览 → XPU(8190) VAE 路由 → CPU 编排门 + 监控 |
| 13 | BSAI-H3-MotionFix | `BSAI-H3-MotionFix\workflows\` | `BSAI_H3_MotionFix_多硬件协同.json` | BSAI_H3_MotionFix | NPU 检测源帧人脸 → CUDA 运动修复参数向导（输出步数/注意力/负向词建议）→ XPU VAE 路由对检测帧编解码往返预览 → CPU 编排/监控 |
| 14 | BSAI-ComfyUI-FaceRefine | `BSAI-ComfyUI-FaceRefine\examples\` | `BSAI_ComfyUI_FaceRefine_多硬件协同.json` | BSAIFaceRefine、H3FaceStitch、H3FaceTrackCrop | NPU 人脸检测（本插件 H3FaceTrackCrop）输出 crops+transform → CUDA 人脸 crop 高清修复 → H3FaceStitch 凭 transform 贴回原帧 → XPU VAE 路由对缝合结果编解码往返 → CPU 编排/监控 |
| 15 | BSAI_ComfyUI_Nodes | `BSAI_ComfyUI_Nodes\example_workflows\`（新建） | `BSAI_ComfyUI_Nodes_多硬件协同.json` | BSAI_DrawTextOverlay（另有 ImageSequenceToVideo、MergeImages、QwenNodes 等 20+ 节点可选） | NPU 检测帧人脸 → CUDA 字幕叠加通用图像工具直出 + 送 XPU VAE 解码预览 → XPU(8190) VAE 路由 → CPU 编排/监控 |
| 16 | BSAI-ComfyUI-Orchestrator | `BSAI-ComfyUI-Orchestrator\example_workflows\`（新建） | `BSAI_ComfyUI_Orchestrator_多硬件协同.json` | BSAIOrchestratorGate、BSAIOrchestratorStatus | CPU 编排门（登记任务/取租约/按水位决策 VAE 分流）+ 状态面板为核心；NPU→CUDA→XPU 全链可视化承载 |
| 17 | BSAI-ComfyUI-Hardware-Perf | `BSAI-ComfyUI-Hardware-Perf\example_workflows\`（新建） | `BSAI_ComfyUI_Hardware_Perf_多硬件协同.json` | BSAIHardwarePerfMonitor | 监控面板独立轮询展示 RTX5090/XPU/NPU/CPU 实时水位为核心；同链 NPU→CUDA→XPU→CPU 编排全程可视 |
| 18 | BSAI-MiniMAX-H3-Prompt | `BSAI-MiniMAX-H3-Prompt\example_workflows\`（新建） | `BSAI_MiniMAX_H3_Prompt_多硬件协同.json` | BSAI_H3_ModelLoader、"BSAI_MiniMAX H3 prompt"、BSAI_H3_DirectPrompt | 本地 LLM 产出 STRING 提示词直供正向 CLIPTextEncode.text → NPU 前置检测参考帧 → CUDA 采样 → XPU 解码 → CPU 编排门 + 监控 |
| 19 | BSAI-Qwen-Prompt-Enhancer | `BSAI-Qwen-Prompt-Enhancer\examples\` | `BSAI_Qwen_Prompt_Enhancer_多硬件协同.json` | BSAI_Qwen_Prompt_Enhancer、BSAI_Jev_Schema、BSAI_Jev_Decision、MarkdownNote、QwenImage21_Prompt_Template | 提示词增强器输出喂正向 CLIPTextEncode、Jev Schema/Decision 结构化决策演示 → NPU 前置检测 → CUDA 采样 → XPU 解码 → CPU 编排/监控 |
| 20 | BSAI-ComfyUI_Contextual-Series | `BSAI-ComfyUI_Contextual-Series\example_workflows\` | `BSAI_ComfyUI_Contextual_Series_多硬件协同.json` | BSAI_ContextualSeriesLoad、BSAI_AssetLibraryInput、BSAI_ImageBatchSplitter | 上下文帧加载器 images 直供 NPU 人脸检测输入源 → CUDA 独立文生图采样链 → XPU 解码 → CPU 编排门 + 监控 |
| 21 | BSAI-Asset-Library-Auto-List | `BSAI-Asset-Library-Auto-List\example_workflows\` | `BSAI_Asset_Library_Auto_List_多硬件协同.json` | BSAI_AssetLibraryAutoList、BSAI_AssetLibraryAutoListByType | 资产库自动列表面板（输出 STRING/INT）→ NPU 前置检测 → CUDA 主采样 → XPU VAE 解码 → CPU 编排/监控 |
| 22 | BSAI-ComfyUI-AimDo-Fix | `BSAI-ComfyUI-AimDo-Fix\example_workflows\`（新建） | `BSAI_ComfyUI_AimDo_Fix_多硬件协同.json` | 无注册节点（纯运行时 shim：启动即修复 aimdo 内存编译崩溃） | 修复生效后的标准多硬件生成链：NPU→CUDA 稳定采样（H3/Qwen Image 2.1）→XPU VAE 解码→CPU 编排/监控，标题注明修复作用 |
| 23 | BSAI-ComfyUI-VRAM-Unlock | `BSAI-ComfyUI-VRAM-Unlock\example_workflows\`（新建） | `BSAI_ComfyUI_VRAM_Unlock_多硬件协同.json` | BSAIVRAMUnlockInfo | 显存解锁信息面板（最左列）→ NPU 检测 → CUDA 满载主采样 → XPU VAE 分流解码 → CPU 编排 + 监控实时显存水位 |
| 24 | BSAI-ComfyUI-VedaSparse | `BSAI-ComfyUI-VedaSparse\example_workflows\` | `BSAI_ComfyUI_VedaSparse_多硬件协同.json` | BSAIVedaSparsePatch、BSAIVedaSparseStats | NPU 人脸检测 → CUDA 在稀疏注意力 patch 加速（Keep 5%/dual-fast 分块）下采样 → XPU VAE 解码 → CPU 编排 + 监控 + 稀疏命中统计面板 |

## 校验结论

- 24/24 个文件通过 `python_embeded\python.exe` 校验：`json.load` 解析成功；所有 links 源/目标节点存在、槽位不越界、连线类型与源输出/目标输入一致；`inputs[].link` 与 `outputs[].links` 双向闭合；`last_node_id`/`last_link_id` 一致。
- 每个工作流 v2 均含：NPU 检测节点（`H3FaceTrackCrop`）+ **NPU 新能力演示链（`BSAINPU_DepthEstimate` + `BSAINPU_LLMGenerate` + `ShowText`）** + 插件自身核心节点 + XPU VAE 路由节点（`BSAIVAEDecodeRouter`/`BSAIVAEEncodeRouter`）+ CPU 编排节点（`BSAIOrchestratorGate`/`BSAIOrchestratorStatus`）+ 监控节点（`BSAIHardwarePerfMonitor`）。
- 每个工作流的节点 `type` 均经源码核实：插件自身节点取自对应插件 `NODE_CLASS_MAPPINGS`，协同节点取自 BSAI-NPU-Service / FaceRefine / VAERouter / Orchestrator / Hardware-Perf 源码，内置节点匹配 ComfyUI core + comfy_extras 源码类（含新版 `io.ComfyNode` API），无臆造类名。
- 全部文件仅写入 G 盘整合包内插件示例目录，未写入 C 盘、未推送 GitHub（按约定待验收后再同步）。
