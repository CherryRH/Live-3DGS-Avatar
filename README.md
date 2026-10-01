# Live3DGSAvatar

**面向视频通话场景的 3DGS 人体重建 —— 实时、可驱动、可持续更新的头部数字人**

以 [RGBAvatar](https://github.com/gapszju/RGBAvatar)（CVPR 2025 Highlight）的约简高斯 blendshape 方案为起点，
本项目正在构建一套完整可用的工程实现：从单目视频重建头部 avatar，实时渲染，并在通话过程中持续更新。

> 当前进度：**P0 环境与基线**、**P1 渲染内核重写** 已完成并验收。
> 渲染管线与参照实现**逐位一致**（254 帧 PSNR 中位 101 dB）。
> 预训练由合作方工作提供，本项目**不自研训练**，仅预留接入点。

---

## 特性

| 能力 | 说明 | 状态 |
|---|---|---|
| **实时渲染** | 给定驱动参数出图；渲染管线与参照实现逐位一致（89 FPS @ 512²） | ✅ 已完成 |
| **模型接入** | 读取合作方产出的预训练模型并驱动渲染 | 待接入 |
| **动态更新** | 流式到达的新帧可持续优化模型 | 待定 |
| **服务端渲染** | 渲染在服务端完成，客户端只需接收视频流 | P3 |
| **图形化程序** | 实时预览、模型加载与导出 | P2 |

## 技术要点

- **表示**：约简高斯 blendshape —— 用轻量 MLP 把 3DMM 参数映射为 20 维权重，线性混合一组可学习高斯基，
  在保持表情细节的同时把模型体积压到可实时渲染的量级（约 6 万高斯）。
- **绑定**：高斯参数存于模板网格的切空间，通过三角面 TBN 与重心插值跟随网格变形，
  因此任意表情与姿态都可直接驱动，无需重新训练。
- **训练加速**：批并行高斯光栅化（两阶段 + 多 CUDA 流）配合颜色初始化，
  大幅降低 GPU-CPU 同步开销。*（训练由合作方提供，本项目只消费模型）*
- **在线重建**：local-global 双采样池平衡"快速适应新帧"与"不遗忘历史帧"。

## 环境要求

- Linux（WSL2 可用）
- NVIDIA GPU（CUDA 13.0）
- Python 3.10
- PyTorch 2.14.0+cu130

## 安装

一键脚本会创建独立 conda 环境并编译本地 CUDA 扩展：

```bash
git clone https://github.com/CherryRH/Live-3DGS-Avatar && cd Live3DGSAvatar
bash scripts/setup_env.sh
conda activate live3dgs
```

环境自检：

```bash
python scripts/env_check.py
```

> 本地 CUDA 扩展（高斯光栅化 / nvdiffrast / fused-ssim）均已随仓库提供，
> **无需额外 clone 或联网获取源码**。详见 `docs/ENVIRONMENT.md`。

## 数据准备

需要 FLAME 2020 模型与跟踪器输出的数据集：

```
data/FLAME2020/generic_model.pkl     # FLAME 模板（需自行从官网获取并遵守其许可）
<DATA_DIR>/<SUBJECT>/checkpoint/     # 每帧 3DMM 参数与相机
<DATA_DIR>/<SUBJECT>/images/         # 每帧图像
```

## 使用

```bash
# 查看当前生效的配置（路径是否指对，先看这个）
python scripts/show_config.py

# 统一渲染测试：core vs 参照实现 vs 数据集原图（PSNR / FPS / 显存）
python scripts/render_test.py --frames 20

# 全量 + 并排对比图（两边同批大小）
python scripts/render_test.py --frames -1 --batch-size 4 --dump-diff

# 量化批大小本身的影响（参照内部是串行循环，理论上每帧成本不变）
python scripts/render_test.py --frames 20 --batch-size 4 --sweep-batch 1 4 10

# 只测 core（跳过参照实现，更快）
python scripts/render_test.py --frames 20 --skip-reference

# 数值等价验收门（与参照逐层比对）
python scripts/equivalence_check.py \
    --ply <PATH>/model.ply --data <DATA_DIR> --frames 3
```

> GUI 的命令将在 P2 完成后启用；训练由合作方提供（见 `docs/ARCHITECTURE.md` §6）。

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 架构设计：分层、核心接口、验收门与 P1 验收证据 |
| [`docs/CORE_GUIDE.md`](docs/CORE_GUIDE.md) | **代码导览**：形状流转、设计原因、陷阱清单 |
| [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md) | 坐标 / 矩阵 / 空间 / 精度约定（数值问题的权威来源） |
| [`docs/MIGRATION.md`](docs/MIGRATION.md) | 与参照实现的全部差异登记 |
| [`docs/CONFIG.md`](docs/CONFIG.md) | **配置说明**：路径、渲染参数、优先级、如何改 |
| [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) | 环境搭建、版本矩阵、排错 |

## 目录

```
src/           项目源码（core/ 已完成；data/training/runtime 为后续阶段）
scripts/       环境搭建、自检、等价门、渲染出图
submodules/    本地 CUDA 扩展与第三方源码（vendored）
tests/         单元测试（67 项，无需 GPU）与数值等价门
docs/          架构 / 导览 / 约定 / 差异 / 环境
data/          输入资产（不入库）
output/        渲染产物（不入库）
configs/       系统配置与渲染配置（见 docs/CONFIG.md）
```

## 测试

```bash
python tests/run_tests.py          # 零依赖，未安装 pytest 也能运行
```

## 许可

本项目为学术研究用途。`submodules/` 下的第三方源码受各自许可约束，
详见 `submodules/ATTRIBUTION.md`。
