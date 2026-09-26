# Live3DGSAvatar

**面向视频通话场景的 3DGS 人体重建 —— 实时、可驱动、可持续更新的头部数字人**

以 [RGBAvatar](https://github.com/gapszju/RGBAvatar)（CVPR 2025 Highlight）的约简高斯 blendshape 方案为起点，
本项目正在构建一套完整可用的工程实现：从单目视频重建头部 avatar，实时渲染，并在通话过程中持续更新。

> 当前进度：**P0 环境与基线** 已完成；核心渲染与训练模块正在重写中。

---

## 特性

| 能力 | 说明 | 状态 |
|---|---|---|
| **静态训练** | 输入单目视频序列，重建可驱动的头部高斯 avatar | 重写中 |
| **实时渲染** | 给定驱动参数实时出图，支持任意视角与背景合成 | 重写中 |
| **动态更新** | 流式到达的新帧可持续优化模型，抗遗忘采样 | 规划中 |
| **服务端渲染** | 渲染在服务端完成，客户端只需接收视频流 | 规划中 |
| **图形化程序** | 训练监控、实时预览、模型导出 | 规划中 |

## 技术要点

- **表示**：约简高斯 blendshape —— 用轻量 MLP 把 3DMM 参数映射为 20 维权重，线性混合一组可学习高斯基，
  在保持表情细节的同时把模型体积压到可实时渲染的量级（约 6 万高斯）。
- **绑定**：高斯参数存于模板网格的切空间，通过三角面 TBN 与重心插值跟随网格变形，
  因此任意表情与姿态都可直接驱动，无需重新训练。
- **训练加速**：批并行高斯光栅化（两阶段 + 多 CUDA 流）配合颜色初始化，
  大幅降低 GPU-CPU 同步开销。
- **在线重建**：local-global 双采样池平衡"快速适应新帧"与"不遗忘历史帧"。

## 环境要求

- Linux（WSL2 可用）
- NVIDIA GPU（CUDA 13.0）
- Python 3.10
- PyTorch 2.14.0+cu130

## 安装

一键脚本会创建独立 conda 环境并编译本地 CUDA 扩展：

```bash
git clone <this-repo> && cd Live3DGSAvatar
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
# 训练
python -m live3dgsavatar.app.cli train --subject <SUBJECT> --work-name <NAME>

# 渲染
python -m live3dgsavatar.app.cli render --subject <SUBJECT> --work-name <NAME>
```

> 命令将于核心模块重写完成后启用。

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 架构设计：分层、核心接口、迁移映射、验收门 |
| [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) | 环境搭建、版本矩阵、排错 |

## 目录

```
src/           项目源码
scripts/       环境搭建脚本与自检工具
submodules/    本地 CUDA 扩展与第三方源码（vendored）
docs/          架构与环境文档
configs/       训练配置
tests/         单元测试与数值等价测试
data/          输入资产（不入库）
output/        训练与渲染产物（不入库）
```

## 许可

本项目为学术研究用途。`submodules/` 下的第三方源码受各自许可约束，
详见 `submodules/ATTRIBUTION.md`。
