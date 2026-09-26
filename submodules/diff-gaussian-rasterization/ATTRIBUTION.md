# 第三方源码归属说明

本目录下的 `diff_gaussian_rasterization/` 是**外部源码的 vendored 副本**，不是本项目原创代码。

## 来源

| 项 | 值 |
|---|---|
| 上游仓库 | https://github.com/graphdeco-inria/diff-gaussian-rasterization （原始 3DGS 光栅化器，Inria / GRAPHDECO） |
| 取得途径 | 经由 https://github.com/gapszju/RGBAvatar 的 `submodules/diff-gaussian-rasterization` |
| 源仓库 commit | `903653b689a8b860df30f165e242ccd6ed3f5583`（RGBAvatar，2026-04-24） |
| 拷贝日期 | 见 git 首次提交记录 |
| 许可 | 见 `LICENSE.md`（非商业研究/评估用途） |

## 与上游的差异

RGBAvatar 相对原始 3DGS 光栅化器做了以下扩展（**这些差异已包含在本副本中**）：

| 文件 | 差异 |
|---|---|
| `cuda_rasterizer/fast_forward.cu` | 新增。按累计权重阈值一次性写入高斯颜色 |
| `cuda_rasterizer/forward.cu` | 新增 `est_color` / `est_weight` / `target_image` 输出 |
| `cuda_utils/linear_blending.cu` | 新增。blendshape 线性混合（前向 + 反向） |
| `cuda_utils/mesh_binding.cu` | 新增。网格绑定（TBN 旋转 + 重心插值平移） |
| `cuda_utils/face_tbn.cu` | 新增。逐三角面 TBN |
| `cuda_utils/rotation.cu` | 新增。矩阵↔四元数、四元数乘法 |
| `rasterize_points.cu` | 新增 `batch_rasterize_gaussians1/2` 两阶段批并行入口 |
| `diff_gaussian_rasterization/__init__.py` | 新增 `BatchGaussianRasterizer` 与 `BatchRasterizationTensors` |

## 本副本的删减

仅删除了与构建/运行无关的内容，未改动任何源码：

- `build/`（上游构建产物）
- `*.egg-info/`
- `third_party/glm/doc/`（17 MB Doxygen 文档）
- `third_party/glm/test/`（GLM 自身的测试）

保留 `third_party/glm/glm/`（编译必需的头文件）与 `third_party/stbi_image_write.h`。

## 修改约定

本项目对本目录的任何修改都必须在 `docs/MIGRATION.md` 中登记，并注明动机。
未登记的修改视为违规。
