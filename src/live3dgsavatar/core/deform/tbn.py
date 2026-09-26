"""逐三角面 TBN（切空间正交基）。

对应参照实现的 `submodules/diff-gaussian-rasterization/cuda_utils/face_tbn.cu`
与 `utils.compute_face_tbn`。**两者数值等价、但矩阵布局互为转置**：

- CUDA 版 `TBNs[idx] = transpose(mat3(tangent, bitangent, normal))`（glm 列主序）
- Python 版 `tbn = stack([tangent, bitangent, normal], dim=-1)`

本模块提供**两种返回形态**并由调用方显式选择，避免再次出现「靠约定隐含布局」的问题：

- :func:`compute_face_tbn`         → [B, F, 3, 3]，**列为** TBN（等价 CUDA 内核的输出）
- :func:`compute_face_tbn_column_stacked` → 同上（语义化别名，供 Binder 使用）
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_face_tbn(
    face_vertices: torch.Tensor,   # [B, F, 3, 3]
    face_uvs: torch.Tensor,        # [F, 3, 2] （或可广播的 [B, F, 3, 2]）
) -> torch.Tensor:
    """逐面 TBN。

    Args:
        face_vertices: [B, F, 3, 3] 每个三角面的三个顶点
        face_uvs:      [F, 3, 2] 每个三角面的三个 UV

    Returns:
        [B, F, 3, 3]，**第 j 列是第 j 个基向量** `(tangent, bitangent, normal)`。
        与 CUDA 内核 `compute_face_tbn` 的输出一致。

    公式（见 docs/CONVENTIONS.md §3.1）::

        e1, e2   = v1 - v0, v2 - v0
        d1, d2   = uv1 - uv0, uv2 - uv0
        f        = 1 / (d1.x·d2.y - d2.x·d1.y)
        tangent   = (e1·d2.y - e2·d1.y) · f
        bitangent = (e2·d1.x - e1·d2.x) · f
        normal    = cross(e1, e2)
    """
    if face_vertices.ndim != 4 or face_vertices.shape[-2:] != (3, 3):
        raise ValueError(f"face_vertices 形状应为 [B, F, 3, 3]，实际 {tuple(face_vertices.shape)}")
    if face_uvs.ndim != 3 or face_uvs.shape[-2:] != (3, 2):
        raise ValueError(f"face_uvs 形状应为 [F, 3, 2]，实际 {tuple(face_uvs.shape)}")

    face_uvs = face_uvs.to(dtype=face_vertices.dtype, device=face_vertices.device)

    v0, v1, v2 = face_vertices.unbind(-2)          # 各 [B, F, 3]
    uv0, uv1, uv2 = face_uvs.unbind(-2)            # 各 [F, 2]

    edge1 = v1 - v0
    edge2 = v2 - v0
    duv1 = uv1 - uv0
    duv2 = uv2 - uv0

    denom = duv1[..., 0] * duv2[..., 1] - duv2[..., 0] * duv1[..., 1]
    f = 1.0 / denom                                 # [F]
    f = f.unsqueeze(-1).unsqueeze(0)                # [1, F, 1] 供广播

    tangent = f * (duv2[..., 1].unsqueeze(-1) * edge1 - duv1[..., 1].unsqueeze(-1) * edge2)
    bitangent = f * (-duv2[..., 0].unsqueeze(-1) * edge1 + duv1[..., 0].unsqueeze(-1) * edge2)
    normal = torch.cross(edge1, edge2, dim=-1)

    tbn = torch.stack([tangent, bitangent, normal], dim=-1)   # [B, F, 3, 3]，列为基
    return F.normalize(tbn, dim=-2)                            # 逐列归一化


def compute_face_tbn_column_stacked(
    face_vertices: torch.Tensor,
    face_uvs: torch.Tensor,
) -> torch.Tensor:
    """语义化别名：明确「列为基向量」这一布局约定，供 `Binder` 使用。"""
    return compute_face_tbn(face_vertices, face_uvs)


def compute_face_normal(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """逐面法线 [B, F, 3]（仅用于可视化，不参与绑定）。"""
    faces = faces.to(torch.int64)
    v0 = vertices[..., faces[..., 0], :]
    v1 = vertices[..., faces[..., 1], :]
    v2 = vertices[..., faces[..., 2], :]
    return F.normalize(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)
