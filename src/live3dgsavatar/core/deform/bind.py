"""绑定：把切空间高斯搬到世界空间。

    xyz_world  = R @ xyz_tangent + Σ_i bary_i · v_i
    rot_world  = q(R) ⊗ rot_tangent

其中 `R = TBN[face_id]`，**列 j 是第 j 个基向量**（见 `tbn.py`），
`bary = binding_face_bary`。

## 为什么是 `R` 而不是 `Rᵀ`（一个曾反复踩的坑）

`mesh_binding` 是**逐元素**的，`R` 是否正交只影响「用什么算子把切空间坐标
变到世界空间」，不影响「哪个算子对」。判据只能是**与参照一致**，而不是自证。

**证据链**（两条独立证据互相印证）：

1. 本项目等价门曾在位置项报差 **9.87e-02** —— 当时本项目用 `Rᵀ`，
   而参照 `gaussian_deform_batch` 产出 `R·x`（`binding_rotations @ gs.xyz`）：
   `‖Rᵀ − R‖` 在非正交 `R` 下正是该量级；
2. 改为 `R·x` 后，`scripts/render_test.py` 中 core 与参照的 254 帧渲染
   **PSNR 中位 101 dB、max|Δ| = 1/255** —— 逐位一致。

**易混淆点**：CUDA `face_tbn.cu` 内部 `TBNs[idx] = transpose(mat3(t,b,n))`
（**行**为基），而 `cuda_utils` 之外、Python 侧的 `utils.compute_face_tbn` 与
本项目 `tbn.py` 都是**列**为基。两者互为转置，切勿跨来源比对。
`mesh_binding` 接收的是**调用方传入**的 `face_tbns`，因此其内部那次 `transpose`
是针对「传入布局」的修正。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..types import GaussianSet, Mesh
from .tbn import compute_face_tbn


def quaternion_multiply(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """四元数乘法（WXYZ）。形状按 batch 广播，等价于 `utils.quaternion_multiply`。"""
    vector = (
        p[..., None, 0] * q[..., 1:]
        + q[..., None, 0] * p[..., 1:]
        + torch.cross(p[..., 1:], q[..., 1:], dim=-1)
    )
    last = p[..., 0] * q[..., 0] - torch.sum(p[..., 1:] * q[..., 1:], dim=-1)
    return torch.cat((last[..., None], vector), dim=-1)


def matrix_to_quaternion(m: torch.Tensor) -> torch.Tensor:
    """旋转矩阵 → 单位四元数（WXYZ）。

    与 CUDA 的 `matrix_to_quaternion`、Python 的 `utils.matrix_to_quaternion` 同算法
    （Shepperd 方法，按迹/对角元选分支）。
    """
    if m.shape[-2:] != (3, 3):
        raise ValueError(f"期望 [..., 3, 3]，实际 {tuple(m.shape)}")
    shape = m.shape[:-2]
    flat = m.reshape(-1, 3, 3)
    n = flat.shape[0]
    dtype, device = m.dtype, m.device

    diag = torch.stack([flat[:, 0, 0], flat[:, 1, 1], flat[:, 2, 2]], dim=1)   # [n, 3]
    trace = diag.sum(dim=1, keepdim=True)                                      # [n, 1]
    decision = torch.cat([diag, trace], dim=1)                                 # [n, 4]
    choices = decision.argmax(dim=1)                                           # [n]

    quat = torch.empty(n, 4, dtype=dtype, device=device)
    idx1 = torch.nonzero(choices != 3, as_tuple=True)[0]
    idx2 = torch.nonzero(choices == 3, as_tuple=True)[0]

    if idx1.numel():
        i = choices[idx1]                                    # [m]
        j = (i + 1) % 3
        k = (j + 1) % 3
        rows = torch.arange(i.numel(), device=device)
        f = flat[idx1]                                       # [m, 3, 3]
        quat[idx1, i + 1] = 1 - trace[idx1, 0] + 2 * f[rows, i, i]
        quat[idx1, j + 1] = f[rows, j, i] + f[rows, i, j]
        quat[idx1, k + 1] = f[rows, k, i] + f[rows, i, k]
        quat[idx1, 0] = f[rows, k, j] - f[rows, j, k]

    if idx2.numel():
        f = flat[idx2]
        quat[idx2, 1] = f[:, 2, 1] - f[:, 1, 2]
        quat[idx2, 2] = f[:, 0, 2] - f[:, 2, 0]
        quat[idx2, 3] = f[:, 1, 0] - f[:, 0, 1]
        quat[idx2, 0] = 1 + trace[idx2, 0]

    quat = torch.nn.functional.normalize(quat, dim=1)
    return quat.reshape(*shape, 4)


@dataclass
class Binding:
    """高斯与模板三角面的绑定关系（由 UV 光栅化一次性确定，不参与训练）。

    Attributes:
        face_id:   [N] int64 每个高斯绑定的三角面下标（0-based）
        face_bary: [N, 3] 该面内的重心坐标 `(u, v, 1-u-v)`
        valid_mask: [H*W] bool UV 图上哪些 texel 绑定了高斯
    """

    face_id: torch.Tensor
    face_bary: torch.Tensor
    valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.face_id.ndim != 1:
            raise ValueError(f"face_id 形状应为 [N]，实际 {tuple(self.face_id.shape)}")
        if self.face_bary.shape != (self.face_id.shape[0], 3):
            raise ValueError(
                f"face_bary 形状应为 [N, 3]，实际 {tuple(self.face_bary.shape)}"
                f"（N = {self.face_id.shape[0]}）")
        if self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask 应为 bool")

    @property
    def num_gaussians(self) -> int:
        return self.face_id.shape[0]

    def to(self, device: torch.device | str) -> "Binding":
        return Binding(self.face_id.to(device), self.face_bary.to(device),
                       self.valid_mask.to(device))


class MeshBinder:
    """`Binder` 的实现：网格 + 绑定 → 世界空间高斯。

    对应 CUDA 的 `mesh_binding`。无 batch 维之外的隐藏状态，可复用。
    """

    def __init__(self, binding: Binding) -> None:
        self.binding = binding

    def bind(self, gaussians: GaussianSet, mesh: Mesh) -> GaussianSet:
        if gaussians.space != "tangent":
            raise ValueError(
                f"MeshBinder 的输入必须是切空间高斯，实际 space={gaussians.space!r}")
        if gaussians.batch_size != mesh.batch_size:
            raise ValueError(
                f"batch 不一致：高斯 {gaussians.batch_size}，网格 {mesh.batch_size}")

        face_id = self.binding.face_id
        bary = self.binding.face_bary.to(dtype=mesh.verts.dtype, device=mesh.verts.device)

        # 索引合法性：**负索引会被 PyTorch 当作「从末尾数」而静默取到错误的三角面**，
        # 越界索引同样不会报错。必须在 gather 之前拦截。
        n_faces = mesh.num_faces
        if face_id.numel():
            lo, hi = int(face_id.min()), int(face_id.max())
            if lo < 0 or hi >= n_faces:
                raise ValueError(
                    f"binding.face_id 超出网格面数范围 [{lo}, {hi}]，"
                    f"合法区间为 [0, {n_faces - 1}]。"
                    "（负值会被当作从末尾索引，越界值会静默给出错误结果）")

        tri_verts = mesh.verts[:, mesh.faces]                       # [B, F, 3, 3]
        face_tbn = compute_face_tbn(tri_verts, mesh.uvs[mesh.uv_faces])   # [B, F, 3, 3]

        binding_rot = face_tbn[:, face_id]                          # [B, N, 3, 3]
        binding_tri = tri_verts[:, face_id]                         # [B, N, 3, 3]
        offset = (binding_tri * bary.unsqueeze(0).unsqueeze(-1)).sum(dim=-2)  # [B, N, 3]

        # ⚠️ 用 R（不是 Rᵀ）：与参照 `gaussian_deform_batch` 一致。
        #    判据是"与参照逐位一致"，不是自证；两条独立证据见模块 docstring。
        xyz = (binding_rot @ gaussians.xyz.unsqueeze(-1)).squeeze(-1) + offset

        rot = quaternion_multiply(matrix_to_quaternion(binding_rot), gaussians.rotation)

        return GaussianSet(
            xyz=xyz,
            rotation=rot,
            scaling=gaussians.scaling,
            opacity=gaussians.opacity,
            color=gaussians.color,
            space="world",
        )
