"""绑定：PLY 路径 与 build_binding 路径 必须一致（**不需要 GPU**）。

**这是等价门的一个盲区**：等价门比对的是
    `avatar.binding_face_id`（来自 PLY）  vs  参照 `binding_face_id`
两者都源自同一份 UV 光栅化结果，因此即使 `build_binding` 本身有 bug，
只要 PLY 里的绑定是对的、而等价门用的是 PLY 的绑定，门禁就不会发现。

本测试用**假的 nvdiffrast 上下文**喂入已知的 `face_id` / `face_uv`，
检查 `build_binding` 的约定：

- `face_id == 0` 表示「无三角形覆盖」→ `valid_mask` 必须排除它；
- 有效 texel 的 `face_id` 必须**减去 1** 变成 0-based（参照实现的做法）；
- `bary = [u, v, 1-u-v]`，且第三分量由前两者导出。
"""

from __future__ import annotations

import sys
import types

import torch

ATOL = 1e-6


class _FakeGLContext:
    """占位上下文；配合下面的假 `nvdiffrast.torch` 使用。"""


def _install_fake_nvdiffrast(face_uv: torch.Tensor, face_id: torch.Tensor):
    """注入一个假的 `nvdiffrast.torch.rasterize`。

    返回 `(H, W)` 对应的 rast_out（形状 `[1, H, W, 4]`），
    其中前两通道为 face_uv、第四通道为 face_id。
    """
    h, w = face_id.shape
    rast = torch.zeros(1, h, w, 4)
    rast[0, ..., :2] = face_uv
    rast[0, ..., 3] = face_id.reshape(h, w)

    module = types.ModuleType("nvdiffrast")
    torch_mod = types.ModuleType("nvdiffrast.torch")

    def rasterize(ctx, verts, faces, resolution=None):
        return rast, None

    torch_mod.rasterize = rasterize
    module.torch = torch_mod

    sys.modules["nvdiffrast"] = module
    sys.modules["nvdiffrast.torch"] = torch_mod


def test_build_binding_face_id_is_zero_based_and_masks_empty() -> None:
    """`face_id == 0` 必须被排除，有效值必须减 1。"""
    from live3dgsavatar.core.deform.binding import compute_uv_rast_info

    # 4 个 texel：0 表示无覆盖，2/3/5 是 1-based 的面号
    face_id = torch.tensor([[0.0, 2.0], [3.0, 5.0]])
    face_uv = torch.tensor([[[0.1, 0.2], [0.25, 0.25]],
                            [[0.5, 0.1], [0.0, 0.0]]])

    _install_fake_nvdiffrast(face_uv, face_id)
    try:
        uv, fid = compute_uv_rast_info(
            uvs=torch.zeros(4, 2), uv_faces=torch.zeros(1, 3, dtype=torch.int32),
            size=(2, 2), glctx=_FakeGLContext())
    finally:
        sys.modules.pop("nvdiffrast", None)
        sys.modules.pop("nvdiffrast.torch", None)

    # compute_uv_rast_info 返回 [H, W, ·]（与参照 compute_rast_info 的语义一致），
    # reshape 成 [H*W, ·] 由 build_binding 负责
    assert uv.shape == (2, 2, 2), f"face_uv 形状应为 [H, W, 2]，实际 {tuple(uv.shape)}"
    assert fid.shape == (2, 2, 1), f"face_id 形状应为 [H, W, 1]，实际 {tuple(fid.shape)}"
    assert torch.allclose(fid.reshape(-1), face_id.reshape(-1))


def test_build_binding_bary_third_component_is_derived() -> None:
    """`bary = [u, v, 1-u-v]`，第三分量必须由前两者导出（不是 uv 的第三列）。"""
    from live3dgsavatar.core.deform.binding import build_binding

    face_id = torch.tensor([[1.0, 1.0], [1.0, 0.0]])          # 最后一个 texel 无覆盖
    face_uv = torch.tensor([[[0.30, 0.20], [0.10, 0.40]],
                            [[0.25, 0.25], [0.0, 0.0]]])

    _install_fake_nvdiffrast(face_uv, face_id)
    try:
        b = build_binding(uvs=torch.zeros(4, 2),
                          uv_faces=torch.zeros(1, 3, dtype=torch.int32),
                          tex_size=2, glctx=_FakeGLContext(), device="cpu")
    finally:
        sys.modules.pop("nvdiffrast", None)
        sys.modules.pop("nvdiffrast.torch", None)

    assert b.num_gaussians == 3, f"应保留 3 个有效 texel，实际 {b.num_gaussians}"
    assert torch.equal(b.face_id, torch.zeros(3, dtype=torch.long)), \
        "1-based 的面号 1 必须转成 0-based 的 0"

    # 重心：前两列取自 face_uv，第三列为 1-u-v
    expected_uv = torch.tensor([[0.30, 0.20], [0.10, 0.40], [0.25, 0.25]])
    assert torch.allclose(b.face_bary[:, :2], expected_uv, atol=ATOL)
    assert torch.allclose(b.face_bary[:, 2], 1.0 - expected_uv.sum(dim=-1), atol=ATOL)
    assert torch.allclose(b.face_bary.sum(dim=-1), torch.ones(3), atol=ATOL), \
        "重心坐标必须和为 1"

    # 掩码必须为 bool 且与有效 texel 对齐
    assert b.valid_mask.dtype == torch.bool
    assert int(b.valid_mask.sum()) == 3


def test_binding_valid_mask_layout_is_row_major() -> None:
    """`valid_mask` 必须是行主序展开的 `[H*W]`，与 `extract_texture` 的 reshape 一致。

    参照实现用 `face_id.reshape(-1)` 得到掩码，再 `reshape(tex, tex, 3)` 铺回纹理。
    若这里改成列主序，纹理图会旋转/镜像，但**高斯数量不变**，
    因此等价门不会报错 —— 属于静默错误。
    """
    from live3dgsavatar.core.deform.binding import build_binding

    # 2x2：只有右上角有覆盖 -> 行主序下下标应为 1
    face_id = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    face_uv = torch.tensor([[[0.0, 0.0], [0.2, 0.3]],
                            [[0.0, 0.0], [0.0, 0.0]]])

    _install_fake_nvdiffrast(face_uv, face_id)
    try:
        b = build_binding(uvs=torch.zeros(4, 2),
                          uv_faces=torch.zeros(1, 3, dtype=torch.int32),
                          tex_size=2, glctx=_FakeGLContext(), device="cpu")
    finally:
        sys.modules.pop("nvdiffrast", None)
        sys.modules.pop("nvdiffrast.torch", None)

    idx = torch.nonzero(b.valid_mask).reshape(-1)
    assert idx.tolist() == [1], f"行主序下有效 texel 下标应为 1，实际 {idx.tolist()}"
