"""从 UV 图构建高斯与模板三角面的绑定关系。

对应参照实现的 `BindingModel.binding()`：用 nvdiffrast 在 UV 域做一次光栅化，
把 `tex_size × tex_size` 的每个有效 texel 变成一个高斯。

    face_uv, face_id = compute_rast_info(uvs, uv_faces, (tex_size, tex_size))
    valid   = face_id > 0                     # 0 表示该 texel 没落到三角形上
    face_id = face_id[valid] - 1              # 参照实现转成 0-based
    bary    = [face_uv[valid], 1 - uv.sum(-1)]

**一个有效 texel = 一个高斯**，参数与 UV 图逐像素对齐 —— 这是 RGBAvatar 能让纹理图
可视化/可编辑的原因，也是 `num_gaussians` 由 `tex_size` 决定的根源。
"""

from __future__ import annotations

import torch

from .bind import Binding


def compute_uv_rast_info(
    uvs: torch.Tensor,        # [Vuv, 2] in [0, 1]
    uv_faces: torch.Tensor,   # [F, 3] int32
    size: tuple[int, int],
    glctx,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在 UV 域光栅化，返回 `(face_uv [H*W, 2], face_id [H*W, 1])`。

    等价于参照实现的 `diff_renderer.texture.compute_rast_info`。

    把 UV 从 `[0,1]` 映射到裁剪空间 `[-1,1]`，并补 `z=0, w=1`，
    然后用 `dr.rasterize` 光栅化；`face_id == 0` 表示该像素没有三角形覆盖。
    """
    import nvdiffrast.torch as dr  # noqa: PLC0415

    verts_clip = uvs * 2.0 - 1.0
    verts_clip = torch.nn.functional.pad(verts_clip, [0, 1], value=0.0)
    verts_clip = torch.nn.functional.pad(verts_clip, [0, 1], value=1.0)

    rast_out, _ = dr.rasterize(
        glctx, verts_clip.unsqueeze(0), uv_faces.to(torch.int32),
        resolution=(int(size[0]), int(size[1])),
    )
    rast = rast_out.squeeze(0)                      # [H, W, 4]
    return rast[..., :2], rast[..., 3:]             # face_uv, face_id


def build_binding(
    uvs: torch.Tensor,
    uv_faces: torch.Tensor,
    tex_size: int,
    glctx,
    device: torch.device | str = "cpu",
) -> Binding:
    """构建 `Binding`。

    Args:
        uvs: [Vuv, 2] 模板 UV
        uv_faces: [F, 3] UV 三角面
        tex_size: UV 图边长；`num_gaussians` 即由此决定（有效 texel 数）
        glctx: `nvdiffrast.torch.RasterizeGLContext` 或 `RasterizeCudaContext`
        device: 输出张量所在设备

    Returns:
        `Binding`，其中 `face_id` / `face_bary` 只保留有效 texel，
        `valid_mask` 为 `[tex_size²]` 的 bool 掩码（用于 `extract_texture`）。
    """
    face_uv, face_id = compute_uv_rast_info(uvs, uv_faces, (tex_size, tex_size), glctx)
    face_uv = face_uv.reshape(-1, 2)
    face_id = face_id.reshape(-1)

    valid = face_id > 0
    if not bool(valid.any()):
        raise RuntimeError(
            f"UV 光栅化没有得到任何有效 texel（tex_size={tex_size}）。"
            "请检查 uvs/uv_faces 是否落在 [0,1]² 内，以及 glctx 是否可用")

    # face_id == 0 表示无覆盖 → 转 0-based
    ids = (face_id[valid] - 1).to(torch.long)
    bary_uv = face_uv[valid]
    bary = torch.cat([bary_uv, 1.0 - bary_uv.sum(dim=-1, keepdim=True)], dim=-1)

    return Binding(
        face_id=ids.to(device),
        face_bary=bary.to(device),
        valid_mask=valid.to(device),
    )
