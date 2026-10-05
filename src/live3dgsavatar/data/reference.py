"""从参照实现（RGBAvatar）抽取出的「加载 + 前向」管线。

**层次**：这是**数据层**（`data/`）。放在包里（而非 `tests/`）的理由：
`core/` 与 `app/` 都可能需要读取参照侧的数据集，而 **`src/` 不应依赖 `tests/`**
（`tests/` 不是包，只能靠 `sys.path` 找到；曾因此 `ModuleNotFoundError: No module named 'tests'`）。


**参照仓库为只读**：这里只 import，不修改。

作用：为 `scripts/equivalence_check.py` 提供与 `core/` 逐阶段对应的参照输出，
使等价测试能定位到**具体哪一级**出现偏差，而不是只看到一个渲染结果不同。

需要注意的参照实现约束（都在此为调用方抹平）：
- `FlameConfig` 的路径是相对 cwd 的 `./data/FLAME2020/...`，因此必须先 chdir 到参照仓库根；
- `BindingModel.__init__` 需要 nvdiffrast 的 GL/CUDA 上下文；
- 参照模型参数默认在 CUDA 上。
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

# --------------------------------------------------------------------- 环境 --


# 仓库根：src/live3dgsavatar/data/reference.py → 上溯 4 层
REPO_ROOT = Path(__file__).resolve().parents[3]


def _add_sys_path(root: Path) -> None:
    if str(root) not in sys.path:
        sys.path.append(str(root))


@contextmanager
def reference_workspace(reference_root: Path):
    """进入参照仓库的 cwd（其模型路径依赖相对路径），退出时恢复。

    只需参照仓库根 —— 本项目包已可正常导入（见模块 docstring），
    参照侧的 `from submodules.flame import ...` 走它自己 root 的 sys.path。
    """
    _add_sys_path(reference_root)
    cwd = os.getcwd()
    os.chdir(reference_root)
    try:
        yield
    finally:
        os.chdir(cwd)


# ----------------------------------------------------------------- 数据类 --


@dataclass
class ReferenceBundle:
    """参照侧的加载结果，供逐阶段比对。"""

    flame_model: object
    model: object                # FLAMEBindingModel
    dataset: object              # FLAMEDataset
    camera: object               # IntrinsicsCamera
    config: dict                 # 构造用的模型配置（tex_size / num_basis_* 等）
    template_faces: torch.Tensor
    template_uvs: torch.Tensor
    template_uv_faces: torch.Tensor


# ------------------------------------------------------------------- 构造 --


def build_reference(
    reference_root: Path,
    data_dir: Path,
    ply_path: Path,
    tex_size: int = 256,
    num_basis_in: int = 129,
    num_basis_blend: int = 20,
    mlp_hidden: tuple[int, ...] = (128, 128),
    split: str = "all",
) -> ReferenceBundle:
    """加载参照模型与数据集。

    Args:
        data_dir: 数据集目录（`<data_root>/<subject>`）
        ply_path: 预训练 `.ply`
        tex_size / num_basis_* / mlp_hidden: 模型结构，必须与 `.ply` 一致
        split: 数据集切分
    """
    with reference_workspace(reference_root):
        import nvdiffrast.torch as dr  # noqa: PLC0415

        from camera import IntrinsicsCamera  # noqa: PLC0415
        from dataset import FLAMEDataset  # noqa: PLC0415
        from model import FLAMEBindingModel  # noqa: PLC0415
        from submodules.flame import FLAME, FlameConfig  # noqa: PLC0415
        from utils import Struct  # noqa: PLC0415

        glctx = dr.RasterizeGLContext()
        flame_model = FLAME(FlameConfig()).cuda()
        dataset = FLAMEDataset(flame_model, str(data_dir), split=split,
                               pin_memory=False, use_shape_weight=True,
                               use_pose_weight=True)

        cfg = Struct(tex_size=tex_size, num_basis_in=num_basis_in,
                     num_basis_blend=num_basis_blend, use_blend=True,
                     use_weight_proj=True, use_mlp_proj=True,
                     init_scaling=0.0008, init_opacity=0.5)
        model = FLAMEBindingModel(cfg, flame_model, glctx)
        model.load_ply(str(ply_path))

        camera = IntrinsicsCamera(
            K=dataset.camera_intri,
            R=dataset.camera_extri[:3, :3],
            T=dataset.camera_extri[:3, 3],
            width=dataset.image_width,
            height=dataset.image_height,
        ).cuda()

        return ReferenceBundle(
            flame_model=flame_model,
            model=model,
            dataset=dataset,
            camera=camera,
            config={"tex_size": tex_size, "num_basis_in": num_basis_in,
                    "num_basis_blend": num_basis_blend, "mlp_hidden": mlp_hidden},
            template_faces=flame_model.faces.detach().clone(),
            template_uvs=flame_model.uvs.to(torch.float32).detach().clone(),
            template_uv_faces=flame_model.uv_faces.to(torch.int32).detach().clone(),
        )


# ------------------------------------------------------ 逐阶段的参照输出 --


def reference_uv_rast(reference_root: Path, bundle: ReferenceBundle):
    """参照的 UV 域光栅化原始输出。

    返回 `(face_uv [H*W, 2], face_id [H*W, 1])`，等价于参照
    `diff_renderer.texture.compute_rast_info`。用于把绑定的偏差**定位到光栅化本身**，
    而不是只看绑定后的 xyz（那样分不清是 UV 光栅化错了还是 mesh_binding 错了）。
    """
    from diff_renderer.texture import compute_rast_info  # noqa: PLC0415

    return compute_rast_info(
        uvs=bundle.flame_model.uvs.to(torch.float32),
        uv_faces=bundle.flame_model.uv_faces.to(torch.int32),
        size=(bundle.config["tex_size"], bundle.config["tex_size"]),
        glctx=bundle.model.glctx,
    )


def reference_blend_weights(model, blend_weight: torch.Tensor) -> torch.Tensor:
    """参照的 `project_weight`：`[B, D] → [B, K]`。"""
    return model.project_weight(blend_weight)


def reference_blended_attributes(model, blend_weight: torch.Tensor):
    """参照的混合结果（**激活前**的属性，激活前不参与比较）。

    返回 `(xyz, rotation_raw, color, opacity_logit, scaling_log)`，形状 `[B, N, ·]`。
    """
    weights = model.project_weight(blend_weight)
    from diff_gaussian_rasterization import linear_blending  # noqa: PLC0415

    xyz, rot, color = linear_blending(
        weights,
        model._xyz, model._rotation, model._feature_dc,
        model._xyz_b, model._rotation_b, model._feature_b,
    )
    opacity = model._opacity.unsqueeze(0).expand(weights.shape[0], -1, -1)
    scaling = model._scaling.unsqueeze(0).expand(weights.shape[0], -1, -1)
    return xyz, rot, color, opacity, scaling


def reference_deformed_gaussians(model, mesh_verts: torch.Tensor, blend_weight: torch.Tensor):
    """参照的 `gaussian_deform_batch` 输出（世界空间）。"""
    return model.gaussian_deform_batch(mesh_verts, blend_weight)


def reference_render(reference_root: Path, camera, bg_color: torch.Tensor, gaussians):
    """参照的 `diff_renderer.render_gs_batch`。"""
    from diff_renderer import render_gs_batch  # noqa: PLC0415

    return render_gs_batch(camera, bg_color, gaussians)
