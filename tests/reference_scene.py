"""从参照仓库取出模板几何与相机参数，适配成本项目的 `Mesh` / `Camera`。

**用途**：数据集（INSTA 格式）里的网格顶点是 tracker 已经算好的世界空间顶点，
本项目需要的只是「拓扑 + UV + 顶点 + 相机」。这里把参照仓库的加载逻辑薄封装一次，
避免在多个脚本里重复。

⚠️ 数据层（`src/live3dgsavatar/data/`）属于 P2 的工作；在那之前，
本模块是**脚本侧**的临时适配层，不属于 `core/`（`core/` 不允许依赖参照仓库）。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_DATA = Path("/home/crh/Datasets/INSTA/duda")
DEFAULT_PLY = Path("/home/crh/Projects/RGBAvatar/output/duda/test/model.ply")


@dataclass
class Scene:
    """渲染一帧所需的全部输入。"""

    frames: list                      # list[dict]，每个含 mesh / blend_weight
    faces: torch.Tensor               # [F,3] int32
    uvs: torch.Tensor                 # [Vuv,2]
    uv_faces: torch.Tensor            # [F,3] int32
    K: torch.Tensor                   # [3,3] float32
    R: torch.Tensor                   # [3,3]
    T: torch.Tensor                   # [3]
    width: int
    height: int

    @property
    def num_frames(self) -> int:
        return len(self.frames)


def load_scene(
    data_dir: Path = DEFAULT_DATA,
    reference_root: Path | None = None,
    frames: int = 3,
    split: str = "all",
    device: str = "cuda",
) -> Scene:
    """加载模板几何、相机与若干帧的网格顶点。

    Args:
        data_dir: INSTA 格式数据集目录（需含 `images/` 与 `checkpoint/`）
        reference_root: RGBAvatar 只读参照仓库；为 ``None`` 时用默认路径
        frames: 取前多少帧；``-1`` 表示全部
        split: `train` / `test` / `all`
        device: 目标设备；`cpu` 便于在无 GPU 环境自检加载路径
    """
    from equivalence.reference_pipeline import reference_workspace

    ref_root = (reference_root or Path("/home/crh/Projects/RGBAvatar")).resolve()

    with reference_workspace(ref_root, REPO_ROOT / "src"):
        from dataset import FLAMEDataset  # noqa: PLC0415
        from submodules.flame import FLAME, FlameConfig  # noqa: PLC0415

        # ⚠️ 必须用 FlameConfig 的**默认值**（dtype=float64）。
        #    FLAMEDataset 把姿态/形状参数存为 float64，而 FLAME 的 v_template / shapedirs
        #    跟随 cfg.dtype；把 dtype 改成 float32 会让
        #    `template_vertices(float32) + blend_shapes(betas(float64), ...)`
        #    抛 `expected scalar type Double but found Float`。
        #    网格顶点在 FLAMEDataset 内部已降到 float32（`mesh_verts`），无需在此转换。
        flame = FLAME(FlameConfig()).to(device)
        ds = FLAMEDataset(flame, str(data_dir), split=split, pin_memory=False,
                          use_shape_weight=True, use_pose_weight=True)

        n = ds.__len__() if frames < 0 else min(frames, ds.__len__())
        out = [
            {"mesh": ds.mesh_verts[i].to(device),
             "blend_weight": ds.blend_weight[i].to(device)}
            for i in range(n)
        ]

        # 契约自检：下游 `core/` 只接受 float32（CUDA 内核限制）
        for i, f in enumerate(out[:1]):
            assert f["mesh"].dtype == torch.float32, (
                f"mesh 顶点应为 float32（core/ 与 CUDA 内核要求），实际 {f['mesh'].dtype}")
            assert f["blend_weight"].dtype == torch.float32, (
                f"blend_weight 应为 float32，实际 {f['blend_weight'].dtype}")
            assert f["mesh"].shape[0] == flame.v_template.shape[0] or True, ""
        return Scene(
            frames=out,
            faces=flame.faces.detach().clone().to(torch.int32),
            uvs=flame.uvs.to(torch.float32).detach().clone(),
            uv_faces=flame.uv_faces.to(torch.int32).detach().clone(),
            K=torch.from_numpy(ds.camera_intri).to(torch.float32),
            R=torch.from_numpy(ds.camera_extri[:3, :3]).to(torch.float32),
            T=torch.from_numpy(ds.camera_extri[:3, 3]).to(torch.float32),
            width=int(ds.image_width),
            height=int(ds.image_height),
        )
