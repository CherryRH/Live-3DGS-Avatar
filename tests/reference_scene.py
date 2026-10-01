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

from live3dgsavatar.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# ⚠️ 必须在**导入 FLAME 之前**打补丁（`FLAME.__init__` 里 pickle.load → import chumpy）。
#    `live3dgsavatar/__init__.py` 导入即施加 numpy 兼容补丁。
import live3dgsavatar  # noqa: E402, F401  （勿删：导入即生效）



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

    def frames_tensor(self, idx: list[int], device: str = "cuda"):
        """按索引取网格顶点，堆成 `[B, V, 3]`。"""
        return torch.stack([self.frames[i]["mesh"] for i in idx]).to(device)

    def weights_tensor(self, idx: list[int], device: str = "cuda"):
        """按索引取驱动参数，堆成 `[B, D]`。"""
        return torch.stack([self.frames[i]["blend_weight"] for i in idx]).to(device)


def load_scene(
    config: Config | None = None,
    frames: int | None = None,
    subject: str | None = None,
    device: str | None = None,
) -> Scene:
    """加载模板几何、相机与若干帧的网格顶点。

    所有路径与切分来自配置（`configs/system.yaml`），**本模块不含任何硬编码路径**。

    Args:
        config: 配置对象；``None`` 时用 `load_config()` 读仓库根的 configs/
        frames: 取前多少帧；``None`` 用 `render.frames`；``-1`` 表示全部
        subject: 覆盖 `subject`（数据集主体名）
        device: 目标设备；``None`` 用 `runtime.device`（``cpu`` 便于无 GPU 自检）
    """
    cfg = config or load_config()
    if subject is not None:
        cfg.set("subject", subject)
    if device is None:
        device = str(cfg.get("runtime.device", "cuda"))

    data_dir = Path(cfg.paths.data_root) / str(cfg.subject)
    ref_root = cfg.paths.reference_root
    if ref_root is None:
        raise ValueError(
            "缺少 paths.reference_root（参照仓库根目录）。"
            "请在 configs/system.yaml 设置或设 LIVE3DGS_REFERENCE_ROOT。")
    ref_root = Path(ref_root)
    split = str(cfg.runtime.split)
    if frames is None:
        frames = int(cfg.get("render.frames", 1))
    if not data_dir.exists():
        raise FileNotFoundError(
            f"数据集目录不存在：{data_dir}\n"
            f"（由 paths.data_root={cfg.paths.data_root} + subject={cfg.subject} 推出；"
            "请检查 configs/system.yaml 或设 LIVE3DGS_DATA_ROOT）")

    from equivalence.reference_pipeline import reference_workspace

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
        ds = FLAMEDataset(
            flame, str(data_dir), split=split,
            pin_memory=bool(cfg.get("dataset.pin_memory", False)),
            use_shape_weight=bool(cfg.get("dataset.use_shape_weight", True)),
            use_pose_weight=bool(cfg.get("dataset.use_pose_weight", True)))

        n = ds.__len__() if frames < 0 else min(frames, ds.__len__())
        out = [
            {"mesh": ds.mesh_verts[i].to(device),
             "blend_weight": ds.blend_weight[i].to(device)}
            for i in range(n)
        ]

        # 契约自检：下游 `core/` 只接受 float32（CUDA 内核限制）
        for f in out[:1]:
            assert f["mesh"].dtype == torch.float32, (
                f"mesh 顶点应为 float32（core/ 与 CUDA 内核要求），实际 {f['mesh'].dtype}")
            assert f["blend_weight"].dtype == torch.float32, (
                f"blend_weight 应为 float32，实际 {f['blend_weight'].dtype}")
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
