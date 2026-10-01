"""端到端数值等价测试（**需要 GPU**）。

与 `scripts/equivalence_check.py` **共用 `equivalence/stages.py` 的同一套比对逻辑与判据**，
因此脚本与测试不会漂移。以测试形式固化，便于回归。

无 GPU、缺少参照仓库或缺少预训练模型时**自动跳过**（记为「跳过」，不计入通过），
这样 `python tests/run_tests.py` 在开发机上仍然是全绿的。

判据（docs/CONVENTIONS.md §6.1）：
- 参数：逐位一致（`max|Δ| = 0`）
- 中间属性：`max|Δ| < 1e-5`
- 渲染图：`PSNR > 60 dB` 或 `max|Δ| < 1e-3`
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from live3dgsavatar.config import load_config
from support import REFERENCE_ROOT, REPO_ROOT

# 路径一律来自配置（configs/system.yaml），**此处不硬编码**。
# 换数据集/模型只需改配置，或设 LIVE3DGS_DATA_ROOT / LIVE3DGS_REFERENCE_ROOT。
_CFG = load_config()
_SUBJECT = str(_CFG.subject)
_DATA_ROOT = Path(_CFG.paths.data_root)
PLY = Path(_CFG.get("paths.model_ply") or (
    (Path(REFERENCE_ROOT) / "output" / _SUBJECT
     / str(_CFG.get("paths.model_subdir", "test")) / "model.ply")))
DATA = _DATA_ROOT / _SUBJECT


def _require_environment() -> None:
    """不满足条件就 `pytest.skip`（运行器记为「跳过」，不是「通过」）。"""
    if not torch.cuda.is_available():
        pytest.skip("无可用 GPU")
    if not REFERENCE_ROOT.is_dir():
        pytest.skip(f"参照仓库不存在：{REFERENCE_ROOT}（用 RGBA_REF 指定）")
    if not PLY.exists():
        pytest.skip(f"预训练模型不存在：{PLY}（检查 configs/system.yaml 的 paths.model_ply）")
    if not DATA.is_dir():
        pytest.skip(f"数据集不存在：{DATA}（检查 configs/system.yaml 的 paths.data_root）")


def test_forward_pass_matches_reference() -> None:
    """形状 → 参数 → 绑定构建 → 混合 → 绑定 → 渲染，全链路与参照实现比对。"""
    _require_environment()

    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.deform import build_binding
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.types import Camera, Mesh
    from equivalence.reference_pipeline import build_reference, reference_uv_rast
    from equivalence.stages import (
        EquivalenceReport,
        compare_binding,
        compare_blend,
        compare_deformed,
        compare_parameters,
        compare_render,
    )

    report = EquivalenceReport()

    ref = build_reference(
        reference_root=REFERENCE_ROOT, src_root=REPO_ROOT / "src",
        data_dir=DATA, ply_path=PLY)
    cfg = AvatarConfig(tex_size=ref.config["tex_size"],
                       num_basis_in=ref.config["num_basis_in"],
                       num_basis_blend=ref.config["num_basis_blend"],
                       mlp_hidden=tuple(ref.config["mlp_hidden"]))
    avatar = load_ply(PLY, config=cfg, device="cuda")

    # 1. 参数（逐位一致）
    compare_parameters(report, avatar, ref.model, PLY)

    # 2. 绑定构建：UV 域光栅化本身
    compare_binding(
        report,
        uv_builder=lambda: build_binding(
            uvs=ref.flame_model.uvs.to(torch.float32),
            uv_faces=ref.flame_model.uv_faces.to(torch.int32),
            tex_size=ref.config["tex_size"], glctx=ref.model.glctx, device="cuda"),
        ref_uv_rast=lambda: reference_uv_rast(REFERENCE_ROOT, ref),
        ref_model=ref.model,
    )

    # 3~5. 混合 → 绑定 → 渲染
    n = min(2, len(ref.dataset))
    mesh_ref = ref.dataset.mesh_verts[:n].cuda()
    bw = ref.dataset.blend_weight[:n].cuda()
    mesh_mine = Mesh(verts=mesh_ref.to(torch.float32),
                     faces=ref.template_faces.to(torch.int32),
                     uvs=ref.template_uvs, uv_faces=ref.template_uv_faces)
    camera = Camera.from_intrinsics_extrinsics(
        K=ref.dataset.camera_intri, R=ref.dataset.camera_extri[:3, :3],
        T=ref.dataset.camera_extri[:3, 3],
        width=ref.dataset.image_width, height=ref.dataset.image_height).to("cuda")

    compare_blend(report, avatar, ref.model, bw)
    gs_mine, gs_ref = compare_deformed(report, avatar, ref.model, mesh_ref, mesh_mine, bw)
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    compare_render(report, REFERENCE_ROOT, camera, ref.camera, gs_mine, gs_ref, bg)

    # 汇总失败项，一次性把全部偏差报出来（而不是在第一个 assert 就中断）
    if report.n_fail:
        details = "\n".join(f"  [{r.stage}] {r.name}: {r.detail}"
                            for r in report.rows if not r.ok)
        raise AssertionError(f"等价门有 {report.n_fail} 项失败：\n{details}")
