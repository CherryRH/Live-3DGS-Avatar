"""等价门的分层比对实现 —— `scripts/equivalence_check.py` 与
`tests/equivalence/test_forward_equivalence.py` **共用同一份**，避免两边判据漂移。

分层顺序即偏差定位顺序::

    1. 参数       PLY 读入是否逐位一致
    2. 绑定构建   UV 域光栅化（哪些 texel 成为高斯、绑到哪个面）
    3. 混合       权重投影与 blendshape 线性混合
    4. 绑定       mesh_binding（切空间 → 世界空间）
    5. 渲染       光栅化输出

任一层失败，之后的层必然失败，因此报告是**自上而下定位**的。

注意：本模块只做比对，不做加载。加载逻辑在 `reference_pipeline.py`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ATOL_ATTR = 1e-5
ATOL_EXACT = 0.0          # PLY 原样存取，应逐位一致
PSNR_DB = 60.0
ATOL_IMAGE = 1e-3


@dataclass
class StageResult:
    stage: str
    name: str
    ok: bool
    detail: str


@dataclass
class EquivalenceReport:
    rows: list[StageResult] = field(default_factory=list)

    def add(self, stage: str, name: str, ok: bool, detail: str) -> bool:
        self.rows.append(StageResult(stage, name, ok, detail))
        return ok

    def to_json(self) -> list[dict]:
        return [{"stage": r.stage, "name": r.name, "ok": r.ok, "detail": r.detail}
                for r in self.rows]

    @property
    def n_fail(self) -> int:
        return sum(1 for r in self.rows if not r.ok)

    def dump(self) -> int:
        width = max((len(r.name) for r in self.rows), default=10)
        print(f"\n{'=' * 78}\nP1 数值等价门\n{'=' * 78}")
        stage = None
        for r in self.rows:
            if r.stage != stage:
                print(f"\n[{r.stage}]")
                stage = r.stage
            mark = "\033[1;32m  ok \033[0m" if r.ok else "\033[1;31mFAIL\033[0m"
            print(f"  [{mark}] {r.name.ljust(width)}  {r.detail}")
        print(f"\n{'=' * 78}")
        print("结论：" + ("全部通过 ✓" if self.n_fail == 0 else f"{self.n_fail} 项失败 ✗"))
        print(f"{'=' * 78}")
        return 1 if self.n_fail else 0


# ------------------------------------------------------------------ 第 1 层 --


def compare_parameters(report: EquivalenceReport, avatar, ref_model, ply_path: Path) -> None:
    """PLY 读入的参数必须与参照逐位一致（含 weight_module）。"""
    from support import compare

    pairs = [
        ("_xyz", avatar.xyz, "xyz"),
        ("_opacity", avatar.opacity, "opacity(logit)"),
        ("_scaling", avatar.scaling, "scaling(log)"),
        ("_rotation", avatar.rotation, "rotation"),
        ("_feature_dc", avatar.color, "color(SH DC)"),
        ("_xyz_b", avatar.xyz_b, "xyz_b"),
        ("_rotation_b", avatar.rotation_b, "rotation_b"),
        ("_feature_b", avatar.color_b, "color_b"),
    ]
    for ref_name, mine, label in pairs:
        ok, msg = compare(mine.detach().cpu(),
                          getattr(ref_model, ref_name).detach().cpu(),
                          ATOL_EXACT, label)
        report.add("参数", label, ok, msg)

    # weight_module 在 PLY 里是补零的单列，按参照架构的参数个数截取
    from plyfile import PlyData

    flat_mine = torch.cat([p.detach().reshape(-1)
                           for p in avatar.weight_module.state_dict().values()])
    col = np.asarray(PlyData.read(str(ply_path)).elements[0]["weight_module"],
                     dtype="float32")
    if flat_mine.numel() > col.shape[0]:
        report.add("参数", "weight_module", False,
                   f"参数量 {flat_mine.numel()} 超过 PLY 单列长度 {col.shape[0]}")
    else:
        flat_ref = torch.from_numpy(col[: flat_mine.numel()])
        ok, msg = compare(flat_mine.cpu(), flat_ref, ATOL_EXACT, "weight_module")
        report.add("参数", "weight_module", ok, msg)

    # 通道维/形状一致性（参照不检查，这里显式确认）
    report.add("参数", "通道数一致",
               avatar.num_gaussians == ref_model.num_gaussian and
               avatar.num_basis == ref_model._xyz_b.shape[0],
               f"N={avatar.num_gaussians}, K={avatar.num_basis}")


# ------------------------------------------------------------------ 第 2 层 --


def compare_binding(report: EquivalenceReport, uv_builder, ref_uv_rast, ref_model) -> None:
    """UV 域光栅化与由此得到的绑定。

    Args:
        uv_builder: 无参可调用，返回本项目 `Binding`
        ref_uv_rast: 无参可调用，返回参照的 `(face_uv [H*W,2], face_id [H*W,1])`
        ref_model: 用于与参照模型自身缓存的绑定交叉核对
    """
    from support import compare

    face_uv_ref, face_id_ref = ref_uv_rast()
    face_uv_ref = face_uv_ref.reshape(-1, 2)
    face_id_ref = face_id_ref.reshape(-1)

    binding = uv_builder()
    valid_ref = face_id_ref > 0

    ok, msg = compare(binding.valid_mask.detach().cpu(), valid_ref.detach().cpu(),
                      ATOL_EXACT, "valid_mask（有效 texel 集合）")
    report.add("绑定构建", "valid_mask", ok, msg)

    n_ref = int(valid_ref.sum())
    same_n = binding.num_gaussians == n_ref
    report.add("绑定构建", "高斯数量", same_n, f"{binding.num_gaussians} vs {n_ref}")
    if not same_n:
        report.add("绑定构建", "face_id / face_bary", False,
                   "高斯数量不一致，跳过逐项比对")
        return

    ids_ref = (face_id_ref[valid_ref] - 1).to(torch.long)
    ok, msg = compare(binding.face_id.detach().cpu(), ids_ref.cpu(), ATOL_EXACT, "face_id")
    report.add("绑定构建", "face_id", ok, msg)

    bary_ref = torch.cat(
        [face_uv_ref[valid_ref],
         1.0 - face_uv_ref[valid_ref].sum(dim=-1, keepdim=True)], dim=-1)
    ok, msg = compare(binding.face_bary.detach().cpu(), bary_ref.cpu(),
                      ATOL_ATTR, "face_bary")
    report.add("绑定构建", "face_bary", ok, msg)

    cached = ref_model.binding_face_id.detach().cpu().long()
    same = cached.shape == binding.face_id.shape and bool(
        (binding.face_id.detach().cpu() == cached).all())
    report.add("绑定构建", "与参照模型缓存一致", same, f"N={binding.num_gaussians}")


# ------------------------------------------------------------------ 第 3 层 --


def compare_blend(report: EquivalenceReport, avatar, ref_model,
                  blend_weight: torch.Tensor) -> None:
    """权重投影与 blendshape 线性混合（激活前/后都比）。"""
    from support import compare

    from live3dgsavatar.data.reference import (
        reference_blended_attributes,
        reference_blend_weights,
    )

    field = avatar.build_blend_field(use_cuda_kernel=False)

    ok, msg = compare(field.project_weight(blend_weight).detach(),
                      reference_blend_weights(ref_model, blend_weight).detach(),
                      ATOL_ATTR, "project_weight")
    report.add("混合", "project_weight", ok, msg)

    xyz_ref, rot_ref, color_ref, op_ref, sc_ref = reference_blended_attributes(
        ref_model, blend_weight)
    mine = field(blend_weight)
    for label, a, b in [
        ("xyz", mine.xyz, xyz_ref),
        ("rotation(normalized)", mine.rotation,
         torch.nn.functional.normalize(rot_ref, dim=-1)),
        ("color", mine.color, color_ref),
        ("opacity(activated)", mine.opacity, torch.sigmoid(op_ref)),
        ("scaling(activated)", mine.scaling, torch.exp(sc_ref)),
    ]:
        ok, msg = compare(a.detach(), b.detach(), ATOL_ATTR, label)
        report.add("混合", label, ok, msg)


# ------------------------------------------------------------------ 第 4 层 --


def compare_deformed(report: EquivalenceReport, avatar, ref_model,
                     mesh_ref: torch.Tensor, mesh_mine, blend_weight: torch.Tensor):
    """mesh_binding（切空间 → 世界空间）。**全部属性都与参照判等。**

    ⚠️ 不要在这里标注"位置项有意偏离"：经复查，
    本项目与参照都用 `R @ xyz`（`R` 的列为基向量），位置应当一致。
    当时看到的 9.87e-02 差异是本项目多用了一次转置（`Rᵀ`）造成的 bug，
    已在 `core/deform/bind.py` 修正。

    实测：修正后位置差 `max|Δ| = 2.98e-08`，渲染 `PSNR = 133 dB`（逐位一致）。
    """
    from support import compare

    from live3dgsavatar.data.reference import reference_deformed_gaussians

    gs_ref = reference_deformed_gaussians(ref_model, mesh_ref, blend_weight)
    gs_mine = avatar.deform(mesh_mine, blend_weight, use_cuda_kernel=False)
    for label, a, b in [
        ("xyz", gs_mine.xyz, gs_ref.xyz),
        ("rotation", gs_mine.rotation, gs_ref.rotation),
        ("opacity", gs_mine.opacity, gs_ref.opacity),
        ("scaling", gs_mine.scaling, gs_ref.scaling),
        ("color", gs_mine.color, gs_ref.sh),
    ]:
        ok, msg = compare(a.detach(), b.detach(), ATOL_ATTR, label)
        report.add("绑定", label, ok, msg)
    return gs_mine, gs_ref


# ------------------------------------------------------------------ 第 5 层 --


def compare_render(report: EquivalenceReport, reference_root: Path, camera, ref_camera,
                   gs_mine, gs_ref, bg: torch.Tensor) -> None:
    """渲染输出：`PSNR > 60 dB` 或 `max|Δ| < 1e-3`（与参照判等）。"""
    from support import compare_image

    from live3dgsavatar.data.reference import reference_render
    from live3dgsavatar.core.render import SimpleRasterizer

    out_ref = reference_render(reference_root, ref_camera, bg, gs_ref)
    out_mine = SimpleRasterizer().render(gs_mine, camera, bg)
    for label, a, b in [("color", out_mine.color, out_ref["color"]),
                        ("alpha", out_mine.alpha, out_ref["alpha"])]:
        ok, msg = compare_image(a.detach(), b.detach(), PSNR_DB, ATOL_IMAGE, label)
        report.add("渲染", label, ok, msg)

    # 附加信息：形状与有限性
    report.add("渲染", "输出形状一致",
               tuple(out_mine.color.shape) == tuple(out_ref["color"].shape),
               f"{tuple(out_mine.color.shape)}")
    report.add("渲染", "输出全部有限",
               bool(torch.isfinite(out_mine.color).all()), "无 NaN / Inf")

