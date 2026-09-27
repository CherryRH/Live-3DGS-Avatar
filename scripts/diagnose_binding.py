#!/usr/bin/env python3
"""绑定层的原理性验证（默认需要 GPU；`--dry-run` 可在 CPU 上自检脚本本身）。

## 为什么不再"对着参照比"

等价门发现绑定层 `xyz` 有 9.87e-02 的 max 误差，但误差分布是**平滑长尾**
（中位数 6.9e-03 → 分位 0.99 为 3.5e-02 → max 9.87e-02），不是统一量级的系统偏差。
这不能简单断定"实现错了"——也可能是参照实现本身不严谨。

参照实现是研究原型，**自身就有已知缺陷**（`save_ply` 不可用、
Python 回退版与 CUDA 版语义不一致、`linear_blending` 静默返回全零），
因此**不能作为正确性的最终判据**。本脚本改用**独立推导的运动学不变量**判定。

| 编号 | 不变量 | 违反说明什么 |
|---|---|---|
| I1 | 刚性等变：模板做刚体变换 T 后，绑定结果应等于原结果施加 T | 变换用错空间/顺序（最强判据） |
| I2 | `Σ bary = 1` | 顶点组合非仿射，位置可能跑出三角形平面 |
| I3 | offset 到三角面距离为 0 | 权重或顶点取错 |
| I4a | TBN 剪切程度（**仅报告，非错误**） | — |
| I5/I6 | TBN=I、局部 xyz=0 时，CUDA 算子输出应精确等于 `Σ baryᵢ·vᵢ` | 算子与公式不一致 |
| I8 | TBN 行列式/条件数 → 可逆性 | 退化面导致变换爆炸 |
| I9 | TBN 正交化后，与参照的差异是否变小 | 判定 `Rᵀ` 在非正交 `R` 下是否成立 |

**重要**：`I4` 曾把"TBN 应该正交"当作不变量，这是**错的**。
代数上 `normalize(t)·normalize(b) = −cos(∠A')`，正交当且仅当 UV 三角形在顶点 a 处为直角；
真实 FLAME UV 上 `|cos|` 中位数约 0.68，只有 0.47% 的面接近正交。
详见 docs/MIGRATION.md D 节。

用法::

    conda activate live3dgs
    python scripts/diagnose_binding.py              # 真实数据（需 GPU）
    python scripts/diagnose_binding.py --dry-run    # CPU 合成数据，只验证脚本自身
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="绑定层的原理性验证")
    p.add_argument("--ply", type=Path,
                   default=Path("/home/crh/Projects/RGBAvatar/output/duda/test/model.ply"))
    p.add_argument("--data", type=Path, default=Path("/home/crh/Datasets/INSTA/duda"))
    p.add_argument("--reference", type=Path, default=Path("/home/crh/Projects/RGBAvatar"))
    p.add_argument("--top", type=int, default=5, help="打印最差多少个高斯")
    p.add_argument("--dry-run", action="store_true",
                   help="用 CPU 合成数据跑通全部代码路径（验证脚本本身，不需要 GPU）")
    return p.parse_args()


def _fmt(t, n: int = 6) -> str:
    return "[" + ", ".join(f"{x:+.5f}" for x in t.reshape(-1)[:n].tolist()) + "]"


def _line(name: str, value: str, ok: bool | None = None, note: str = "") -> None:
    """打印一行结论。`ok=None` 表示"仅报告，不做判定"。"""
    tail = f"   {note}" if note else ""
    if ok is None:
        print(f"  [  --  ] {name:<40} {value}{tail}")
    else:
        mark = "\033[1;32m  ok  \033[0m" if ok else "\033[1;31m FAIL \033[0m"
        print(f"  [{mark}] {name:<40} {value}{tail}")


@dataclass
class Data:
    """分析所需的全部张量。真实与合成数据共用，便于 dry-run 自检。"""

    mesh: object
    xyz_tan: object
    tri: object
    tbn: object
    fid: object
    bary: object
    theirs: object
    faces: object
    uvs: object
    uvf: object
    label: str = ""
    device: str = "cpu"


def load_fake_data(seed: int = 0, device: str = "cpu") -> Data:
    """构造形状同构的合成数据，用于 `--dry-run`。"""
    import torch

    torch.manual_seed(seed)
    B = 1
    side = 15                       # 15x15 顶点栅格 → 14x14x2 = 392 个三角面
    V = side * side
    F = (side - 1) * (side - 1) * 2
    N = 2000

    # 规则三角剖分（避免随机面产生的退化三角形与 NaN）
    idx = torch.arange(V).reshape(side, side)
    quads = []
    for i in range(side - 1):
        for j in range(side - 1):
            a, b = idx[i, j], idx[i, j + 1]
            c, d = idx[i + 1, j], idx[i + 1, j + 1]
            quads += [[a, b, c], [b, d, c]]
    faces = torch.tensor(quads, dtype=torch.long)

    # UV：带一点剪切，保证 TBN 非正交但可逆
    gx, gy = torch.meshgrid(torch.linspace(0, 1, side), torch.linspace(0, 1, side),
                            indexing="ij")
    uvs = torch.stack([(gx + 0.25 * gy).reshape(-1), (gy + 0.15 * gx).reshape(-1)], dim=-1)
    uvf = faces.clone()

    mesh = torch.randn(B, V, 3) * 0.1
    xyz_tan = torch.randn(B, N, 3) * 0.01
    fid = torch.randint(0, F, (N,))
    raw = torch.rand(N, 3) + 0.05
    bary = raw / raw.sum(-1, keepdim=True)

    tri = mesh[:, faces]
    tbn = _tbn_of(mesh, faces, uvs, uvf)      # reference 模式 → 非正交，可走到 I4/I8/I9

    R = tbn[:, fid]
    bv = tri[:, fid]
    off = (bv * bary.unsqueeze(0).unsqueeze(-1)).sum(-2)
    theirs = torch.matmul(R.transpose(-1, -2), xyz_tan.unsqueeze(-1)).squeeze(-1) + off
    theirs = theirs + torch.randn_like(theirs) * 1e-3   # 扰动，走到"最差 N 个"与离群分支

    return Data(mesh=mesh, xyz_tan=xyz_tan, tri=tri, tbn=tbn, fid=fid, bary=bary,
                theirs=theirs, faces=faces, uvs=uvs, uvf=uvf,
                label="合成数据", device=device)


def load_real_data(args) -> Data:
    import torch

    from equivalence.reference_pipeline import (
        build_reference, reference_blended_attributes)
    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.io import load_ply

    ref = build_reference(
        reference_root=args.reference.resolve(), src_root=REPO_ROOT / "src",
        data_dir=args.data.resolve(), ply_path=args.ply.resolve())
    model = ref.model
    dev = "cuda"

    mesh = ref.dataset.mesh_verts[:1].to(dev)
    bw = ref.dataset.blend_weight[:1].to(dev)
    faces, uvs, uvf = model.template_faces, ref.template_uvs, ref.template_uv_faces

    xyz_tan = reference_blended_attributes(model, bw)[0]
    tri = mesh[:, faces]
    tbn = _tbn_of(mesh, faces, uvs, uvf)
    with torch.no_grad():
        theirs = model.gaussian_deform_batch(mesh, bw).xyz

    cfg = AvatarConfig(tex_size=256, num_basis_in=129, num_basis_blend=20,
                       mlp_hidden=(128, 128), use_weight_proj=True)
    binding = load_ply(args.ply.resolve(), config=cfg, device=dev).to_binding()

    return Data(mesh=mesh, xyz_tan=xyz_tan, tri=tri, tbn=tbn, fid=binding.face_id,
                bary=binding.face_bary, theirs=theirs, faces=faces, uvs=uvs,
                uvf=uvf, label="真实数据", device=dev)


def _tbn_of(mesh, faces, uvs, uvf):
    from live3dgsavatar.core.deform.tbn import compute_face_tbn

    return compute_face_tbn(mesh[:, faces], uvs[uvf], mode="reference")


def _orthonormalize(tbn):
    import torch
    import torch.nn.functional as F

    t_raw, n_raw = tbn[..., :, 0], tbn[..., :, 2]
    t_orth = F.normalize(t_raw - (t_raw * n_raw).sum(-1, keepdim=True) * n_raw, dim=-1)
    return torch.stack([t_orth, torch.cross(n_raw, t_orth, dim=-1), n_raw], dim=-1)


# ------------------------------------------------------------------ 分析 --


def analyse(d: Data, top: int) -> int:
    import torch

    dev = d.device
    mesh, tri, tbn = d.mesh, d.tri, d.tbn
    fid, bary, xyz_tan, theirs = d.fid, d.bary, d.xyz_tan, d.theirs
    faces, uvs, uvf = d.faces, d.uvs, d.uvf

    # 形状自检：早失败比形状广播导致的静默错误好得多
    B = mesh.shape[0]
    N = fid.shape[0]
    assert tri.shape[:2] == (B, faces.shape[0]), f"tri 形状异常 {tuple(tri.shape)}"
    assert tbn.shape[:2] == (B, faces.shape[0]), f"tbn 形状异常 {tuple(tbn.shape)}"
    assert xyz_tan.shape == (B, N, 3), f"xyz_tan 应为 [B,N,3]，实际 {tuple(xyz_tan.shape)}"
    assert theirs.shape == (B, N, 3), f"theirs 应为 [B,N,3]，实际 {tuple(theirs.shape)}"
    assert bary.shape == (N, 3), f"bary 应为 [N,3]，实际 {tuple(bary.shape)}"

    R = tbn[:, fid]
    bv = tri[:, fid]
    off_formula = (bv * bary.unsqueeze(0).unsqueeze(-1)).sum(-2)
    assert off_formula.shape == (B, N, 3), f"offset 形状异常 {tuple(off_formula.shape)}"

    def bind_xyz(mesh_v, tbn_f, xyz, transpose: bool = True):
        """`tbn_f` 传**逐面**的 TBN `[B,F,3,3]`；本函数内部按 fid 取到 `[B,N,3,3]`。"""
        b = (mesh_v[:, faces][:, fid] * bary.unsqueeze(0).unsqueeze(-1)).sum(-2)
        r = tbn_f[:, fid]
        m = r.transpose(-1, -2) if transpose else r
        out = torch.matmul(m, xyz.unsqueeze(-1)).squeeze(-1) + b
        assert out.shape == xyz.shape, f"绑定输出形状 {tuple(out.shape)} 应等于 {tuple(xyz.shape)}"
        return out

    out1 = bind_xyz(mesh, tbn, xyz_tan)

    print("\n" + "=" * 78)
    print("一、绑定变换的基本性质（独立于参照实现）")
    print("=" * 78)

    s = bary.sum(-1)
    _line("I2 重心之和 == 1", f"max|Σ-1| = {(s - 1).abs().max().item():.3e}",
          bool((s - 1).abs().max() < 1e-5))

    gram = R.transpose(-1, -2) @ R
    eye = torch.eye(3, device=dev).expand_as(gram)
    orth = (gram - eye).abs().amax(dim=(-1, -2))[0]
    _line("I4a TBN 剪切程度 max|RᵀR-I|（非错误）", f"{orth.max().item():.3e}")
    print(f"           非正交的高斯数 = {int((orth > 1e-4).sum())} / {orth.numel()}"
          f"  （UV 参数化非正交所致，属正常）")

    det = torch.linalg.det(R)
    det_f = det[torch.isfinite(det)]
    _line("I8a TBN 行列式（应显著非零）",
          f"min|det|={det_f.abs().min().item():.3e}  "
          f"median={det_f.abs().median().item():.3e}"
          if det_f.numel() else "无有限值",
          bool(det_f.numel() and det_f.abs().min() > 1e-6))

    # 非有限值（退化面会产生 NaN）会让 torch.linalg.cond 直接抛异常，先屏蔽
    finite = torch.isfinite(R).all(dim=(-1, -2)).all(dim=-1)[0]      # [N]
    n_bad = int((~finite).sum())
    if n_bad:
        _line("I8b 非有限 TBN（退化面）", f"{n_bad} / {finite.numel()} 个高斯",
              False, "（compute_face_tbn 对退化面会产生 NaN）")
    cond = torch.linalg.cond(R[finite].unsqueeze(0)) if bool(finite.any()) else None
    if cond is not None:
        _line("I8b TBN 条件数",
              f"median={cond.median().item():.2f}  max={cond.max().item():.3e}",
              bool(cond.max() < 1e4),
              f"（>1e4 的高斯数 = {int((cond > 1e4).sum())}）")
    else:
        _line("I8b TBN 条件数", "无有限 TBN，跳过", False)

    n_hat = R[..., :, 2]
    dist = ((off_formula - bv[..., 0, :]) * n_hat).sum(-1)
    _line("I3 offset 离面偏差", f"max = {dist.abs().max().item():.3e}",
          bool(dist.abs().max() < 1e-4))

    print("\n  [I4 深挖] TBN 各列模长与列间夹角")
    col_norm = R.norm(dim=-2)
    for j, nm in enumerate(("tangent", "bitangent", "normal")):
        cn = col_norm[0, :, j]
        print(f"    {nm:<10} 模长 median={cn.median().item():.3e}"
              f"  p99={torch.quantile(cn, 0.99).item():.3e}  max={cn.max().item():.3e}")

    Rn = torch.nn.functional.normalize(R, dim=-2)
    for nm, dd in (("tangent·bitangent", (Rn[0, :, :, 0] * Rn[0, :, :, 1]).sum(-1)),
                   ("tangent·normal", (Rn[0, :, :, 0] * Rn[0, :, :, 2]).sum(-1)),
                   ("bitangent·normal", (Rn[0, :, :, 1] * Rn[0, :, :, 2]).sum(-1))):
        print(f"    cos({nm:<18}) median={dd.abs().median().item():.3e}"
              f"  max={dd.abs().max().item():.3e}")

    uv0, uv1, uv2 = uvs[uvf][:, 0], uvs[uvf][:, 1], uvs[uvf][:, 2]
    e1, e2 = uv1 - uv0, uv2 - uv0
    area = (e1[:, 0] * e2[:, 1] - e2[:, 0] * e1[:, 1]).abs() * 0.5
    print(f"\n    UV 三角形面积 median={area.median().item():.3e}"
          f"  p1={torch.quantile(area, 0.01).item():.3e}  min={area.min().item():.3e}")
    print(f"    面积 < 1e-6 的面数 = {int((area < 1e-6).sum())} / {area.numel()}")

    gram_f = tbn.transpose(-1, -2) @ tbn
    eye_f = torch.eye(3, device=dev).expand_as(gram_f)
    orth_face = (gram_f - eye_f).abs().amax(dim=(-1, -2))[0]
    print(f"    逐面 TBN 正交偏差 median={orth_face.median().item():.3e}"
          f"  min={orth_face.min().item():.3e}  max={orth_face.max().item():.3e}")
    order = torch.argsort(orth_face, descending=True)
    for i in (order[0].item(), order[len(order) // 2].item(), order[-1].item()):
        cl = tbn[0, i].norm(dim=0)
        print(f"      face {i}: 正交偏差={orth_face[i].item():.3e}"
              f"  列模长=({cl[0].item():.3e},{cl[1].item():.3e},{cl[2].item():.3e})")

    print("\n" + "=" * 78)
    print("二、I1 刚性等变性（不依赖任何实现，最强判据）")
    print("=" * 78)

    torch.manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(3, 3, device=dev))
    if float(torch.det(Q)) < 0:
        Q[:, 0] = -Q[:, 0]
    t = torch.randn(3, device=dev) * 0.05

    mesh2 = mesh @ Q.T + t
    tbn2 = _tbn_of(mesh2, faces, uvs, uvf)
    out2 = bind_xyz(mesh2, tbn2, xyz_tan)
    d1 = (out2 - (out1 @ Q.T + t)).abs().max().item()
    _line("I1 刚性等变（Rᵀ·x）", f"max|Δ| = {d1:.3e}", bool(d1 < 1e-3))

    alt1 = bind_xyz(mesh, tbn, xyz_tan, transpose=False)
    alt2 = bind_xyz(mesh2, tbn2, xyz_tan, transpose=False)
    d1_alt = (alt2 - (alt1 @ Q.T + t)).abs().max().item()
    _line("I1 刚性等变（R·x）", f"max|Δ| = {d1_alt:.3e}", bool(d1_alt < 1e-3))
    print("           ⟹ 只有满足刚性等变的那一种才可能是正确形式")
    if d1 < 1e-3 and d1_alt >= 1e-3:
        print("           判定：Rᵀ·x 正确")
    elif d1_alt < 1e-3 and d1 >= 1e-3:
        print("           判定：R·x 正确（Rᵀ·x 不满足等变性 ⟹ 参照的 Rᵀ 形式可疑）")
    elif d1 < 1e-3 and d1_alt < 1e-3:
        print("           判定：两者都满足（可能本例 TBN 近似正交，区分度不足）")
    else:
        print("           判定：两者都不满足 ⟹ 问题不在乘法顺序，而在更外层")

    print("\n  [I9] 正交化对照：把 TBN 做 Gram-Schmidt 后重新绑定")
    tbn_o = _orthonormalize(tbn)
    gram_o = tbn_o.transpose(-1, -2) @ tbn_o
    eye_o = torch.eye(3, device=dev).expand_as(gram_o)
    orth_o = (gram_o - eye_o).abs().amax(dim=(-1, -2))[0]
    _line("I9a 正交化后 max|RᵀR-I|", f"{orth_o.max().item():.3e}",
          bool(orth_o.max() < 1e-4))

    orth_mine = torch.matmul(tbn_o[:, fid].transpose(-1, -2),
                             xyz_tan.unsqueeze(-1)).squeeze(-1) + off_formula
    d_before = (out1 - theirs).abs().max().item()
    d_after = (orth_mine - theirs).abs().max().item()
    _line("I9b 正交化前 与参照 max|Δ|", f"{d_before:.3e}")
    _line("I9c 正交化后 与参照 max|Δ|", f"{d_after:.3e}", bool(d_after < d_before),
          "（通过 ⟹ 非正交 TBN 确实污染了变换）")

    print("\n" + "=" * 78)
    print("三、I5/I6 CUDA 算子的直接对照")
    print("=" * 78)
    print("   （I5：TBN=I、局部 xyz=0 时，算子输出应精确等于 Σ baryᵢ·vᵢ）")
    _probe_operator(tri, bary, fid, xyz_tan, off_formula, dev)

    print("\n" + "=" * 78)
    print("四、与参照实现的差异分布（判断长尾来源，不作为对错判据）")
    print("=" * 78)

    err = (out1 - theirs).abs().amax(dim=-1)[0]
    for q_ in (0.5, 0.9, 0.99, 0.999, 1.0):
        print(f"    分位 {q_:>5}: {torch.quantile(err, q_).item():.3e}")
    span = float((mesh.max() - mesh.min()).item())
    print(f"    均值 = {err.mean().item():.3e}   中位数 = {err.median().item():.3e}")
    print(f"    网格跨度 = {span:.4f}，最大误差占跨度 "
          f"{100 * err.max().item() / max(span, 1e-12):.1f}%")

    k = min(top, int(err.numel()))
    print(f"\n  最差 {k} 个高斯：")
    for g in torch.topk(err, k).indices.tolist():
        shear = float((R[0, g].T @ R[0, g] - torch.eye(3, device=dev)).abs().max())
        print(f"    g={g:<6} err={err[g].item():.3e} face={int(fid[g]):<6}"
              f" bary_min={float(bary[g].min()):.2e} TBN剪切={shear:.2e}")

    big = err > 1e-3
    nb = int(big.sum())
    print(f"\n    err > 1e-3 的比例 = {100.0 * float(big.float().mean()):.2f}%  ({nb} 个)")
    if nb:
        print(f"    离群高斯中 TBN 剪切 > 1e-4 的比例 = "
              f"{float((orth[big] > 1e-4).float().mean()):.3f}")
        print(f"    离群高斯中 bary_min < 1e-3 的比例 = "
              f"{float((bary[big].min(-1).values < 1e-3).float().mean()):.3f}")
        print(f"    离群高斯涉及的面数 = {int(fid[big].unique().numel())} / {tri.shape[1]}")
    return 0


def _probe_operator(tri, bary, fid, xyz_tan, off_formula, dev) -> None:
    """直接调用 mesh_binding，判定其输出等于 Rᵀ·x 还是 R·x，以及是否等于公式值。"""
    import torch

    if dev == "cpu":
        # 该算子是 CUDA 内核：在 CPU 张量上**静默返回全零**（不报错），
        # 因此 CPU 下这些对照没有意义。同类问题也出现在 linear_blending 上。
        print("  [  --  ] dry-run(CPU)：跳过算子对照 —— CUDA 算子在 CPU 张量上静默返回全零")
        print("           （真实运行请在 GPU 上执行本脚本）")
        return

    try:
        from diff_gaussian_rasterization import mesh_binding
    except Exception as e:                                  # pragma: no cover
        print(f"  [  --  ] 无法导入 mesh_binding：{type(e).__name__}: {e}")
        return

    def eye_like():
        return torch.eye(3, device=dev).reshape(1, 1, 3, 3).expand(
            tri.shape[0], tri.shape[1], 3, 3).contiguous()

    # ⚠️ 该算子要求索引为 int32；int64 会抛 "expected scalar type Int but found Long"
    fid32 = fid.to(torch.int32).contiguous()

    try:
        fv1 = tri[:, :1].contiguous()
        tb1 = torch.eye(3, device=dev).reshape(1, 1, 3, 3).contiguous()
        one_fid = torch.zeros(1, dtype=torch.int32, device=dev).contiguous()
        z1 = torch.zeros(1, 1, 3, device=dev).contiguous()
        q1 = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=dev).contiguous()

        for label, b3 in [("bary=[1,0,0]", [1.0, 0.0, 0.0]),
                          ("bary=[0,1,0]", [0.0, 1.0, 0.0]),
                          ("bary=[1/3]*3", [1 / 3, 1 / 3, 1 / 3])]:
            bb = torch.tensor([b3], device=dev, dtype=torch.float32).contiguous()
            got, _ = mesh_binding(z1, q1, fv1, tb1, bb, one_fid)
            want = (fv1[0, 0] * bb[0].unsqueeze(-1)).sum(0)
            dd = (got[0, 0] - want).abs().max().item()
            _line(f"I5 算子 == Σ baryᵢvᵢ {label}", f"max|Δ| = {dd:.3e}",
                  bool(dd < 1e-6))
            if dd >= 1e-6:
                print(f"           算子={_fmt(got[0, 0])}   公式={_fmt(want)}")

        n = int(xyz_tan.shape[1])
        z_all = torch.zeros(1, n, 3, device=dev).contiguous()
        q_all = q1.repeat(1, n, 1).contiguous()
        out_op, _ = mesh_binding(z_all, q_all, tri.contiguous(), eye_like(),
                                 bary.contiguous(), fid32)
        d_off = (out_op - off_formula).abs().max().item()
        _line("I6 全量：算子offset == 公式", f"max|Δ| = {d_off:.3e}",
              bool(d_off < 1e-5))

        theta = 0.7
        c, sn = float(torch.cos(torch.tensor(theta))), float(torch.sin(torch.tensor(theta)))
        Rz = torch.tensor([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]],
                          device=dev).reshape(1, 1, 3, 3).contiguous()
        loc = torch.tensor([[[0.0, 0.0, 1.0]]], device=dev).contiguous()
        b0 = torch.tensor([[1.0, 0.0, 0.0]], device=dev).contiguous()
        got, _ = mesh_binding(loc, q1, fv1, Rz, b0, one_fid)
        got_vec = got[0, 0] - fv1[0, 0, 0]
        unit = torch.tensor([0.0, 0.0, 1.0], device=dev)
        d_T = (got_vec - Rz[0, 0].T @ unit).abs().max().item()
        d_R = (got_vec - Rz[0, 0] @ unit).abs().max().item()
        _line("I6 非对称TBN：Rᵀ·x vs R·x",
              f"对Rᵀ偏差={d_T:.2e} 对R偏差={d_R:.2e}",
              bool(d_T < 1e-5 or d_R < 1e-5))
        print(f"           算子输出 = {_fmt(got_vec)}")
        print(f"           Rᵀ·x     = {_fmt(Rz[0, 0].T @ unit)}")
        print(f"           R·x      = {_fmt(Rz[0, 0] @ unit)}  "
              f"（两者相等 ⟺ R 正交）")
    except Exception as e:
        print(f"  [  --  ] mesh_binding 调用失败：{type(e).__name__}: {str(e)[:120]}")


def main() -> int:
    args = parse_args()

    if args.dry_run:
        print("[dry-run] CPU 合成数据，仅验证脚本自身（不访问 GPU / 模型 / 数据集）")
        d = load_fake_data(device="cpu")
        print(f"[dry-run] N={int(d.fid.numel())}  F={int(d.faces.shape[0])}  "
              f"V={int(d.mesh.shape[1])}")
        return analyse(d, args.top)

    import torch

    if not torch.cuda.is_available():
        print("[error] 需要 GPU（可用 --dry-run 在 CPU 上验证脚本本身）")
        return 1

    d = load_real_data(args)
    print(f"\n[info] 数据：{d.label}  N={int(d.fid.numel())}  F={int(d.faces.shape[0])}")
    return analyse(d, args.top)


if __name__ == "__main__":
    raise SystemExit(main())
