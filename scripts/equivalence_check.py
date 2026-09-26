#!/usr/bin/env python3
"""P1 数值等价验收门 —— 与参照实现（RGBAvatar）逐阶段比对。

判据（见 docs/CONVENTIONS.md §6.1）：

- 中间属性（权重 / 混合结果 / 绑定后的 xyz 与 rotation）：`max|Δ| < 1e-5`
- 最终渲染图 `color` / `alpha`：**`PSNR > 60 dB` 或 `max|Δ| < 1e-3`**

比对是**分层**的，因此一旦失败能直接定位到哪一级出现偏差，而不是只看到"图不一样"。

⚠️ 需要可用的 NVIDIA GPU（光栅化器与 nvdiffrast 都需要）。

用法::

    conda activate live3dgs
    python scripts/equivalence_check.py \
        --ply  /home/crh/Projects/RGBAvatar/output/duda/test/model.ply \
        --data /home/crh/Datasets/INSTA/duda \
        --frames 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT / "tests"))     # 复用 tests/ 下的支撑模块
sys.path.insert(0, str(SRC_ROOT))

ATOL_ATTR = 1e-5
PSNR_DB = 60.0
ATOL_IMAGE = 1e-3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P1 数值等价验收门")
    p.add_argument("--reference", type=Path,
                   default=Path("/home/crh/Projects/RGBAvatar"),
                   help="RGBAvatar 参照仓库（只读）")
    p.add_argument("--ply", type=Path, required=True, help="预训练 model.ply")
    p.add_argument("--data", type=Path, required=True, help="数据集目录")
    p.add_argument("--frames", type=int, default=3, help="比对帧数（默认 3）")
    p.add_argument("--tex-size", type=int, default=256)
    p.add_argument("--num-basis-in", type=int, default=129)
    p.add_argument("--num-basis-blend", type=int, default=20)
    p.add_argument("--mlp-hidden", type=int, nargs="*", default=[128, 128])
    p.add_argument("--skip-render", action="store_true",
                   help="只比对中间属性（不加载 nvdiffrast 光栅化）")
    p.add_argument("--out", type=Path, default=Path("output/equivalence"))
    args = p.parse_args()

    # chdir 到参照仓库之前必须绝对化（参照的 FLAME 路径依赖其 cwd）
    for name in ("reference", "ply", "data", "out"):
        setattr(args, name, Path(getattr(args, name)).expanduser().resolve())
    return args

def _preflight(args) -> list[str]:
    """不依赖 GPU 的前置检查，避免"跑到第 5 步才发现路径不对"。"""
    problems: list[str] = []
    if not args.reference.is_dir():
        problems.append(f"参照仓库不存在：{args.reference}")
    elif not (args.reference / "camera" / "camera.py").exists():
        problems.append(f"{args.reference} 不像 RGBAvatar 仓库（缺 camera/camera.py）")
    if not args.ply.exists():
        problems.append(f"预训练模型不存在：{args.ply}")
    if not args.data.is_dir():
        problems.append(f"数据集目录不存在：{args.data}")
    else:
        for sub in ("images", "checkpoint"):
            if not (args.data / sub).is_dir():
                problems.append(f"数据集缺子目录：{args.data / sub}")
    if args.reference.is_dir():
        flame = args.reference / "data" / "FLAME2020" / "generic_model.pkl"
        if not flame.exists():
            problems.append(
                f"参照仓库缺 FLAME 模型：{flame}\n"
                f"        提示：参照实现的 FlameConfig 用相对路径，因此必须放在参照仓库内")
    return problems


def main() -> int:
    args = parse_args()

    print(f"[info] 参照仓库 : {args.reference}")
    print(f"[info] 模型     : {args.ply}")
    print(f"[info] 数据集   : {args.data}")
    print(f"[info] 输出     : {args.out}")

    print("\n[0/6] 前置检查（不需 GPU）…")
    problems = _preflight(args)
    if problems:
        for item in problems:
            print(f"  \033[1;31m[error]\033[0m {item}")
        return 1
    print("       路径与资源就绪")

    import torch

    if not torch.cuda.is_available():
        print("  \033[1;31m[error]\033[0m 等价门需要 GPU：torch.cuda.is_available() 为 False")
        print("          请在可访问 GPU 设备的终端中运行。")
        return 1

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

    # ------------------------------------------------------------ 加载两侧 --
    print("\n[1/6] 加载参照实现…")
    ref = build_reference(
        reference_root=args.reference, src_root=SRC_ROOT, data_dir=args.data,
        ply_path=args.ply, tex_size=args.tex_size, num_basis_in=args.num_basis_in,
        num_basis_blend=args.num_basis_blend, mlp_hidden=tuple(args.mlp_hidden),
    )
    print(f"       参照高斯数 = {ref.model.num_gaussian}, "
          f"分辨率 = {ref.dataset.image_width}x{ref.dataset.image_height}")

    print("[2/6] 加载本项目 core/…")
    cfg = AvatarConfig(tex_size=args.tex_size, num_basis_in=args.num_basis_in,
                       num_basis_blend=args.num_basis_blend,
                       mlp_hidden=tuple(args.mlp_hidden), use_weight_proj=True)
    avatar = load_ply(args.ply, config=cfg, device="cuda")
    print(f"       本项目高斯数 = {avatar.num_gaussians}, K = {avatar.num_basis}")

    # ---------------------------------------------------------- 1. 参数层 --
    compare_parameters(report, avatar, ref.model, args.ply)

    # ------------------------------------------------------ 2. 绑定构建层 --
    print("[3/6] 比对 UV 绑定构建…")
    compare_binding(
        report,
        uv_builder=lambda: build_binding(
            uvs=ref.flame_model.uvs.to(torch.float32),
            uv_faces=ref.flame_model.uv_faces.to(torch.int32),
            tex_size=args.tex_size, glctx=ref.model.glctx, device="cuda"),
        ref_uv_rast=lambda: reference_uv_rast(args.reference, ref),
        ref_model=ref.model,
    )

    # ---------------------------------------------------------- 3. 混合层 --
    n = min(args.frames, len(ref.dataset))
    mesh_ref = ref.dataset.mesh_verts[:n].cuda()
    bw = ref.dataset.blend_weight[:n].cuda()
    mesh_mine = Mesh(verts=mesh_ref.to(torch.float32),
                     faces=ref.template_faces.to(torch.int32),
                     uvs=ref.template_uvs, uv_faces=ref.template_uv_faces)
    camera = Camera.from_intrinsics_extrinsics(
        K=ref.dataset.camera_intri, R=ref.dataset.camera_extri[:3, :3],
        T=ref.dataset.camera_extri[:3, 3],
        width=ref.dataset.image_width, height=ref.dataset.image_height).to("cuda")

    print(f"[4/6] 比对驱动权重与混合属性（{n} 帧）…")
    compare_blend(report, avatar, ref.model, bw)

    # ---------------------------------------------------------- 4. 绑定层 --
    print("[5/6] 比对绑定后的世界空间高斯…")
    gs_mine, gs_ref = compare_deformed(report, avatar, ref.model, mesh_ref, mesh_mine, bw)

    # ---------------------------------------------------------- 5. 渲染层 --
    if args.skip_render:
        print("[6/6] 跳过渲染比对（--skip-render）")
    else:
        print("[6/6] 比对渲染结果…")
        bg = torch.zeros(3, dtype=torch.float32, device="cuda")
        compare_render(report, args.reference, camera, ref.camera, gs_mine, gs_ref, bg)

    # ---------------------------------------------------------------- 落盘 --
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "equivalence.json").write_text(
        json.dumps(report.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {args.out / 'equivalence.json'}")
    return report.dump()


if __name__ == "__main__":
    raise SystemExit(main())
