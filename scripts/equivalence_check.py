#!/usr/bin/env python3
"""P1 数值等价验收门 —— 与参照实现（RGBAvatar）逐阶段比对。

判据（见 docs/CONVENTIONS.md §6.1）：

- 中间属性（权重 / 混合结果 / 绑定后的 xyz 与 rotation）：`max|Δ| < 1e-5`
- 最终渲染图 `color` / `alpha`：**`PSNR > 60 dB` 或 `max|Δ| < 1e-3`**

比对是**分层**的，因此一旦失败能直接定位到哪一级出现偏差，而不是只看到"图不一样"。

⚠️ 需要可用的 NVIDIA GPU（光栅化器与 nvdiffrast 都需要）。

用法::

    conda activate live3dgs
    python scripts/equivalence_check.py                 # 全部取配置默认值
    python scripts/equivalence_check.py --frames 3      # 只比对 3 帧
    python scripts/equivalence_check.py --skip-render   # 不加载光栅化

所有路径与模型结构来自 `configs/system.yaml` 与 `configs/render.yaml`
（见 docs/CONFIG.md）；命令行只做覆盖。
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

# ⚠️ 必须早于任何可能触发 chumpy 的导入（见 docs/ENVIRONMENT.md §3.10）
import live3dgsavatar  # noqa: E402, F401  导入即施加 numpy 兼容补丁
from live3dgsavatar.config import (  # noqa: E402
    Config, load_config, reference_model_dir, resolve_model_ply)

ATOL_ATTR = 1e-5
PSNR_DB = 60.0
ATOL_IMAGE = 1e-3


def _normalize_argv(argv: list[str]) -> list[str]:
    """把 RGBAvatar 风格的下划线参数名归一化成本项目的短横线写法。

    参照实现的 CLI 用 `--work_name`（见 `RGBAvatar/render.py`），
    本项目统一用 `--work-name`。为了让两边的命令可以直接互相复制，
    这里把 `--a_b` 归一化为 `--a-b`，**两种写法都接受**。
    """
    out = []
    for token in argv:
        if token.startswith("--") and "_" in token:
            name, sep, value = token.partition("=")
            if name.count("-") == 2 and "_" in name:      # 形如 --work_name
                token = name.replace("_", "-") + (sep + value if sep else "")
        out.append(token)
    return out


def parse_args() -> argparse.Namespace:
    """命令行参数 = **对配置文件的覆盖层**（默认量集中在 configs/）。"""
    p = argparse.ArgumentParser(
        description="P1 数值等价验收门",
        epilog="默认值来自 configs/*.yaml；用 python scripts/show_config.py 查看。")
    p.add_argument("--config-dir", type=Path, default=None)
    p.add_argument("--reference", type=Path, default=None,
                   help="参照仓库（只读），覆盖 paths.reference_root")
    p.add_argument("--subject", default=None,
                   help="人物名 / 数据集主体名，同时也是模型的一级目录名（默认 configs/system.yaml 的 subject）")
    p.add_argument("--work-name", default=None,
                   help="工作名（默认 configs/system.yaml 的 work_name）")
    p.add_argument("--ply", type=Path, default=None,
                   help="预训练 model.ply，覆盖 paths.model_ply")
    p.add_argument("--data", type=Path, default=None,
                   help="数据集目录，覆盖 <data_root>/<subject>")
    p.add_argument("--frames", type=int, default=None)
    p.add_argument("--tex-size", type=int, default=None)
    p.add_argument("--num-basis-in", type=int, default=None)
    p.add_argument("--num-basis-blend", type=int, default=None)
    p.add_argument("--mlp-hidden", type=int, nargs="+", default=None)
    p.add_argument("--skip-render", action="store_true",
                   help="只比对中间属性（不加载 nvdiffrast 光栅化）")
    p.add_argument("--out", type=Path, default=None,
                   help="报告输出目录；默认 <output_dir>/equivalence")
    raw = p.parse_args(_normalize_argv(sys.argv[1:]))
    return _finalize(raw, load_config(raw.config_dir))


def _finalize(args: argparse.Namespace, cfg: Config) -> argparse.Namespace:
    """把命令行覆盖合并进配置，并解析出本脚本要用的全部值。"""
    if args.subject is not None:
        cfg.set("subject", args.subject)
    if args.work_name is not None:
        cfg.set("work_name", args.work_name)
    if args.reference is not None:
        cfg.set("paths.reference_root", Path(args.reference).expanduser().resolve())
    if args.ply is not None:
        cfg.set("paths.model_ply", Path(args.ply).expanduser().resolve())
    if args.frames is not None:
        cfg.set("render.frames", args.frames)
    for cli, key in (("tex_size", "model.network.tex_size"),
                     ("num_basis_in", "model.network.num_basis_in"),
                     ("num_basis_blend", "model.network.num_basis_blend"),
                     ("mlp_hidden", "model.network.mlp_hidden")):
        value = getattr(args, cli)
        if value is not None:
            cfg.set(key, value)

    subject = str(cfg.subject)
    work_name = str(cfg.work_name)
    ref_root = cfg.paths.reference_root
    if ref_root is None:
        raise SystemExit("[error] 缺少 paths.reference_root（参照仓库根目录）")
    # 等价门两边用**各自**的模型：core 用本项目 models/，参照用它自己的 output/
    model_ply = resolve_model_ply(cfg)
    ref_dir = reference_model_dir(cfg)
    ref_ply = (ref_dir / "model.ply") if ref_dir is not None else model_ply
    data_dir = args.data
    if data_dir is None:
        data_dir = Path(cfg.paths.data_root) / subject
    out_dir = args.out
    if out_dir is None:
        out_dir = Path(cfg.paths.output_dir) / "equivalence"

    args.subject = subject
    args.work_name = work_name
    args.reference = Path(ref_root)
    args.ply = Path(model_ply)
    args.ref_ply = Path(ref_ply)
    args.data = Path(data_dir)
    # 参照的 FLAME 路径依赖其 cwd，故在 chdir 之前必须绝对化
    args.out = Path(out_dir).expanduser().resolve()
    args.frames = int(cfg.get("render.frames", 1))
    args.tex_size = int(cfg.get("model.network.tex_size", 256))
    args.num_basis_in = int(cfg.get("model.network.num_basis_in", 129))
    args.num_basis_blend = int(cfg.get("model.network.num_basis_blend", 20))
    hidden = cfg.get("model.network.mlp_hidden")
    args.mlp_hidden = list(hidden) if hidden else []
    args.split = str(cfg.runtime.split)
    args.cfg = cfg

    if args.frames != -1 and args.frames < 1:
        raise SystemExit(f"[error] --frames 必须 ≥ 1 或 -1，实际 {args.frames}")
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
    from live3dgsavatar.data.reference import build_reference, reference_uv_rast
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
        reference_root=args.reference, data_dir=args.data,
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
