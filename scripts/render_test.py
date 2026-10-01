#!/usr/bin/env python3
"""统一渲染测试 —— 一份脚本给出三类结论（**需要 GPU**）。

对每个渲染帧 `i`，同时取得三方图像：

    ① core     本项目 `core/` 的渲染
    ② reference 参照实现（RGBAvatar）的渲染       ← 同一份 `model.ply`
    ③ dataset   数据集原图（GT）

并报告：

| 对比 | 指标 | 含义 |
|---|---|---|
| core vs reference | PSNR / `max\\|Δ\\|` | **复现度**：本项目是否等价于参照 |
| core vs dataset | PSNR（全图 + mask 内） | **重建质量**：模型渲染得对不对 |
| reference vs dataset | PSNR（同样两项） | 参照的同一指标，作为对照基准 |

性能：core 与 reference 的**单帧耗时 / FPS / 峰值显存**并排给出。

数据集原图是 **RGBA**：`A` 是人像 mask、背景为黑。
因此除全图 PSNR 外，另给 **mask 内 PSNR**（`A > 250` 的像素），后者才反映人像区域质量。

产物::

    output/render_test/images/          core 渲染的 PNG
    output/render_test/reference/       参照渲染的 PNG
    output/render_test/diff/             可选：并排对比图（--dump-diff）
    output/render_test/report.json       全部指标

用法::

    conda activate live3dgs
    python scripts/render_test.py --frames 20
    python scripts/render_test.py --frames -1 --batch-size 10 --dump-diff
    python scripts/render_test.py --dry-run          # CPU 自检，不需 GPU
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT / "src"))

# ⚠️ 必须在**导入任何可能触发 chumpy 的模块之前**打补丁。
#    `live3dgsavatar/__init__.py` 导入即施加 numpy 兼容补丁；
#    而参照侧的 `FLAME.__init__` 会 pickle.load → import chumpy。
#    若等 `main()` 里才 import 本项目，顺序就反了（曾因此报
#    `cannot import name 'int' from 'numpy'`）。
import live3dgsavatar  # noqa: E402, F401  （勿删：导入即生效）
from live3dgsavatar.config import (  # noqa: E402
    Config, load_config, reference_model_dir, resolve_model_ply)


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
    """命令行参数 = **对配置文件的覆盖层**。

    任何参数不传时都用 `configs/*.yaml` 的值；这里不设业务默认量
    （默认量集中在配置文件，见 docs/CONFIG.md）。
    """
    p = argparse.ArgumentParser(
        description="统一渲染测试（core vs 参照 vs 数据集原图）",
        epilog="默认值来自 configs/system.yaml 与 configs/render.yaml；"
               "用 python scripts/show_config.py 查看实际生效值。")
    p.add_argument("--config-dir", type=Path, default=None,
                   help="配置目录；默认仓库根的 configs/")
    p.add_argument("--subject", default=None,
                   help="人物名 / 数据集主体名，同时也是模型的一级目录名（默认 configs/system.yaml 的 subject）")
    p.add_argument("--work-name", default=None,
                   help="工作名（默认 configs/system.yaml 的 work_name，如 test）")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="产物根目录；默认 <paths.output_dir>/render_test")
    p.add_argument("--data-root", type=Path, default=None,
                   help="数据集根目录，覆盖 paths.data_root")
    p.add_argument("--reference", type=Path, default=None,
                   help="参照仓库（只读），覆盖 paths.reference_root")
    p.add_argument("--ply", type=Path, default=None,
                   help="模型 .ply，覆盖 paths.model_ply")
    p.add_argument("--frames", type=int, default=None,
                   help="渲染帧数；-1 表示全部（默认 render.frames）")
    p.add_argument("--batch-size", type=int, default=None,
                   help="**两边共用**的批大小（默认 render.batch_size）")
    p.add_argument("--tex-size", type=int, default=None,
                   help="UV 纹理边长（默认 model.network.tex_size）")
    p.add_argument("--num-basis-in", type=int, default=None)
    p.add_argument("--num-basis-blend", type=int, default=None)
    p.add_argument("--mlp-hidden", type=int, nargs="+", default=None)
    p.add_argument("--white-bg", action="store_true", default=None,
                   help="用白底渲染（默认为配置的 render.background）")
    p.add_argument("--dump-diff", action="store_true", default=None,
                   help="额外输出并排对比图（默认 test.dump_diff）")
    p.add_argument("--sweep-batch", type=int, nargs="+", default=None,
                   metavar="B",
                   help="额外用这些批大小各跑一次参照，量化批大小的影响"
                        "（参照的 render_gs_batch 是串行循环，理论上每帧成本不变）")
    p.add_argument("--skip-reference", action="store_true",
                   help="只测 core vs dataset（不加载参照实现）")
    p.add_argument("--dry-run", action="store_true", help="CPU 自检，不访问数据集/GPU")
    return p.parse_args(_normalize_argv(sys.argv[1:]))


def _validate(args: argparse.Namespace) -> None:
    """参数校验：非法值必须在开跑前报错。

    这类错误若留到深处会给出难读的堆栈（`range() arg 3 must not be zero`），
    或更糟 —— 静默跑 0 帧然后报告一份空结果。
    """
    if args.batch_size < 1:
        raise SystemExit(f"[error] --batch-size 必须 ≥ 1，实际 {args.batch_size}")
    if args.frames != -1 and args.frames < 1:
        raise SystemExit(f"[error] --frames 必须 ≥ 1 或 -1（全部），实际 {args.frames}")
    if args.sweep_batch is not None and any(b < 1 for b in args.sweep_batch):
        raise SystemExit(f"[error] --sweep-batch 全部必须 ≥ 1，实际 {args.sweep_batch}")
    if args.tex_size < 1:
        raise SystemExit(f"[error] --tex-size 必须 ≥ 1，实际 {args.tex_size}")
    if len(args.mlp_hidden) == 0:
        raise SystemExit("[error] --mlp-hidden 不能为空列表（写 null 表示不用 MLP）")


def _finalize(args: argparse.Namespace, cfg: Config) -> argparse.Namespace:
    """把命令行覆盖合并进配置，并解析出本脚本要用的全部值。

    优先级：命令行 > 环境变量 > configs/*.yaml > 内置兜底（见 load_config）。
    """
    # 命令行 → 配置（None 表示用户没传，保留配置值）
    if args.subject is not None:
        cfg.set("subject", args.subject)
    if args.work_name is not None:
        cfg.set("work_name", args.work_name)
    if args.data_root is not None:
        cfg.set("paths.data_root", Path(args.data_root).expanduser().resolve())
    if args.reference is not None:
        cfg.set("paths.reference_root",
                Path(args.reference).expanduser().resolve())
    if args.ply is not None:
        cfg.set("paths.model_ply", Path(args.ply).expanduser().resolve())
    if args.frames is not None:
        cfg.set("render.frames", args.frames)
    if args.batch_size is not None:
        cfg.set("render.batch_size", args.batch_size)
    if args.tex_size is not None:
        cfg.set("model.network.tex_size", args.tex_size)
    if args.num_basis_in is not None:
        cfg.set("model.network.num_basis_in", args.num_basis_in)
    if args.num_basis_blend is not None:
        cfg.set("model.network.num_basis_blend", args.num_basis_blend)
    if args.mlp_hidden is not None:
        cfg.set("model.network.mlp_hidden", args.mlp_hidden)
    if args.white_bg:
        cfg.set("render.background", [1.0, 1.0, 1.0])
    if args.dump_diff:
        cfg.set("test.dump_diff", True)

    subject = str(cfg.subject)
    work_name = str(cfg.work_name)
    data_root = Path(cfg.paths.data_root)
    ref_root = cfg.paths.reference_root
    out_root = args.output_dir
    if out_root is None:
        out_root = Path(cfg.paths.output_dir) / "render_test"
    else:
        out_root = Path(out_root).expanduser().resolve()

    # 模型路径：本项目 models/ 优先，找不到再回退参照仓库（见 docs/CONFIG.md）
    model_ply = resolve_model_ply(cfg)

    data_dir = data_root / subject
    args.subject = subject
    args.work_name = work_name
    args.data_root = data_root
    args.data = data_dir
    args.reference = Path(ref_root) if ref_root is not None else None
    args.ply = model_ply
    args.output_dir = out_root
    args.gt_dir = data_dir / str(cfg.get("paths.image_subdir", "images"))
    args.frames = int(cfg.get("render.frames", 1))
    args.batch_size = int(cfg.get("render.batch_size", 1))
    args.tex_size = int(cfg.get("model.network.tex_size", 256))
    args.num_basis_in = int(cfg.get("model.network.num_basis_in", 129))
    args.num_basis_blend = int(cfg.get("model.network.num_basis_blend", 20))
    hidden = cfg.get("model.network.mlp_hidden")
    args.mlp_hidden = list(hidden) if hidden else []
    args.background = [float(x) for x in cfg.get("render.background",
                                                 [0.0, 0.0, 0.0])]
    args.sh_degree = int(cfg.get("render.sh_degree", 0))
    args.scaling_modifier = float(cfg.get("render.scaling_modifier", 1.0))
    args.use_weight_proj = bool(cfg.get("model.network.use_weight_proj", True))
    args.split = str(cfg.runtime.split)
    args.device = str(cfg.runtime.device)
    args.dump_diff = bool(cfg.get("test.dump_diff", False))
    args.cfg = cfg

    _validate(args)
    return args


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """`[0,255]` uint8 图像的 PSNR；完全相同返回 inf。"""
    a = a.astype(np.float64) / 255.0
    b = b.astype(np.float64) / 255.0
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))


def psnr_masked(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
    """只在 `mask` 为真的像素上算 PSNR；mask 为空返回 None。"""
    if not bool(mask.any()):
        return None
    a = a[mask].astype(np.float64) / 255.0
    b = b[mask].astype(np.float64) / 255.0
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).max())


class Accum:
    """累积一组标量，最后给出中位/最小/均值。"""

    def __init__(self) -> None:
        self.values: list[float] = []

    def add(self, v: float | None) -> None:
        if v is not None:
            self.values.append(v)

    def summary(self) -> dict | None:
        """统计量只在**有限值**上计算；`inf`（完全相同）单独计数。

        否则 `inf` 会污染中位数/最小值（例如自比时中位会变成 inf）。
        """
        if not self.values:
            return None
        finite = [v for v in self.values if np.isfinite(v)]
        n_inf = len(self.values) - len(finite)
        if not finite:
            return {"n": len(self.values), "n_infinite": n_inf,
                    "median": None, "min": None, "mean": None}
        return {
            "n": len(self.values),
            "n_infinite": n_inf,
            "median": float(np.median(finite)),
            "min": float(np.min(finite)),
            "mean": float(np.mean(finite)),
        }


# ------------------------------------------------------------------ 渲染 --


def render_reference(args, scene, frames: list[int]):
    """用参照实现渲染并返回 `{frame_index: HxWx3 uint8}`。"""
    from equivalence.reference_pipeline import build_reference, reference_render

    # 参照实现用**它自己的同名模型**（output/<subject>/<work_name>/model.ply），
    # 本项目 core 用 models/ 下的模型；两者本应内容一致，此处刻意分开取，
    # 以免"对照"变成"自己跟自己比"。
    ref_dir = reference_model_dir(args.cfg)
    ref_ply = (ref_dir / "model.ply") if ref_dir is not None else None
    if ref_ply is None or not ref_ply.exists():
        raise SystemExit(
            f"[error] 参照模型不存在：{ref_ply}\n"
            "（参照实现渲染需要它自己的模型；换模型时请同步两边，"
            "或用 --skip-reference 只测本项目）")
    ref = build_reference(
        reference_root=args.reference, src_root=REPO_ROOT / "src",
        data_dir=args.data, ply_path=ref_ply, tex_size=args.tex_size,
        num_basis_in=args.num_basis_in, num_basis_blend=args.num_basis_blend,
        mlp_hidden=tuple(args.mlp_hidden), split=args.split,
    )
    bg = torch.tensor(args.background, dtype=torch.float32, device="cuda")

    images: dict[int, np.ndarray] = {}
    per_frame_ms: list[float] = []

    def run_batch(idx: list[int]):
        mesh = scene.frames_tensor(idx, device="cuda")
        bw = scene.weights_tensor(idx, device="cuda")
        with torch.no_grad():
            gs = ref.model.gaussian_deform_batch(mesh, bw)
            pkg = reference_render(args.reference, ref.camera, bg, gs)
        return pkg["color"]

    # 预热
    run_batch(frames[:1])
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for start in range(0, len(frames), args.batch_size):
        chunk = frames[start:start + args.batch_size]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        color = run_batch(chunk)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        per_frame_ms.extend([dt / len(chunk)] * len(chunk))
        arr = (color.clamp(0, 1).permute(0, 2, 3, 1) * 255).byte().cpu().numpy()
        for j, i in enumerate(chunk):
            images[i] = arr[j]

    return images, {
        "mean_ms_per_frame": float(np.mean(per_frame_ms)),
        "median_ms_per_frame": float(np.median(per_frame_ms)),
        "fps": 1000.0 / float(np.mean(per_frame_ms)),
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "batch_size": args.batch_size,
        "num_gaussian": int(ref.model.num_gaussian),
    }


def render_core(args, scene, frames: list[int], out_dir: Path, save: bool = True):
    """用本项目 `core/` 渲染。

    **批大小与参照对齐**（都用 `args.batch_size`）。两边都是「批内逐帧」的语义
    ——参照的 `render_gs_batch` 也是个 `for i in range(bs)` 串行循环 ——
    所以对齐批大小后，计时口径才真正一致。
    """
    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.render import SimpleRasterizer
    from live3dgsavatar.core.types import Camera, Mesh

    cfg = AvatarConfig(tex_size=args.tex_size, num_basis_in=args.num_basis_in,
                       num_basis_blend=args.num_basis_blend,
                       mlp_hidden=tuple(args.mlp_hidden),
                       use_weight_proj=args.use_weight_proj)
    avatar = load_ply(args.ply, config=cfg, device="cuda")
    camera = Camera.from_intrinsics_extrinsics(
        K=scene.K, R=scene.R, T=scene.T, width=scene.width, height=scene.height).to("cuda")
    bg = torch.tensor(args.background, dtype=torch.float32, device="cuda")
    rasterizer = SimpleRasterizer()

    def run_batch(idx: list[int]):
        verts = scene.frames_tensor(idx, device="cuda")           # [B, V, 3]
        weights = scene.weights_tensor(idx, device="cuda")        # [B, D]
        mesh = Mesh(verts=verts, faces=scene.faces,
                    uvs=scene.uvs, uv_faces=scene.uv_faces)
        with torch.no_grad():
            gs = avatar.deform(mesh, weights)
            return rasterizer.render(gs, camera, bg)

    run_batch(frames[:1])                       # 预热
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    from PIL import Image

    images: dict[int, np.ndarray] = {}
    per_frame_ms: list[float] = []
    for start in range(0, len(frames), args.batch_size):
        chunk = frames[start:start + args.batch_size]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = run_batch(chunk)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        per_frame_ms.extend([dt / len(chunk)] * len(chunk))

        arr = (out.color.clamp(0, 1).permute(0, 2, 3, 1) * 255).byte().cpu().numpy()
        for j, i in enumerate(chunk):
            images[i] = arr[j]
            if save:
                Image.fromarray(arr[j]).save(out_dir / f"{i:05d}.png")

    return images, {
        "mean_ms_per_frame": float(np.mean(per_frame_ms)),
        "median_ms_per_frame": float(np.median(per_frame_ms)),
        "fps": 1000.0 / float(np.mean(per_frame_ms)),
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "batch_size": args.batch_size,
        "num_gaussian": int(avatar.num_gaussians),
    }


# -------------------------------------------------------------- GT 读取 --


def load_gt(args, frames: list[int]):
    """读取数据集原图（RGBA）。返回 `{i: (rgb HxWx3, mask HxW bool)}`。"""
    from PIL import Image

    if not args.gt_dir.is_dir():
        return None
    files = sorted(args.gt_dir.iterdir())
    out = {}
    for i in frames:
        if i >= len(files):
            continue
        arr = np.array(Image.open(files[i]))
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[2] == 4:
            out[i] = (arr[..., :3], arr[..., 3] > 250)   # alpha 即 mask
        else:
            out[i] = (arr[..., :3], np.ones(arr.shape[:2], dtype=bool))
    return out or None


# ------------------------------------------------------------------ 主流程 --


def main() -> int:
    raw = parse_args()
    if raw.dry_run:
        # dry-run 也要走完整配置解析，才能顺带验证配置本身
        return _dry_run(_finalize(raw, load_config(raw.config_dir)))

    args = _finalize(raw, load_config(raw.config_dir))

    if not torch.cuda.is_available():
        print("[error] 需要 GPU（可用 --dry-run 在 CPU 上自检）")
        return 1
    for label, path in (("模型", args.ply), ("数据集", args.data)):
        if not path.exists():
            print(f"[error] {label}不存在：{path}")
            return 1

    print(f"[info] 本项目模型: {args.ply}")
    print(f"[info] 参照模型  : "
          f"{reference_model_dir(args.cfg) / 'model.ply' if args.reference else '（跳过）'}")
    print(f"[info] 数据集   : {args.data}")
    print(f"[info] 模型名/工作名: {args.subject} / {args.work_name}")
    print(f"[info] 参照仓库 : {args.reference}")
    print(f"[info] 输出     : {args.output_dir}")
    print(f"[info] 配置     : {len(args.cfg.as_plain())} 节，"
          f"批大小={args.batch_size}，帧数={args.frames}，设备={args.device}")
    print(f"[info] 帧数     : {'全部' if args.frames < 0 else args.frames}"
          f"{'  （跳过参照）' if args.skip_reference else ''}")

    from reference_scene import load_scene

    print("\n[1/4] 加载模板几何与相机…")
    scene = load_scene(args.cfg, frames=args.frames)
    if scene.num_frames == 0:
        print("[error] 数据集未提供任何帧")
        return 1
    if args.frames != -1 and scene.num_frames < args.frames:
        print(f"[warn] 请求 {args.frames} 帧，数据集只有 {scene.num_frames} 帧，"
              "按可用帧数执行")
    n_total = scene.num_frames
    frames = list(range(n_total))
    print(f"       {n_total} 帧，{scene.faces.shape[0]} 面，"
          f"{scene.width}x{scene.height}")

    out_dir = args.output_dir
    img_dir, ref_dir = out_dir / "images", out_dir / "reference"
    for d in (img_dir, ref_dir):
        d.mkdir(parents=True, exist_ok=True)
        for p in d.glob("*.png"):
            p.unlink()

    print(f"\n[2/4] 用 core/ 渲染 {len(frames)} 帧…")
    core_imgs, core_perf = render_core(args, scene, frames, img_dir)
    print(f"       {core_perf['mean_ms_per_frame']:.2f} ms/帧，"
          f"{core_perf['fps']:.1f} FPS，峰值 {core_perf['peak_memory_mib']:.0f} MiB")

    ref_imgs, ref_perf = None, None
    if not args.skip_reference:
        print(f"\n[3/4] 用参照实现渲染 {len(frames)} 帧（batch={args.batch_size}）…")
        ref_imgs, ref_perf = render_reference(args, scene, frames)
        from PIL import Image

        for i, arr in ref_imgs.items():
            Image.fromarray(arr).save(ref_dir / f"{i:05d}.png")
        print(f"       {ref_perf['mean_ms_per_frame']:.2f} ms/帧，"
              f"{ref_perf['fps']:.1f} FPS，峰值 {ref_perf['peak_memory_mib']:.0f} MiB")
    else:
        print("\n[3/4] 跳过参照渲染（--skip-reference）")

    sweep = None
    if args.sweep_batch:
        print(f"\n[*] 批大小扫描（参照）：{args.sweep_batch}")
        sweep = _sweep_reference(args, scene, frames)

    print("\n[4/4] 对比…")
    gt = load_gt(args, frames)
    if gt is None:
        print(f"       [warn] 找不到数据集原图（{args.gt_dir}），跳过 GT 对比")
    elif len(gt) < len(frames):
        print(f"       [warn] 数据集原图只有 {len(gt)} 张，少于要渲染的 "
              f"{len(frames)} 帧；GT 对比将只覆盖前 {len(gt)} 帧")

    acc = {
        "core_vs_ref_psnr": Accum(), "core_vs_ref_maxdiff": Accum(),
        "core_vs_gt_psnr": Accum(), "core_vs_gt_psnr_masked": Accum(),
        "ref_vs_gt_psnr": Accum(), "ref_vs_gt_psnr_masked": Accum(),
    }
    per_frame: list[dict] = []

    for i in frames:
        row: dict = {"frame": i}
        if ref_imgs is not None and i in ref_imgs:
            d = max_abs_diff(core_imgs[i], ref_imgs[i])
            p = psnr(core_imgs[i], ref_imgs[i])
            acc["core_vs_ref_psnr"].add(p)
            acc["core_vs_ref_maxdiff"].add(d)
            row.update(core_vs_ref_psnr=p, core_vs_ref_maxdiff=d)
        if gt is not None and i in gt:
            g_rgb, g_mask = gt[i]
            p = psnr(core_imgs[i], g_rgb)
            pm = psnr_masked(core_imgs[i], g_rgb, g_mask)
            acc["core_vs_gt_psnr"].add(p)
            acc["core_vs_gt_psnr_masked"].add(pm)
            row.update(core_vs_gt_psnr=p, core_vs_gt_psnr_masked=pm)
            if ref_imgs is not None and i in ref_imgs:
                p2 = psnr(ref_imgs[i], g_rgb)
                pm2 = psnr_masked(ref_imgs[i], g_rgb, g_mask)
                acc["ref_vs_gt_psnr"].add(p2)
                acc["ref_vs_gt_psnr_masked"].add(pm2)
                row.update(ref_vs_gt_psnr=p2, ref_vs_gt_psnr_masked=pm2)
        per_frame.append(row)

    if args.dump_diff and gt is not None:
        _dump_diff(out_dir / "diff", frames, core_imgs, ref_imgs, gt)

    report = {
        "subject": args.subject,
        "work_name": args.work_name,
        "ply": str(args.ply), "data": str(args.data),
        "frames": len(frames), "resolution": [scene.width, scene.height],
        "background": args.background,
        "config": args.cfg.resolved(),
        "core_perf": core_perf, "reference_perf": ref_perf,
        "reference_batch_sweep": sweep,
        "summary": {k: v.summary() for k, v in acc.items()},
        "per_frame": per_frame,
        "paths": {"images": str(img_dir),
                  "reference": str(ref_dir) if ref_imgs else None,
                  "report": str(out_dir / "report.json")},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    _print_summary(report)
    return 0



def _sweep_reference(args, scene, batch_sizes: list[int]) -> dict:
    """用不同批大小各渲染一遍参照，返回 `{批大小: 性能}`。

    参照的 `render_gs_batch` 内部是 `for i in range(bs)` 串行循环，
    因此批大小**理论上不该改变每帧成本**。若实测显著变化，
    说明差异来自 Python/分配开销的摊薄，而非算法效率。
    """
    out: dict = {}
    for bs in batch_sizes:
        if bs < 1:
            print(f"      [skip] 非法批大小 {bs}")
            continue
        sub = argparse.Namespace(**vars(args))
        sub.batch_size = bs
        # 必须喂**真实数量的帧**：每帧耗时 = 批耗时 / 批内帧数，
        # 若只给 1 帧再按 bs 除，测出来的值会被系统性低估。
        n = max(1, min(bs, scene.num_frames))
        _, perf = render_reference(sub, scene, list(range(n)))
        out[str(bs)] = perf
        print(f"      batch={bs}（{n} 帧）: {perf['mean_ms_per_frame']:.2f} ms/帧")
    return out


def _print_summary(report: dict) -> None:
    s = report["summary"]
    W = 78
    print("\n" + "=" * W)
    print(f"渲染测试报告 —— {report['subject']}/{report['work_name']}，"
          f"{report['frames']} 帧，"
          f"{report['resolution'][0]}x{report['resolution'][1]}")
    print("=" * W)

    def line(label, key, unit="", fmt=".2f"):
        v = s.get(key)
        if not v:
            return
        extra = f"（{v['n_infinite']} 帧完全相同）" if v.get("n_infinite") else ""
        if v.get("median") is None:
            print(f"  {label:<34} 全部帧完全相同" + extra)
            return
        print(f"  {label:<34} 中位 {v['median']:{fmt}} {unit}"
              f"  最小 {v['min']:{fmt}} {unit}  (n={v['n']}){extra}")

    print("\n[复现度] core  vs  参照实现（同一份 model.ply）")
    if s.get("core_vs_ref_psnr"):
        line("PSNR", "core_vs_ref_psnr", "dB")
        line("max|Δ|", "core_vs_ref_maxdiff", "/255", ".1f")
    else:
        print("  （已跳过参照渲染）")

    print("\n[重建质量] core  vs  数据集原图")
    if s.get("core_vs_gt_psnr"):
        line("PSNR（全图）", "core_vs_gt_psnr", "dB")
        line("PSNR（mask 内）", "core_vs_gt_psnr_masked", "dB")
    else:
        print("  （无 GT 图）")

    if s.get("ref_vs_gt_psnr"):
        print("\n[重建质量] 参照实现  vs  数据集原图（对照）")
        line("PSNR（全图）", "ref_vs_gt_psnr", "dB")
        line("PSNR（mask 内）", "ref_vs_gt_psnr_masked", "dB")

    c = report["core_perf"]
    r = report["reference_perf"]

    print("\n[性能]")
    if not r:
        print(f"  平均耗时 (ms/帧)          {c['mean_ms_per_frame']:>10.2f}")
        print(f"  中位耗时 (ms/帧)          {c['median_ms_per_frame']:>10.2f}")
        print(f"  吞吐 (FPS)                {c['fps']:>10.1f}")
        print(f"  峰值显存 (MiB)            {c['peak_memory_mib']:>10.0f}")
        print(f"  批大小                    {c['batch_size']:>10}")
        print(f"\n  报告：{report['paths']['report']}")
        print("=" * W)
        return

    same_bs = c["batch_size"] == r["batch_size"]
    print(f"  {'批大小':<24} {c['batch_size']:>12} {r['batch_size']:>12}"
          f"   {'（已对齐）' if same_bs else '⚠️ 未对齐'}")
    print(f"  {'平均耗时 (ms/帧)':<24} {c['mean_ms_per_frame']:>12.2f}"
          f" {r['mean_ms_per_frame']:>12.2f}")
    print(f"  {'中位耗时 (ms/帧)':<24} {c['median_ms_per_frame']:>12.2f}"
          f" {r['median_ms_per_frame']:>12.2f}")
    print(f"  {'吞吐 (FPS)':<24} {c['fps']:>12.1f} {r['fps']:>12.1f}")
    print(f"  {'峰值显存 (MiB)':<24} {c['peak_memory_mib']:>12.0f}"
          f" {r['peak_memory_mib']:>12.0f}")
    if c["mean_ms_per_frame"] > 0:
        print(f"\n  core / 参照 耗时比 = "
              f"{c['mean_ms_per_frame'] / r['mean_ms_per_frame']:.2f}×")
    if not same_bs:
        print("  ⚠️ 批大小不一致，耗时不可直接比较（用同一个 --batch-size 重跑）")

    sweep = report.get("reference_batch_sweep")
    if sweep:
        print("\n[批大小对参照每帧耗时的影响]")
        print(f"  {'批大小':<10} {'平均耗时 (ms/帧)':>18} {'吞吐 (FPS)':>12}")
        for bs, v in sweep.items():
            print(f"  {bs:<10} {v['mean_ms_per_frame']:>18.2f} {v['fps']:>12.1f}")
        vals = [v["mean_ms_per_frame"] for v in sweep.values()]
        spread = (max(vals) - min(vals)) / max(vals) * 100 if max(vals) else 0
        print(f"  → 极差 {spread:.1f}%；接近 0 说明批大小基本只摊薄 Python 开销，"
              "不是性能差异的来源")

    print(f"\n  报告：{report['paths']['report']}")
    print("=" * W)


def _dump_diff(diff_dir: Path, frames: list[int], core_imgs, ref_imgs, gt) -> None:
    """输出并排对比图：core | reference | dataset。"""
    from PIL import Image

    diff_dir.mkdir(parents=True, exist_ok=True)
    for p in diff_dir.glob("*.png"):
        p.unlink()
    for i in frames:
        if i not in core_imgs or i not in gt:
            continue
        panels = [core_imgs[i]]
        if ref_imgs is not None and i in ref_imgs:
            panels.append(ref_imgs[i])
        panels.append(gt[i][0])
        h = max(p.shape[0] for p in panels)
        sep = np.full((h, 4, 3), 255, dtype=np.uint8)
        merged = np.concatenate(
            [np.concatenate([p, sep], axis=1) for p in panels], axis=1)
        Image.fromarray(merged).save(diff_dir / f"{i:05d}.png")


def _dry_run(args) -> int:
    """CPU 自检：用真实模型 + 合成帧走通 core 渲染链路的形状与约定。"""
    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.render import SimpleRasterizer, to_kernel_matrix
    from live3dgsavatar.core.types import Camera, Mesh

    print("[dry-run] 参照的 FLAMEDataset 硬编码 .cuda()，CPU 下无法加载数据集；")
    print("          本模式只验证 core/ 链路与指标函数的正确性。")

    if not args.ply.exists():
        print(f"[error] 模型不存在：{args.ply}")
        return 1

    cfg = AvatarConfig(tex_size=args.tex_size, num_basis_in=args.num_basis_in,
                       num_basis_blend=args.num_basis_blend,
                       mlp_hidden=tuple(args.mlp_hidden),
                       use_weight_proj=args.use_weight_proj)
    avatar = load_ply(args.ply, config=cfg, device="cpu")
    print(f"  avatar      N={avatar.num_gaussians}, K={avatar.num_basis}")

    n_faces = int(avatar.binding_face_id.max()) + 1
    n_verts = int(avatar.binding_face_id.numel())
    torch.manual_seed(0)
    faces = torch.stack(
        [torch.arange(0, n_faces * 3, 3, dtype=torch.int32) % n_verts,
         torch.arange(1, n_faces * 3 + 1, 3, dtype=torch.int32) % n_verts,
         torch.arange(2, n_faces * 3 + 2, 3, dtype=torch.int32) % n_verts], dim=1)
    uvs = torch.rand(n_verts, 2)
    mesh = Mesh(verts=torch.randn(1, n_verts, 3) * 0.1, faces=faces,
                uvs=uvs, uv_faces=faces)
    gs = avatar.deform(mesh, torch.randn(1, cfg.num_basis_in) * 0.1)
    print(f"  deform      xyz={tuple(gs.xyz.shape)} space={gs.space} "
          f"有限={bool(torch.isfinite(gs.xyz).all())}")

    K = torch.tensor([[400.0, 0, 256], [0, 400.0, 256], [0, 0, 1]])
    cam = Camera.from_intrinsics_extrinsics(
        K=K, R=torch.eye(3), T=torch.tensor([0.0, 0.0, -0.5]), width=512, height=512)
    km = to_kernel_matrix(cam.full_proj[0])
    print(f"  camera      kernel 矩阵连续={km.is_contiguous()} "
          f"与转置一致={torch.equal(km, cam.full_proj[0].transpose(0, 1))}")

    # 指标函数自检（这部分不依赖 GPU，必须正确）
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = np.zeros((4, 4, 3), dtype=np.uint8)
    assert psnr(a, a) == float("inf"), "相同图 PSNR 应为 inf"

    # 只改一个像素 → 其 3 个通道各差 1.0 ⟹ Σd² = 3，MSE = 3/(4·4·3) = 1/16
    b[0, 0] = 255
    p = psnr(a, b)
    expect = 10.0 * np.log10(16.0)          # = 12.04 dB
    assert abs(p - expect) < 1e-6, (
        f"单像素全白应为 {expect:.2f} dB（MSE=1/16），实测 {p:.2f} dB")
    m_empty = np.zeros((4, 4), dtype=bool)
    assert psnr_masked(a, b, m_empty) is None, "空 mask 应返回 None"
    m_one = np.zeros((4, 4), dtype=bool)
    m_one[0, 0] = True
    pm = psnr_masked(a, b, m_one)
    assert pm is not None and abs(pm - 0.0) < 1e-9, (
        f"mask 内只有 1 个像素且二者差 255，PSNR 应为 0 dB，实测 {pm}")
    assert max_abs_diff(a, a) == 0.0
    assert max_abs_diff(a, b) == 255.0
    print(f"  指标函数     PSNR/psnr_masked/max_abs_diff 自检通过"
          f"（1/16 像素差 → {p:.2f} dB）")

    try:
        SimpleRasterizer().render(gs, cam, torch.zeros(3))
    except Exception as e:
        print(f"  rasterizer  预期在无 CUDA 时失败：{type(e).__name__}")
    else:
        print("  rasterizer  竟然在 CPU 上返回了结果（异常）")

    # ---- 参数校验（曾因 --sweep-batch 用 nargs="*" 拿到空列表而崩）----
    import argparse as _ap

    def _fake(**over):
        base = dict(batch_size=4, frames=3, sweep_batch=None, tex_size=256,
                    mlp_hidden=[128, 128])
        base.update(over)
        return _ap.Namespace(**base)

    _validate(_fake())                                   # 合法，不应抛
    for bad, why in ((dict(batch_size=0), "batch_size=0"),
                     (dict(batch_size=-1), "batch_size<0"),
                     (dict(frames=0), "frames=0"),
                     (dict(sweep_batch=[0]), "sweep_batch 含 0"),
                     (dict(tex_size=0), "tex_size=0"),
                     (dict(mlp_hidden=[]), "mlp_hidden 空列表")):
        try:
            _validate(_fake(**bad))
        except SystemExit:
            pass
        else:
            raise AssertionError(f"{why} 应被 _validate 拒绝")
    print("  参数校验     batch_size / frames / sweep_batch / tex_size 非法值均被拒绝 ✓")

    # ---- 摘要打印（曾因 c / r 未定义而崩，且只在实际渲染时才走到）----
    def _mk(bs, ms, fps):
        return {"batch_size": bs, "mean_ms_per_frame": ms,
                "median_ms_per_frame": ms, "fps": fps, "peak_memory_mib": 100.0}

    fake = {
        "subject": "dry", "work_name": "test",
        "frames": 3, "resolution": [512, 512],
        "paths": {"report": "<dry-run 不写文件>"},
        "core_perf": _mk(4, 8.0, 125.0),
        "reference_perf": None,
        "reference_batch_sweep": None,
        "summary": {k: None for k in
                    ("core_vs_ref_psnr", "core_vs_ref_maxdiff", "core_vs_gt_psnr",
                     "core_vs_gt_psnr_masked", "ref_vs_gt_psnr",
                     "ref_vs_gt_psnr_masked")},
    }
    import io
    from contextlib import redirect_stdout

    for perf in (None, _mk(4, 2.0, 500.0)):
        fake["reference_perf"] = perf
        fake["reference_batch_sweep"] = ({"1": _mk(1, 2.1, 476.0),
                                          "4": _mk(4, 2.0, 500.0)}
                                         if perf else None)
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_summary(fake)
        assert "报告" in buf.getvalue()
    print("  摘要打印     core-only / core+参照 / 带 sweep 三种形态均正常 ✓")

    # ---- 配置解析结果（路径必须绝对、模型解析可用）----
    assert Path(args.ply).is_absolute(), f"模型路径应为绝对：{args.ply}"
    assert Path(args.data).is_absolute(), f"数据集路径应为绝对：{args.data}"
    assert Path(args.output_dir).is_absolute(), "输出目录应为绝对"
    assert args.data.name == args.subject, (
        f"数据集目录应以 subject 结尾：{args.data} vs {args.subject}")
    _validate(args)
    print(f"  配置         subject={args.subject}/{args.work_name} "
          f"batch={args.batch_size} frames={args.frames} "
          f"tex={args.tex_size} bg={args.background} ✓")
    print(f"  模型路径     {args.ply}")

    print("\n[dry-run] 渲染链路、指标函数、参数校验、摘要打印、配置自检通过 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
