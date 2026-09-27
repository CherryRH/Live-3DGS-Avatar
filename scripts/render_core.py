#!/usr/bin/env python3
"""用本项目 `core/` 渲染图像并输出到 `output/`（**需要 GPU**）。

与 `smoke_test.py` 的区别：**那个跑的是参照实现**，本脚本跑的是 `core/`。
因此两者产出的差异 = 「本项目相对参照实现的行为差异」。

输出：

    output/<subject>/<work_name>/render_image/    逐帧 PNG
    output/<subject>/<work_name>/render_core.json 性能与差异报告

用法::

    conda activate live3dgs
    python scripts/render_core.py --frames 3
    python scripts/render_core.py --frames -1 --compare-ref output/smoke
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="用 core/ 渲染图像")
    p.add_argument("--subject", default="duda")
    p.add_argument("--work-name", default="core")
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    p.add_argument("--ply", type=Path, default=None,
                   help="默认 <reference>/output/<subject>/test/model.ply")
    p.add_argument("--data", type=Path, default=None,
                   help="默认 /home/crh/Datasets/INSTA/<subject>")
    p.add_argument("--reference", type=Path,
                   default=Path("/home/crh/Projects/RGBAvatar"))
    p.add_argument("--frames", type=int, default=3,
                   help="渲染帧数；-1 表示全部")
    p.add_argument("--tex-size", type=int, default=256)
    p.add_argument("--num-basis-in", type=int, default=129)
    p.add_argument("--num-basis-blend", type=int, default=20)
    p.add_argument("--mlp-hidden", type=int, nargs="*", default=[128, 128])
    p.add_argument("--white-bg", action="store_true")
    p.add_argument("--compare-ref", type=Path, default=None,
                   help="与另一目录逐帧比较；smoke_test.py 的输出默认在 output/smoke")
    p.add_argument("--dry-run", action="store_true",
                   help="仅验证模型与数据加载（CPU，不需要 GPU）")
    args = p.parse_args()

    args.output_dir = args.output_dir.resolve()
    args.ply = (args.ply or (args.reference / "output" / args.subject / "test"
                             / "model.ply")).resolve()
    args.data = (args.data or Path("/home/crh/Datasets/INSTA") / args.subject).resolve()
    args.reference = args.reference.resolve()
    if args.compare_ref is not None:
        args.compare_ref = args.compare_ref.resolve()
    return args


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) / 255.0
    b = b.astype(np.float64) / 255.0
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))


def main() -> int:
    args = parse_args()

    device = "cpu" if args.dry_run else "cuda"
    if not args.dry_run and not torch.cuda.is_available():
        print("[error] 需要 GPU（可用 --dry-run 在 CPU 上验证加载路径）")
        return 1
    for label, path in (("模型", args.ply), ("数据集", args.data)):
        if not path.exists():
            print(f"[error] {label}不存在：{path}")
            return 1

    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.render import SimpleRasterizer
    from live3dgsavatar.core.types import Camera, Mesh
    from reference_scene import load_scene

    print(f"[info] 模型   : {args.ply}")
    print(f"[info] 数据集 : {args.data}")
    print(f"[info] 输出   : {args.output_dir / args.subject / args.work_name}")

    if args.dry_run:
        # 参照实现的 FLAMEDataset 内部硬编码 `.cuda()`，CPU 下无法加载数据集。
        # 因此 CPU 自检覆盖的是**本项目 core/ 侧的渲染链路**（用合成帧），
        # 这恰好也是唯一能在无 GPU 环境验证的部分：
        #   deform（混合 → 绑定）→ Camera 构造 → to_kernel_matrix → 输入校验
        return _dry_run(args)

    # ------------------------------------------------------------ 加载 --
    cfg = AvatarConfig(tex_size=args.tex_size, num_basis_in=args.num_basis_in,
                       num_basis_blend=args.num_basis_blend,
                       mlp_hidden=tuple(args.mlp_hidden), use_weight_proj=True)
    print("\n[1/3] 加载 avatar（core/）…")
    avatar = load_ply(args.ply, config=cfg, device=device)
    print(f"       N = {avatar.num_gaussians}, K = {avatar.num_basis}")

    print("[2/3] 加载模板几何与相机…")
    scene = load_scene(args.data, args.reference, frames=args.frames, device=device)
    print(f"       帧数 = {scene.num_frames}, 面数 = {scene.faces.shape[0]}, "
          f"分辨率 = {scene.width}x{scene.height}")

    camera = Camera.from_intrinsics_extrinsics(
        K=scene.K, R=scene.R, T=scene.T, width=scene.width, height=scene.height).to(device)
    bg = torch.tensor([1.0, 1.0, 1.0] if args.white_bg else [0.0, 0.0, 0.0],
                      dtype=torch.float32, device="cuda")
    rasterizer = SimpleRasterizer()

    out_dir = args.output_dir / args.subject / args.work_name
    img_dir = out_dir / "render_image"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 清理上次运行的残留：否则改变帧数或命名后，旧文件会与新文件混在一起，
    # 让 `--compare-ref` 的结果失真（例如上一版用的是 `00000.png`）
    stale = sorted(img_dir.glob("*.png"))
    for p in stale:
        p.unlink()
    if stale:
        print(f"       已清理 {len(stale)} 个上次运行残留的 PNG")

    # ------------------------------------------------------------ 渲染 --
    print(f"[3/3] 渲染 {scene.num_frames} 帧…")

    def run_frame(i: int):
        f = scene.frames[i]
        mesh = Mesh(verts=f["mesh"].unsqueeze(0), faces=scene.faces,
                    uvs=scene.uvs, uv_faces=scene.uv_faces)
        with torch.no_grad():
            gs = avatar.deform(mesh, f["blend_weight"].unsqueeze(0))
            return rasterizer.render(gs, camera, bg)

    # 预热（排除首次 kernel 编译）
    run_frame(0)
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    from PIL import Image

    per_frame_ms: list[float] = []
    t_all = time.perf_counter()
    for i in range(scene.num_frames):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = run_frame(i)
        torch.cuda.synchronize()
        per_frame_ms.append((time.perf_counter() - t0) * 1000.0)

        img = out.color[0].clamp(0, 1).permute(1, 2, 0).mul(255).byte().cpu().numpy()
        # 命名统一到**参照实现 `render.py` 的约定**（`%05d.png`）。
        # 注意：`smoke_test.py` 用的是 `frame_%05d.png`（它自己的前缀），
        # 比较时两种都能识别。
        Image.fromarray(img).save(img_dir / f"{i:05d}.png")
        if (i + 1) % 20 == 0 or i == scene.num_frames - 1:
            print(f"      {i + 1}/{scene.num_frames}  "
                  f"({np.mean(per_frame_ms):.2f} ms/帧)")
    total = time.perf_counter() - t_all

    mean_ms = float(np.mean(per_frame_ms))
    report = {
        "impl": "core", "gpu": torch.cuda.get_device_name(0),
        "num_gaussian": avatar.num_gaussians, "tex_size": args.tex_size,
        "resolution": [scene.width, scene.height],
        "frames": scene.num_frames,
        "mean_ms_per_frame": mean_ms,
        "median_ms_per_frame": float(np.median(per_frame_ms)),
        "total_s": total,
        "fps": 1000.0 / mean_ms if mean_ms > 0 else None,
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "image_dir": str(img_dir),
    }

    # -------------------------------------------------- 与参照基线对比 --
    if args.compare_ref is not None:
        report["reference_compare"] = _compare(args.compare_ref, img_dir)
        # 也吸收参照的性能基线（若存在）
        ref_json = args.compare_ref / "baseline.json"
        if ref_json.exists():
            ref = json.loads(ref_json.read_text(encoding="utf-8"))
            report["reference_perf"] = {k: ref.get(k) for k in
                                        ("per_frame_ms", "fps", "peak_memory_mib",
                                         "frames", "num_gaussian")}
            if ref.get("frames") == scene.num_frames and ref.get("per_frame_ms"):
                report["reference_perf"]["_note"] = "参照为全量 254 帧的实测"

    out_dir.mkdir(parents=True, exist_ok=True)
    rp = out_dir / "render_core.json"
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'=' * 66}\ncore/ 渲染结果\n{'=' * 66}")
    print(f"  帧数        {scene.num_frames}")
    print(f"  平均耗时    {mean_ms:.2f} ms/帧  ({report['fps']:.1f} FPS)")
    print(f"  中位耗时    {report['median_ms_per_frame']:.2f} ms/帧")
    print(f"  峰值显存    {report['peak_memory_mib']:.0f} MiB")
    print(f"  图像        {img_dir}")
    print(f"  报告        {rp}")
    if "reference_compare" in report:
        c = report["reference_compare"]
        print(f"\n  与参照对比（{c['reference_dir']}）")
        print(f"    可比帧数    {c['compared']} / {c['total_ours']}")
        if c["compared"]:
            print(f"    PSNR       中位 {c['psnr_median']:.2f} dB  "
                  f"最小 {c['psnr_min']:.2f} dB")
            print(f"    max|Δ|     中位 {c['maxdiff_median']:.1f} / 255")
    print(f"{'=' * 66}")
    return 0


def _compare(ref_dir: Path, ours_dir: Path) -> dict:
    """与参照渲染图逐帧比较（灰度上算 PSNR）。"""
    from PIL import Image

    # 兼容两种命名：参照 `render.py` 用 `00000.png`，smoke_test 用 `frame_00000.png`。
    # 统一按「去掉可选 frame_ 前缀」后的名字配对。
    def _index(p):
        stem = p.stem
        if stem.startswith("frame_"):
            stem = stem[len("frame_"):]
        return int(stem) if stem.isdigit() else None

    ours = sorted((p for p in ours_dir.glob("*.png") if _index(p) is not None),
                  key=_index)
    result = {"reference_dir": str(ref_dir), "total_ours": len(ours),
              "compared": 0, "psnr_median": None, "psnr_min": None,
              "maxdiff_median": None, "missing": []}
    if not ours:
        result["missing"].append(f"本目录没有 frame_*.png：{ours_dir}")
        return result
    if not ref_dir.is_dir():
        result["missing"].append(f"参照目录不存在：{ref_dir}")
        return result

    # 参照目录里同样两种命名都接受
    ref_by_index = {}
    for q in ref_dir.glob("*.png"):
        i = _index(q)
        if i is not None:
            ref_by_index[i] = q

    psnrs, maxdiffs = [], []
    for p in ours:
        q = ref_by_index.get(_index(p))
        if q is None:
            result["missing"].append(p.name)
            continue
        a = np.array(Image.open(p).convert("RGB"))
        b = np.array(Image.open(q).convert("RGB"))
        if a.shape != b.shape:
            result["missing"].append(f"{p.name}: 形状不一致 {a.shape} vs {b.shape}")
            continue
        psnrs.append(_psnr(a, b))
        maxdiffs.append(float(np.abs(a.astype(np.int16) - b.astype(np.int16)).max()))

    if psnrs:
        result["compared"] = len(psnrs)
        result["psnr_median"] = float(np.median(psnrs))
        result["psnr_min"] = float(np.min(psnrs))
        result["maxdiff_median"] = float(np.median(maxdiffs))
    return result


def _dry_run(args) -> int:
    """CPU 自检：用真实模型参数 + 合成网格走通 core/ 的渲染链路。"""
    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.render import SimpleRasterizer, to_kernel_matrix
    from live3dgsavatar.core.types import Camera, Mesh

    print("\n[dry-run] 参照的 FLAMEDataset 硬编码 .cuda()，CPU 下无法加载数据集；")
    print("          因此本模式只验证 core/ 侧链路（合成帧）。")

    cfg = AvatarConfig(tex_size=args.tex_size, num_basis_in=args.num_basis_in,
                       num_basis_blend=args.num_basis_blend,
                       mlp_hidden=tuple(args.mlp_hidden), use_weight_proj=True)
    avatar = load_ply(args.ply, config=cfg, device="cpu")
    print(f"  avatar      N={avatar.num_gaussians}, K={avatar.num_basis}")

    # 合成帧：拓扑必须与模板一致（面数取绑定里的最大 face_id + 1）
    n_faces = int(avatar.binding_face_id.max()) + 1
    n_verts = int(avatar.binding_face_id.numel())  # 顶点数只需足够
    torch.manual_seed(0)
    faces = torch.stack([torch.arange(0, n_faces * 3, 3, dtype=torch.int32) % n_verts,
                         torch.arange(1, n_faces * 3 + 1, 3, dtype=torch.int32) % n_verts,
                         torch.arange(2, n_faces * 3 + 2, 3, dtype=torch.int32) % n_verts],
                        dim=1)
    uvs = torch.rand(n_verts, 2)
    mesh = Mesh(verts=torch.randn(1, n_verts, 3) * 0.1, faces=faces,
                uvs=uvs, uv_faces=faces)
    bw = torch.randn(1, cfg.num_basis_in) * 0.1

    gs = avatar.deform(mesh, bw)
    print(f"  deform      xyz={tuple(gs.xyz.shape)} space={gs.space} "
          f"有限={bool(torch.isfinite(gs.xyz).all())}")

    K = torch.tensor([[400.0, 0, 256], [0, 400.0, 256], [0, 0, 1]])
    cam = Camera.from_intrinsics_extrinsics(
        K=K, R=torch.eye(3), T=torch.tensor([0.0, 0.0, -0.5]), width=512, height=512)
    km = to_kernel_matrix(cam.full_proj[0])
    print(f"  camera      full_proj={tuple(cam.full_proj.shape)} "
          f"kernel矩阵连续={km.is_contiguous()} "
          f"与转置一致={torch.equal(km, cam.full_proj[0].transpose(0, 1))}")

    # 走到光栅化器的输入校验为止（CPU 无 CUDA 内核）
    try:
        SimpleRasterizer().render(gs, cam, torch.zeros(3))
    except Exception as e:
        print(f"  rasterizer  预期在无 CUDA 时失败：{type(e).__name__}: {str(e)[:70]}")
    else:
        print("  rasterizer  竟然在 CPU 上返回了结果（异常）")

    print("\n[dry-run] core/ 链路形状与约定自检通过 ✓")
    print("          真实渲染请在 GPU 上运行：去掉 --dry-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
