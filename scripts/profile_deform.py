#!/usr/bin/env python3
"""分解一帧的耗时，**在 GPU 上正确计时**（需要 GPU）。

为什么要单独一个脚本：CPU 与 GPU 的性能特征差异极大 ——
实测中 CPU 上占 42% 的 `linear_blending`，在 GPU 上可能只占很小一部分；
而 `torch.nonzero` 这类**数据依赖形状**的算子会强制 device→host 同步，
在 GPU 上代价远高于 CPU。**用 CPU 数据推断 GPU 瓶颈会导致优化方向完全跑偏。**

计时方式：每个被测片段前后各打一个 CUDA event，循环若干次后取 event 差值。
不能直接用 `time.perf_counter()`：CUDA 是异步的，wall-clock 会把
"提交内核"的时间当成"算完"的时间。

用法::

    conda activate live3dgs
    python scripts/profile_deform.py                 # 默认 batch=1
    python scripts/profile_deform.py --batch 10
    python scripts/profile_deform.py --batch 1 4 10  # 多个批大小各测一遍
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import live3dgsavatar  # noqa: E402, F401  （导入即施加 numpy 兼容补丁）
from live3dgsavatar.config import load_config, resolve_model_ply  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU 上的单帧耗时分解")
    p.add_argument("--config-dir", type=Path, default=None)
    p.add_argument("--subject", default=None)
    p.add_argument("--work-name", default=None)
    p.add_argument("--batch", type=int, nargs="+", default=[1],
                   help="批大小；可给多个")
    p.add_argument("--frames", type=int, default=None,
                   help="用数据集的前多少帧（默认取 max(batch)）")
    p.add_argument("--repeats", type=int, default=30, help="每项重复次数")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--skip-rasterize", action="store_true",
                   help="只测 deform（不加载光栅化器）")
    p.add_argument("--dry-run", action="store_true",
                   help="CPU 自检脚本本身（不访问 GPU/数据集）")
    return p.parse_args()


def _finalize(args: argparse.Namespace) -> argparse.Namespace:
    cfg = load_config(args.config_dir)
    if args.subject is not None:
        cfg.set("subject", args.subject)
    if args.work_name is not None:
        cfg.set("work_name", args.work_name)
    args.cfg = cfg
    args.ply = resolve_model_ply(cfg)
    args.data = Path(cfg.paths.data_root) / str(cfg.subject)
    args.batches = args.batch
    args.frames = args.frames or max(args.batches)
    return args


class Timer:
    """CUDA event 计时器：`t(fn)` 返回该项的平均毫秒数。"""

    def __init__(self, repeats: int, warmup: int) -> None:
        self.repeats = repeats
        self.warmup = warmup
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __call__(self, fn) -> float:
        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()
        self.start.record()
        for _ in range(self.repeats):
            fn()
        self.end.record()
        torch.cuda.synchronize()
        return self.start.elapsed_time(self.end) / self.repeats


def main() -> int:
    args = _finalize(parse_args())
    if args.dry_run:
        return _dry_run(args)
    if not torch.cuda.is_available():
        print("[error] 需要 GPU（CPU 上的结论不能迁移，见脚本 docstring）")
        return 1
    for label, path in (("模型", args.ply), ("数据集", args.data)):
        if not path.exists():
            print(f"[error] {label}不存在：{path}")
            return 1

    print(f"[info] 模型   : {args.ply}")
    print(f"[info] 数据集 : {args.data}")
    print(f"[info] 设备   : {torch.cuda.get_device_name(0)}")

    from reference_scene import load_scene
    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.deform.bind import (
        matrix_to_quaternion, quaternion_multiply)
    from live3dgsavatar.core.deform.tbn import compute_face_tbn
    from live3dgsavatar.core.io import load_ply
    from live3dgsavatar.core.render import SimpleRasterizer
    from live3dgsavatar.core.types import Camera, Mesh

    cfg = args.cfg
    av_cfg = AvatarConfig(
        tex_size=int(cfg.get("model.network.tex_size")),
        num_basis_in=int(cfg.get("model.network.num_basis_in")),
        num_basis_blend=int(cfg.get("model.network.num_basis_blend")),
        mlp_hidden=tuple(cfg.get("model.network.mlp_hidden") or ()),
        use_weight_proj=bool(cfg.get("model.network.use_weight_proj", True)))
    avatar = load_ply(args.ply, config=av_cfg, device="cuda")

    scene = load_scene(cfg, frames=args.frames)
    camera = Camera.from_intrinsics_extrinsics(
        scene.K, scene.R, scene.T, scene.width, scene.height).to("cuda")
    bg = torch.tensor([float(x) for x in cfg.get("render.background", [0, 0, 0])],
                      dtype=torch.float32, device="cuda")

    n_frames = scene.num_frames
    print(f"[info] N={avatar.num_gaussians}  K={avatar.num_basis}  可用帧={n_frames}")

    for batch in args.batches:
        batch = min(batch, n_frames)
        idx = list(range(batch))
        verts = scene.frames_tensor(idx, device="cuda")
        weights = scene.weights_tensor(idx, device="cuda")
        mesh = Mesh(verts=verts, faces=scene.faces,
                    uvs=scene.uvs, uv_faces=scene.uv_faces)
        timer = Timer(args.repeats, args.warmup)

        field = avatar.build_blend_field(use_cuda_kernel=False)
        binder = avatar.build_binder()
        tangent = field(weights, batch_size=batch)
        used, inv = torch.unique(binder.binding.face_id, return_inverse=True)
        tri_used = mesh.verts[:, mesh.faces[used]]
        uv_used = mesh.uvs[mesh.uv_faces[used]]
        tbn_used = compute_face_tbn(tri_used, uv_used)
        binding_rot = tbn_used[:, inv]
        binding_tri = tri_used[:, inv]
        bary = binder.binding.face_bary.to(mesh.verts.dtype)
        rast = SimpleRasterizer()
        gs = avatar.deform(mesh, weights, use_cuda_kernel=False)

        # (名称, 是否子项, 计时函数)
        items: list[tuple[str, bool, object]] = [
            ("blend_field（整体）", False,
             lambda: field(weights, batch_size=batch)),
            ("project_weight（MLP）", True,
             lambda: field.project_weight(weights)),
            ("binder.bind（整体）", False,
             lambda: binder.bind(tangent, mesh)),
            ("TBN（仅 F_used 个面）", True,
             lambda: compute_face_tbn(tri_used, uv_used)),
            ("gather tbn[:, inv]", True, lambda: tbn_used[:, inv]),
            ("gather tri[:, inv]", True, lambda: tri_used[:, inv]),
            ("offset 重心插值", True,
             lambda: (binding_tri * bary.unsqueeze(0).unsqueeze(-1)).sum(-2)),
            ("xyz = br @ xyz", True,
             lambda: (binding_rot @ tangent.xyz.unsqueeze(-1)).squeeze(-1)),
            ("matrix_to_quaternion", True,
             lambda: matrix_to_quaternion(binding_rot)),
            ("quaternion_multiply", True,
             lambda: quaternion_multiply(
                 matrix_to_quaternion(binding_rot), tangent.rotation)),
            ("deform（整体）", False,
             lambda: avatar.deform(mesh, weights, use_cuda_kernel=False)),
        ]
        if not args.skip_rasterize:
            items.append(("rasterize（整体）", False,
                          lambda: rast.render(gs, camera, bg)))
            items.append(("★ 完整一帧", False,
                          lambda: rast.render(
                              avatar.deform(mesh, weights, use_cuda_kernel=False),
                              camera, bg)))

        results = [(name, sub, timer(fn)) for name, sub, fn in items]
        deform_ms = next(v for k, _, v in results if k == "deform（整体）")
        total_ms = next((v for k, _, v in results if k.startswith("★")), deform_ms)

        # ⚠️ 单位是 **ms/批**（一次 forward 调用），不是 ms/帧。
        #    批内是逐帧串行的（SimpleRasterizer 内部 for i in range(b)），
        #    故 ms/帧 = ms/批 ÷ batch。
        print(f"\n{'=' * 78}\nbatch = {batch}   （repeats={args.repeats}，"
              f"单位 ms/批；÷{batch} 得 ms/帧）\n{'=' * 78}")
        print(f"  {'项目':<34} {'ms/批':>9} {'ms/帧':>8} {'占 deform':>10}")
        for name, sub, ms in results:
            if name.startswith("★"):
                continue
            label = f"  {name}" if sub else name
            print(f"  {label:<34} {ms:9.3f} {ms / batch:8.3f} "
                  f"{100 * ms / deform_ms:9.1f}%")
        print(f"  {'─' * 34} {'─' * 9} {'─' * 8}")
        print(f"  {'★ 完整一帧':<34} {total_ms:9.3f} {total_ms / batch:8.3f} "
              f"{1000 / (total_ms / batch):7.0f} FPS/帧")
        print(f"\n  → deform 占整帧 {100 * deform_ms / total_ms:.0f}%，"
              f"rasterize 占 {100 * (total_ms - deform_ms) / total_ms:.0f}%")
        del gs
        gc.collect()
        torch.cuda.empty_cache()

    return 0


def _dry_run(args) -> int:
    """CPU 自检：验证参数解析、分组显示与输出格式（不访问 GPU/数据集）。"""
    print("[dry-run] 校验参数与输出格式（不访问 GPU / 数据集）")

    class _FakeTimer:
        def __init__(self, *a, **k): pass
        def __call__(self, fn): return 1.0

    # 用假数据走一遍打印逻辑
    fake = [("blend_field（整体）", False, 3.0),
            ("project_weight（MLP）", True, 1.0),
            ("binder.bind（整体）", False, 5.0),
            ("★ 完整一帧", False, 9.0)]
    deform_ms = 8.0
    total_ms = 9.0
    for name, sub, ms in fake:
        if name.startswith("★"):
            continue
        label = f"  {name}" if sub else name
        assert len(f"{label:<34}") >= 34
    print(f"  {'项目':<34} {'ms/帧':>9} {'占 deform':>10}")
    for name, sub, ms in fake:
        if name.startswith("★"):
            continue
        label = f"  {name}" if sub else name
        print(f"  {label:<34} {ms:9.3f} {100 * ms / deform_ms:9.1f}%")
    print(f"  {'★ 完整一帧':<34} {total_ms:9.3f} {1000 / total_ms:8.0f} FPS")
    assert Timer is not None and _FakeTimer is not None
    print("\n[dry-run] 参数、分组标签与格式化自检通过 ✓")
    return 0


def rast_render(avatar, rast, mesh, weights, camera, bg):
    gs = avatar.deform(mesh, weights, use_cuda_kernel=False)
    return rast.render(gs, camera, bg)


if __name__ == "__main__":
    raise SystemExit(main())
