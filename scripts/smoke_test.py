#!/usr/bin/env python3
"""GPU 冒烟测试 + 基线性能采集 —— Live3DGSAvatar P0

用参照实现（RGBAvatar，只读）加载一个预训练模型，在若干帧上跑完整前向
（blend → bind → rasterize），输出渲染图与性能基线。

这同时是 P1"数值等价门"的参照产物生成器。

⚠️ 需要可用的 NVIDIA GPU。

用法:
    conda activate live3dgs
    python scripts/smoke_test.py \
        --ply  /home/crh/Projects/RGBAvatar/output/duda/test/model.ply \
        --data /home/crh/Datasets/INSTA/duda \
        --frames 3 --out output/smoke
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

# 参照仓库默认位置：从脚本位置推导项目根，再取同级目录；可用 --rgbavatar / RGBA_REF 覆盖
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RGBAVATAR = Path(os.environ.get("RGBA_REF", REPO_ROOT.parent / "RGBAvatar"))


def to_py(value):
    """把 numpy / torch 标量转成原生 Python 类型，供 json 序列化。

    参照实现里取自 `torch.load` 的字段（如 `img_size`）是 numpy 标量，
    `json.dumps` 无法直接序列化 `np.int64`。
    """
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    # 先判数组：np.ndarray 同时有 .item() 与 .tolist()，但 .item() 仅对单元素有效
    if hasattr(value, "ndim") and getattr(value, "ndim") > 0:
        return value.tolist()
    if hasattr(value, "item"):          # numpy / torch 标量
        try:
            return value.item()
        except (ValueError, RuntimeError):
            return str(value)
    if hasattr(value, "tolist"):        # 其他序列类型
        return value.tolist()
    return str(value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live3DGSAvatar P0 GPU 冒烟测试")
    p.add_argument("--rgbavatar", type=Path, default=DEFAULT_RGBAVATAR,
                   help="RGBAvatar 只读参照仓库路径")
    p.add_argument("--ply", type=Path, required=True, help="预训练 model.ply")
    p.add_argument("--data", type=Path, required=True, help="数据集目录（含 images/ 与 checkpoint/）")
    p.add_argument("--frames", type=int, default=-1,
                   help="渲染帧数；-1 表示全部（默认，渲染总耗时/帧数即为单帧耗时）")
    p.add_argument("--batch-size", type=int, default=2, help="批大小（默认 2，6GB 显存请勿超过 4）")
    p.add_argument("--tex-size", type=int, default=256)
    p.add_argument("--num-basis-in", type=int, default=129)
    p.add_argument("--num-basis-blend", type=int, default=20)
    p.add_argument("--out", type=Path, default=Path("output/smoke"),
                   help="输出目录，相对路径按**调用时的工作目录**解析")
    p.add_argument("--white-bg", action="store_true")
    p.add_argument("--alpha", action="store_true", help="同时保存 alpha 通道")
    args = p.parse_args()

    # ⚠️ 必须在此处（chdir 之前）把路径绝对化。
    # load_reference() 会 chdir 到参照仓库，否则相对路径会被解析到参照仓库内部，
    # 把输出写进只读参照仓库。
    for name in ("rgbavatar", "ply", "data", "out"):
        setattr(args, name, Path(getattr(args, name)).expanduser().resolve())
    return args


def apply_compat() -> None:
    """应用第三方兼容性补丁（必须在任何 chumpy 相关导入之前）。

    参照仓库的数据集/模型加载会间接 import chumpy 来反序列化 FLAME pkl，
    而 chumpy 0.70 与 numpy 2.x 不兼容。补丁由 src/live3dgsavatar/compat/ 提供，
    与主代码共用同一处修复，避免两套逻辑漂移。
    """
    src = REPO_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import live3dgsavatar  # noqa: F401, PLC0415  （导入即应用补丁）
    from live3dgsavatar import compat  # noqa: PLC0415

    if compat.patched_aliases:
        print(f"[info] 已应用 numpy 兼容补丁: {', '.join(compat.patched_aliases)}")


def load_reference(rgbavatar: Path) -> dict:
    """把只读参照仓库挂到 sys.path，返回其模块集合。"""
    if not rgbavatar.is_dir():
        sys.exit(f"[error] 找不到参照仓库：{rgbavatar}（用 --rgbavatar 指定）")
    # 参照实现用相对路径（data/FLAME2020/...）定位 FLAME 模型
    os.chdir(rgbavatar)
    sys.path.insert(0, str(rgbavatar))
    apply_compat()

    from camera import IntrinsicsCamera                      # noqa: PLC0415
    from diff_renderer import render_gs_batch                # noqa: PLC0415
    from dataset import FLAMEDataset                         # noqa: PLC0415
    from model import FLAMEBindingModel                      # noqa: PLC0415
    from submodules.flame import FLAME, FlameConfig          # noqa: PLC0415
    from torch.utils.data import DataLoader, Subset          # noqa: PLC0415
    from utils import Struct                                 # noqa: PLC0415
    return dict(IntrinsicsCamera=IntrinsicsCamera, render_gs_batch=render_gs_batch,
                FLAMEDataset=FLAMEDataset, FLAMEBindingModel=FLAMEBindingModel,
                FLAME=FLAME, FlameConfig=FlameConfig, Struct=Struct,
                DataLoader=DataLoader, Subset=Subset)


def main() -> int:
    args = parse_args()

    import numpy as np
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        sys.exit(
            "[error] torch.cuda.is_available() == False。\n"
            "        请在可访问 GPU 设备的终端中运行（WSL2 下需能打开 /dev/dxg）。"
        )
    dev = torch.cuda.get_device_name(0)
    print(f"[info] GPU: {dev}")

    import nvdiffrast.torch as dr

    ref = load_reference(args.rgbavatar)

    # ---------------------------------------------------------- 建模型 ----
    glctx = dr.RasterizeGLContext()
    flame_model = ref["FLAME"](ref["FlameConfig"]()).cuda()
    dataset = ref["FLAMEDataset"](flame_model, str(args.data), split="all",
                                  pin_memory=True, use_shape_weight=True, use_pose_weight=True)
    cfg = ref["Struct"](tex_size=args.tex_size, num_basis_in=args.num_basis_in,
                        num_basis_blend=args.num_basis_blend, use_blend=True,
                        use_weight_proj=True, use_mlp_proj=True,
                        init_scaling=0.0008, init_opacity=0.5)
    gaussian_model = ref["FLAMEBindingModel"](cfg, flame_model, glctx)
    gaussian_model.load_ply(str(args.ply))
    print(f"[info] 高斯数量: {gaussian_model.num_gaussian}")

    camera = ref["IntrinsicsCamera"](
        K=dataset.camera_intri,
        R=dataset.camera_extri[:3, :3],
        T=dataset.camera_extri[:3, 3],
        width=dataset.image_width, height=dataset.image_height,
    ).cuda()
    print(f"[info] 分辨率: {dataset.image_width}x{dataset.image_height}")

    bg = torch.tensor([1.0, 1.0, 1.0] if args.white_bg else [0.0, 0.0, 0.0],
                      dtype=torch.float32, device="cuda")

    # ---------------------------------------------------------- 渲染 ----
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[info] 输出目录: {args.out}")       # 绝对路径，便于确认没写进参照仓库
    n = len(dataset) if args.frames < 0 else min(args.frames, len(dataset))
    loader = ref["DataLoader"](ref["Subset"](dataset, list(range(n))),
                               batch_size=args.batch_size, shuffle=False)

    def run() -> tuple[torch.Tensor, list[int]]:
        """跑完整前向，返回拼好的图像与帧号。"""
        out, ids = [], []
        for bi, data in enumerate(loader):
            mesh = data["mesh"].cuda()
            bw = data["blend_weight"].cuda()
            gaussian = gaussian_model.gaussian_deform_batch(mesh, bw)
            pkg = ref["render_gs_batch"](camera, bg, gaussian)
            img = pkg["color"]
            if args.alpha:
                img = torch.cat([img, pkg["alpha"]], dim=1)
            out.append(img)
            ids.extend(range(bi * args.batch_size, bi * args.batch_size + mesh.shape[0]))
        return torch.cat(out, dim=0), ids

    with torch.no_grad():                       # 预热：排除首次 kernel 编译
        run()
    torch.cuda.synchronize()
    # 预热完成后先回收再重置，避免预热的分配被计入「峰值显存」
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        image, idx = run()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    n = len(idx)
    print(f"\n{'=' * 58}\nP0 基线（{n} 帧，batch={args.batch_size}）\n{'=' * 58}")
    print(f"  总耗时      {dt * 1000:8.1f} ms")
    print(f"  单帧耗时    {dt / n * 1000:8.1f} ms")
    print(f"  吞吐        {n / dt:8.1f} FPS")
    print(f"  峰值显存    {peak_mib:8.1f} MiB")
    print(f"  高斯数量    {gaussian_model.num_gaussian}")
    print(f"  分辨率      {dataset.image_width}x{dataset.image_height}")
    print(f"{'=' * 58}\n")

    arr = (image.permute(0, 2, 3, 1) * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    for j, i in enumerate(idx):
        p = args.out / f"frame_{i:05d}.png"
        Image.fromarray(arr[j]).save(p)
        print(f"[out] {p}")

    report = {
        "gpu": str(dev),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "num_gaussian": to_py(gaussian_model.num_gaussian),
        # ⚠️ image_width/height 来自 torch.load 的 numpy 标量（int64），
        #    必须转成 Python int，否则 json.dumps 抛 TypeError
        "resolution": [to_py(dataset.image_width), to_py(dataset.image_height)],
        "batch_size": to_py(args.batch_size),
        "frames": to_py(n),
        "total_ms": to_py(dt * 1000),
        "per_frame_ms": to_py(dt / n * 1000),
        "fps": to_py(n / dt),
        "peak_memory_mib": to_py(peak_mib),
        "tex_size": to_py(args.tex_size),
        "num_basis_in": to_py(args.num_basis_in),
        "num_basis_blend": to_py(args.num_basis_blend),
    }
    rp = args.out / "baseline.json"
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {rp}\n\n冒烟测试通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
