"""等价测试的共用支撑。

提供参照实现（RGBAvatar）的加载与比较工具。参照仓库为**只读**，测试只导入不修改。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

# 项目根 = tests/support.py 的上一级
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"


def _find_reference() -> Path:
    """定位参照仓库：环境变量 RGBA_REF 优先，其次仓库同级目录。"""
    env = os.environ.get("RGBA_REF")
    if env:
        return Path(env).expanduser().resolve()
    for candidate in (REPO_ROOT.parent / "RGBAvatar", Path.home() / "Projects" / "RGBAvatar"):
        if candidate.is_dir():
            return candidate
    # 都找不到时给出一个最有信息量的默认值，交由 require_reference 报错
    return REPO_ROOT.parent / "RGBAvatar"


REFERENCE_ROOT = _find_reference()


def ensure_src_on_path() -> None:
    """把 src/ 挂到 sys.path，使 core 可被导入（无需安装）。"""
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))


def require_reference() -> Path:
    if not REFERENCE_ROOT.is_dir():
        raise FileNotFoundError(
            f"找不到参照仓库 {REFERENCE_ROOT}。"
            "设置环境变量 RGBA_REF，或把 RGBAvatar 放在本项目同级目录。"
        )
    return REFERENCE_ROOT


def load_module_from(path: Path, name: str) -> ModuleType:
    """按文件路径加载模块，避免与已安装的同名包冲突。

    参照仓库内部使用顶层包名（如 `from diff_renderer import ...`），
    因此先把参照仓库根目录挂到 sys.path 上。
    """
    root = str(REFERENCE_ROOT)
    if root not in sys.path:
        sys.path.append(root)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_REF_CAMERA: ModuleType | None = None


def reference_camera_module() -> ModuleType:
    """参照实现的 `camera/camera.py`（纯 numpy/torch，不依赖 nvdiffrast）。"""
    global _REF_CAMERA
    if _REF_CAMERA is None:
        root = require_reference()
        _REF_CAMERA = load_module_from(root / "camera" / "camera.py", "_rgba_reference_camera")
    return _REF_CAMERA


# ----------------------------------------------------------------- 比较工具 --


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().to(torch.float64) - b.detach().to(torch.float64)).abs().max())


def compare(
    a: torch.Tensor,
    b: torch.Tensor,
    atol: float,
    name: str,
) -> tuple[bool, str]:
    """返回 (是否通过, 说明)。形状不一致直接判失败。"""
    if tuple(a.shape) != tuple(b.shape):
        return False, f"{name}: 形状不一致 {tuple(a.shape)} vs {tuple(b.shape)}"
    d = max_abs_diff(a, b)
    ok = d <= atol
    return ok, f"{name}: max|Δ| = {d:.3e} (atol {atol:.0e})"


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """两图之间的 PSNR（按 [0,1] 动态范围）。"""
    a64 = a.detach().to(torch.float64)
    b64 = b.detach().to(torch.float64)
    mse = float(((a64 - b64) ** 2).mean())
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def compare_image(
    a: torch.Tensor,
    b: torch.Tensor,
    psnr_db: float,
    atol: float,
    name: str,
) -> tuple[bool, str]:
    """图像的等价判据：`PSNR > psnr_db` **或** `max|Δ| < atol`。"""
    d = max_abs_diff(a, b)
    p = psnr(a, b)
    ok = (p > psnr_db) or (d < atol)
    return ok, f"{name}: PSNR = {p:.2f} dB, max|Δ| = {d:.3e}"
