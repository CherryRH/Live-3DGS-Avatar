"""相机数值等价测试（**不需要 GPU**）。

逐元素比对 `core.types.Camera` 与参照实现 `camera/camera.py` 的
`w2v` 与 `full_proj` —— 这两个矩阵是整条渲染链路的基座，
一旦不一致，后面所有渲染结果都无意义。

判据：max|Δ| < 1e-6（纯矩阵乘法，float32 下应几乎逐位一致）。

注意：测试写成普通循环而非 pytest 参数化，以便在**未安装 pytest** 时
也能用 `python tests/run_tests.py` 运行（见该脚本说明）。
"""

from __future__ import annotations

import numpy as np
import torch

from support import REFERENCE_ROOT, compare, reference_camera_module, require_reference

ATOL = 1e-6

# 参照实现 `IntrinsicsCamera` 从 K 取出 cx/cy 后直接与 Python 数值运算并赋给张量，
# 若 K 是 **float32** 数组会抛 `TypeError: can't assign a numpy.float32 to a torch.FloatTensor`。
# 这是参照实现的既有行为（不改动参照仓库），因此测试统一用 float64 构造 K。
K_DTYPE = np.float64


def _ref():
    require_reference()
    return reference_camera_module()


def _random_pose(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """构造一个合理的 world→camera 位姿（R 为正交阵，T 为平移）。"""
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))          # 保证 det = +1
    t = rng.normal(scale=0.1, size=3)
    return q.astype(K_DTYPE), t.astype(K_DTYPE)


def test_intrinsics_camera_matches_reference() -> None:
    from live3dgsavatar.core.types import Camera

    ref = _ref()
    for seed in (0, 1, 2):
        for size in ((512, 512), (640, 480)):
            w, h = size
            rng = np.random.default_rng(seed)
            R, T = _random_pose(rng)

            # 非中心主点，确保 cx/cy 路径被覆盖
            fx, fy = 1200.0, 1180.0
            cx, cy = w * 0.48, h * 0.52
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=K_DTYPE)

            reference = ref.IntrinsicsCamera(K=K, R=R, T=T, width=w, height=h)
            cam = Camera.from_intrinsics_extrinsics(K=K, R=R, T=T, width=w, height=h)

            tag = f"seed={seed}, size={w}x{h}"
            ok, msg = compare(cam.w2c[0].cpu(), reference.get_w2v.cpu(), ATOL,
                              f"[{tag}] w2c vs reference.get_w2v")
            assert ok, msg
            ok, msg = compare(cam.full_proj[0].cpu(), reference.get_full_proj.cpu(), ATOL,
                              f"[{tag}] full_proj")
            assert ok, msg


def test_projection_depth_range_is_0_1() -> None:
    """深度必须映射到 [0,1]（OpenGL 约定）；znear→0，zfar→1。"""
    from live3dgsavatar.core.types import Camera

    w = h = 256
    K = np.array([[200.0, 0.0, w / 2], [0.0, 200.0, h / 2], [0.0, 0.0, 1.0]], dtype=K_DTYPE)
    cam = Camera.from_intrinsics_extrinsics(
        K=K, R=np.eye(3, dtype=K_DTYPE), T=np.zeros(3, dtype=K_DTYPE),
        width=w, height=h, znear=0.01, zfar=100.0)
    P = cam.proj[0]

    def depth(z_cam: float) -> float:
        p = torch.tensor([[0.0, 0.0, z_cam, 1.0]], dtype=torch.float32)
        clip = p @ P.T
        return float(clip[0, 2] / clip[0, 3])

    assert abs(depth(0.01) - 0.0) < 1e-5, "znear 应对应 depth = 0"
    assert abs(depth(100.0) - 1.0) < 1e-4, "zfar 应对应 depth = 1"
    assert 0.0 < depth(1.0) < 1.0


def test_focal_fov_roundtrip() -> None:
    """fov 与 focal 必须自洽：`focal = W / (2·tan(fov_x/2))`。"""
    from live3dgsavatar.core.types import Camera

    ref = _ref()
    w, h = 640, 480
    fx, fy = 900.0, 880.0
    K = np.array([[fx, 0.0, w / 2], [0.0, fy, h / 2], [0.0, 0.0, 1.0]], dtype=K_DTYPE)
    cam = Camera.from_intrinsics_extrinsics(
        K=K, R=np.eye(3, dtype=K_DTYPE), T=np.zeros(3, dtype=K_DTYPE), width=w, height=h)

    assert abs(float(cam.tan_fov_x[0]) - w / (2 * fx)) < 1e-6
    assert abs(float(cam.tan_fov_y[0]) - h / (2 * fy)) < 1e-6
    assert abs(float(cam.fov_x[0]) - ref.focal2fov(fx, w)) < 1e-6


def test_perspective_camera_matches_reference() -> None:
    """由 fovy 构造时应与 `PerspectiveCamera` 的 fov_x 推导一致。"""
    import math

    from live3dgsavatar.core.types import Camera

    ref = _ref()
    w, h = 800, 600
    fov_y = math.pi / 3
    reference = ref.PerspectiveCamera(fov_y=fov_y, width=w, height=h)
    cam = Camera.from_fov(fov_y=fov_y, width=w, height=h)

    assert abs(float(cam.fx[0]) - float(reference.fx)) < 1e-4
    assert abs(float(cam.fov_x[0]) - float(reference.fov_x)) < 1e-6
    ok, msg = compare(cam.proj[0].cpu(), reference.get_proj.cpu(), 1e-6, "proj")
    assert ok, msg


def test_camera_position_matches_reference() -> None:
    """`position` 应等于参照的 `get_pos`（view→world 的平移）。"""
    from live3dgsavatar.core.types import Camera

    ref = _ref()
    rng = np.random.default_rng(7)
    R, T = _random_pose(rng)
    K = np.array([[500.0, 0.0, 128.0], [0.0, 500.0, 128.0], [0.0, 0.0, 1.0]], dtype=K_DTYPE)
    reference = ref.IntrinsicsCamera(K=K, R=R, T=T, width=256, height=256)
    cam = Camera.from_intrinsics_extrinsics(K=K, R=R, T=T, width=256, height=256)

    ok, msg = compare(cam.position[0].cpu(), reference.get_pos.cpu(), 1e-6, "position")
    assert ok, msg


def test_reference_root_is_readonly_policy() -> None:
    """健全性检查：确认定位到的是参照仓库根目录。"""
    root = require_reference()
    assert (root / "camera" / "camera.py").exists(), f"{root} 不像 RGBAvatar 仓库"
    assert root == REFERENCE_ROOT
