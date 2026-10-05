"""核心层端到端测试（**不需要 GPU**）。

覆盖三个在「有 GPU 才跑」的等价门里容易被掩盖的环节：

1. `GaussianAvatar.deform` 的**完整链**（混合 → 绑定）在 CPU 上必须能跑通，
   且张量形状 / dtype / 设备一致、四元数是单位长度；
2. `blend_weight=None`（`blend_start_iter` 之前的行为）必须等价于「权重全零 + 基态」；
3. 相机矩阵边界转换与光栅化封装里的**输入校验**——这些是 GPU 路径上
   唯一能在 CPU 上验证的部分，也是出错时最难定位的部分（症状通常是全黑或镜像）。
"""

from __future__ import annotations

import numpy as np
import torch

ATOL = 1e-5


# ------------------------------------------------------------ 共用构造 ----


def _tiny_mesh(b: int = 1, v: int = 9, f: int = 8, verts: torch.Tensor | None = None):
    """构造一个拓扑合法的小网格（顶点落在球面上，避免退化面）。

    `verts` 可覆盖自动生成的顶点（用于取同一批网格的某一帧切片）。
    """
    from live3dgsavatar.core.types import Mesh

    if verts is None:
        torch.manual_seed(21)
        verts = torch.nn.functional.normalize(torch.randn(b, v, 3), dim=-1)
    # 面：循环三元组，索引都在 [0, v)
    faces = torch.tensor([[i % v, (i + 1) % v, (i + 3) % v] for i in range(f)],
                         dtype=torch.int32)
    uvs = torch.rand(v, 2) * 0.9 + 0.05          # 落在 (0,1) 内，避免边界 texel 抖动
    uv_faces = faces.clone()
    return Mesh(verts=verts, faces=faces, uvs=uvs, uv_faces=uv_faces)


def _tiny_avatar(n: int = 64, k: int = 3, tex_size: int = 8, d: int = 7):
    """构造带合法绑定的小 avatar。高斯数由 `n` 直接给定（不真跑 UV 光栅化）。"""
    from live3dgsavatar.core.avatar import AvatarConfig, GaussianAvatar
    from live3dgsavatar.core.deform import Binding

    torch.manual_seed(22)
    f = 8
    binding = Binding(
        face_id=torch.randint(0, f, (n,)),
        face_bary=torch.nn.functional.normalize(torch.rand(n, 3) + 0.2, dim=-1),
        valid_mask=torch.ones(tex_size * tex_size, dtype=torch.bool),
    )
    cfg = AvatarConfig(tex_size=tex_size, num_basis_in=d, num_basis_blend=k,
                       mlp_hidden=(), use_weight_proj=True)
    return GaussianAvatar(cfg, binding)


# ------------------------------------------------------- deform 全链路 ----


def test_deform_runs_on_cpu_and_is_wellformed() -> None:
    """混合 → 绑定 全链路：形状、dtype、设备与四元数单位性。"""
    avatar = _tiny_avatar()
    mesh = _tiny_mesh(b=2)
    bw = torch.randn(2, avatar.config.num_basis_in)

    gs = avatar.deform(mesh, bw)

    assert gs.space == "world", "deform 的输出必须是世界空间"
    assert gs.batch_size == 2 and gs.num_gaussians == avatar.num_gaussians
    assert gs.xyz.shape == (2, avatar.num_gaussians, 3)
    assert gs.color.shape == (2, avatar.num_gaussians, 1, 3)
    assert gs.xyz.dtype == mesh.verts.dtype, f"dtype 不一致：{gs.xyz.dtype} vs {mesh.verts.dtype}"

    norms = gs.rotation.norm(dim=-1)
    assert float((norms - 1.0).abs().max()) < ATOL, "绑定后的四元数必须是单位长度"

    # 不透明度与尺度必须落在物理范围内（激活已生效）
    assert float(gs.opacity.min()) >= 0.0 and float(gs.opacity.max()) <= 1.0
    assert float(gs.scaling.min()) > 0.0


def test_deform_batch_dimension_follows_mesh() -> None:
    """batch 维必须来自 mesh；不一致时必须报错而不是广播出错误结果。"""
    avatar = _tiny_avatar()
    for b in (1, 3):
        bw = torch.randn(b, avatar.config.num_basis_in)
        gs = avatar.deform(_tiny_mesh(b=b), bw)
        assert gs.batch_size == b, f"batch 应为 {b}，实际 {gs.batch_size}"

    # mesh 与 blend_weight 的 batch 不一致应报错
    try:
        avatar.deform(_tiny_mesh(b=2), torch.randn(3, avatar.config.num_basis_in))
    except ValueError:
        pass
    else:
        raise AssertionError("mesh 与 blend_weight 的 batch 不一致应报错")


def test_deform_without_blend_weight_uses_base_only() -> None:
    """`blend_weight=None` 必须等价于「权重全零」。

    对应参照实现在 `blend_start_iter` 之前跳过混合的行为。
    """
    avatar = _tiny_avatar()
    mesh = _tiny_mesh(b=2)

    gs_none = avatar.deform(mesh, None)
    gs_zero = avatar.deform(mesh, torch.zeros(2, avatar.config.num_basis_in))

    for name in ("xyz", "rotation", "opacity", "scaling", "color"):
        a = getattr(gs_none, name)
        b = getattr(gs_zero, name)
        d = float((a.double() - b.double()).abs().max())
        assert d < ATOL, f"None 与全零权重不等价（{name}）：max|Δ| = {d:.3e}"


def test_blend_weight_actually_affects_output() -> None:
    """健全性检查：基不为零时，不同权重必须给出不同结果。

    防止出现「混合被静默跳过、永远返回基态」这类在等价门里也会通过的错误
    （因为参照与我们都返回基态时比对是一致的）。
    """
    from live3dgsavatar.core.avatar import AvatarConfig, GaussianAvatar
    from live3dgsavatar.core.deform import Binding

    n, k, f = 32, 4, 8
    torch.manual_seed(23)
    binding = Binding(
        face_id=torch.randint(0, f, (n,)),
        face_bary=torch.nn.functional.normalize(torch.rand(n, 3) + 0.2, dim=-1),
        valid_mask=torch.ones(64, dtype=torch.bool),
    )
    cfg = AvatarConfig(tex_size=8, num_basis_in=6, num_basis_blend=k,
                       mlp_hidden=(), use_weight_proj=True)
    avatar = GaussianAvatar(cfg, binding)
    with torch.no_grad():
        avatar.xyz_b.normal_(0, 0.1)
        avatar.rotation_b.normal_(0, 0.1)
        avatar.color_b.normal_(0, 0.1)

    mesh = _tiny_mesh(b=1)
    out_a = avatar.deform(mesh, torch.zeros(1, 6))
    out_b = avatar.deform(mesh, torch.full((1, 6), 2.0))
    d = float((out_a.xyz - out_b.xyz).detach().abs().max())
    assert d > 1e-4, "不同驱动权重给出了相同结果，说明混合没有生效"


def test_weight_module_is_actually_used() -> None:
    """加权投影模块（MLP）必须真的参与计算。"""
    from live3dgsavatar.core.avatar import AvatarConfig, GaussianAvatar
    from live3dgsavatar.core.deform import Binding

    n, f = 32, 8
    torch.manual_seed(24)
    binding = Binding(
        face_id=torch.randint(0, f, (n,)),
        face_bary=torch.nn.functional.normalize(torch.rand(n, 3) + 0.2, dim=-1),
        valid_mask=torch.ones(64, dtype=torch.bool),
    )
    cfg = AvatarConfig(tex_size=8, num_basis_in=6, num_basis_blend=3,
                       mlp_hidden=(5,), use_weight_proj=True)
    avatar = GaussianAvatar(cfg, binding)
    with torch.no_grad():
        avatar.xyz_b.normal_(0, 0.1)

    probe = torch.ones(2, 6)
    before = avatar.build_blend_field().project_weight(probe).detach().clone()
    assert before.shape == (2, 3), f"project_weight 输出应为 [B, K]，实际 {tuple(before.shape)}"

    # 改动 MLP 权重后，同一输入必须给出不同投影结果
    with torch.no_grad():
        for param in avatar.weight_module.parameters():
            param.add_(0.5)
    after = avatar.build_blend_field().project_weight(probe).detach()
    assert not torch.allclose(before, after), "改动 MLP 权重后投影结果未变，说明 MLP 未参与计算"


# --------------------------------------------------- 相机矩阵边界 ----


def test_kernel_matrix_transpose_roundtrip() -> None:
    """`to_kernel_matrix` 做一次转置且返回连续张量（内核按列主序读取）。"""
    from live3dgsavatar.core.render import to_kernel_matrix

    torch.manual_seed(25)
    m = torch.randn(4, 4)
    k = to_kernel_matrix(m)
    assert k.shape == (4, 4) and k.is_contiguous()
    assert torch.equal(k, m.transpose(0, 1)), "必须是转置"
    assert torch.equal(to_kernel_matrix(k), m), "转置两次应回到原矩阵"

    # 投影矩阵的组合顺序：full_proj = proj @ w2c，不可交换
    from live3dgsavatar.core.types import Camera

    K = torch.tensor([[200.0, 0, 32], [0, 200.0, 32], [0, 0, 1]])
    cam = Camera.from_intrinsics_extrinsics(
        K=K, R=torch.eye(3), T=torch.tensor([0.0, 0.0, -1.0]), width=64, height=64)
    assert torch.allclose(cam.full_proj[0], cam.proj[0] @ cam.w2c[0], atol=1e-6)
    assert not torch.allclose(cam.proj[0] @ cam.w2c[0], cam.w2c[0] @ cam.proj[0],
                              atol=1e-3), "顺序不可交换，说明测试用例没覆盖真实情形"

    # 形状错误要报错
    for bad in (torch.randn(3, 3), torch.randn(4, 3)):
        try:
            to_kernel_matrix(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{tuple(bad.shape)} 应被拒绝")


def test_rasterizer_rejects_wrong_space_and_batch() -> None:
    """光栅化/绑定封装必须在**进入 CUDA 内核之前**拒绝错误输入。

    这些校验是 GPU 路径上唯一能在 CPU 上验证的部分；少了它们，
    错误输入会以「全黑」「三角形乱飞」的形式出现在渲染结果里，极难定位。
    """
    from live3dgsavatar.core.deform import MeshBinder
    from live3dgsavatar.core.types import Camera, GaussianSet, Mesh

    avatar = _tiny_avatar(n=16, k=2, tex_size=4, d=5)
    mesh = _tiny_mesh(b=1)
    binder = MeshBinder(avatar.to_binding())

    gs_world = GaussianSet(
        xyz=torch.zeros(1, 16, 3), rotation=torch.zeros(1, 16, 4),
        scaling=torch.ones(1, 16, 3), opacity=torch.ones(1, 16, 1),
        color=torch.zeros(1, 16, 1, 3), space="world")
    try:
        binder.bind(gs_world, mesh)
    except ValueError:
        pass
    else:
        raise AssertionError("Binder 应拒绝 world 空间输入（防止重复绑定）")

    # 非世界空间的高斯不得进入光栅化器（在导入 CUDA 扩展前就被拦下）
    from live3dgsavatar.core.render.rasterizer import SimpleRasterizer

    gs_tangent = GaussianSet(
        xyz=torch.zeros(1, 16, 3), rotation=torch.zeros(1, 16, 4),
        scaling=torch.ones(1, 16, 3), opacity=torch.ones(1, 16, 1),
        color=torch.zeros(1, 16, 1, 3), space="tangent")
    cam = Camera.from_intrinsics_extrinsics(
        K=torch.tensor([[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]]),
        R=torch.eye(3), T=torch.zeros(3), width=64, height=64)
    try:
        SimpleRasterizer().render(gs_tangent, cam, torch.zeros(3))
    except ValueError as e:
        assert "世界空间" in str(e), f"报错信息应说明空间要求，实际：{e}"
    else:
        raise AssertionError("Rasterizer 应拒绝非世界空间的高斯")


def test_normalize_bg_broadcasts_correctly() -> None:
    """背景色规整：`[3]` → `[B,3]`，训练态的 `[B,3,1,1]` 也要能 reshape。"""
    from live3dgsavatar.core.render.rasterizer import _normalize_bg

    one = torch.tensor([1.0, 0.0, 0.0])
    assert _normalize_bg(one, 4).shape == (4, 3)
    assert torch.allclose(_normalize_bg(one, 4)[2], one)

    per_sample = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])
    out = _normalize_bg(per_sample, 2)
    assert out.shape == (2, 3) and torch.allclose(out[1], per_sample[1])

    # 训练态的 [B,3,1,1] 随机背景色
    random_bg = torch.rand(2, 3, 1, 1)
    assert _normalize_bg(random_bg, 2).shape == (2, 3)

    try:
        _normalize_bg(torch.rand(3, 3), 2)
    except ValueError:
        pass
    else:
        raise AssertionError("背景色行数与 batch 不匹配应报错")


def test_binding_validation_rejects_bad_shapes() -> None:
    """Binding 与 GaussianSet 的构造期校验。"""
    from live3dgsavatar.core.deform import Binding
    from live3dgsavatar.core.types import GaussianSet

    # face_bary 的 N 与 face_id 不一致
    try:
        Binding(face_id=torch.zeros(4, dtype=torch.long),
                face_bary=torch.zeros(5, 3), valid_mask=torch.ones(4, dtype=torch.bool))
    except ValueError:
        pass
    else:
        raise AssertionError("face_id 与 face_bary 的 N 不一致应报错")

    # valid_mask 必须是 bool
    try:
        Binding(face_id=torch.zeros(4, dtype=torch.long),
                face_bary=torch.zeros(4, 3), valid_mask=torch.ones(4))
    except ValueError:
        pass
    else:
        raise AssertionError("valid_mask 非 bool 应报错")

    # GaussianSet 的颜色必须是单阶 SH
    try:
        GaussianSet(xyz=torch.zeros(1, 4, 3), rotation=torch.zeros(1, 4, 4),
                    scaling=torch.ones(1, 4, 3), opacity=torch.ones(1, 4, 1),
                    color=torch.zeros(1, 4, 3, 3), space="tangent")   # SH 维写成 3
    except ValueError:
        pass
    else:
        raise AssertionError("多阶 SH 应被拒绝（本项目 sh_degree=0）")


def test_batched_deform_equals_per_frame_deform() -> None:
    """**回归测试**：批量 deform 必须与逐帧 deform 数值等价。

    `scripts/render_test.py` 为与参照对齐批大小而走批量路径；
    若批量与逐帧有差异，性能对比就失去意义（比的可能不是同一件事）。

    ⚠️ 逐帧的网格**直接切自批网格**（`verts` / `faces` / `uvs` 全部取自同一份），
    不能各自调 `_tiny_mesh` —— 那样 `uvs` 会因 RNG 状态不同而不同，
    导致测的其实是**不同的 UV 参数化**，而不是批量与逐帧的差异。
    """
    from live3dgsavatar.core.types import Mesh

    avatar = _tiny_avatar()
    b = 3
    mesh = _tiny_mesh(b=b)
    bw = torch.randn(b, avatar.config.num_basis_in)

    def frame_mesh(i: int) -> Mesh:
        return Mesh(verts=mesh.verts[i:i + 1], faces=mesh.faces,
                    uvs=mesh.uvs, uv_faces=mesh.uv_faces)

    with torch.no_grad():
        batched = avatar.deform(mesh, bw)
        singles = [avatar.deform(frame_mesh(i), bw[i:i + 1]) for i in range(b)]

    assert batched.xyz.shape == (b, avatar.num_gaussians, 3)
    for i in range(b):
        for name in ("xyz", "rotation", "color", "opacity", "scaling"):
            a = getattr(batched, name)[i]
            c = getattr(singles[i], name)[0]
            d = float((a - c).abs().max())
            assert d < 1e-5, (
                f"批量与逐帧在第 {i} 帧的 {name} 不一致：max|Δ| = {d:.3e}")

    # 用例必须真的在跑多个不同的帧，否则测不出批量维度的问题
    spread = max(float((mesh.verts[i] - mesh.verts[0]).abs().max()) for i in range(b))
    assert spread > 1e-3, "各帧网格应不同，否则用例失去意义"


def test_build_binder_reuses_instance() -> None:
    """**回归测试**：`build_binder()` 必须返回**同一个**实例。

    `MeshBinder` 内部缓存了「去重后的绑定面 + 反查索引」（纯模板依赖）。
    若每次调用都新建实例，该缓存每帧失效，`torch.unique` 就会每帧重跑
    （CPU 上约 2 ms）—— **比不做该优化还慢**。
    曾真实发生：`deform()` 每帧调 `build_binder()`，导致优化变成回退。
    """
    avatar = _tiny_avatar()
    first = avatar.build_binder()
    second = avatar.build_binder()
    assert first is second, (
        "build_binder() 每次返回新实例会让 MeshBinder 的面索引缓存失效，"
        "务必复用同一实例")


def test_face_index_cache_is_not_recomputed_per_frame() -> None:
    """面索引缓存必须命中：同一 binder 多次 bind 不应重跑 `torch.unique`。"""
    import torch as _torch

    from live3dgsavatar.core.types import GaussianSet

    avatar = _tiny_avatar()
    binder = avatar.build_binder()
    mesh = _tiny_mesh(b=1)
    gs = GaussianSet(
        xyz=_torch.zeros(1, avatar.num_gaussians, 3),
        rotation=_torch.tensor([[[1.0, 0, 0, 0]]]).repeat(1, avatar.num_gaussians, 1),
        scaling=_torch.ones(1, avatar.num_gaussians, 3),
        opacity=_torch.ones(1, avatar.num_gaussians, 1),
        color=_torch.zeros(1, avatar.num_gaussians, 1, 3),
        space="tangent")

    calls = {"n": 0}
    real_unique = _torch.unique

    def counting_unique(*a, **kw):
        calls["n"] += 1
        return real_unique(*a, **kw)

    _torch.unique = counting_unique
    try:
        for _ in range(5):
            binder.bind(gs, mesh)
    finally:
        _torch.unique = real_unique

    assert calls["n"] <= 1, (
        f"5 次 bind 触发了 {calls['n']} 次 torch.unique；面索引缓存未生效")
