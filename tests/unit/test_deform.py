"""TBN / 混合 / 绑定的数值等价测试（**不需要 GPU**）。

对照方式：
- TBN：与参照实现的 Python 版 `utils.compute_face_tbn` 逐元素比较
  （该文件是纯 torch，可直接按路径加载）。
- 混合：与独立的 `einsum` 实现比较，二者互为交叉验证。
- 绑定：与「照着 CUDA `mesh_binding.cu` 直译」的参考实现比较。
"""

from __future__ import annotations

import numpy as np
import torch

from support import compare, load_module_from, require_reference

ATOL = 1e-5


def _reference_utils():
    """参照仓库的 `utils.py`（纯 torch，但含 nvdiffrast 之外的依赖，按需加载）。"""
    root = require_reference()
    return load_module_from(root / "utils.py", "_rgba_reference_utils")


# --------------------------------------------------------------------- TBN --


def test_face_tbn_matches_reference_python() -> None:
    from live3dgsavatar.core.deform.tbn import compute_face_tbn

    ref = _reference_utils()
    torch.manual_seed(0)
    b, f = 2, 7
    verts = torch.randn(b, f, 3, 3)
    uvs = torch.rand(f, 3, 2)

    mine = compute_face_tbn(verts, uvs)
    theirs = ref.compute_face_tbn(verts, uvs)
    ok, msg = compare(mine, theirs, ATOL, "compute_face_tbn vs reference.utils")
    assert ok, msg


def test_face_tbn_layout_is_column_basis() -> None:
    """TBN 的第 j 列必须是第 j 个基向量（与 CUDA 内核输出布局一致）。

    用一个已知的平直三角面验证：法线应等于 `cross(e1, e2)` 的归一化方向。
    """
    from live3dgsavatar.core.deform.tbn import compute_face_tbn

    # 单个面，位于 z = 0 平面，UV 与顶点轴对齐 → tangent=+x, bitangent=+y, normal=+z
    verts = torch.tensor([[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]])
    uvs = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    tbn = compute_face_tbn(verts, uvs)[0, 0]          # [3, 3]

    tangent, bitangent, normal = tbn[:, 0], tbn[:, 1], tbn[:, 2]
    assert torch.allclose(tangent, torch.tensor([1.0, 0.0, 0.0]), atol=1e-6), tangent
    assert torch.allclose(bitangent, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6), bitangent
    assert torch.allclose(normal, torch.tensor([0.0, 0.0, 1.0]), atol=1e-6), normal


def test_face_tbn_bases_have_unit_norm() -> None:
    """TBN 三列必须各自单位长。

    ⚠️ 只断言**逐列单位长**，不断言列间正交：参照实现是逐列 normalize，
    而真实 UV 常带剪切，tangent/bitangent 并不正交（那是 UV 参数化的属性，
    不是实现的 bug）。断言正交会写出一个恒失败的测试。
    """
    from live3dgsavatar.core.deform.tbn import compute_face_tbn

    torch.manual_seed(3)
    verts = torch.randn(2, 11, 3, 3)
    uvs = torch.rand(11, 3, 2)
    tbn = compute_face_tbn(verts, uvs)                # [B, F, 3, 3]，列为基

    col_norms = tbn.norm(dim=-2)                      # [B, F, 3]
    d = float((col_norms - 1.0).abs().max())
    assert d < 1e-5, f"TBN 存在非单位列，max|Δ| = {d:.3e}"

    # 至少法线与三角面法线方向一致（正交性中唯一由几何保证的一条）
    e1 = verts[..., 1, :] - verts[..., 0, :]
    e2 = verts[..., 2, :] - verts[..., 0, :]
    geo_normal = torch.nn.functional.normalize(torch.cross(e1, e2, dim=-1), dim=-1)
    dot = (tbn[..., :, 2] * geo_normal).sum(dim=-1)
    assert float((dot.abs() - 1.0).abs().max()) < 1e-5, "normal 与几何法线不平行"


# ------------------------------------------------------------------- blend --


def _einsum_linear_blending(weights, base, basis):
    return base.unsqueeze(0) + torch.einsum("bk,kn...->bn...", weights, basis)


def test_linear_blending_matches_einsum() -> None:
    from live3dgsavatar.core.deform.blend import linear_blending

    torch.manual_seed(1)
    for base_shape in [(5, 3), (5, 4), (5, 1, 3)]:
        for k in (1, 4, 20):
            b = 3
            weights = torch.randn(b, k)
            base = torch.randn(*base_shape)
            basis = torch.randn(k, *base_shape)
            mine = linear_blending(weights, base, basis)
            theirs = _einsum_linear_blending(weights, base, basis)
            # 容差 1e-5：K=20 时 float32 的求和顺序差异可达 ~2e-6
            ok, msg = compare(mine, theirs, 1e-5, f"linear_blending base={base_shape} K={k}")
            assert ok, msg


def test_linear_blending_shape_errors_are_loud() -> None:
    """形状错误必须报错，而不是像参照实现那样静默给出错误结果。"""
    from live3dgsavatar.core.deform.blend import linear_blending

    base = torch.randn(5, 3)
    basis = torch.randn(4, 5, 3)

    for bad in [
        (torch.randn(3), base, basis),                 # weights 缺 batch 维
        (torch.randn(3, 7), base, basis),              # K 与 basis 不符
        (torch.randn(3, 4), base, torch.randn(4, 4, 3)),  # basis 的 N 不符
    ]:
        try:
            linear_blending(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"应当报错但通过了：{[(t.shape) for t in bad]}")


def test_blend_field_activation_matches_reference_formulas() -> None:
    """激活必须与参照一致：opacity=sigmoid、scaling=exp、rotation=normalize。"""
    from live3dgsavatar.core.deform.blend import GaussianBlendField

    torch.manual_seed(2)
    n, k, d = 9, 3, 6
    field = GaussianBlendField(
        base_xyz=torch.zeros(n, 3),
        base_rotation=torch.nn.functional.normalize(torch.randn(n, 4), dim=-1),
        base_color=torch.zeros(n, 1, 3),
        base_opacity=torch.randn(n, 1),
        base_scaling=torch.randn(n, 3),
        basis_xyz=torch.randn(k, n, 3),
        basis_rotation=torch.randn(k, n, 4),
        basis_color=torch.randn(k, n, 1, 3),
        weight_module=None,
    )
    blend_weight = torch.randn(2, d)
    # weight_module 为 None 时，权重应为 D 维；此处只用前 k 列以保持形状一致
    gs = field(blend_weight[:, :k])

    assert gs.space == "tangent"
    assert gs.batch_size == 2 and gs.num_gaussians == n
    assert torch.allclose(gs.opacity[0, :, 0], torch.sigmoid(field.base_opacity[:, 0]))
    assert torch.allclose(gs.scaling[0], torch.exp(field.base_scaling))
    norms = gs.rotation.norm(dim=-1)
    assert float((norms - 1.0).abs().max()) < 1e-5, "rotation 必须是单位四元数"


def test_blend_field_rejects_extra_unsqueeze() -> None:
    """参照实现注释警告过：blend_weight 多 unsqueeze 一次会产生错误结果。"""
    from live3dgsavatar.core.deform.blend import GaussianBlendField

    n, k = 4, 2
    field = GaussianBlendField(
        base_xyz=torch.zeros(n, 3), base_rotation=torch.zeros(n, 4),
        base_color=torch.zeros(n, 1, 3), base_opacity=torch.zeros(n, 1),
        base_scaling=torch.zeros(n, 3),
        basis_xyz=torch.zeros(k, n, 3), basis_rotation=torch.zeros(k, n, 4),
        basis_color=torch.zeros(k, n, 1, 3),
    )
    for bad_shape in [(k,), (1, 1, k)]:
        try:
            field(torch.zeros(*bad_shape))
        except ValueError:
            pass
        else:
            raise AssertionError(f"blend_weight 形状 {bad_shape} 应被拒绝")


# -------------------------------------------------------------------- bind --


def _reference_binding(gs_xyz, gs_rot, tri_verts, face_tbn, bary, face_id):
    """照着 CUDA `mesh_binding.cu` 直译的参考实现（列为基 → 用 Rᵀ）。"""
    binding_rot = face_tbn[:, face_id]                       # [B, N, 3, 3]
    v = tri_verts[:, face_id]                                # [B, N, 3, 3]
    offset = (v * bary.unsqueeze(0).unsqueeze(-1)).sum(dim=-2)
    xyz = (binding_rot.transpose(-1, -2) @ gs_xyz.unsqueeze(-1)).squeeze(-1) + offset
    return xyz, binding_rot


def test_bind_matches_reference_translation() -> None:
    from live3dgsavatar.core.deform import Binding, MeshBinder
    from live3dgsavatar.core.deform.bind import matrix_to_quaternion, quaternion_multiply
    from live3dgsavatar.core.deform.tbn import compute_face_tbn
    from live3dgsavatar.core.types import GaussianSet, Mesh

    torch.manual_seed(4)
    b, f, v, n = 2, 5, 8, 13
    verts = torch.randn(b, v, 3)
    faces = torch.tensor([[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 0], [1, 3, 5]])
    uvs = torch.rand(v, 2)
    uv_faces = faces.clone()
    face_id = torch.randint(0, f, (n,))
    bary = torch.rand(n, 3)
    bary = bary / bary.sum(dim=1, keepdim=True)

    mesh = Mesh(verts=verts, faces=faces, uvs=uvs, uv_faces=uv_faces)
    binding = Binding(face_id=face_id, face_bary=bary,
                      valid_mask=torch.ones(4, dtype=torch.bool))
    binder = MeshBinder(binding)

    gs = GaussianSet(
        xyz=torch.randn(b, n, 3),
        rotation=torch.nn.functional.normalize(torch.randn(b, n, 4), dim=-1),
        scaling=torch.rand(b, n, 3) + 0.1,
        opacity=torch.rand(b, n, 1),
        color=torch.randn(b, n, 1, 3),
        space="tangent",
    )
    out = binder.bind(gs, mesh)

    assert out.space == "world", "Binder 必须输出 world 空间"

    tri = verts[:, faces]
    tbn = compute_face_tbn(tri, uvs[uv_faces])
    ref_xyz, binding_rot = _reference_binding(gs.xyz, gs.rotation, tri, tbn, bary, face_id)

    ok, msg = compare(out.xyz, ref_xyz, ATOL, "bound xyz")
    assert ok, msg

    # 旋转应为 binding_rot 的四元数与局部四元数之积
    ref_rot = quaternion_multiply(matrix_to_quaternion(binding_rot), gs.rotation)
    ok, msg = compare(out.rotation, ref_rot, ATOL, "bound rotation")
    assert ok, msg


def test_bind_rejects_world_space_input() -> None:
    from live3dgsavatar.core.deform import Binding, MeshBinder
    from live3dgsavatar.core.types import GaussianSet, Mesh

    mesh = Mesh(verts=torch.randn(1, 4, 3),
                faces=torch.tensor([[0, 1, 2]]), uvs=torch.rand(4, 2),
                uv_faces=torch.tensor([[0, 1, 2]]))
    binder = MeshBinder(Binding(torch.zeros(1, dtype=torch.long), torch.tensor([[1.0, 0.0, 0.0]]),
                                torch.ones(2, dtype=torch.bool)))
    gs = GaussianSet(xyz=torch.zeros(1, 1, 3), rotation=torch.zeros(1, 1, 4),
                     scaling=torch.ones(1, 1, 3), opacity=torch.ones(1, 1, 1),
                     color=torch.zeros(1, 1, 1, 3), space="world")
    try:
        binder.bind(gs, mesh)
    except ValueError:
        pass
    else:
        raise AssertionError("MeshBinder 应拒绝 world 空间输入（防重复绑定）")


def test_bind_batch_mismatch_is_loud() -> None:
    from live3dgsavatar.core.deform import Binding, MeshBinder
    from live3dgsavatar.core.types import GaussianSet, Mesh

    mesh = Mesh(verts=torch.randn(2, 4, 3),
                faces=torch.tensor([[0, 1, 2]]), uvs=torch.rand(4, 2),
                uv_faces=torch.tensor([[0, 1, 2]]))
    binder = MeshBinder(Binding(torch.zeros(1, dtype=torch.long), torch.tensor([[1.0, 0.0, 0.0]]),
                                torch.ones(2, dtype=torch.bool)))
    gs = GaussianSet(xyz=torch.zeros(3, 1, 3), rotation=torch.zeros(3, 1, 4),
                     scaling=torch.ones(3, 1, 3), opacity=torch.ones(3, 1, 1),
                     color=torch.zeros(3, 1, 1, 3), space="tangent")
    try:
        binder.bind(gs, mesh)
    except ValueError:
        pass
    else:
        raise AssertionError("batch 不一致应报错")
