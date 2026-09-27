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


def test_face_tbn_bases_may_be_skewed() -> None:
    """TBN 三列各自单位长，但 **tangent 与 bitangent 一般不正交**。

    代数上 `normalize(t)·normalize(b) = −cos(∠A')`（A' 为 UV 三角形在顶点 a 处的内角），
    两者正交当且仅当该角为直角。一般 UV 图不是正交参数化。

    ⚠️ 本测试明确**不断言正交**：曾误把正交性当正确性判据（见 MIGRATION D 节）。
    这里反过来固化"非正交是正常现象"，防止以后有人去"修"它。
    """
    from live3dgsavatar.core.deform.tbn import compute_face_tbn

    # 用一个明显非直角的 UV 三角形：顶点 a 处 u 方向与 v 方向夹角 45°
    verts = torch.tensor([[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]])
    uvs = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]])   # a 处内角 45°
    tbn = compute_face_tbn(verts, uvs)[0, 0]                      # [3, 3]，列为基

    # 逐列单位长
    col_norms = tbn.norm(dim=0)
    assert float((col_norms - 1.0).abs().max()) < 1e-5, f"列模长非 1：{col_norms}"

    # tangent 与 bitangent 应当**不正交**
    cos_tb = float((tbn[:, 0] * tbn[:, 1]).sum())
    assert abs(cos_tb) > 1e-3, (
        f"本例 UV 在顶点 a 处内角为 45°，tangent/bitangent 不应正交，"
        f"实测 cos = {cos_tb:.3e}")

    # 理论值：cos = -cos(45°) = -0.7071
    assert abs(cos_tb + 2 ** -0.5) < 1e-4, f"cos 应约等于 -0.7071，实测 {cos_tb:.6f}"

    # 法线仍必须与几何法线平行（这条是由几何保证的）
    e1 = verts[0, 0, 1] - verts[0, 0, 0]
    e2 = verts[0, 0, 2] - verts[0, 0, 0]
    geo_n = torch.nn.functional.normalize(torch.cross(e1, e2, dim=-1), dim=-1)
    assert abs(float((tbn[:, 2] * geo_n).sum()) - 1.0) < 1e-5


def test_face_tbn_orthonormal_mode_is_orthonormal() -> None:
    """`mode="orthonormal"` 必须给出正交基，且保持法线与 tangent 所在平面。"""
    from live3dgsavatar.core.deform.tbn import compute_face_tbn

    torch.manual_seed(31)
    verts = torch.randn(2, 9, 3, 3)
    uvs = torch.rand(9, 3, 2)

    ref_mode = compute_face_tbn(verts, uvs, mode="reference")
    orth = compute_face_tbn(verts, uvs, mode="orthonormal")

    # reference 模式大概率非正交
    g_ref = ref_mode.transpose(-1, -2) @ ref_mode
    dev_ref = float((g_ref - torch.eye(3)).abs().max())
    assert dev_ref > 1e-3, "本例应存在 UV 剪切；若否则测试失去意义"

    # orthonormal 模式必须正交
    g_o = orth.transpose(-1, -2) @ orth
    dev_o = float((g_o - torch.eye(3)).abs().max())
    assert dev_o < 1e-5, f"orthonormal 模式不正交：max|RᵀR-I| = {dev_o:.3e}"

    # 法线不变（以几何法线为准）
    assert torch.allclose(orth[..., :, 2], ref_mode[..., :, 2], atol=1e-6)

    # tangent 仍在原 tangent 与法线张成的平面内（只去掉法向分量）
    n = ref_mode[..., :, 2]
    t = ref_mode[..., :, 0]
    t_plane = t - (t * n).sum(-1, keepdim=True) * n
    t_plane = torch.nn.functional.normalize(t_plane, dim=-1)
    assert torch.allclose(orth[..., :, 0], t_plane, atol=1e-5)


def test_face_tbn_rejects_unknown_mode() -> None:
    from live3dgsavatar.core.deform.tbn import compute_face_tbn

    try:
        compute_face_tbn(torch.randn(1, 1, 3, 3), torch.rand(1, 3, 2), mode="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("未知 mode 应报错")


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


# --------------------------------------------------------------- 刚性等变 --


def _nondegenerate_mesh(b: int = 1, side: int = 6, shear: float = 0.35):
    """规则栅格三角剖分 + 带剪切的 UV，保证 TBN 非正交但可逆。"""
    from live3dgsavatar.core.types import Mesh

    v = side * side
    idx = torch.arange(v).reshape(side, side)
    tris = []
    for i in range(side - 1):
        for j in range(side - 1):
            a, bb, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j], idx[i + 1, j + 1]
            tris += [[a, bb, c], [bb, d, c]]
    faces = torch.tensor(tris, dtype=torch.int32)

    gx, gy = torch.meshgrid(torch.linspace(0, 1, side), torch.linspace(0, 1, side),
                            indexing="ij")
    # 带剪切的 UV：tangent 与 bitangent 不正交
    uvs = torch.stack([(gx + shear * gy).reshape(-1),
                       (gy + shear * gx).reshape(-1)], dim=-1)

    torch.manual_seed(0)
    verts = torch.randn(b, v, 3) * 0.1
    return Mesh(verts=verts, faces=faces, uvs=uvs, uv_faces=faces)


def test_binding_is_rigidly_equivariant() -> None:
    """**核心正确性判据**：绑定必须关于刚体变换等变。

    若模板做刚体变换 `T = (Q, t)`，则
    `bind(T(M), x) == Q · bind(M, x) + t`。这条性质**不依赖任何参照实现**。

    ⚠️ 它同时给出矩阵乘法顺序的判据。旋转矩阵满足 `R(QM) = Q·R(M)`（已验证），
    于是：

        R(QM)  · x = Q·(R(M)·x)        ← R·x 形式**严格等变**
        R(QM)ᵀ · x ≠ Q·(R(M)ᵀ·x)      ← Rᵀ·x 形式仅在 R 正交时等变

    真实网格的 TBN 一般**不正交**（UV 参数化带剪切），故 `Rᵀ·x` 不等变。
    参照实现的 `mesh_binding` 用的正是 `Rᵀ·x`。见 docs/MIGRATION.md D 节。
    """
    from live3dgsavatar.core.deform.tbn import compute_face_tbn

    mesh = _nondegenerate_mesh()
    faces = mesh.faces
    fid = torch.arange(4).repeat(mesh.num_faces // 4)[:mesh.num_faces] % mesh.num_faces
    n = fid.numel()
    # ⚠️ 重心必须**和为 1**：normalize 除的是 L2 范数，不能用它构造重心
    raw = torch.rand(n, 3) + 0.1
    bary = raw / raw.sum(dim=-1, keepdim=True)
    assert float((bary.sum(-1) - 1).abs().max()) < 1e-6, "重心之和必须为 1"
    xyz_tan = torch.randn(1, n, 3) * 0.01

    def rotation_term(mesh_v, xyz, transpose: bool):
        tbn = compute_face_tbn(mesh_v[:, faces], mesh.uvs[mesh.uv_faces])
        r = tbn[:, fid]
        m = r.transpose(-1, -2) if transpose else r
        return torch.matmul(m, xyz.unsqueeze(-1)).squeeze(-1)

    def offset_term(mesh_v):
        tv = mesh_v[:, faces][:, fid]
        return (tv * bary.unsqueeze(0).unsqueeze(-1)).sum(-2)

    # 用例必须确实非正交，否则两种形式不可区分
    tbn0 = compute_face_tbn(mesh.verts[:, faces], mesh.uvs[mesh.uv_faces])
    shear = float((tbn0[0, 0].T @ tbn0[0, 0] - torch.eye(3)).abs().max())
    assert shear > 1e-3, f"用例 TBN 近似正交（偏差 {shear:.2e}），无法区分乘法顺序"

    torch.manual_seed(1)
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if float(torch.det(Q)) < 0:
        Q[:, 0] = -Q[:, 0]
    t = torch.randn(3) * 0.05
    mesh2 = mesh.verts @ Q.T + t

    # TBN 的变换律（本测试的基石）
    tbn2 = compute_face_tbn(mesh2[:, faces], mesh.uvs[mesh.uv_faces])
    d_tbn = float((tbn2[0] - torch.einsum("ij,fjk->fik", Q, tbn0[0])).abs().max())
    assert d_tbn < 1e-5, f"TBN 应满足 R(QM) = Q·R(M)，实测 max|Δ| = {d_tbn:.3e}"

    # 旋转项：只有 R·x 等变
    d_R = float((rotation_term(mesh2, xyz_tan, False)
                 - rotation_term(mesh.verts, xyz_tan, False) @ Q.T).abs().max())
    d_T = float((rotation_term(mesh2, xyz_tan, True)
                 - rotation_term(mesh.verts, xyz_tan, True) @ Q.T).abs().max())
    assert d_R < 1e-5, f"R·x 应严格等变，实测 max|Δ| = {d_R:.3e}"
    assert d_T > 1e-3, (
        f"Rᵀ·x 在非正交 TBN 下应不等变（这是关键结论），实测 max|Δ| = {d_T:.3e}；"
        "若通过则说明用例失去了区分力")

    # 位置项：重心组合必须等变（Σ bary = 1 是前提）
    d_off = float((offset_term(mesh2) - (offset_term(mesh.verts) @ Q.T + t)).abs().max())
    assert d_off < 1e-5, f"重心插值应等变，实测 max|Δ| = {d_off:.3e}"


def test_orthonormal_mode_makes_transpose_equal_inverse() -> None:
    """正交化的**真正**好处：`Rᵀ = R⁻¹`，从而 `Rᵀ·x` 这一写法重新成立。

    ⚠️ 这里**不能**断言"`Rᵀ·x` 与 `R·x` 相等" —— 那是错的。
    正交只给出 `Rᵀ = R⁻¹`；`Rᵀ` 与 `R` 本身一般不同（例如 90° 旋转阵，
    反对称部分 ||Rᵀ−R|| ≈ 2，但它完全正交）。
    曾把 `||Rᵀx − Rx||` 与 `||RᵀR−I||` 混为一谈，写出错误的不等式。
    """
    from live3dgsavatar.core.deform.tbn import compute_face_tbn

    mesh = _nondegenerate_mesh()
    faces = mesh.faces
    xyz = torch.randn(1, 8, 3) * 0.01

    T = compute_face_tbn(mesh.verts[:, faces], mesh.uvs[mesh.uv_faces],
                         mode="orthonormal")
    M = T[0, 0]

    # 正交性
    assert float((M.T @ M - torch.eye(3)).abs().max()) < 1e-5
    assert abs(float(torch.det(M)) - 1.0) < 1e-4, "应仍是右手系（det = +1）"

    # 关键性质：转置等于逆 —— 这才是 Rᵀ·x 成立的前提
    d_inv = float((M.T - torch.linalg.inv(M)).abs().max())
    assert d_inv < 1e-5, f"正交化后应满足 Rᵀ = R⁻¹，实测 max|Δ| = {d_inv:.3e}"

    # 对照：reference 模式下 Rᵀ ≠ R⁻¹（这正是问题所在）
    T_ref = compute_face_tbn(mesh.verts[:, faces], mesh.uvs[mesh.uv_faces],
                             mode="reference")
    M_ref = T_ref[0, 0]
    d_ref = float((M_ref.T - torch.linalg.inv(M_ref)).abs().max())
    assert d_ref > 1e-2, (
        f"reference 模式下应明显有 Rᵀ ≠ R⁻¹，实测 {d_ref:.3e}；"
        "若一致则说明用例的 UV 剪切不足")

    # 并说明：正交化**不**意味着 Rᵀ·x == R·x
    a = torch.matmul(T[:, :8].transpose(-1, -2), xyz.unsqueeze(-1)).squeeze(-1)
    b = torch.matmul(T[:, :8], xyz.unsqueeze(-1)).squeeze(-1)
    assert float((a - b).abs().max()) > 1e-3, (
        "Rᵀ 与 R 一般不同；若相等说明该面恰好对称，用例不具代表性")


def test_normalize_is_not_a_barycentric_constructor() -> None:
    """**回归测试**：`F.normalize` 除的是 L2 范数，不是求和。

    曾用它构造重心坐标，导致 `Σ bary != 1`，进而让等变性测试假失败。
    本测试把这个坑固定下来。
    """
    raw = torch.tensor([[1.0, 2.0, 3.0]])
    wrong = torch.nn.functional.normalize(raw, dim=-1)
    right = raw / raw.sum(dim=-1, keepdim=True)

    assert abs(float(wrong.sum()) - 1.0) > 0.1, "F.normalize 不应给出和为 1 的结果"
    assert abs(float(right.sum()) - 1.0) < 1e-6
    assert torch.allclose(wrong.norm(dim=-1), torch.ones(1), atol=1e-6), \
        "F.normalize 保证 L2 范数为 1"
