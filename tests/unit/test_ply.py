"""PLY 序列化测试（**不需要 GPU**）。

重点验证三件事：
1. 往返一致（参数逐元素相同）；
2. **互操作**：写出的文件能被参照实现的 `load_ply` 读入（这是等价门的前提）；
3. 修掉的两个参照缺陷生效：`save_ply` 可用（K4）、配置与文件不一致时报错（K3）。
"""

from __future__ import annotations

import re
from contextlib import contextmanager

import numpy as np
import torch

from support import REFERENCE_ROOT, load_module_from, require_reference

ATOL = 0.0  # PLY 是 float32 原样存取，应逐位一致


def _tiny_avatar(n: int = 512, k: int = 3, tex_size: int = 32):
    """构造一个小 avatar：随机参数 + 合法绑定。"""
    from live3dgsavatar.core.avatar import AvatarConfig, GaussianAvatar

    torch.manual_seed(11)
    from live3dgsavatar.core.deform import Binding

    binding = Binding(
        face_id=torch.randint(0, 4, (n,)),
        face_bary=torch.nn.functional.normalize(torch.rand(n, 3) + 0.1, dim=-1),
        valid_mask=torch.ones(tex_size * tex_size, dtype=torch.bool),
    )
    cfg = AvatarConfig(tex_size=tex_size, num_basis_in=7, num_basis_blend=k,
                       mlp_hidden=(), use_weight_proj=True)
    avatar = GaussianAvatar(cfg, binding)
    with torch.no_grad():
        avatar.xyz.normal_(0, 0.01)
        avatar.opacity.normal_(0, 0.5)
        avatar.scaling.normal_(0, 0.1)
        avatar.rotation.normal_()
        avatar.color.normal_(0, 0.2)
        avatar.xyz_b.normal_(0, 0.01)
        avatar.rotation_b.normal_(0, 0.1)
        avatar.color_b.normal_(0, 0.2)
    return avatar


@contextmanager
def _no_cuda_stub():
    """在无 GPU 环境下打桩参照实现的硬编码 `.cuda()` 调用。

    参照的 `GaussianModel.__init__` 与 `load_ply` 里有 `nn.Module.cuda()` 和
    `torch.from_numpy(...).cuda()` 两类硬编码调用。我们只验证 PLY 读写路径，
    不涉及任何 GPU 计算，因此把这两者临时替换为 no-op。
    """
    original_module_cuda = torch.nn.Module.cuda
    original_tensor_cuda = torch.Tensor.cuda

    torch.nn.Module.cuda = lambda self, *a, **k: self       # type: ignore[method-assign]
    torch.Tensor.cuda = lambda self, *a, **k: self          # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.nn.Module.cuda = original_module_cuda         # type: ignore[method-assign]
        torch.Tensor.cuda = original_tensor_cuda            # type: ignore[method-assign]


def _make_reference_gaussian_model(module_name: str, num_basis_in: int, num_basis_blend: int):
    """构造参照的 `GaussianModel`（已打桩 `.cuda()`）。"""
    require_reference()
    from types import SimpleNamespace

    ref_gaussian = load_module_from(
        REFERENCE_ROOT / "model" / "gaussian.py", module_name)
    with _no_cuda_stub():
        return ref_gaussian.GaussianModel(SimpleNamespace(
            use_mlp_proj=True, num_basis_in=num_basis_in,
            num_basis_blend=num_basis_blend))


def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().cpu().to(torch.float64) - b.detach().cpu().to(torch.float64)).abs().max())


def test_ply_roundtrip_is_exact(tmp_path) -> None:
    from live3dgsavatar.core.io import load_ply, save_ply

    avatar = _tiny_avatar()
    path = tmp_path / "avatar.ply"
    save_ply(avatar, path)
    assert path.exists() and path.stat().st_size > 0

    # 不传 config：应从 PLY 内嵌的 gaussian_config 还原
    loaded = load_ply(path, config=None, device="cpu")

    assert loaded.num_gaussians == avatar.num_gaussians
    assert loaded.num_basis == avatar.num_basis
    assert loaded.config.mlp_hidden == avatar.config.mlp_hidden
    assert loaded.config.num_basis_in == avatar.config.num_basis_in

    for name in ("xyz", "opacity", "scaling", "rotation", "color",
                 "xyz_b", "rotation_b", "color_b"):
        d = _max_diff(getattr(avatar, name), getattr(loaded, name))
        assert d <= ATOL, f"{name} 往返不一致：max|Δ| = {d:.3e}"

    assert torch.equal(loaded.binding_face_id, avatar.binding_face_id)
    d = _max_diff(loaded.binding_face_bary, avatar.binding_face_bary)
    assert d <= ATOL, f"face_bary 往返不一致：{d:.3e}"

    # weight_module 参数也应还原
    ref_flat = torch.cat([p.detach().reshape(-1) for p in avatar.weight_module.state_dict().values()])
    got_flat = torch.cat([p.detach().reshape(-1) for p in loaded.weight_module.state_dict().values()])
    d = _max_diff(ref_flat, got_flat)
    assert d <= ATOL, f"weight_module 往返不一致：max|Δ| = {d:.3e}"


def test_read_ply_metadata_reports_config(tmp_path) -> None:
    from live3dgsavatar.core.io import read_ply_metadata, save_ply

    avatar = _tiny_avatar(n=512, k=5)
    path = tmp_path / "a.ply"
    save_ply(avatar, path)

    meta = read_ply_metadata(path)
    assert meta["num_gaussians"] == 512
    assert meta["num_basis_blend_from_file"] == 5
    assert meta["has_binding"] is True
    assert meta["config"]["num_basis_blend"] == 5
    assert meta["config"]["mlp_hidden"] == []


def test_load_rejects_basis_count_mismatch(tmp_path) -> None:
    """K3：配置与文件列数不一致必须报错（参照实现会静默出错）。"""
    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.io import load_ply, save_ply

    save_ply(_tiny_avatar(k=3), tmp_path / "a.ply")
    bad = AvatarConfig(tex_size=32, num_basis_in=7, num_basis_blend=9, mlp_hidden=())
    try:
        load_ply(tmp_path / "a.ply", config=bad, device="cpu")
    except ValueError as e:
        assert "num_basis" in str(e), f"报错信息应说明 num_basis 不一致，实际：{e}"
    else:
        raise AssertionError("基数量不一致应报错")


def test_ply_property_schema_matches_reference(tmp_path) -> None:
    """属性 schema 必须覆盖参照实现 `load_ply` 读取的全部键。

    这是 PLY 互操作里**真正容易出错**的部分（命名、顺序、整型 dtype）。
    这里直接从参照源码里解析它读取的属性名，再逐个核对我们的文件，
    比「在 CPU 上真的调一次参照 load_ply」更快，且不依赖参照的硬编码 MLP 架构
    （参照 MLP 固定为 D→128→128→K，要求高斯数 ≥ 3.5 万，CPU 上代价过高；
    完整的互操作验证在 `scripts/equivalence_check.py` 中于 GPU 上完成）。
    """
    require_reference()
    from live3dgsavatar.core.io import read_ply_metadata, save_ply

    source = (REFERENCE_ROOT / "model" / "gaussian.py").read_text(encoding="utf-8")
    section = source[source.index("def load_ply"):source.index("def load_weight_module")]

    # 参照读取的固定属性
    fixed = set(re.findall(r'plydata\.elements\[0\]\["([a-z_]+)"\]', section))
    # 参照读取的带下标属性前缀
    prefixes = set(re.findall(r'\["([a-z_]+)_\{\}"', section))
    prefixes |= set(re.findall(r'startswith\("([a-z_]+_)"\)', section))
    prefixes |= set(re.findall(r'"([a-z_]+_)(?:\{\}|0)"', section))

    save_ply(_tiny_avatar(n=512, k=3, tex_size=32), tmp_path / "a.ply")
    from plyfile import PlyData

    props = {p.name for p in PlyData.read(str(tmp_path / "a.ply")).elements[0].properties}

    missing = sorted(n for n in fixed if n not in props)
    assert not missing, f"缺少参照会读取的固定属性：{missing}"

    for prefix in sorted(prefixes):
        assert any(p.startswith(prefix) for p in props), \
            f"缺少参照会读取的属性前缀 {prefix!r}"

    # 绑定列必须存在且 face_id 为整型（参照按 int32 读）
    assert "face_id" in props and "face_bary_0" in props
    el = PlyData.read(str(tmp_path / "a.ply")).elements[0]
    # 从结构化数据里取实际 numpy dtype（PlyProperty 不直接暴露类型）
    dtype = el.data.dtype
    assert dtype["face_id"] == np.int32, f"face_id 应为 int32，实际 {dtype['face_id']}"

    # 其余数值列必须是 float32
    for name in ("x", "y", "z", "opacity", "scale_0", "rot_0", "f_dc_0", "xyz_b_0",
                 "rot_b_0", "f_dc_b_0", "weight_module", "f_rest_0"):
        assert dtype[name] == np.float32, f"{name} 应为 float32，实际 {dtype[name]}"


def test_save_rejects_weight_module_larger_than_gaussians(tmp_path) -> None:
    """weight_module 被压进单列，因此参数量必须 ≤ 高斯数；超出要报错而非写出坏文件。"""
    from live3dgsavatar.core.avatar import AvatarConfig, GaussianAvatar
    from live3dgsavatar.core.deform import Binding
    from live3dgsavatar.core.io import save_ply

    n, tex = 64, 8            # 64 个高斯
    binding = Binding(torch.zeros(n, dtype=torch.long),
                      torch.tensor([[1.0, 0.0, 0.0]]).repeat(n, 1),
                      torch.ones(tex * tex, dtype=torch.bool))
    cfg = AvatarConfig(tex_size=tex, num_basis_in=129, num_basis_blend=20,
                       mlp_hidden=(128, 128), use_weight_proj=True)
    avatar = GaussianAvatar(cfg, binding)   # 参数量 35732 >> 64

    try:
        save_ply(avatar, tmp_path / "bad.ply")
    except ValueError as e:
        assert "tex_size" in str(e), f"报错应给出 tex_size 建议，实际：{e}"
    else:
        raise AssertionError("参数量超过高斯数时应报错")


def test_load_rejects_tex_size_too_small_for_gaussians(tmp_path) -> None:
    """**回归测试**：`tex_size` 装不下文件里的高斯数时必须报错。

    `tex_size` 不参与存储（它只决定 UV texel 的容量上界 `tex_size²`），
    因此**无法**从文件反推；但配错时必须显式报出，
    否则会把错误配置静默带进后续的保存/重建。
    曾实测：`tex_size=128`（上界 16384）加载 60353 个高斯**不报错**。
    """
    from live3dgsavatar.core.avatar import AvatarConfig
    from live3dgsavatar.core.io import load_ply, save_ply

    path = tmp_path / "m.ply"
    avatar = _tiny_avatar(n=64, k=3, tex_size=8)          # 上界 8² = 64，刚好
    save_ply(avatar, path)

    # 正确配置可加载（config 从文件的自描述注释还原）
    loaded = load_ply(path, device="cpu")
    assert loaded.num_gaussians == 64

    # tex_size 太小 → 明确报错
    bad = AvatarConfig(tex_size=4, num_basis_in=7, num_basis_blend=3,
                       mlp_hidden=(), use_weight_proj=True)
    try:
        load_ply(path, config=bad, device="cpu")
    except ValueError as e:
        assert "tex_size" in str(e), f"报错信息应提到 tex_size，实际：{e}"
    else:
        raise AssertionError("tex_size=4 装不下 64 个高斯，应报错")
