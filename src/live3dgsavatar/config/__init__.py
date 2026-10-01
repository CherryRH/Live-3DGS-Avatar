"""配置加载：把 `configs/*.yaml` 变成可点访问的对象。

**为什么放在 `src/live3dgsavatar/config/` 而不是 core/**
    `core/` 是纯计算层，不允许依赖配置、路径、YAML（见
    `docs/ARCHITECTURE.md` §2.3）。配置属于**应用层**基础设施。

用法::

    from live3dgsavatar.config import load_config

    cfg = load_config()                 # 读仓库根的 configs/
    cfg.paths.data_root                 # 已解析为绝对 Path
    cfg.get("render.batch_size", 4)     # 带默认值的点路径访问
    cfg.render.batch_size = 8           # 可写（就地覆盖默认量）

优先级（高 → 低）::

    命令行参数  >  环境变量  >  configs/*.yaml  >  代码内置兜底值

`configs/local.yaml` 若存在，会在 `system.yaml` / `render.yaml` 之后**深合并**，
用于放本机私有覆盖（该文件已被 .gitignore 忽略）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

__all__ = ["Config", "load_config", "repo_root", "CONFIG_DIR"]

# 仓库根：src/live3dgsavatar/config/__init__.py → 上溯 4 层
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"

# 环境变量 → 点路径。便于换机器 / CI 而不改文件。
_ENV_OVERRIDES = {
    "LIVE3DGS_DATA_ROOT": "paths.data_root",
    "LIVE3DGS_OUTPUT_DIR": "paths.output_dir",
    "LIVE3DGS_REFERENCE_ROOT": "paths.reference_root",
    "LIVE3DGS_SUBJECT": "subject",
    "LIVE3DGS_DEVICE": "runtime.device",
}

# 代码内置兜底：仅当 YAML 缺失该字段时生效。
# **刻意保持最小**——真正的默认量应当在 configs/*.yaml 里，便于用户查看与修改。
_FALLBACK: dict[str, Any] = {
    "subject": "duda",
    "paths.data_root": "data/INSTA",
    "paths.output_dir": "output",
    "paths.reference_root": None,
    "paths.model_ply": None,
    "paths.model_subdir": "test",
    "paths.image_subdir": "images",
    "runtime.device": "cuda",
    "runtime.split": "all",
    "smoke.max_frames": None,
    "render.background": [0.0, 0.0, 0.0],
    "render.scaling_modifier": 1.0,
    "render.sh_degree": 0,
    "render.batch_size": 1,
    "render.frames": 1,
    "render.max_batch_size": 10,
    "render.max_gaussian_size": 60353,
    "render.num_streams": 3,
    "model.network.tex_size": 256,
    "model.network.num_basis_in": 129,
    "model.network.num_basis_blend": 20,
    "model.network.mlp_hidden": [128, 128],
    "model.network.use_weight_proj": True,
    "dataset.use_shape_weight": True,
    "dataset.use_pose_weight": True,
    "dataset.pin_memory": False,
    "test.dump_diff": False,
}

# 需要解析为绝对路径的字段（相对路径按**仓库根**解析，而非 cwd）
_PATH_FIELDS = (
    "paths.data_root",
    "paths.output_dir",
    "paths.reference_root",
    "paths.model_ply",
)

# 仓库根下不存在的相对路径，按 cwd 解析更符合直觉的字段。
# （例如 `--output-dir out` 通常指用户当前目录）
_CWD_RELATIVE = ("paths.output_dir",)


class Config(dict):
    """支持点路径访问的嵌套配置。

    - `cfg.paths.data_root` 等价于 `cfg["paths"]["data_root"]`
    - 嵌套 dict 会自动包装成 `Config`
    - `.get("a.b.c", default)` 支持点路径
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(
                f"配置项 {name!r} 不存在。可用键：{sorted(self.keys())}") from exc
        return _wrap(value)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        """按**点路径**取值，与 `dict.get` 兼容（无点时行为相同）。"""
        if "." not in key:
            return _wrap(super().get(key, default))
        node: Any = self
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return _wrap(node)

    def set(self, key: str, value: Any) -> None:
        """按点路径赋值，中间层不存在时自动创建。"""
        parts = key.split(".")
        node: Any = self
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = value

    def as_plain(self) -> dict:
        """转回普通 dict（便于写 JSON / 打印）。"""
        return _unwrap(self)

    def resolved(self) -> dict:
        """返回一份便于阅读的最终生效配置（Path 转 str）。"""
        def enc(v: Any) -> Any:
            if isinstance(v, Path):
                return str(v)
            if isinstance(v, dict):
                return {k: enc(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [enc(x) for x in v]
            return v

        return enc(self.as_plain())

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"Config({self.as_plain()!r})"


def _wrap(value: Any) -> Any:
    return Config(value) if isinstance(value, dict) else value


def _unwrap(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并：`override` 里的标量覆盖 `base`，嵌套 dict 逐层合并。"""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    import yaml  # 延迟导入：未装 PyYAML 时只有真正读配置才报错

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} 顶层必须是映射（mapping），实际是 {type(data).__name__}")
    return data


def _resolve_path(value: Any, field: str) -> Any:
    """把路径字段解析为绝对 `Path`。

    - `~` 展开
    - 绝对路径原样
    - 相对路径：`output_dir` 按 **cwd**（用户直觉），其余按**仓库根**
    """
    if value is None:
        return None
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p
    base = Path.cwd() if field in _CWD_RELATIVE else REPO_ROOT
    return (base / p).resolve()


def load_config(
    config_dir: Path | str | None = None,
    files: Iterable[str] = ("system.yaml", "render.yaml", "local.yaml"),
    env: bool = True,
) -> Config:
    """加载配置。

    Args:
        config_dir: 配置目录；默认仓库根的 `configs/`
        files: 按顺序加载并深合并的文件名；缺失的自动跳过
        env: 是否应用环境变量覆盖

    Raises:
        ValueError: 必填项缺失或类型不符（例如缺 `paths.data_root`）
    """
    cdir = Path(config_dir).resolve() if config_dir else CONFIG_DIR

    merged: dict[str, Any] = {}
    for name in files:
        merged = deep_merge(merged, _load_yaml(cdir / name))

    cfg = Config(merged)

    # 内置兜底：只填 YAML 里没有的键
    for key, value in _FALLBACK.items():
        if cfg.get(key, _MISSING) is _MISSING:
            cfg.set(key, value)

    if env:
        for var, key in _ENV_OVERRIDES.items():
            if var in os.environ:
                cfg.set(key, os.environ[var])

    # 路径解析
    for field in _PATH_FIELDS:
        if cfg.get(field, None) is not None:
            cfg.set(field, _resolve_path(cfg.get(field), field))

    _validate(cfg)
    return cfg


_MISSING = object()


def _validate(cfg: Config) -> None:
    data_root = cfg.get("paths.data_root")
    if data_root is None:
        raise ValueError(
            "缺少 paths.data_root。请在 configs/system.yaml 中设置，"
            "或设环境变量 LIVE3DGS_DATA_ROOT。")

    for key, expect in (("render.batch_size", int),
                        ("render.frames", int),
                        ("model.network.tex_size", int),
                        ("model.network.num_basis_in", int),
                        ("model.network.num_basis_blend", int),
                        ("runtime.device", str)):
        value = cfg.get(key)
        if not isinstance(value, expect):
            raise ValueError(
                f"配置项 {key} 应为 {expect.__name__}，实际 "
                f"{type(value).__name__}（{value!r}）")

    if cfg.get("runtime.device") not in ("cuda", "cpu"):
        raise ValueError(
            f"runtime.device 只能是 'cuda' 或 'cpu'，实际 {cfg.get('runtime.device')!r}")

    hidden = cfg.get("model.network.mlp_hidden")
    if hidden is not None and not isinstance(hidden, list):
        raise ValueError(
            f"model.network.mlp_hidden 应为列表，实际 {type(hidden).__name__}")


def repo_root() -> Path:
    """仓库根目录。"""
    return REPO_ROOT
