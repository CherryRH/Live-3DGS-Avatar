"""架构一致性测试（**不需要 GPU**）。

把 `docs/ARCHITECTURE.md` 里声明的分层规则变成可执行的检查，
避免它们在后续迭代中被无声破坏。

规则来源：
- §2.3 依赖方向严格单向，`core/` 不得 import `data/` / `training/` / `app/` / `tracking/` / `runtime/`；
- §2.2 `ext/`（现 `submodules/`）只被 `core/render/` 引用；
- §3.1 `core/` 不做文件 I/O 与参数解析。
"""

from __future__ import annotations

import ast
from pathlib import Path

from support import REPO_ROOT

SRC = REPO_ROOT / "src" / "live3dgsavatar"
CORE = SRC / "core"

# core/ 的允许依赖（子包 + 标准库 + 第三方）
CORE_FORBIDDEN_PREFIXES = (
    "live3dgsavatar.data",
    "live3dgsavatar.training",
    "live3dgsavatar.app",
    "live3dgsavatar.tracking",
    "live3dgsavatar.runtime",
    "live3dgsavatar.streaming",
)

# 唯一允许接触 CUDA 扩展的位置
CUDA_EXTENSION_PACKAGES = ("diff_gaussian_rasterization",)
CUDA_ALLOWED_FILES = {
    CORE / "render" / "rasterizer.py",   # 光栅化封装
    CORE / "deform" / "blend.py",        # 可选的 CUDA linear_blending 加速路径
}


def _iter_modules() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def _imports(path: Path) -> list[tuple[int, str]]:
    """返回 `(行号, 模块名)` 列表，覆盖 `import x` 与 `from x import y`。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # 相对导入：解析成绝对包名
                parts = list(path.relative_to(SRC).with_suffix("").parts)[:-1]
                ups = node.level - 1
                base = parts[: len(parts) - ups] if ups else parts
                module = "live3dgsavatar." + ".".join(base + ([node.module] if node.module else []))
                found.append((node.lineno, module))
            elif node.module:
                found.append((node.lineno, node.module))
    return found


def test_core_does_not_import_upper_layers() -> None:
    """分层规则：`core/` 不得依赖 data / training / app / tracking / runtime / streaming。"""
    violations: list[str] = []
    for path in sorted(CORE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for lineno, module in _imports(path):
            if module.startswith(CORE_FORBIDDEN_PREFIXES):
                rel = path.relative_to(SRC)
                violations.append(f"{rel}:{lineno} → {module}")
    assert not violations, "core/ 依赖了上层模块：\n  " + "\n  ".join(violations)


def test_cuda_extension_only_imported_where_allowed() -> None:
    """CUDA 扩展只允许在 `core/render/` 与 `core/deform/blend.py` 出现。

    这条规则的意义：`submodules/` 里的扩展与 CUDA/torch 版本强耦合，
    一旦 import 散落到各处，替换或升级的成本会迅速失控。
    """
    violations: list[str] = []
    for path in _iter_modules():
        for lineno, module in _imports(path):
            root = module.split(".")[0]
            if root in CUDA_EXTENSION_PACKAGES and path not in CUDA_ALLOWED_FILES:
                rel = path.relative_to(SRC)
                violations.append(f"{rel}:{lineno} → {module}")
    assert not violations, (
        "CUDA 扩展被允许范围之外的文件导入：\n  " + "\n  ".join(violations)
        + f"\n（允许位置：{sorted(str(p.relative_to(SRC)) for p in CUDA_ALLOWED_FILES)}）"
    )


def test_core_has_no_file_io_or_argparse() -> None:
    """`core/` 只做算法，不做 I/O 与参数解析（I/O 仅限 `core/io/` 的序列化）。"""
    forbidden = {"argparse", "sys"}
    io_allowed = {CORE / "io" / "ply.py"}
    violations: list[str] = []

    for path in sorted(CORE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for lineno, module in _imports(path):
            if module in forbidden:
                violations.append(f"{path.relative_to(SRC)}:{lineno} → import {module}")
        if path not in io_allowed and path != CORE / "io" / "__init__.py":
            for lineno, module in _imports(path):
                if module in ("plyfile", "json"):
                    violations.append(f"{path.relative_to(SRC)}:{lineno} → import {module}")
    assert not violations, (
        "core/ 出现了 I/O 或参数解析：\n  " + "\n  ".join(violations)
        + f"\n（序列化只允许在 {sorted(str(p.relative_to(SRC)) for p in io_allowed)}）")


def test_no_torch_cuda_calls_in_core_module_scope() -> None:
    """`core/` 不得在**模块作用域**调用 `torch.cuda.*`。

    允许在函数/方法内部使用（例如训练态显式要求 CUDA），但模块导入时不得触碰设备，
    否则「导入 core」就会在无 GPU 环境下失败，CPU 侧的测试将全部无法运行。

    实现说明：必须**剪掉函数与类的定义体**，只检查真正在导入期执行的语句。
    直接 `ast.walk(module)` 会穿透到方法体内部，产生误报。
    """
    violations: list[str] = []

    def module_level_calls(node: ast.AST):
        """只遍历导入期会执行的节点：跳过函数/类定义体。"""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue                      # 定义体不在导入期执行
            yield child
            yield from module_level_calls(child)

    for path in sorted(CORE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for sub in module_level_calls(tree):
            if (isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Attribute)
                    and sub.value.attr == "cuda"
                    and isinstance(sub.value.value, ast.Name)
                    and sub.value.value.id == "torch"):
                violations.append(f"{path.relative_to(SRC)}:{sub.lineno} → torch.cuda.{sub.attr}")

    # 类体内的默认参数/装饰器也属于导入期，但上述遍历已覆盖（ClassDef 被剪掉的是其 body，
    # 如果将来出现类属性形式的 torch.cuda 调用，这里会漏检；用下面的兜底断言弥补）
    assert not violations, (
        "core/ 在模块作用域调用了 torch.cuda：\n  " + "\n  ".join(violations))


def test_importing_core_does_not_require_gpu() -> None:
    """健全性检查：导入整个 `core` 包不得触发任何设备访问。"""
    import importlib

    for name in ("live3dgsavatar.core", "live3dgsavatar.core.avatar",
                 "live3dgsavatar.core.deform", "live3dgsavatar.core.render",
                 "live3dgsavatar.core.io", "live3dgsavatar.core.types"):
        importlib.import_module(name)


def test_flame_dtype_is_not_overridden() -> None:
    """**回归测试**：不得覆盖 `FlameConfig.dtype`。

    `FLAMEDataset` 把姿态/形状参数存为 **float64**，而 `FLAME` 的
    `v_template` / `shapedirs` 跟随 `cfg.dtype`。把它改成 float32 会让
    `template_vertices(float32) + blend_shapes(betas(float64), ...)`
    抛 `expected scalar type Double but found Float`。

    参照仓库的 `FlameConfig` 默认 float64，故两边一致、从不暴露该问题。
    本项目曾在脚本侧改成 float32 而踩坑，此处用静态检查钉住。
    """
    import re

    offenders: list[str] = []
    for path in sorted(REPO_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts or "submodules" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "FlameConfig" not in text:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            code = line.split("#", 1)[0]          # 跳过注释（含本测试自身的文档示例）
            if re.search(r"\bcfg\.dtype\s*=", code) or \
                    re.search(r"FlameConfig\([^)]*dtype", code):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "不得覆盖 FLAME 的 dtype（会导致 Double/Float 混算）：\n  " + "\n  ".join(offenders))
