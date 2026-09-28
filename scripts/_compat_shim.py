"""共享兼容性补丁 —— 供 scripts/ 下的独立脚本使用。

**权威实现在 `src/live3dgsavatar/compat/__init__.py`**，本文件是它的精简副本：
`env_check.py` 与 `render_test.py` 必须在「未安装本项目、未设置 PYTHONPATH」的
条件下也能运行，因此不能依赖 `src/`。

⚠️ 修改 `src/live3dgsavatar/compat/__init__.py` 中的补丁列表时，必须同步修改本文件
   （两个文件各有一份 `_NUMPY_ALIASES`）。

当前补丁：chumpy 0.70 与 numpy 2.x 不兼容 —— `chumpy/__init__.py:11` 执行
    from numpy import bool, int, float, complex, object, unicode, str, nan, inf
这些别名在 numpy 1.24 弃用、2.0 移除。原因与移除条件见 docs/ENVIRONMENT.md §3.6。
"""

from __future__ import annotations

import warnings

# ⚠️ 过滤器必须在 import numpy **之前**装好：
# numpy 2.2 对已移除的别名做了 module __getattr__ 弃用转发，连 hasattr(np, "object")
# 这样的**读取**都会触发 FutureWarning，而告警的归属模块是 numpy 自己。
# chumpy 导入期会触发大量同类告警；补丁本身是有意为之，重复刷屏会掩盖真问题。
warnings.filterwarnings("ignore", category=FutureWarning, module=r"chumpy(\.|$)")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"chumpy(\.|$)")
# 探测 `hasattr(np, "object")` 本身就会触发 numpy 的弃用转发告警（归属模块是本文件）。
# 这是补丁的有意行为，精确按消息静音，不牵连其他告警。
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*np\.(object|str)\b.*")

import numpy as np  # noqa: E402

_NUMPY_ALIASES = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}


def apply() -> tuple[str, ...]:
    """补齐 numpy 2.x 移除的标量别名。返回实际补齐的名称（幂等）。"""
    patched = []
    for name, builtin in _NUMPY_ALIASES.items():
        if not hasattr(np, name):
            setattr(np, name, builtin)
            patched.append(name)
    return tuple(patched)


def verify_chumpy() -> tuple[bool, str]:
    """应用补丁后验证 chumpy 是否可用。返回 (是否可用, 说明)。"""
    apply()
    try:
        import chumpy  # noqa: F401, PLC0415
    except Exception as e:  # pragma: no cover
        return False, f"{type(e).__name__}: {e}"
    return True, "OK"
