"""Live3DGSAvatar.

包初始化时应用第三方兼容性补丁，必须早于任何可能 import chumpy 的代码。
"""

from . import compat as _compat  # noqa: F401  （导入即生效，勿删）

__all__ = ["_compat"]
