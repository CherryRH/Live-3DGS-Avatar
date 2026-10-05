"""GUI 应用层（P2）。

- `protocol.py`：协议编解码与背压工具（**纯函数，不依赖 FastAPI/torch**）
- `session.py` ：渲染会话（唯一接触 GPU 与 `core/` 的部分）
- `server.py`  ：FastAPI 应用 + WebSocket 推流循环 + CLI

协议见 `docs/GUI_PROTOCOL.md`。启动用顶层 `bash run_gui.sh`，
或 `python -m live3dgsavatar.app`。

`server` / `session` 需要 FastAPI 与 torch。未安装时本模块仍可导入
（`protocol` 始终可用），这样没有 Web 依赖的环境也能跑 `protocol` 的单测。
"""

from . import protocol

__all__ = ["protocol"]

try:                                     # pragma: no cover - 依赖环境
    from .server import create_app, main

    __all__ += ["create_app", "main"]
except ImportError as _e:                # 缺 fastapi / uvicorn 等
    _IMPORT_ERROR = _e

    def create_app(*_a, **_k):           # type: ignore[misc]
        raise ImportError(
            f"GUI 后端依赖未安装（{_e}）。请先运行：bash setup_env.sh")

    def main(*_a, **_k) -> int:          # type: ignore[misc]
        raise ImportError(
            f"GUI 后端依赖未安装（{_e}）。请先运行：bash setup_env.sh")

    __all__ += ["create_app", "main"]
