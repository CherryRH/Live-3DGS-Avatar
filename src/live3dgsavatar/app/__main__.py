"""`python -m live3dgsavatar.app` 入口。

刻意**不直接** `from .server import main` —— 那样在缺依赖时会抛出
一长串 ModuleNotFoundError 堆栈，用户看不出该做什么。这里转成一句可执行的提示。
"""

import sys


def _run() -> int:
    try:
        from .server import main
    except ImportError as e:
        print(f"[error] GUI 后端依赖未安装：{e}", file=sys.stderr)
        print("        请先运行：bash setup_env.sh", file=sys.stderr)
        return 1
    return main()


raise SystemExit(_run())
