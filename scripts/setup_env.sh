#!/usr/bin/env bash
# =============================================================================
# Live3DGSAvatar 环境初始化脚本（P0）
#
# 在独立 conda 环境中完成：CUDA 13.0 + PyTorch + 三个本地扩展
# （nvdiffrast / fused-ssim / diff-gaussian-rasterization，全部从 submodules/ 构建）
#
# 用法：
#   bash scripts/setup_env.sh                                     # 建在 miniconda3/envs/live3dgs（推荐）
#   ENV_PREFIX=$PWD/.conda/envs/live3dgs bash scripts/setup_env.sh # 建在工作区内
#   FORCE=1 bash scripts/setup_env.sh                             # 已存在则重建
#
# 幂等：重复执行会跳过已完成步骤。
# =============================================================================

# ⚠️ 注意：这里刻意**不加 -u（nounset）**。
# conda 环境里 gcc_linux-64/gxx_linux-64 的 deactivate 钩子会在第 130 行引用
# _CONDA_PYTHON_SYSCONFIGDATA_NAME_USED（该变量仅在 activate 时赋值），
# 一旦开启 -u，conda 自己的钩子就会以 "unbound variable" 致命退出。
# 这是 cuda-toolkit → cuda-compiler → cuda-nvcc → gcc_linux-64 的传递依赖导致的，非本项目代码问题。
# 详见 docs/ENVIRONMENT.md §7。
set -eo pipefail

# ---------------------------------------------------------------- 配置区 ----
PY_VER="${PY_VER:-3.10}"
ENV_NAME="${ENV_NAME:-live3dgs}"
CUDA_CHANNEL="${CUDA_CHANNEL:-nvidia/label/cuda-13.0.0}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"   # RTX 3060 = sm_86
MAX_JOBS="${MAX_JOBS:-1}"                             # WSL 内存有限，限制并发
NVCC_THREADS="${NVCC_THREADS:-2}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMODULES="$REPO_ROOT/submodules"

# 若指定 ENV_PREFIX 则用 -p，否则用 -n（建在 conda 默认位置）
if [[ -n "${ENV_PREFIX:-}" ]]; then
    ENV_SPEC=(-p "$ENV_PREFIX"); ACTIVATE_TARGET="$ENV_PREFIX"
else
    ENV_SPEC=(-n "$ENV_NAME");   ACTIVATE_TARGET="$ENV_NAME"
fi

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------- 0. conda 与前置检查 ----
log "0/7 检查前置条件"
command -v conda >/dev/null 2>&1 || die "未找到 conda。请先安装 miniconda。"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

for pkg in diff-gaussian-rasterization nvdiffrast fused-ssim; do
    [[ -d "$SUBMODULES/$pkg" ]] || die "缺少 submodules/$pkg（请确认仓库完整）"
done

# ------------------------------------------------------- 1. 创建环境 ----
env_exists() {
    if [[ -n "${ENV_PREFIX:-}" ]]; then
        [[ -x "$ENV_PREFIX/bin/python" ]]
    else
        conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"
    fi
}

if env_exists; then
    if [[ "${FORCE:-0}" == "1" ]]; then
        warn "环境 $ACTIVATE_TARGET 已存在，FORCE=1 → 删除重建"
        conda env remove "${ENV_SPEC[@]}" -y
    else
        warn "环境 $ACTIVATE_TARGET 已存在，跳过创建（FORCE=1 可重建）"
    fi
fi
if ! env_exists; then
    log "1/7 创建 conda 环境（python $PY_VER）"
    conda create "${ENV_SPEC[@]}" "python=$PY_VER" -y
fi

# shellcheck disable=SC1091
conda activate "$ACTIVATE_TARGET"
log "已激活：$CONDA_PREFIX"
python -V

# --------------------------------------------- 2. CUDA Toolkit 13.0 ----
log "2/7 安装 CUDA Toolkit 13.0（conda channel: $CUDA_CHANNEL）"
if [[ -x "$CONDA_PREFIX/bin/nvcc" ]]; then
    warn "环境内已有 nvcc，跳过"
else
    conda install -y -c "$CUDA_CHANNEL" cuda-toolkit
fi
"$CONDA_PREFIX/bin/nvcc" --version | tail -2

# ------------------------------------------------- 3. PyTorch ----
log "3/7 安装 PyTorch（由 pip 解析与 CUDA 13 匹配的版本）"
if python -c "import torch" 2>/dev/null; then
    warn "torch 已安装，跳过"
else
    pip install torch torchvision
fi
python - <<'PY'
import torch, torchvision
print(f"  torch       {torch.__version__}")
print(f"  torch cuda  {torch.version.cuda}")
print(f"  torchvision {torchvision.__version__}")
assert torch.version.cuda is not None, "torch 是 CPU-only 构建，请检查 pip 安装源"
PY

# ------------------------------------------- 4. 构建工具 + 依赖 ----
# ⚠️ 顺序关键，两步都不能省：
#   (a) 本地扩展与 chumpy 都用 --no-build-isolation 安装，pip 不会自动准备
#       构建依赖，必须先装好 setuptools/wheel/ninja。
#   (b) chumpy 必须**单独**装，不能列进 requirements.txt：
#       它缺少 PEP 517 元数据，在 `pip install -r` 的依赖解析阶段会报
#       `Failed to build 'chumpy' when getting requirements to build wheel`。
log "4/7 安装构建工具与 Python 依赖"
pip install --upgrade pip setuptools wheel ninja

# chumpy：FLAME generic_model.pkl 的反序列化依赖，必须单独安装
if python -c "import chumpy" 2>/dev/null; then
    warn "chumpy 已安装，跳过"
else
    pip install chumpy --no-build-isolation
fi

pip install -r "$REPO_ROOT/requirements.txt"

# 校验 numpy 与 chumpy 的实际落点（chumpy 依赖 numpy 2.x 兼容补丁）
python - <<'PY'
import numpy
print(f"  numpy  {numpy.__version__}")
if numpy.__version__.split('.')[0] >= '2':
    print("  提示：numpy 2.x 下 chumpy 需要兼容补丁，"
          "已由 src/live3dgsavatar/compat/ 在导入期处理（见 docs/ENVIRONMENT.md §3.6）")
PY

# ------------------------------------- 5. nvdiffrast / fused-ssim ----
export TORCH_CUDA_ARCH_LIST
export MAX_JOBS NVCC_THREADS

log "5/7 安装 nvdiffrast（submodules/）"
if python -c "import nvdiffrast" 2>/dev/null; then
    warn "nvdiffrast 已安装，跳过"
else
    pip install "$SUBMODULES/nvdiffrast" --no-build-isolation
fi

log "5/7 安装 fused-ssim（submodules/，可选）"
if python -c "import fused_ssim" 2>/dev/null; then
    warn "fused-ssim 已安装，跳过"
else
    pip install "$SUBMODULES/fused-ssim" --no-build-isolation
fi

# ------------------------------------------------ 6. 光栅化扩展 ----
log "6/7 编译光栅化扩展（TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST）"

# 供 cuobjdump / nvcc 定位 CUDA 13 工具链；两者路径一致时无副作用
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$REPO_ROOT/.conda/torch_extensions}"
mkdir -p "$TORCH_EXTENSIONS_DIR"

if python -c "from diff_gaussian_rasterization import _C" 2>/dev/null; then
    warn "光栅化扩展已可导入，跳过编译"
else
    pip install -e "$SUBMODULES/diff-gaussian-rasterization" --no-build-isolation
fi

# 立即验证：光栅化扩展必须可导入且算子齐全，否则立刻失败而不是留到第 7 步
python - <<'PY'
import re, sys
from pathlib import Path
from diff_gaussian_rasterization import _C
ext_cpp = Path("submodules/diff-gaussian-rasterization/ext.cpp")
registered = re.findall(r'm\.def\(\s*"([^"]+)"', ext_cpp.read_text(encoding="utf-8"))
missing = [s for s in registered if not hasattr(_C, s)]
print(f"  光栅化扩展：{len(registered) - len(missing)}/{len(registered)} 算子")
if missing:
    sys.exit(f"  [error] 扩展缺少算子：{missing}")
PY

# ------------------------------------------------- 7. 自检 ----
log "7/7 环境自检"
python "$REPO_ROOT/scripts/env_check.py"

cat <<EOF

$(printf '\033[1;32m')环境就绪$(printf '\033[0m')

  激活：
      conda activate $ACTIVATE_TARGET

  GPU 冒烟测试（需可用的 NVIDIA GPU）：
      python scripts/smoke_test.py \\
          --ply  /home/crh/Projects/RGBAvatar/output/duda/test/model.ply \\
          --data /home/crh/Datasets/INSTA/duda \\
          --frames 3 --out output/smoke

EOF
