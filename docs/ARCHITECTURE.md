# Live3DGSAvatar 架构设计文档

> 视频通话场景 3DGS 人体重建 —— 第一阶段（头部）架构设计
> 状态：**待审核**　版本：v0.1　最后更新：待定

---

## 0. 本文档的边界

**目标**：定义一个**足够简单、可落地、可扩展**的工程架构，把 RGBAvatar 的算法能力内化成本项目自己的代码，并支撑后续的视频通话场景扩展。

**明确不做**（本阶段）：
- 不追求架构完备性。层级只到"能隔离变化"为止，不做插件框架、不做通用中间表示、不做多后端抽象。
- 不考虑非 NVIDIA 平台。CUDA 13.0 为唯一目标工具链。
- 不引入 VoluMe 等前馈路线。见附录 A（未来工作）。

**评审要点**：请重点看 §3（接口契约）、§4（目录结构）、§5（迁移映射）、§9（验收门）四节。其余为实现细节。

---

## 1. 目标与硬约束

### 1.1 产品目标

第一阶段的理想产物：**一个支持静态训练、实时渲染、动态更新的图形化程序**。

| 能力 | 含义 | 对应后端 |
|---|---|---|
| 静态训练 | 给定单目视频序列，重建一个可驱动头部 avatar | `training/` |
| 实时渲染 | 给定驱动参数，实时出图（目标 ≥ 60 FPS @ 512²） | `core/render/` |
| 动态更新 | 流式到达的帧可持续优化模型（在线重建） | `training/online.py` |

### 1.2 已确定的硬约束

| 编号 | 约束 | 来源 |
|---|---|---|
| C1 | 目标平台为 NVIDIA GPU，CUDA **13.0** | 用户决策。实测 11.8 反而不兼容，官方文档不准确，统一 13.0 |
| C2 | 训练与推理技术栈：PyTorch + 自研 CUDA 光栅化扩展 | 复用 RGBAvatar。已实测可用：torch `2.14.0+cu130` |
| C3 | 渲染部署形态：**服务端渲染，推流给用户端** | 用户决策（实验室算力充足） |
| C4 | 覆盖范围：**仅头部** | RGBAvatar 拓扑边界所限；上半身为未来工作 |
| C5 | 用途：现阶段纯学术/自用 | 商用未定；架构上不封死替换 3DMM/tracker 的路径即可 |
| C6 | `RGBAvatar/` 为**只读参照**，所有实现落在 `Live3DGSAvatar/` | 约束 |
| C7 | 数据集**只用 `duda`** | 这是唯一由项目作者亲手跑通 tracker → INSTA → RGBAvatar 全链路的数据；其余数据集已删除 |
| C8 | 独立 conda 环境 `live3dgs` | 与参照仓库的 `rgbavatar` 环境隔离，避免互相污染 |

### 1.3 本机开发环境

| 项 | 值 |
|---|---|
| Conda 环境 | **`live3dgs`**（新建，python 3.10.21）—— 参照环境为 `rgbavatar` |
| PyTorch | 2.14.0+cu130 |
| CUDA Toolkit | 13.0（conda 环境内 nvcc 13.0.48 / 系统 13.0.88，二者皆可用） |
| GPU | RTX 3060 Laptop, 6 GB |
| 系统 | WSL2, 16 核, 7 GB RAM |

搭建脚本、版本矩阵、排错表见 **`docs/ENVIRONMENT.md`**。

> **说明**：6 GB 显存是本机训练的限制项，但不是架构约束——训练可放实验室服务器。本机足以完成 P0–P1（复现 + 重写验证）。

---

## 2. 系统总览

### 2.1 一句话架构

**一个只读的 `GaussianSet`（单帧高斯集合）+ 三条纯函数管线（blend / bind / rasterize）+ 两个门面（训练、运行时）。**

### 2.2 分层图

```
┌─────────────────────────────────────────────────────────────┐
│  app/          图形化程序 · CLI · 服务端入口                  │
│                训练监控面板 / 实时预览 / 会话管理              │
├─────────────────────────────────────────────────────────────┤
│  runtime/      门面层：把"模型+数据集+渲染器"组装成可用对象     │
│                AvatarRuntime: train() / render() / update()  │
│                Session: 服务端渲染会话（预留）                 │
├─────────────────────────────────────────────────────────────┤
│  training/     训练循环 · 损失 · 优化器 · 采样池               │
│                offline.py（离线）  online.py（在线/流式）      │
├─────────────────────────────────────────────────────────────┤
│  core/         算法核心（无训练逻辑、无 I/O、无 GUI）          │
│  ├─ types.py        GaussianSet / Mesh / Camera / Frame      │
│  ├─ template/       FLAME / FuHead 模板（封装，不改数学）      │
│  ├─ deform/         blend: 权重→高斯    bind: 法空间→世界     │
│  ├─ render/         rasterize 封装（训练态/推理态两条路径）    │
│  └─ avatar.py       GaussianAvatar（参数容器 + 序列化）       │
├─────────────────────────────────────────────────────────────┤
│  submodules/   唯一允许出现 CUDA 的地方（vendored 源码）        │
│                diff_gaussian_rasterization/（vendored）       │
├─────────────────────────────────────────────────────────────┤
│  data/         数据集读取（FLAME 格式 / NeRSemble 格式）       │
│  tracking/     驱动参数求解（接口 + 离线实现 + 实时占位）      │
│  streaming/    服务端渲染推流（预留）                          │
├─────────────────────────────────────────────────────────────┤
│  compat/       与 RGBAvatar 的对照实现（仅用于回归测试）       │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 依赖方向（严格单向）

```
app → runtime → training → core → ext
                  ↓         ↑
                data ───────┘
                tracking ───┘
```

规则：
- `core/` **不得** import `training/`、`data/`、`app/`、`tracking/`。
- `core/` **不得**做文件 I/O 与参数解析（`avatar.py` 的 `save/load` 除外，见 §3.6）。
- `submodules/` 只被 `core/render/` 引用，其他任何地方不得直接 import。

---

## 3. 核心接口契约

这是全文最重要的部分。**只有这 5 个类型 + 3 个协议**，其余都是实现。

### 3.1 约定

1. **Batch-first**：所有张量第一维恒为 batch（单帧时为 1），避免 RGBAvatar 中单帧/批量两套签名并存的问题。
2. **尺度约定**：`xyz` 为米；`scaling` 为**激活后**的正尺度（非 log）；`opacity` 为 `[0,1]`；`rotation` 为**单位四元数**，WXYZ 序。
3. **坐标空间显式标注**：高斯参数固定在 **tangent space（切空间）**，只有 `binder` 的输出是 **world space**。命名后缀 `_tan` / `_world` 强制标注。
4. **激活函数归属**：`sigmoid`/`exp`/`normalize` 在 `GaussianSet` 的构造侧完成；`GaussianSet` 内部永远是"可直接渲染的物理量"。

### 3.2 `GaussianSet` —— 跨层唯一数据契约

```python
# core/types.py
from dataclasses import dataclass
import torch

@dataclass
class GaussianSet:
    """一批可直接交给光栅化器的高斯。张量形状的 B 维恒存在。"""
    xyz:      torch.Tensor  # [B, N, 3]   位置。按 space 字段解释其空间
    rotation: torch.Tensor  # [B, N, 4]   单位四元数 (WXYZ)
    scaling:  torch.Tensor  # [B, N, 3]   正尺度
    opacity:  torch.Tensor  # [B, N, 1]   [0, 1]
    color:    torch.Tensor  # [B, N, 1, 3] SH 0 阶系数（DC）

    space: str = "tangent"  # "tangent" | "world"，显式标注，防止空间混用

    def to(self, device) -> "GaussianSet": ...
    def detach(self) -> "GaussianSet": ...
    def __len__(self) -> int: ...   # N
```

**与 RGBAvatar 的差异**：
- 原名 `GaussianAttributes`，改名以强调"这是一组高斯"。
- **新增 `space` 字段**。RGBAvatar 靠调用顺序隐式区分切空间/世界空间的高斯，这是最容易出错的地方。
- **固定 batch 维**。RGBAvatar 的 `GaussianAttributes` 有时是 `[N,3]` 有时是 `[B,N,3]`，导致 `render_gs` 与 `render_gs_batch` 两份代码（且其中一份已标注 `# legacy`）。

### 3.3 `Mesh` / `Camera` / `Frame`

```python
# core/types.py
@dataclass
class Mesh:
    verts: torch.Tensor  # [B, V, 3]  世界坐标
    faces: torch.Tensor  # [F, 3]     int32，拓扑恒定
    uvs:   torch.Tensor  # [Vuv, 2]   模板 UV，恒定
    uv_faces: torch.Tensor  # [F, 3]  int32，恒定

@dataclass
class Camera:
    K: torch.Tensor       # [B, 3, 3] 或 [3,3]
    w2c: torch.Tensor     # [B, 4, 4] world→camera（注意：不是 view matrix 的转置混淆）
    width: int
    height: int
    # fov_x / fov_y / position 由属性派生，不重复存储

@dataclass
class Frame:
    """一帧的全部输入。训练与推理共用。"""
    mesh: Mesh
    blend_weight: torch.Tensor  # [B, D] FLAME 参数（离线 129 维 / NeRSemble 100 维）
    camera: Camera
    image: torch.Tensor | None = None   # [B, 3, H, W] GT，推理时为 None
    mask:  torch.Tensor | None = None   # [B, 1, H, W]
```

**`Camera` 的明确约定**（这是 RGBAvatar 里最混乱的部分）：
- 内部**只存 `w2c`**（world→camera）与 `K`。
- `w2v`（view matrix）与 `full_proj` 通过方法派生，不缓存多份。
- 所有矩阵采用**行主序、右乘列向量**（`p_clip = P @ V @ p_world`）的数学约定；传给 CUDA 扩展时在 `core/render/` 内部统一转置为列主序，**这个转置只允许出现在一个地方**。

### 3.4 `BlendField` —— 权重 → 高斯

```python
# core/deform/blend.py
class BlendField(Protocol):
    """把驱动参数映射为切空间高斯。对应 RGBAvatar 的 linear_blending + MLP。"""

    def __call__(self, blend_weight: torch.Tensor) -> GaussianSet:
        """blend_weight: [B, D]  →  GaussianSet(space='tangent')"""
```

### 3.5 `Binder` —— 切空间 → 世界空间

```python
# core/deform/bind.py
class Binder(Protocol):
    """把切空间高斯按模板网格变形搬到世界空间。对应 RGBAvatar 的 mesh_binding。"""

    def bind(self, gaussians: GaussianSet, mesh: Mesh) -> GaussianSet:
        """GaussianSet(space='tangent') + Mesh  →  GaussianSet(space='world')"""
```

**为什么把 blend 与 bind 拆成两个协议**：这是本架构为"未来上半身"预留的**唯一关键扩展点**。头部用 `BlendField = 20 基线性混合`、`Binder = 三角面 TBN`；未来身体用 `BlendField = LBS 权重`、`Binder = 混合拓扑拼接`。**渲染层完全不需要改动。**

### 3.6 `Rasterizer`

```python
# core/render/rasterizer.py
class Rasterizer(Protocol):
    def render(
        self,
        gaussians: GaussianSet,   # space='world'
        camera: Camera,
        bg_color: torch.Tensor,   # [B, 3]
        target_image: torch.Tensor | None = None,  # 仅训练态需要
    ) -> "RenderOutput": ...
```

```python
@dataclass
class RenderOutput:
    color: torch.Tensor       # [B, 3, H, W]
    alpha: torch.Tensor       # [B, 1, H, W]
    est_color: torch.Tensor | None = None  # [B, N, 3]   仅训练态
    est_weight: torch.Tensor | None = None # [B, N]      仅训练态
    radii: torch.Tensor | None = None      # [B, N]      仅训练态
```

两个实现：

| 实现 | 用途 | 后端 |
|---|---|---|
| `BatchRasterizer` | 训练（需要梯度、`est_color/est_weight`、批并行） | `BatchGaussianRasterizer` + `_BatchRasterizeGaussians` |
| `SimpleRasterizer` | 推理（无梯度、逐帧循环） | `GaussianRasterizer` |

> **修正 RGBAvatar 的一处混乱**：其 `diff_renderer/gaussian.py` 里有 `render_gs` 和 `render_gs_batch`（标注 `# legacy`）两份，`model/mv_reconstruction.py` 里还有第三份 `render_gs_batch`。本设计**把三者合并为一套协议 + 两个后端实现**，语义差异由"是否传 `target_image`"表达。

### 3.7 `GaussianAvatar` —— 参数容器

```python
# core/avatar.py
@dataclass
class AvatarConfig:
    tex_size: int = 256
    num_basis_in: int = 129      # 输入 FLAME 参数维度 D
    num_basis_blend: int = 20    # 约简后的基数量 K
    use_blend: bool = True
    use_weight_proj: bool = True
    use_mlp_proj: bool = True
    init_scaling: float = 0.0008
    init_opacity: float = 0.5

class GaussianAvatar:
    """参数容器。持有可学习参数；不持有优化器、不持有相机、不做渲染。"""

    # ---- 静态参数（[N, ·]）----
    xyz:      Parameter  # [N, 3]
    opacity:  Parameter  # [N, 1]    (logit)
    scaling:  Parameter  # [N, 3]    (log)
    rotation: Parameter  # [N, 4]
    color_dc: Parameter  # [N, 1, 3]

    # ---- Blendshape 基（[K, N, ·]）----
    xyz_b:      Parameter  # [K, N, 3]
    rotation_b: Parameter  # [K, N, 4]
    color_b:    Parameter  # [K, N, 1, 3]

    # ---- 权重复约简 MLP（D → K）----
    weight_module: nn.Module

    # ---- 绑定信息（由模板决定，非学习参数）----
    binding_face_id:   Tensor  # [N]    int32
    binding_face_bary: Tensor  # [N, 3]
    valid_mask:        Tensor  # [tex_size²] bool

    def attribute_names(self) -> list[str]: ...      # 供优化器分组
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path, cfg: AvatarConfig) -> "GaussianAvatar": ...
    def build_blend_field(self) -> BlendField: ...
    def build_binder(self, template) -> Binder: ...
```

**参数量（bala，N=60349, K=20）**：`xyz_b/rotation_b/color_b = 60349×20×10 ≈ 12.07M` 参数 ≈ 48 MB，是模型主体。

### 3.8 门面

```python
# runtime/avatar_runtime.py
class AvatarRuntime:
    """把模板 + avatar + 数据集 + 渲染器组装成可直接使用的对象。
    这是 app 层唯一需要认识的类。"""

    @classmethod
    def setup(cls, cfg: RuntimeConfig) -> "AvatarRuntime": ...

    def train(self, dataset, steps: int) -> "TrainStats": ...
    def render(self, frame: Frame) -> RenderOutput: ...
    def update(self, frames: Iterable[Frame]) -> "UpdateStats": ...   # 在线更新
    def snapshot(self) -> "AvatarSnapshot": ...
```

四件事：`setup / train / render / update`。不多不少。

---

## 4. 目录结构

```
Live3DGSAvatar/
├── docs/
│   ├── ARCHITECTURE.md          ← 本文档
│   ├── ENVIRONMENT.md           ← 环境搭建、版本矩阵、排错（P0 已完成）
│   ├── CONVENTIONS.md           ← 坐标系与数值约定（P1 产出）
│   └── MIGRATION.md             ← 与 RGBAvatar 的行为差异清单（P1 产出）
├── configs/                     ← 当前为空；P1 落地 offline.yaml / online.yaml
├── src/live3dgsavatar/
│   ├── core/
│   │   ├── types.py             # GaussianSet / Mesh / Camera / Frame / RenderOutput
│   │   ├── avatar.py            # GaussianAvatar / AvatarConfig
│   │   ├── template/
│   │   │   ├── flame.py         # 封装 submodules/flame（不改数学）
│   │   │   └── fuhead.py
│   │   ├── deform/
│   │   │   ├── blend.py         # BlendField：权重→切空间高斯（linear_blending + MLP）
│   │   │   ├── bind.py          # Binder：切空间→世界（TBN + 重心）
│   │   │   └── tbn.py           # compute_face_tbn 的参考实现
│   │   └── render/
│   │       ├── rasterizer.py    # Rasterizer 协议 + 两个实现
│   │       └── camera_utils.py  # 矩阵转置/列主序转换的唯一入口
│   ├── data/
│   │   ├── flame_dataset.py     # INSTA 格式（当前仅 duda）
│   │   └── sampler.py           # local-global 采样池
│   ├── training/
│   │   ├── losses.py            # l1 / ssim / lpips / alpha / sparsity / orth
│   │   ├── offline.py           # 离线训练循环
│   │   └── online.py            # 在线/流式训练循环
│   ├── tracking/
│   │   ├── base.py              # Tracker 协议
│   │   └── offline_flame.py     # 读取 metrical-tracker 的 checkpoint 输出
│   ├── runtime/
│   │   ├── avatar_runtime.py    # AvatarRuntime 门面
│   │   └── session.py           # 服务端渲染会话（骨架，P3 填充）
│   ├── streaming/               # P3：编码/传输
│   │   └── __init__.py
│   ├── app/
│   │   ├── cli.py               # train / render / update 三个子命令
│   │   └── gui.py               # P2：图形化程序
│   └── compat/
│       └── rgba_avatar.py       # 与 RGBAvatar 的对照实现（仅测试用）
├── ext/ 与 third_party/ 已取消 ─────────────────────────────────────────
│   diff-gaussian-rasterization / nvdiffrast / fused-ssim 统一并入 submodules/
├── submodules/                  ← 全部 vendored，统一从此处构建
│   ├── ATTRIBUTION.md           #   来源 / 许可 / 差异登记
│   ├── diff-gaussian-rasterization/   # 唯一的 CUDA 内核
│   ├── nvdiffrast/                    # UV 域光栅化
│   ├── fused-ssim/                    # 多视角 SSIM（可选）
│   ├── flame/                         # P1 拷入
│   └── fuhead/                        # P1 拷入
├── tests/
│   ├── unit/                    # 单测
│   ├── equivalence/             # 数值等价测试（黄金测试）
│   └── fixtures/                # 小规模固定输入
├── scripts/
│   ├── setup_env.sh             # 一键建环境（幂等）—— P0 已完成
│   ├── env_check.py             # 环境自检（无 GPU 可跑）—— P0 已完成
│   ├── smoke_test.py            # GPU 冒烟 + 基线性能采集 —— P0 已完成
│   └── export_baseline.py       # 从 RGBAvatar 导出基线产物（P1）
├── data/                        # 输入资产（gitignore）
│   └── FLAME2020/               # generic_model.pkl / flame_uv.npz / eyelid
├── output/                      # 产物（gitignore）
│   └── <subject>/<work_name>/   #   model.ply + config.yaml + render_image/
├── requirements.txt
└── README.md
```

### 4.1 vendored 源码的统一处置

**所有外部源码一律放在 `submodules/`，并从 `submodules/` 构建。**
不再区分 `ext/` 与 `third_party/`，也不再从 `~/Libraries/` 构建。

| 来源 | 处置 | 理由 |
|---|---|---|
| `submodules/diff-gaussian-rasterization` | vendor | 本项目唯一的 CUDA 内核，必须能独立编译、固定版本、按需修改 |
| `~/Libraries/nvdiffrast` | vendor 到 `submodules/nvdiffrast/` | 见 §4.2 |
| `~/Libraries/fused-ssim` | vendor 到 `submodules/fused-ssim/` | 见 §4.2 |
| `submodules/flame`, `submodules/fuhead` | vendor（P1） | 纯 Python，LBS 数学正确，不重写；需明确归属与许可 |
| `model/`, `diff_renderer/`, `camera/`, `dataset/` | **重写为 `src/`** | 这是本项目的主要工作量 |
| `train_*.py`, `render*.py`, `utils.py` | **重写为 `app/` + `training/`** | RGBAvatar 的脚本层与本项目形态差异最大 |

`submodules/ATTRIBUTION.md` 登记来源 commit、原许可、本项目的删减与修改点。

### 4.2 为什么 `nvdiffrast` / `fused-ssim` 也要 vendor

这两个库原本放在 `~/Libraries/`（源码 + 手工构建）。**直接把 `~/Libraries/...` 交给 `pip install` 会失败**：

```
error: could not delete 'build/lib.linux-x86_64-cpython-310/nvdiffrast/__init__.py': Permission denied
```

原因不是权限属主（文件属主就是当前用户），而是这两个源码目录里残留了此前手工构建的 `build/` 目录，而构建工具需要在源码树内删除/重建它。

**结论**：`~/Libraries/` 只作为「上游源码的获取来源」，**不作为构建源**。全部 vendor 到 `submodules/` 后：构建可复现（源码随仓库固定）、不依赖工作区外路径的可写性、与 CUDA/torch 版本解耦（换环境重新编译即可）。

### 4.3 数据集布局（当前仅 `duda`）

```
/home/crh/Datasets/INSTA/duda/          # 外部路径，不入库
├── images/       254 张 PNG             ← RGBAvatar 读取
├── checkpoint/   254 个 .frame          ← RGBAvatar 读取（FLAME 参数 + 相机）
├── flame/exp/                           ← INSTA 中间产物（当前不用）
├── meshes/ depth/ matted/ seg_mask/     ← INSTA/tracker 中间产物（当前不用）
└── transforms*.json  canonical.obj
```

数据集只需 `images/` 与 `checkpoint/` 两项即可驱动 RGBAvatar 的 `FLAMEDataset`；其余为 tracker/INSTA 的中间产物，保留但本项目不读取。

> **数据集范围限制**：本阶段**只以 `duda` 为基准**。所有质量验收（P1 数值等价门、P2 质量对齐门）均以 duda 为准。新增数据集前需先确认其 tracker 链路可复现。

---

## 5. RGBAvatar → Live3DGSAvatar 映射表

| RGBAvatar | Live3DGSAvatar | 处置 | 备注 |
|---|---|---|---|
| `diff_renderer/gaussian.py::GaussianAttributes` | `core/types.py::GaussianSet` | 重写 | 加 `space` 字段，固定 batch 维 |
| `diff_renderer/gaussian.py::render_gs` | `core/render/rasterizer.py::SimpleRasterizer` | 重写 | 推理路径 |
| `diff_renderer/gaussian.py::render_gs_batch` | 合并入上者 | **删除** | 原标注 `# legacy`，逐帧循环 |
| `diff_renderer/batch_gaussian.py::BatchGaussianRenderer` | `core/render/rasterizer.py::BatchRasterizer` | 重写 | 训练路径 |
| `diff_renderer/texture.py::compute_rast_info` | `core/deform/bind.py::build_binding` | 保留逻辑 | 用 nvdiffrast 在 UV 域光栅化 |
| `model/gaussian.py::GaussianModel` | `core/avatar.py::GaussianAvatar` | 重写 | 参数与 PLY 拆开 |
| `model/binding.py::BindingModel` | `core/deform/{blend,bind}.py` | **拆分** | 见 §3.4/§3.5 |
| `model/binding.py::FLAMEBindingModel` | `core/deform/` + `core/template/flame.py` | 重写 | |
| `model/binding.py::FuHeadBindingModel` | 同上 | 保留 | 在线路径的备选模板 |
| `model/reconstruction.py::Reconstruction` | `training/offline.py` | 重写 | 单目 |
| `model/mv_reconstruction.py::MultiViewReconstruction` | `training/offline.py`（多视角变体） | 重写 | 去重复 |
| `model/mv_reconstruction.py::render_gs_batch` | 合并入 `SimpleRasterizer` | **删除** | 第三份重复实现 |
| `camera/camera.py` | `core/types.py::Camera` + `camera_utils.py` | 重写 | 消除多份矩阵缓存 |
| `dataset/flame_dataset.py` | `data/flame_dataset.py` | 重写 | 可读旧格式（回归用） |
| `dataset/sampler.py` | `data/sampler.py` | 保留逻辑 | local-global 池 |
| `submodules/flame` | `submodules/flame` | vendor（P1） | |
| `submodules/fuhead` | `submodules/fuhead` | vendor（P1） | |
| `submodules/diff-gaussian-rasterization` | `submodules/diff-gaussian-rasterization` | vendor | |
| `utils.py` | 按职责拆分 | 重写 | `l1_loss`/`ssim` → `training/losses.py`；`Struct` → 用 dataclass 替代 |

---

## 6. 训练管线

### 6.1 离线训练（`training/offline.py`）

```
DataLoader(batch=10)
  └─ Frame{mesh, blend_weight, camera, image, mask}
        │
        ├─ blend:  weight[B,129] ─MLP→ w[B,20] ─linear_blending→ GaussianSet(tangent)
        │          ⚠ blend_start_iter 之前 skip，仅用基态
        ├─ bind:   GaussianSet(tangent) + Mesh → GaussianSet(world)
        ├─ render: BatchRasterizer(world, Camera, bg, target_image) → RenderOutput
        ├─ loss:   L1(颜色) [+ SSIM + LPIPS + alpha + sparsity + orth，按 config 开关]
        ├─ backward + Adam.step
        └─ fast_forward: 用 est_color/est_weight 一次性写入 color_dc
```

关键超参对齐 `configs/offline.yaml`：`blend_start_iter=3000`、`use_fast_forward=True`、`random_bg_color=True`、`position_lr_max_steps=30000`、`iteration=50000`。

### 6.2 在线/流式训练（`training/online.py`）

与离线共用 `blend/bind/rasterize/losses`，**唯一区别是数据来源与采样策略**：

```
新帧到达 ──► 加入 local pool M_l (FIFO, cap=150)
              └ 溢出时 ──► reservoir sampling 进 global pool M_g (cap=1000)
每步采样 B 个样本，其中 70% 来自 M_l，30% 来自 M_g
```

参数：`local_pool_max_size=150`, `global_pool_max_size=1000`, `max_global_ratio=0.7`。

> **架构上的关键认识**：在线训练与离线训练的差异**只是采样器**，`GaussianAvatar` 与渲染器完全复用。因此 `training/offline.py` 与 `online.py` 应共享同一个 `TrainStep` 实现，只在 `Iterable[Frame]` 的来源上分叉。

### 6.3 损失函数

| 损失 | 触发条件 | 实现来源 |
|---|---|---|
| L1 颜色 | `lambda_l1 > 0` | 自实现 |
| SSIM | `lambda_ssim > 0` | 自实现（单目）/ `fused_ssim`（多视角） |
| LPIPS | `lambda_lpips > 0` 且 `iter > 20000` | `lpips` |
| alpha | `lambda_alpha > 0` | 自实现 |
| sparsity | `lambda_sparsity > 0` 且 `use_weight_proj` | 自实现（MLP 输出 L1） |
| orth | `lambda_orth > 0` 且已启用 blend | 自实现（基正交性） |

---

## 7. 渲染与实时性

### 7.1 推理路径（本阶段重点）

```
Frame(mesh, blend_weight, camera)
  └─ blend → bind → SimpleRasterizer → RenderOutput
```

推理时 `blend_weight` 来自 `tracking/`。本阶段 `tracking/offline_flame.py` 复用数据集中已拟合的 FLAME 参数（与训练同源），**不引入新的 tracker 依赖**——这保证第一阶段可以完全复现 RGBAvatar 的渲染结果。

### 7.2 实时性的三个来源

1. **不做梯度**：推理路径只走 forward，不分配 `grad_*` 缓冲。
2. **不做逐帧重建**：模型参数固定，每帧只做 `blend + bind + rasterize`。RGBAvatar 实测约 400 FPS（RTX 3090，含动画），这是本架构的性能天花板参考。
3. **零拷贝**：`GaussianSet` 的 5 个张量直接从 `GaussianAvatar` 的参数视图构造，避免 `expand/cat` 产生的隐式拷贝。

### 7.3 服务端渲染（P3 骨架，本阶段只留接口）

`runtime/session.py` 定义：

```python
class Session:
    def __init__(self, avatar: GaussianAvatar, runtime: AvatarRuntime): ...
    def push_drive_params(self, params: torch.Tensor) -> None: ...  # 接收驱动流
    def render_frame(self, view: Camera) -> RenderOutput: ...       # 按观看者视角渲染
    def push_observation(self, frame: Frame) -> None: ...           # 供在线更新
```

`streaming/` 本阶段只留空模块与 TODO，**不实现**。服务端渲染的具体传输方案（WebRTC / 编码器选型）留待 P3 决策。

---

## 8. 环境与复现

> **权威来源**：`docs/ENVIRONMENT.md`。本节只保留架构相关的要点——完整的搭建步骤、
> 双 CUDA 工具链说明、排错表、验证记录都在那里。

### 8.1 固定版本矩阵

| 组件 | 版本 | 说明 |
|---|---|---|
| Python | 3.10.21 | CUDA 扩展编译产物为 `cpython-310` |
| PyTorch | **2.14.0+cu130** | 实测可用，不再改动 |
| CUDA Toolkit | **13.0** | 环境内 nvcc 13.0.48 / 系统 13.0.88，二者皆可用 |
| torchvision | 0.29.0+cu130 | |
| nvdiffrast | 0.4.0 | vendored 到 `submodules/nvdiffrast/` |
| fused-ssim | 1.0.0 | vendored 到 `submodules/fused-ssim/` |
| GPU 架构 | `sm_86` | `TORCH_CUDA_ARCH_LIST=8.6` |
| 环境名 | **`live3dgs`** | 与参照仓库的 `rgbavatar` 隔离（约束 C8） |

> **重要**：CUDA 13.0 是本项目基线。RGBAvatar 官方 README 建议 11.8，但在本环境下 11.8 反而不可用。

### 8.2 三条构建规则

1. **一切从 `submodules/` 构建，不从 `~/Libraries/` 构建。** 后者残留手工构建的 `build/` 目录，
   会导致 `pip install` 在源码树内删除失败；vendor 到工作区后正常。理由见 §4.2。
2. **所有本地扩展用 `--no-build-isolation`**，且**必须先把 `setuptools/wheel/ninja` 装进环境**——
   否则 pip 不会自动准备构建依赖，报 `Failed to build installable wheels for ...`。
3. **扩展用 `pip install -e`（editable）**：`.so` 落在源码目录内，便于用 `git status` 发现产物与源码不同步。

一键脚本 `scripts/setup_env.sh` 已把上述规则固化，幂等可重跑。

### 8.3 数据与模型资产（当前仅 `duda`）

| 资产 | 路径 | 状态 |
|---|---|---|
| FLAME2020 | `data/FLAME2020/generic_model.pkl` | ✅ 已有（gitignore） |
| flame_uv | `data/FLAME2020/flame_uv.npz` | ✅ 已有 |
| eyelid | `data/FLAME2020/{l,r}_eyelid.npy` | ✅ 已有 |
| 预训练模型（参照） | `RGBAvatar/output/duda/test/model.ply` | ✅ 只读参照 |
| 数据集 | `/home/crh/Datasets/INSTA/duda`（254 帧） | ✅ 已有 |
| 基线产物 | `output/smoke/`（由 `scripts/smoke_test.py` 生成） | ⏳ 待 GPU 运行 |

> **`data/checkpoints` 不需要**：RGBAvatar 的数据集契约只有"多个 subject/work 下的 `checkpoint/` + `images/`"，
> 而这两者都在数据集目录内（`<DATA_DIR>/<SUBJECT>/{checkpoint,images}`）。本项目训练产出的
> `model.ply` 归属于**训练输出**，因此放在 `output/<subject>/<work_name>/`，不新建 `data/checkpoints/`。

> **数据集范围限制**：本阶段只以 `duda` 为基准，所有质量验收均以 duda 为准（约束 C7）。
> `NeRSemble` 相关代码路径本阶段不引入。

---

## 9. 验收门（每个阶段必须可量化）

| 阶段 | 目标 | 验收标准 | 状态 |
|---|---|---|---|
| **P0 基线与环境** | 环境可复现 + 基线可复现 | ① 单条命令从零建环境 ② `scripts/env_check.py` 全绿 ③ 用 `duda` 的 `model.ply` 跑通渲染并记录 **FPS / 峰值显存 / 单帧耗时** | ✅ 全部完成 |
| **P1 只读内核重写** | `core/` 完成，行为与 RGBAvatar 一致 | **数值等价门**：分层比对参照实现，中间属性 `max\|Δ\| < 1e-5`，渲染图 **`PSNR > 60 dB` 或 `max\|Δ\| < 1e-3`**；`tests/equivalence/` 全绿 | 代码 ✅ / 等价门 ⏳ 待 GPU 运行 |
| **P2 训练重写 + GUI** | 离线训练复现 + 图形化程序 | ① 在 `duda` 上 PSNR 与论文差距 **< 1 dB** ② GUI 可启动训练、实时预览、导出 | 未开始 |
| **P3 服务化** | 服务端渲染 + 推流 | 端到端延迟 **< 150 ms**（目标 100 ms）；单路稳定 10 分钟 | 未开始 |
| **P4 动态更新** | 在线训练 | 按帧顺序在线重建，PSNR 与离线差距 **< 1 dB** | 未开始 |

**P0 与 P1 之间不可跳过**：数值等价门是本次重写的安全网。没有它，无法区分"架构改进"与"引入了 bug"。

### 9.1 P1 交付物与当前状态

| 交付物 | 状态 |
|---|---|
| `docs/CONVENTIONS.md` | ✅ 坐标/矩阵/空间/精度约定 |
| `docs/MIGRATION.md` | ✅ 4 项缺陷修复 + 6 项有意偏离登记 + 1 项未解决偏差 |
| `docs/CORE_GUIDE.md` | ✅ 代码导览：形状流转、设计原因、陷阱清单、"验证 X 跑哪条命令" |
| `core/types.py` · `core/avatar.py` · `core/io/ply.py` | ✅ 类型契约、参数容器、PLY 互操作 |
| `core/deform/{tbn,bind,blend,binding}.py` | ✅ TBN / 绑定 / 混合 / UV 绑定构建 |
| `core/render/{camera_utils,rasterizer}.py` | ✅ 矩阵边界 + 两个光栅化后端 |
| `tests/run_tests.py` · `tests/unit/` | ✅ **零依赖**运行器，**35 项**，全部无需 GPU（含架构一致性检查） |
| `tests/equivalence/` · `scripts/equivalence_check.py` | ✅ 已交付并执行；发现绑定层 `xyz` 偏差（见 MIGRATION D 节） |
| `scripts/diagnose_binding.py` | ✅ 绑定层**原理性验证**（运动学不变量，不以参照为判据） |

### 9.2 分层规则的可执行化

`docs/ARCHITECTURE.md` §2.3 声明的依赖规则由 `tests/unit/test_architecture.py` 强制检查：

| 检查 | 内容 |
|---|---|
| 分层单向 | `core/` 不得 import `data/` / `training/` / `app/` / `tracking/` / `runtime/` / `streaming/` |
| CUDA 隔离 | `diff_gaussian_rasterization` 只允许在 `core/render/rasterizer.py` 与 `core/deform/blend.py` 出现 |
| 无 I/O | `core/` 不得 import `argparse` / `sys`；`plyfile` / `json` 仅限 `core/io/ply.py` |
| 导入期无设备访问 | `core/` 模块作用域不得调用 `torch.cuda.*`（否则无 GPU 环境下导入即失败） |

最后一条尤其重要：它保证了「导入 `core` 不需要 GPU」这个性质，
CPU 侧的 35 项测试才得以成立。

### 9.3 等价门的执行结构

等价门分**六步**执行，其中比对分五层，任一层失败都能直接定位，
而不是只看到「图不一样」：

| 步骤 | 内容 | 判据 |
|---|---|---|
| 0 前置检查 | 路径、数据集子目录、FLAME 模型是否就位（**不需 GPU**） | 全部存在 |
| 1 参数 | PLY 读入的 8 组参数 + `weight_module` | 逐位一致（`max\|Δ\| = 0`） |
| 2 绑定构建 | UV 域光栅化：`valid_mask` / `face_id` / `face_bary` | 精确 / `1e-5` |
| 3 混合 | `project_weight` + blendshape 线性混合（激活前后） | `max\|Δ\| < 1e-5` |
| 4 绑定 | `mesh_binding` 输出（xyz / rotation / 颜色 / 不透明度 / 尺度） | `max\|Δ\| < 1e-5` |
| 5 渲染 | 光栅化输出 `color` / `alpha` | `PSNR > 60 dB` 或 `max\|Δ\| < 1e-3` |

**第 2 层（绑定构建）单独成层很重要**：UV 光栅化决定了「哪些 texel 变成高斯、每个高斯绑到哪个面」，
若它不一致，后面所有几何比对都必然失败。缺少这一层就只能看到「xyz 不一样」，
无法区分是光栅化错了还是 `mesh_binding` 错了。

比对逻辑集中在 `tests/equivalence/stages.py`，**脚本与测试共用同一份**，避免判据漂移。

---

## 10. 已知问题与设计决策（避免继承 RGBAvatar 的缺陷）

| 编号 | RGBAvatar 的问题 | 本设计的处置 |
|---|---|---|
| K1 | `BindingModel.clone()` 签名与 `__init__` 不匹配（调用 5 参数，签名 3 参数）→ 调用即报错 | 不移植 `clone()`；快照用 `AvatarSnapshot`（state_dict 形式） |
| K2 | `save_ply()` 引用不存在的 `self.binding_face_id` 属性 → 任何模型都保存失败 | 绑定信息归属 `GaussianAvatar`（非 `GaussianModel`），保存路径统一 |
| K3 | `load_ply()` 从 config 读取 `num_basis_blend` 而非文件列数 → config 与 PLY 不一致时静默出错 | PLY 中记录 `num_basis_blend`；加载时校验并报错 |
| K4 | `render_gs_batch` 存在三份重复实现，其中一份标注 `# legacy` | 合并为 `Rasterizer` 协议 + 两个后端（§3.6） |
| K5 | 单帧/批量两套张量形状并存，靠调用方记忆区分 | 固定 batch-first（§3.1） |
| K6 | 切空间/世界空间高斯靠调用顺序区分，无显式标注 | `GaussianSet.space` 字段（§3.2） |
| K7 | `compute_rast_info` 有 `FIXME: precision issue across different devices` | 绑定结果缓存进模型文件；加载时校验 `N` 与 `binding_face_id` 长度一致 |
| K8 | 依赖 `Struct(**dict)` 传参，无类型检查 | 全部替换为 dataclass（`AvatarConfig` / `RuntimeConfig`） |
| K9 | CUDA/torch 版本强耦合，且官方文档版本建议不正确 | `env_check.py` + 构建时版本校验（§8.2） |
| K10 | `load_ply` 中 `f_rest` 恒为 45 个零（sh_degree=0 的占位） | 保留占位以兼容 3DGS 工具链，但明确注释其恒零 |

---

## 11. 里程碑顺序

```
P0 基线与环境 ──► P1 只读内核重写 ──► P2 训练重写 + GUI ──► P3 服务化 ──► P4 动态更新
     (必须)          (数值等价门)         (质量对齐门)        (延迟门)
```

**建议从 P0 + P1 开始**：P0 把环境与基线锁死，P1 把 `core/` 写完并用数值等价门验证。这两步完成后，后续所有工作都在一个可信的基础上进行。

---

## 附录 A：未来工作（本阶段不做，仅记录）

| 方向 | 说明 | 触发条件 |
|---|---|---|
| **上半身/肩颈** | 需引入混合拓扑模板（个性化人体网格挖空面/手 + FLAME 面部 + MANO 手 + 蒙皮权重迁移）与独立的身体高斯集合 | P2 完成后评估 |
| **实时 tracker** | 当前依赖离线 metrical-tracker。实时需自建 "2D landmark → FLAME 系数" 回归器 | P4 后 |
| **前馈重建路线** | VoluMe 类方法（单帧 → U-Net → splatter image）。瓶颈在合成训练数据管线而非网络结构 | 单独预研 |
| **光照解耦** | 当前颜色为 SH DC（视角无关、烘焙）。通话中光照变化无法适应 | 需要 relightable 表示 |
| **多部件合成** | `HeadPart` 之外的 `BodyPart` / `HandsPart`，共用同一 `DeformationField` 接口 | 与上半身同步 |
| **服务端并发** | 单卡多路会话调度与资源隔离 | P3 之后 |
| **商业化替换** | FLAME（非商用）/ FaceWarehouse / DDE 的替代方案 | 商用决策后 |

## 附录 B：术语表

| 术语 | 含义 |
|---|---|
| 切空间 (tangent space) | 以绑定三角面的 TBN 为基的局部坐标系；高斯参数默认存于此 |
| 世界空间 (world space) | 变形后的全局坐标系；光栅化器在此空间工作 |
| 约简基 (reduced blendshapes) | MLP 把 D 维 3DMM 参数压成 K=20 维权重，混合 K 组高斯基 |
| fast forward | 用渲染统计量（`est_color/est_weight`）一次性写入高斯颜色的初始化策略 |
| local-global 池 | 在线训练中"新帧池 + 历史帧池"的双池采样策略，用于抗遗忘 |
| TBN | tangent / bitangent / normal，逐三角面的切空间正交基 |
