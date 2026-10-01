# 配置说明

本项目的**所有可调参数**都集中在 `configs/`，代码中不再有硬编码的路径与业务默认量。

| 文件 | 内容 |
|---|---|
| `configs/system.yaml` | 路径、设备、数据集主体名 |
| `configs/render.yaml` | 渲染参数、模型结构、数据集读取选项 |
| `configs/local.yaml` | **可选**，本机私有覆盖（已 gitignore） |

## 1. 优先级

```
命令行参数  >  环境变量  >  configs/*.yaml  >  代码内置兜底
```

代码内置兜底（`src/live3dgsavatar/config/__init__.py::_FALLBACK`）**刻意保持最小**，
只为保证字段缺失时不 `KeyError`；真正的默认量都在 YAML 里，便于查看与修改。

`configs/local.yaml` 在 `system.yaml` / `render.yaml` 之后**深合并**，用于放不想入库的
本机覆盖（例如另一个数据集路径）。

## 2. 查看最终生效值

```bash
python scripts/show_config.py                 # 全部
python scripts/show_config.py --section paths # 只看路径
python scripts/show_config.py --sources       # 配置文件是否被读到 + 环境变量
python scripts/show_config.py --json          # 机器可读
```

`show_config.py` 还会提示哪些关键路径当前不存在（换机器/未下载时正常）。

## 3. 环境变量

| 变量 | 覆盖字段 |
|---|---|
| `LIVE3DGS_DATA_ROOT` | `paths.data_root` |
| `LIVE3DGS_OUTPUT_DIR` | `paths.output_dir` |
| `LIVE3DGS_REFERENCE_ROOT` | `paths.reference_root` |
| `LIVE3DGS_SUBJECT` | `subject` |
| `LIVE3DGS_DEVICE` | `runtime.device` |

环境变量只需填**路径字符串**，加载时会解析为绝对路径。

## 4. `system.yaml` 字段

| 字段 | 说明 |
|---|---|
| `subject` | 数据集主体名。实际读取 `<data_root>/<subject>/`。切换受试者只改这一处 |
| `paths.data_root` | 数据集根目录；下需含 `<subject>/images/` 与 `<subject>/checkpoint/` |
| `paths.output_dir` | 产物根目录。各脚本写到 `<output_dir>/<脚本名>/` |
| `paths.reference_root` | RGBAvatar 参照仓库根（**只读**）。等价门与渲染测试从中取 FLAME 模板、模板 UV、数据集读取代码与预训练模型 |
| `paths.model_ply` | 预训练模型。`null` 表示按 `model_subdir` 推导 |
| `paths.model_subdir` | 模型产物目录名（参照仓库里常是 `test`）。**与 `runtime.split` 无关** |
| `paths.image_subdir` | 数据集原图子目录（默认 `images`），用作 PSNR 的 GT |
| `runtime.device` | `cuda` 或 `cpu`。`cpu` 仅供不依赖 CUDA 内核的自检 |
| `runtime.split` | 数据集切分：`all` / `train` / `test` |
| `smoke.max_frames` | 采样帧数上限；`null` 表示不限制 |

### 路径写法

三种都支持：

```yaml
data_root: /home/you/Datasets/INSTA    # 绝对路径
data_root: ~/Datasets/INSTA            # ~ 展开
reference_root: ../RGBAvatar           # 相对**仓库根**
output_dir: output                     # output 特殊：相对**当前工作目录**
```

> `output_dir` 是唯一按 cwd 解析的字段 —— `--output-dir out` 通常指用户当前目录。

## 5. `render.yaml` 字段

### `render` —— 渲染

| 字段 | 默认 | 说明 |
|---|---|---|
| `background` | `[0,0,0]` | 背景色。数据集原图为黑底；白底用 `[1,1,1]` |
| `scaling_modifier` | `1.0` | 高斯缩放系数 |
| `sh_degree` | `0` | 球谐阶数。本项目只用 DC 项 |
| `batch_size` | `4` | 批大小。**core 与参照实现共用**，保证计时口径一致。6GB 显存建议 ≤ 4 |
| `frames` | `20` | 默认渲染帧数；`-1` 为全部 |
| `max_batch_size` / `max_gaussian_size` / `num_streams` | `10` / `60353` / `3` | 训练用的预分配上限（训练待接入，先保留） |

> 输出**分辨率**不在此配置 —— 它由数据集相机内参决定。

### `model.network` —— 模型结构

⚠️ **这一节必须与 `.ply` 文件匹配。** 项目的 `.ply` 头里写了
`comment gaussian_config {...}` 自描述结构，加载时会与实际列数交叉校验，
写错会**直接报错**而不是静默读错列。

| 字段 | 默认 | 说明 |
|---|---|---|
| `tex_size` | `256` | UV 纹理边长；高斯数上限为 `tex_size²` |
| `num_basis_in` | `129` | 驱动参数维度（FLAME 姿态/表情等） |
| `num_basis_blend` | `20` | blendshape 基数量 K |
| `mlp_hidden` | `[128,128]` | 权重映射 MLP 隐藏层宽度；`null` 表示不用 MLP |
| `use_weight_proj` | `true` | 是否使用 MLP 权重投影 |

### `dataset` / `test`

| 字段 | 说明 |
|---|---|
| `dataset.use_shape_weight` / `use_pose_weight` | 参照侧 `FLAMEDataset` 的读取选项 |
| `dataset.pin_memory` | 是否锁页内存 |
| `test.dump_diff` | 渲染测试是否输出并排对比图 |

## 6. 脚本参数：只保留关键可调项

命令行是**纯覆盖层**。不传就用配置值，因此日常使用不必带任何参数：

```bash
python scripts/render_test.py                     # 全部取配置
python scripts/render_test.py --frames 5 --batch-size 2   # 临时覆盖
python scripts/equivalence_check.py --skip-render
```

`test_scripts_have_no_business_defaults` 会静态检查脚本里不再出现
`default=256` 这类业务默认量。

## 7. 写自己的脚本时怎么用

```python
from live3dgsavatar.config import load_config

cfg = load_config()                       # 读仓库根的 configs/
data_dir = cfg.paths.data_root / cfg.subject
batch = cfg.get("render.batch_size", 1)   # 点路径 + 默认值
cfg.set("render.batch_size", 8)           # 就地覆盖
print(cfg.resolved())                     # 便于写日志/报告
```

要点：

- 路径字段已是**绝对 `Path`**，直接用；
- `cfg.get("a.b.c", default)` 支持点路径，缺失时返回 `default`；
- 把 `cfg.resolved()` 写进结果文件，便于事后复现；
- **`core/` 不允许依赖配置**（见 `docs/ARCHITECTURE.md` §2.3）；
  配置属于应用层，`core/` 只接受显式传入的参数。

## 8. 相关测试

| 测试 | 覆盖 |
|---|---|
| `test_config.py::test_loads_defaults_from_yaml_files` | 默认量确实来自 YAML |
| `test_config.py::test_env_overrides_yaml` | 环境变量优先于 YAML |
| `test_config.py::test_local_yaml_is_merged_last` | `local.yaml` 深合并 |
| `test_config.py::test_wrong_type_is_rejected` | 类型错误明确报错 |
| `test_config.py::test_no_hardcoded_absolute_paths_in_source` | **代码中无硬编码路径** |
| `test_config.py::test_scripts_have_no_business_defaults` | **脚本无业务默认量** |
