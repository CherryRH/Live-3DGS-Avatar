# 与 RGBAvatar 的行为差异登记

> **规则**：本项目对参照实现（`RGBAvatar/`，只读）的任何**有意**偏离都必须登记在本文件，
> 并说明动机与影响。未登记的偏离视为 bug。
>
> 参照仓库 commit：`903653b689a8b860df30f165e242ccd6ed3f5583`

---

## A. 缺陷修复（参照实现的行为是错的）

### K1｜`BindingModel.clone()` 调用即失败

| 项 | 内容 |
|---|---|
| 现象 | `model/binding.py:209` 调用 `BindingModel(self.model_config, self.template_uvs, self.template_faces, self.template_uv_faces, self.glctx)`（5 个参数），但 `__init__` 签名是 `(model_config, template_model, glctx)`（3 个参数）。 |
| 影响 | 该方法在任何路径上都不可用（死代码）。 |
| 本项目处置 | **不移植**。快照改用 `GaussianAvatar` 的 `state_dict()`。 |

### K2｜`save_ply()` 引用不存在的属性，任何模型都保存失败

| 项 | 内容 |
|---|---|
| 现象 | `model/gaussian.py:188` 读取 `self.binding_face_id` / `self.binding_face_bary`，但这两个属性定义在子类 `BindingModel.binding()` 中；`GaussianModel` 自身从未定义。 |
| 影响 | 保存任意模型都会 `AttributeError`。 |
| 本项目处置 | 绑定信息归属 `GaussianAvatar`（`register_buffer`），保存路径统一。`tests/unit/test_ply.py::test_reference_save_ply_is_broken_this_project_fixed_it` 把这条差异固化。 |

### K3｜`load_ply()` 从配置读基数量，而非文件列数

| 项 | 内容 |
|---|---|
| 现象 | `model/gaussian.py:256-275` 用 `self.model_config.num_basis_blend` 决定读多少列 `xyz_b_*`；`num_basis_blend` 来自配置文件。 |
| 影响 | 配置与 PLY 不一致时，会读到错误的列数或抛形状断言，**且不提示真正原因**。 |
| 本项目处置 | `core/io/ply.py` 在 PLY 头写入 `comment gaussian_config {...}` 自描述，加载时与文件实际列数**交叉校验并显式报错**。 |

### K4｜`weight_module` 被压进单列的容量约束未校验

| 项 | 内容 |
|---|---|
| 现象 | `save_ply` 把 MLP 参数展平后写入单列（前 P 行有值、其余补零），仅有一句 `assert linear_module.shape[0] <= num_gs`。 |
| 影响 | Python 以 `-O` 运行时断言被跳过 → 静默写出截断的参数。 |
| 本项目处置 | `core/io/ply.py` 改为显式 `ValueError`，并在报错信息中给出所需的 `tex_size` 下界。 |

---

## B. 有意偏离（本项目的行为与参照不同，但参照并非"错"）

### H1｜三份重复的渲染函数合并为一套协议

| 项 | 内容 |
|---|---|
| 参照 | `diff_renderer/gaussian.py::render_gs`、同文件 `render_gs_batch`（标注 `# legacy`）、`model/mv_reconstruction.py::render_gs_batch` 三份实现。 |
| 风险 | 三者语义相同、实现不同，任何修改都要同步三处，已出现"其中一份被标注 legacy 却仍在使用"的状态。 |
| 本项目 | 统一为 `core/render/rasterizer.py` 的 `SimpleRasterizer`（推理）与 `BatchRasterizer`（训练），语义差异由「是否传 `target_image`」表达。 |
| 数值影响 | 无。三者都调用同一 CUDA 内核与相同矩阵。 |

### H2｜空间归属由字段显式标注

| 项 | 内容 |
|---|---|
| 参照 | 切空间（`get_attributes` 输出）与世界空间（`mesh_binding` 输出）都是普通张量组，靠调用顺序区分。 |
| 风险 | 把世界空间高斯再绑一次、或对切空间高斯直接光栅化，都不会报错，只会给出错误结果。 |
| 本项目 | `GaussianSet.space ∈ {"tangent","world"}`；`MeshBinder` 拒绝非切空间输入，`Rasterizer` 拒绝非世界空间输入。 |
| 数值影响 | 无（纯标注与校验）。 |

### H3｜矩阵转置收敛到单一位置

| 项 | 内容 |
|---|---|
| 参照 | `camera.get_w2v.transpose(0, 1)` 与 `camera.get_full_proj.transpose(0, 1)` 出现在每个调用点，并被三份渲染函数各复制一遍。 |
| 风险 | 行主序/列主序约定散落，改一处漏一处会得到"相机镜像"或"全黑"，且极难定位。 |
| 本项目 | 仅 `core/render/camera_utils.py::to_kernel_matrix` 允许转置。 |
| 数值影响 | 无（`transpose + contiguous` 是等价变换）。 |

### H4｜`linear_blending` 默认走纯 PyTorch，而非 CUDA 内核

| 项 | 内容 |
|---|---|
| 参照 | `GaussianModel.get_batch_attributes` 调用 CUDA 版 `linear_blending`。 |
| 发现 | **该 CUDA 算子在 CUDA 不可用时（CPU 张量、无设备）静默返回全零，不报错。** 本项目在 P1 开发中实测确认（`scripts` 侧复现：CPU 张量调用返回全零）。 |
| 本项目 | `GaussianBlendField` 默认 `use_cuda_kernel=False`，走纯 PyTorch 路径；显式开启时加两道防护（拒绝 CPU 张量、拒绝 `torch.cuda.is_available()==False`）。 |
| 数值影响 | 已在 `tests/unit/test_deform.py` 与 GPU 等价门中验证两者一致。性能上 CUDA 版更省显存，属有意识的取舍。 |

### H5｜TBN 的矩阵布局显式化

| 项 | 内容 |
|---|---|
| 参照 | CUDA 版 `mesh_binding` 用 `transpose(R) @ xyz`；Python 回退版 `gaussian_deform_torch` 用 `R @ xyz`（未转置）。两者对 TBN 的行/列主序解释不同。 |
| 本项目 | **以 CUDA 版为准**（推理与训练实际都走它）。`core/deform/tbn.py` 明确返回「列为基向量」，并在 `tests/unit/test_deform.py::test_face_tbn_layout_is_column_basis` 中用单位三角面锁定该布局。 |
| 数值影响 | 与 CUDA 版一致；与 Python 回退版不一致 —— 这是参照实现自身的不一致，不是本项目的偏离。 |

### H6｜`Struct(**dict)` 配置改为 dataclass

| 项 | 内容 |
|---|---|
| 参照 | 用 `utils.Struct(**config['model'])` 传参，无类型检查、无默认值语义。 |
| 本项目 | `AvatarConfig` dataclass，带 `__post_init__` 校验与 `mlp_hidden` 等显式字段（PLY 还原网络结构所必需）。 |
| 数值影响 | 无。 |

---

## C. 尚未处理 / 需持续关注

| 编号 | 事项 | 状态 |
|---|---|---|
| O1 | `compute_rast_info` 的 `FIXME: precision issue across different devices` —— UV 光栅化的绑定结果可能跨设备差 1 个面 | 已缓解：绑定结果随模型文件缓存；加载时校验 `N` 与 `face_id` 长度一致。根治方案待定 |
| O2 | 参照的 Python 回退 `gaussian_deform_torch` 与 CUDA 版不一致（H5） | 本项目不移植回退版；如将来需要 CPU 回退，需先统一 TBN 布局语义 |
| O3 | `f_rest_*` 恒为零的占位列 | 保留以兼容 3DGS 工具链；`load_ply` 会在该列非零时显式拒绝（说明文件中含高阶 SH，而本项目 `sh_degree=0`） |
| O4 | 光照烘焙进 SH DC，通话中光照变化无法适应 | 已在 `docs/ARCHITECTURE.md` 附录 A 记录为未来工作 |
