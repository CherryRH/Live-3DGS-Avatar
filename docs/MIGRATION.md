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

---

## D. 未解决项：绑定层的 xyz 偏差（P1 等价门发现的第一个真实缺陷）

### 现象

`scripts/equivalence_check.py` 在 duda 预训练模型（60,353 高斯）上的首次运行结果：

| 层 | 结果 |
|---|---|
| 参数（9 项） | ✅ 全部 `max\|Δ\| = 0`（逐位一致） |
| 绑定构建（5 项） | ✅ 全部通过，含 `face_id` / `face_bary` 精确匹配 |
| 混合（6 项） | ✅ 全部通过（`xyz` 2.2e-08，`rotation` 1.2e-06） |
| **绑定（1 项）** | ❌ **`xyz` max\|Δ\| = 9.874e-02** |
| 绑定（其余 4 项） | ✅ `rotation` 8.3e-07、`opacity`/`scaling`/`color` 通过 |
| 渲染（2 项） | ❌ 由绑定层误差传导：`color` PSNR 22.01 dB、`alpha` 14.54 dB |

### 已排除的可能（有证据）

| 假设 | 排除依据 |
|---|---|
| PLY 读入错 | 参数层 9 项 `max\|Δ\| = 0` |
| UV 绑定构建错 | 绑定构建层 5 项通过，且与参照模型缓存的 `binding_face_id` 完全一致 |
| TBN 计算错 | `rotation` 由 TBN 经 `matrix_to_quaternion` 导出，匹配到 8.3e-07 |
| 切空间混合错 | 混合层 `xyz` 差 2.2e-08 |
| 四元数/矩阵布局 | `rotation` 匹配 |
| `face_id` 为 0-based 的约定 | `tests/unit/test_binding_construction.py` 用假光栅化锁定 |
| `bary = [u, v, 1-u-v]` 的约定 | 同上 |

### 首轮推理（**已被第二轮诊断推翻，保留以便追溯**）

曾推断「`R` 正交 + 切空间 `xyz` 量级 1e-5 ⟹ 误差只能来自 offset」。
**该前提是错的**：实测切空间 `xyz` 量级为 **8e-02**（不是 1e-5），
因此旋转项 `Rᵀ·xyz_tan` 与 offset 同量级，**不能排除旋转项**。

### 第二轮诊断给出的新事实

| 量 | 实测 |
|---|---|
| 切空间 `xyz` 量级 | 7.93e-02 |
| 旋转项量级 | 7.97e-02 |
| offset 量级 | 1.73e-01 |
| 误差 **均值** | **5.20e-04** |
| 误差 max | 9.87e-02 |

**结论修正**：误差在逐高斯均值上只有 5.2e-04，说明它是**少数离群高斯**造成的，
而不是全局性的变换错误。前 3 个高斯两侧仅差约 8e-4。

反解等效重心 `w_eff` 得到 ±400 量级的权值与 4.2e-05 残差，
说明在这些三角形的顶点几何下最小二乘病态，**该线索无效**。

### 第三轮诊断：I4 判据本身是错的（**已修正**）

诊断脚本曾报告 `I4 TBN 正交 max|RᵀR-I| = 9.996e-01`、60314/60353 个高斯"非正交"，
一度被当作缺陷。**这是把错误的假设当成了不变量。**

代数上，对三角形 `(v0,v1,v2)` 与 UV `(a,b,c)`：

```
normalize(tangent) · normalize(bitangent) = −cos(∠A')
```

其中 `A'` 是 UV 三角形在顶点 `a` 处的内角。**两者正交当且仅当该角为直角。**
一般 UV 图不是正交参数化，因此非正交是**正常现象**。

**实测证据**（`flame_uv.npz`，9976 个 UV 三角形，纯 CPU 计算）：

| 量 | 值 |
|---|---|
| `\|cos(tangent,bitangent)\|` 中位数 | **0.676** |
| 近似正交（`\|cos\| < 0.01`）的面占比 | **0.47%** |
| UV 面积中位数 | 2.13e-05 |
| UV 面积 < 1e-8 的**退化面** | **34 个** |

已固化进测试：`tests/unit/test_deform.py::test_face_tbn_bases_may_be_skewed`
用顶点 a 处内角 45° 的 UV 验证 `cos = −cos(45°) = −0.7071`（实测精确命中），
并**明确断言"不正交"**，防止以后有人再把它当 bug 去"修"。

### 但由此引出一个真问题：`Rᵀ` 是否成立

若 `R` 不是正交阵，则 `Rᵀ ≠ R⁻¹`，绑定步骤用 `Rᵀ` 时变换会被 UV 剪切污染。
这才是需要判定的问题（而非"TBN 是否正交"）。

为此新增 `compute_face_tbn(mode="orthonormal")`：以几何法线为准做 Gram-Schmidt，
`(t, n×t, n)` 构成正交基，`R` 成为真正的旋转，`Rᵀ = R⁻¹` 严格成立。
诊断脚本的 **I9 对照实验**会比较"正交化前/后与参照的差异"，用以判定哪种做法正确。

### 第四轮：结论已确立 —— `Rᵀ·x` 在数学上不成立

**判据（不依赖任何参照实现）**：TBN 满足 `R(QM) = Q·R(M)`（实测 3.28e-07），
因此对刚体变换 `T = (Q, t)`：

```
R(QM)  · x = Q·(R(M)·x)       ⟹  R·x 形式严格等变
R(QM)ᵀ · x ≠ Q·(R(M)ᵀ·x)      ⟹  Rᵀ·x 形式仅在 R 正交时等变
```

**实测（合成网格 + 带剪切 UV，纯 PyTorch，无需 GPU）**：

| 项 | 误差 |
|---|---|
| TBN 变换律 `R(QM) − Q·R(M)` | 3.28e-07 ✓ |
| 旋转项 `R(QM)x` vs `Q(R(M)x)` | **4.45e-09 ✓** |
| 旋转项 `R(QM)ᵀx` vs `Q(R(M)ᵀx)` | **4.73e-02 ✗** |

**结论**：`R` 不是正交阵（真实 FLAME UV 上 `|cos(t,b)|` 中位数 0.68），
故 `Rᵀ ≠ R⁻¹`，`Rᵀ·x` 不满足刚性等变。**参照实现的 `mesh_binding` 用的正是
`transpose(binding_rotation) * gs_xyz`，即 `Rᵀ·x`，因此它在数学上不成立。**

这解释了等价门观察到的现象：
- 偏差是**平滑长尾**（中位数 6.9e-03 → max 9.87e-02），因为误差正比于该面的 UV 剪切程度；
- `rotation` 那一项通过（旋转由 `matrix_to_quaternion(R)` 导出，与乘法顺序无关）。

**已固化**：`tests/unit/test_deform.py::test_binding_is_rigidly_equivariant`
断言 `R·x` 等变（< 1e-5）而 `Rᵀ·x` 不等变（> 1e-3），
并附 `test_orthonormal_mode_makes_transpose_equal_inverse` 说明正交化的意义是
`Rᵀ = R⁻¹`，而非"两种写法相等"。

### 本项目应采用的正确形式

```
xyz_world = R · xyz_tan + Σᵢ baryᵢ · vᵢ        # R 为「列为基向量」的矩阵
```

即 `MeshBinder` 应使用 `R`（当前实现用的是 `Rᵀ`）。此改动**尚未落地** —— 需要先在
GPU 上跑通验证（见下方待办），确认后再改默认行为并更新 `docs/CONVENTIONS.md` §3.2。

### 过程中被推翻的三个错误推理（保留供追溯）

| # | 错误推理 | 推翻依据 |
|---|---|---|
| 1 | "切空间 xyz 量级 1e-5 ⟹ 误差必来自 offset" | 实测切空间 xyz 量级为 **8e-02**，旋转项与 offset 同量级 |
| 2 | "TBN 应当正交" | `normalize(t)·normalize(b) = −cos(∠A')`，真实 UV 上中位数 0.68 |
| 3 | "两种乘法顺序都满足等变性，无法区分" | 只有 `R·x` 满足；`Rᵀ·x` 在非正交 `R` 下不等变 |

### 第三轮待检验项

`scripts/diagnose_binding.py`（v2）新增：

1. 按误差排序打印**最差 N 个高斯**的全部输入（face_id / bary / 三顶点 / 切空间 xyz / R 三列 / R 正交性）；
2. **误差来源拆分**：分别「固定 offset 反解 rot_part」「固定 rot_part 反解 offset」
   「改用 `R·xyz`（未转置）」「offset 改用 uv_faces 的顶点」四种对照；
3. **直接调用 `mesh_binding` 算子本身**，用非对称旋转矩阵区分 `Rᵀ·x` 与 `R·x`
   —— 此前只从 CUDA 源码推断其语义，从未直接验证；
4. 离群统计：误差是否集中在少数面、是否与「重心落在三角形边上」相关。

### 待检验的假设（需 GPU）

`scripts/diagnose_binding.py` 会把 `offset` 拆开逐项检验：

| 编号 | 假设 |
|---|---|
| H1 | 绑定的三角面索引错位（`face_id` 整体偏移） |
| H2 | 重心分量顺序错（`u,v,w` 排列不对） |
| H3 | 取错三角形的顶点（用了 `uv_faces` 而非 `faces`） |
| H4 | 重心根本没参与（只用质心即可接近） |

脚本另有**反解**：从参照输出反推等效重心 `w`（最小二乘），直接看 `w` 与本项目 `bary` 的关系，
比逐个试假设收敛更快。

### 状态

**未解决**。P1 等价门因此未通过。这是 P1 的第一个真实缺陷，
也说明等价门的设计是有效的（它把问题定位到了单独一层，而不是给出一个笼统的"图不对"）。
