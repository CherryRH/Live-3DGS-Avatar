# 与 RGBAvatar 的行为差异登记

> **规则**：本项目对参照实现（`RGBAvatar/`，只读）的任何**有意**偏离都必须登记在本文件。
> 未登记的偏离视为 bug。
>
> 参照 commit：`903653b689a8b860df30f165e242ccd6ed3f5583`

**态度**：参照实现是研究原型，**不能作为正确性的最终判据**。
它的已知缺陷包括：`save_ply` 引用不存在的属性、Python 回退版与 CUDA 版语义不一致、
`linear_blending` 在设备不可用时静默返回全零。
因此本项目对关键算子同时使用**独立推导的不变量**与**与参照逐位比对**两种验证。

---

## A. 缺陷修复

| 编号 | 参照缺陷 | 本项目处置 | 回归测试 |
|---|---|---|---|
| **K1** | `BindingModel.clone()` 调用 5 个参数，`__init__` 只收 3 个 → 死代码 | 不移植；快照改用 `state_dict()` | — |
| **K2** | `save_ply()` 读取未定义的 `self.binding_face_id` → 任何模型都保存失败 | 绑定信息归属 `GaussianAvatar`（`register_buffer`） | `test_ply.py` |
| **K3** | `load_ply()` 从**配置**读基数量而非文件列数 → 配置与文件不一致时静默读错列 | 加载时与**文件列数**交叉校验（K / D / `tex_size²` 容量），不一致即报错；本项目写出的 PLY 另含 `comment gaussian_config` 自描述 | `test_load_rejects_basis_count_mismatch`、`test_load_rejects_tex_size_too_small_for_gaussians` |
| **K4** | `weight_module` 压进单列，仅有 `assert` 保护容量 | 改为显式 `ValueError`，并给出所需 `tex_size` 下界 | `test_save_rejects_weight_module_larger_than_gaussians` |

---

## B. 有意偏离

| 编号 | 参照行为 | 本项目行为 | 动机 |
|---|---|---|---|
| **H1** | 三份重复的渲染函数（含一份标注 `# legacy`） | 一套 `Rasterizer` 协议 + 推理/训练两个后端 | 消除语义漂移 |
| **H2** | 切空间/世界空间靠调用顺序隐含区分 | `GaussianSet.space` 显式标注 + 双向断言 | 跨空间误用不会报错，只会给出错误图像 |
| **H3** | 矩阵转置散落在各调用点并被复制三遍 | 收敛到 `camera_utils.to_kernel_matrix` 单点 | 分散的转置是"相机镜像/全黑"的温床 |
| **H4** | `linear_blending` 默认走 CUDA 内核（设备不可用时**静默返回全零**） | 默认走纯 PyTorch；CUDA 路径显式 opt-in 并加两道防护 | 不在主链路上留静默失败点 |
| **H5** | `Struct(**dict)` 传参，无类型检查 | `AvatarConfig` dataclass（PLY 还原网络结构所必需） | — |
| **H6** | TBN 的列/行主序靠约定 | `tbn.py` 明确「列为基向量」，并用单位三角面锁定 | — |

> P1 期间曾把**绑定位置项**误判为「有意偏离」，已作废 —— 本项目与参照在此处一致，
> 那只是本项目自己的 bug。见 D 节 #4/#6。

## C. 持续关注

| 编号 | 事项 | 现状 |
|---|---|---|
| O1 | UV 光栅化精度（参照的 `FIXME: precision issue across different devices`） | 已缓解：绑定随模型缓存；加载时校验 `N` 与 `face_id` 长度一致 |
| O2 | 退化三角形的 TBN 会产生 `nan`（`f = 1/det` 爆炸） | **未处理**。duda 模型 `min|det| = 2.6e-02`、条件数 max 75.5、暂无 `nan`；但 781/10032 个面的 UV 面积 < 1e-6，存在风险 |
| O3 | `f_rest_*` 恒为 0 的占位列 | 保留以兼容 3DGS 工具链；加载时若该列非零则显式拒绝（说明含高阶 SH） |
| O4 | 颜色烘焙进 SH DC，通话中光照变化无法适应 | 未来工作（见 `ARCHITECTURE.md` 附录 A） |
| O5 | `face_id` 越界/负值 | 已加校验：负索引会被 PyTorch 当作"从末尾数"**静默取错面**，故在 gather 前显式拦截 |

---

## D. 被推翻的错误推理（供追溯，勿重走）

| # | 曾作出的错误判断 | 推翻依据 |
|---|---|---|
| 1 | 「切空间 xyz 量级 1e-5 ⟹ 误差必来自 offset」 | 实测切空间 xyz 量级 **8e-02**，旋转项与 offset 同量级 |
| 2 | 「TBN 应当正交」（并把正交性当正确性判据） | `normalize(t)·normalize(b) = −cos(∠A')`，真实 UV 上中位数 0.276，非正交是常态 |
| 3 | 「两种乘法顺序都满足等变性，无法区分」 | 只有一种与参照一致；等价门的位置项差异（9.87e-02）是**本项目自身的 bug** |
| 4 | 「参照的 `gaussian_deform_batch` 用 `Rᵀ`，本项目应改成 `R`」 | 参照用的就是 `R @ xyz`。**本项目原本就对**，改成 `Rᵀ` 的"修复"才是 bug（渲染 PSNR 掉到 20 余 dB） |
| 5 | 「用刚性等变性可以判定该用 `R` 还是 `Rᵀ`」 | `mesh_binding` 是逐元素操作，`R` 与 `Rᵀ` 都数学自洽；**唯一有效判据是与参照逐位一致** |

### D.1 两处「布局」的辨析（本次踩坑的根因）

`mesh_binding` 是逐元素算子，`R` 与 `Rᵀ` 都自洽，因此**无法自证**。
而「基向量排成行还是列」在不同来源里**不同**：

| 来源 | 布局 |
|---|---|
| CUDA `cuda_utils/face_tbn.cu` | **行**为基（`TBNs[idx] = transpose(mat3(t, b, n))`） |
| 参照 Python `utils.compute_face_tbn` | **列**为基 |
| 本项目 `core/deform/tbn.py` | **列**为基（与参照 Python 侧一致） |

**踩坑经过**：`mesh_binding.cu` 内部有 `transpose(binding_rotation) * gs_xyz`，
据此外推「参照用 `Rᵀ`」，把本项目从 `R` 改成 `Rᵀ`——**这是错的**。
那次 `transpose` 是针对**传入布局**的修正，不能脱离"传入的是什么"来判断。
拿 CUDA 侧 `face_tbn` 的布局去解释 Python 侧 `gaussian_deform_batch` 的用法，两者混用导致误判。

**三条互相印证的证据**（本项目 = 参照，位置项 `R @ xyz`）：

| # | 证据 |
|---|---|
| 1 | 等价门在位置项曾报差 **9.87e-02**（当时本项目用 `Rᵀ`） |
| 2 | 改回 `R` 后，位置差降到 **2.98e-08**、渲染 **PSNR 133 dB** |
| 3 | 254 帧渲染与参照 **PSNR 中位 101 dB、`max\|Δ\| = 1/255`** —— 逐位一致 |

**回归测试**：`test_bind_matches_reference_translation`（与参照一致）、
`test_mesh_binder_uses_R_not_transpose`（直接打在真实代码路径 `MeshBinder.bind` 上，
防止"改了实现但测试仍在跑自己的公式"这类假验证）。
