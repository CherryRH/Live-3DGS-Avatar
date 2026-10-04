# GUI 前后端协议（P2）

> **状态**：前端骨架已实现并可独立运行（含 mock 模式）；**后端待做**。
> 本文是先定契约，避免前后端各写各的。

设计目标：**简单好用**。本地单用户工具，不做鉴权、不做多路复用、不做视频编解码。

---

## 0. 为什么这些设计

| 决策 | 理由（有实测依据） |
|---|---|
| 帧走 **WebSocket 原始 RGB8**，不压缩 | 回环实测 **855 MiB/s**；512×512 一帧 768 KiB，在 90 FPS 下只需 68 MiB/s，**余量 13×**。而 PNG 编码 27 ms/帧、WebP 11 ms/帧都会成为新瓶颈（见 `docs/MIGRATION.md` B.1） |
| 每帧**只发 4 字节**帧号，不发几何 | 分辨率与背景色在一次会话内不变，用 JSON `config` 发一次即可 |
| **丢帧而不是排队** | 渲染永不阻塞；GUI 永远显示最新帧。浏览器端 `bufferedAmount` 超阈值也丢，避免延迟累积 |
| 渲染在**独立线程**且独占 CUDA | CUDA 上下文有线程亲和性；控制消息只改状态、不碰 GPU |
| 控制走 JSON 文本帧 | 频率低（人手操作），无需二进制协议 |

---

## 1. 会话流程

```
浏览器                          服务端
  │  GET /                     │  返回静态页面
  │  GET /api/state            │  当前模型/参数（用于初始化控件）
  ├────────────────────────────>│
  │  POST /api/model            │  切换 subject / work_name → 重载模型
  ├────────────────────────────>│
  │  WS  /ws                    │  建立双向通道
  ├────────────────────────────>│
  │  ← {"type":"config", ...}   │  分辨率、背景、可用参数范围
  │  ← {"type":"status", ...}   │  FPS、耗时分解、高斯数、丢帧数（周期推送）
  │  ← <binary 4 字节>           │  一帧 RGB8（紧跟 config.width*height*3 字节）
  │  ──{"type":"params", ...}──>│  改参数
  │  ──{"type":"stream", ...}──>│  暂停/继续
```

**握手顺序固定**：服务端连上后**先**发一条 `config`，再开始推帧。客户端在收到 `config` 之前不建立画布。

---

## 2. HTTP 接口

### `GET /api/state`

初始化控件用。返回当前生效的模型与参数（默认值来自 `configs/*.yaml`）。

```json
{
  "subject": "duda",
  "work_name": "test",
  "model": {
    "path": "models/duda/test/model.ply",
    "num_gaussians": 60353,
    "num_basis": 20,
    "num_basis_in": 129,
    "tex_size": 256
  },
  "render": {
    "background": [0.0, 0.0, 0.0],
    "scaling_modifier": 1.0,
    "scale": 1.0,
    "target_fps": 60
  },
  "dataset": {
    "num_frames": 254,
    "image_subdir": "images"
  }
}
```

### `POST /api/model`

```json
{"subject": "duda", "work_name": "test"}
```

成功返回新的 `/api/state` 同构对象；失败返回 `400` + `{"error": "..."}`。
**切换模型会重建渲染器**，期间帧流暂停。

### `GET /healthz`

返回 `{"ok": true, "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU"}`。

---

## 3. WebSocket

端点 `/ws`。

### 3.1 服务端 → 客户端

#### 文本帧：`config`（连接后第一条）

```json
{
  "type": "config",
  "width": 512,
  "height": 512,
  "background": [0.0, 0.0, 0.0],
  "num_frames": 254
}
```

#### 文本帧：`status`（周期推送，默认 2 Hz）

```json
{
  "type": "status",
  "seq": 1024,
  "fps": 167.7,
  "ms": {
    "frame": 5.96,
    "deform": 3.40,
    "rasterize": 2.10
  },
  "dropped": 12,
  "peak_memory_mib": 95.0,
  "paused": false
}
```

#### 二进制帧：一帧图像

| 偏移 | 长度 | 内容 |
|---|---|---|
| 0 | 4 | `seq`，**uint32 小端** |
| 4 | `width*height*3` | RGB8，行优先，无 padding |

- 几何（宽高）由 `config` 提供，**不在二进制帧里重复**
- 客户端若发现 `seq` 跳号，累加本地 `dropped` 计数（服务端丢帧时不通知）

### 3.2 客户端 → 服务端

#### `params`

只发改动的字段；服务端按字段合并。

```json
{
  "type": "params",
  "render": {"background": [1.0, 1.0, 1.0], "scale": 0.75},
  "drive": {"mode": "dataset", "frame": 42}
}
```

`drive.mode`：

| 值 | 含义 |
|---|---|
| `dataset` | 用数据集第 `frame` 帧的驱动参数（"审查"模式） |
| `random` | 以当前帧为基准加随机扰动（`amplitude`），探查训练分布外行为 |
| `manual` | 用 `drive.manual` 里显式给出的分区系数 |

#### `stream`

```json
{"type": "stream", "paused": true}
```

#### `capture`

```json
{"type": "capture", "name": "snapshot-001"}
```

服务端把当前帧以无损格式写到 `output/gui_captures/`，并回一条 `status` 附带路径。

---

## 4. 前端结构（已实现）

```
web/
├── index.html          # 单页；canvas + 控件面板
├── css/app.css
├── js/
│   ├── main.js         # 入口：装配 client + view + controls
│   ├── protocol.js     # 帧解析（ArrayBuffer → ImageData）
│   ├── client.js       # WebSocket 客户端（重连、丢帧统计、背压）
│   ├── view.js         # canvas 绘制
│   ├── controls.js     # 参数面板 ↔ 消息
│   └── mock.js         # mock 帧源（后端未就绪时用来跑通界面）
└── README.md
```

**无构建步骤**：原生 ES modules，由服务端静态托管即可。

### mock 模式

后端未就绪时用 `python -m http.server` 打开 `web/`，前端自动回退到 mock：
本地生成帧（含运动图案）+ 假 `status`，用于验证
**画布绘制、丢帧统计、参数控件、重连逻辑**。
URL 加 `?mock=1` 强制 mock，`?mock=0` 强制连真后端。

---

## 5. 与配置的关系

后端的默认值一律来自 `configs/*.yaml`（见 `docs/CONFIG.md`）：

| 配置项 | GUI 用途 |
|---|---|
| `paths.models_dir` / `subject` / `work_name` | 模型选择器初值 |
| `render.background` | 背景色控件初值 |
| `render.scaling_modifier` | 缩放控件初值 |
| `render.batch_size` | 单帧渲染时为 1 |

GUI 只会**临时覆盖**这些值，不回写 YAML。用户想持久化就改配置文件。

---

## 6. 未来可做的优化（已登记，暂不做）

| 项 | 预期 | 依据 |
|---|---|---|
| 接入 `mesh_binding` CUDA 内核 | deform 3.4 → ~0.3 ms，整帧 5.96 → ~2.7 ms（**约 370 FPS**） | 参照用自己的内核，单帧 2.36 ms。内核已 vendored，只是从"PyTorch 复现其数学"改成"直接调用"。见 `docs/MIGRATION.md` B.1 |
| 前端提速 | 帧率超过显示器刷新率后收益递减 | 167 FPS 已远超 60-144 Hz 面板 |
| 分辨率缩放 | `render.scale` 是显存与帧率最有效的旋钮 | 6 GB 卡上尤其明显 |
