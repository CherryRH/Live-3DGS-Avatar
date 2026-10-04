# web/ —— 实时预览界面

P2 的图形化前端。**无构建步骤**：原生 ES modules，直接由后端静态托管。

> **状态**：前端骨架完成，可独立运行（mock 模式）；**后端待做**。
> 协议见 [`docs/GUI_PROTOCOL.md`](../docs/GUI_PROTOCOL.md)。

## 为什么不用 Vite / TypeScript / 框架

界面就是「一个 canvas + 一组控件」。引入构建链的代价（`node_modules` 100–200 MB、
构建步骤、版本漂移）换不来对应的收益。这个项目的既有取向是**依赖最少、每层显式**。

以后界面变大（多标签、复杂状态）再引入 Vite 也不迟 —— 原生 ESM 可以平滑迁移。

## 目录

```
web/
├── index.html          单页：顶栏 + canvas 舞台 + 参数面板
├── css/app.css
├── js/
│   ├── main.js         入口：装配数据源 / view / controls
│   ├── protocol.js     帧与消息解析（含长度校验、版本校验）
│   ├── client.js       WebSocket 客户端（重连、丢帧统计、背压）
│   ├── view.js         canvas 绘制（RGB→RGBA，复用缓冲）
│   ├── controls.js     参数面板 ↔ 消息装配
│   └── mock.js         mock 帧源（复刻真实协议）
└── tests/
    ├── all.mjs         汇总入口
    ├── run.mjs         protocol.js / mock.js
    └── dom.mjs         view.js / controls.js（用最小桩）
```

## 运行

### 只看界面（后端未就绪）

```bash
cd web && python3 -m http.server 8912
# 浏览器打开 http://127.0.0.1:8912
```

前端会先尝试连后端；连续两次失败后**自动回退到 mock**，用本地生成的图案
验证画布、丢帧统计、参数控件与重连逻辑。

强制指定：`?mock=1` 用 mock，`?mock=0` 只用后端。

> ⚠️ 用 `file://` 直接打开 `index.html` 时**不能**连后端（浏览器不允许），
> 会自动进入 mock 模式。

### 接真实后端

后端就绪后，由后端在同一端口同时提供静态文件与 `/api/*`、`/ws`
（见 `docs/GUI_PROTOCOL.md`），直接访问后端地址即可。

## 测试

```bash
node web/tests/all.mjs
```

覆盖 `protocol.js`（帧布局、长度校验、版本校验）与 `mock.js`（协议一致性）、
以及 `view.js` / `controls.js`（用最小 DOM 桩）。

**`client.js` 的真实 WebSocket 行为只能在浏览器里验证** ——
重连、背压、丢帧统计需要真实网络栈。

## 设计注记

几个踩过的坑，改代码时请保留：

| 位置 | 注意 |
|---|---|
| `mock.js::start()` | `config` **必须异步派发**。同步派发会让 `start()` 之后才挂监听的地方永久漏掉它（画布不 resize，界面卡在"等待连接…"） |
| `protocol.js::parseFrame` | 长度不符**必须抛错**，不能凑合画 —— 错位的帧看起来像"模型坏了"，极难排查 |
| `view.js::draw` | 复用 `ImageData` 与 RGB→RGBA 缓冲，避免每帧分配；`putImageData` 忽略 CSS 变换 |
| `client.js::_pump` | **只派发最新一帧**，`bufferedAmount` 超限就丢 —— 背面压，避免延迟累积 |
| `controls.js` | 控件只"读值"，消息统一在 `_emitParams` 装配，避免每个控件各写一遍 |

## 未做（按计划）

- 后端（FastAPI + uvicorn）
- 驱动参数的**分区直接控制**（下颌/眼睑/眼球/表情）——需要 FLAME 语义切分
- GT 叠加对照
- 参数快照保存/加载
