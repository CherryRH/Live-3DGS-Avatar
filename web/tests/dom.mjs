/**
 * 前端 DOM 相关模块自检（**不需要浏览器**）：用最小桩验证
 * `view.js` 的 RGB→RGBA 转换与 `controls.js` 的参数装配。
 *
 * canvas 与 WebSocket 的真实行为仍只能在浏览器里验证；
 * 这里覆盖的是**我们自己写的逻辑**，而不是浏览器的实现。
 *
 * 用法::
 *
 *     node web/tests/dom.mjs
 */
import { FrameView } from "../js/view.js";
import { Controls } from "../js/controls.js";

let pass = 0; const fails = [];
const t = (name, fn) => { try { fn(); pass++; console.log(`  \x1b[32m✓\x1b[0m ${name}`); }
  catch (e) { fails.push([name, e]); console.log(`  \x1b[31m✗\x1b[0m ${name}\n      ${e.message}`); } };
const eq = (a, b, m) => { if (a !== b) throw new Error(`${m ?? "不相等"}: 期望 ${b} 实际 ${a}`); };
const ok = (c, m) => { if (!c) throw new Error(m ?? "断言失败"); };

// ---- FrameView 桩：只要 createImageData / putImageData ----
function makeCanvas() {
  return {
    width: 0, height: 0,
    getContext: () => ({
      createImageData: (w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
      putImageData: (img) => { canvas0.lastPainted = img; },
      save(){}, restore(){}, setTransform(){}, fillRect(){}, fillStyle: "",
    }),
    toBlob: (cb) => cb(null),
  };
}
let canvas0;
console.log("\nview.js");
t("RGB→RGBA 转换正确且 alpha=255", () => {
  canvas0 = makeCanvas();
  const v = new FrameView(canvas0);
  v.resize(2, 1, 3);
  v.draw({ seq: 7, pixels: new Uint8ClampedArray([10, 20, 30, 40, 50, 60]) });
  const px = canvas0.lastPainted.data;
  eq(px[0], 10); eq(px[1], 20); eq(px[2], 30); eq(px[3], 255, "alpha");
  eq(px[4], 40); eq(px[5], 50); eq(px[6], 60); eq(px[7], 255, "alpha[1]");
  eq(v.lastSeq, 7);
  eq(v.framesDrawn, 1);
});
t("resize 会设置 canvas 尺寸", () => {
  canvas0 = makeCanvas();
  const v = new FrameView(canvas0);
  v.resize(8, 4, 3);
  eq(canvas0.width, 8); eq(canvas0.height, 4);
});
t("像素不足时明确报错（不画错位图）", () => {
  canvas0 = makeCanvas();
  const v = new FrameView(canvas0);
  v.resize(4, 4, 3);
  let threw = false;
  try { v.draw({ seq: 1, pixels: new Uint8ClampedArray(5) }); } catch (e) { threw = /像素不足/.test(e.message); }
  ok(threw, "应抛「像素不足」");
});
t("未 resize 就 draw 应报错", () => {
  canvas0 = makeCanvas();
  const v = new FrameView(canvas0);
  let threw = false;
  try { v.draw({ seq: 1, pixels: new Uint8ClampedArray(12) }); } catch (e) { threw = /尚未 resize/.test(e.message); }
  ok(threw);
});

// ---- Controls 桩 ----
function el(tag = "input", attrs = {}) {
  const listeners = {};
  return Object.assign({
    tagName: tag.toUpperCase(), value: "", disabled: false, max: "", textContent: "",
    checked: false, options: [],
    addEventListener(ev, fn) { (listeners[ev] ??= []).push(fn); },
    _fire(ev) { (listeners[ev] ?? []).forEach((f) => f()); },
  }, attrs);
}
function makeRoot() {
  const map = {
    "#f-subject": el("input", { value: "duda" }),
    "#f-work-name": el("input", { value: "test" }),
    "#btn-load": el("button"), "#model-info": el("p"),
    "#f-bg": el("select", { value: "0,0,0" }),
    "#f-scaling": el("input", { value: "1" }), "#o-scaling": el("output"),
    "#f-scale": el("input", { value: "1" }), "#o-scale": el("output"),
    "#f-fps": el("input", { value: "60" }), "#o-fps": el("output"),
    "#f-drive-mode": el("select", { value: "dataset" }),
    "#f-frame": el("input", { value: "0" }), "#o-frame": el("output"),
    "#f-play": el("input", { checked: false }),
    "#f-amp": el("input", { value: "0.1" }), "#o-amp": el("output"),
    "#btn-pause": el("button"), "#btn-capture": el("button"), "#btn-reconnect": el("button"),
  };
  return { querySelector: (s) => map[s] ?? null, _map: map };
}

console.log("\ncontrols.js");
t("_emitParams 装配出协议要求的结构", () => {
  const root = makeRoot(); let sent = null;
  const c = new Controls(root, { onParams: (m) => { sent = m; }, onLoadModel(){}, onPause(){}, onCapture(){}, onReconnect(){} });
  root._map["#f-bg"].value = "1,1,1";
  root._map["#f-scaling"].value = "1.25";
  root._map["#f-scale"].value = "0.5";
  root._map["#f-fps"].value = "90";
  root._map["#f-frame"].value = "42";
  root._map["#f-amp"].value = "0.3";
  c._emitParams();
  eq(sent.type, "params");
  ok(Array.isArray(sent.render.background) && sent.render.background[0] === 1, "background");
  eq(sent.render.scaling_modifier, 1.25);
  eq(sent.render.scale, 0.5);
  eq(sent.render.target_fps, 90);
  eq(sent.drive.mode, "dataset");
  eq(sent.drive.frame, 42);
  eq(sent.drive.amplitude, 0.3);
});
t("随机模式下禁用帧滑块、启用扰动幅度", () => {
  const root = makeRoot();
  const c = new Controls(root, { onParams(){}, onLoadModel(){}, onPause(){}, onCapture(){}, onReconnect(){} });
  root._map["#f-drive-mode"].value = "random";
  c._renderOutputs();
  ok(root._map["#f-frame"].disabled, "帧滑块应禁用");
  ok(!root._map["#f-amp"].disabled, "扰动幅度应启用");
});
t("applyState 写入模型信息与帧范围", () => {
  const root = makeRoot();
  const c = new Controls(root, { onParams(){}, onLoadModel(){}, onPause(){}, onCapture(){}, onReconnect(){} });
  c.applyState({ subject: "alice", work_name: "run1",
    model: { num_gaussians: 60000, num_basis: 20, num_basis_in: 129, tex_size: 256 },
    dataset: { num_frames: 100 } });
  eq(root._map["#f-subject"].value, "alice");
  eq(root._map["#f-work-name"].value, "run1");
  ok(/60000/.test(root._map["#model-info"].textContent), "高斯数应显示");
  eq(root._map["#f-frame"].max, "99", "帧范围上限");
});
t("subject 为空时拒绝加载", () => {
  const root = makeRoot(); let called = false;
  const c = new Controls(root, { onParams(){}, onLoadModel(){ called = true; }, onPause(){}, onCapture(){}, onReconnect(){} });
  root._map["#f-subject"].value = "  ";
  root._map["#btn-load"]._fire("click");
  ok(!called, "不应触发加载");
  ok(/不能为空/.test(root._map["#model-info"].textContent));
});

console.log(`\n${"─".repeat(50)}`);
if (fails.length === 0) {
  console.log(`\x1b[32m通过 ${pass}，失败 0\x1b[0m`);
} else {
  console.log(`通过 ${pass}，\x1b[31m失败 ${fails.length}\x1b[0m`);
  for (const [n, e] of fails) console.log(`\n--- ${n} ---\n${e.stack}`);
  process.exit(1);
}
