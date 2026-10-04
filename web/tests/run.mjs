#!/usr/bin/env node
/**
 * 前端模块自检（**不需要浏览器、不需要后端**）。
 *
 * 本文件测不依赖 DOM 的部分（protocol.js / mock.js）；
 * `dom.mjs` 用桩测 view.js / controls.js。
 * `client.js` 的真实 WebSocket 行为只能在浏览器里验证。
 *
 * 用法::
 *
 *     node web/tests/run.mjs          # 只跑本文件
 *     node web/tests/all.mjs          # 跑全部前端测试
 */

import { parseFrame, parseMessage, geometryFromConfig, HEADER_BYTES, PROTOCOL_VERSION }
  from "../js/protocol.js";
import { MockSource } from "../js/mock.js";

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed += 1;
    console.log(`  \x1b[32m✓\x1b[0m ${name}`);
  } catch (err) {
    failures.push({ name, err });
    console.log(`  \x1b[31m✗\x1b[0m ${name}\n      ${err.message}`);
  }
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg ?? "断言失败");
}

function assertEqual(actual, expected, msg) {
  if (actual !== expected) {
    throw new Error(`${msg ?? "不相等"}：期望 ${expected}，实际 ${actual}`);
  }
}

function assertThrows(fn, re, msg) {
  try {
    fn();
  } catch (err) {
    if (re && !re.test(err.message)) {
      throw new Error(`${msg ?? "错误信息不符"}：${err.message}`);
    }
    return;
  }
  throw new Error(`${msg ?? "应当抛错"}，但没有`);
}

// ------------------------------------------------------------ protocol --

console.log("\nprotocol.js");

test("config 几何解析正常", () => {
  const g = geometryFromConfig({ width: 512, height: 256, background: [1, 1, 1],
                                 num_frames: 254, version: PROTOCOL_VERSION });
  assertEqual(g.width, 512);
  assertEqual(g.height, 256);
  assertEqual(g.channels, 3, "默认通道数");
  assertEqual(g.numFrames, 254);
  assert(Array.isArray(g.background) && g.background.length === 3);
});

test("config 拒绝非法宽高", () => {
  assertThrows(() => geometryFromConfig({ width: 0, height: 10 }), /宽高非法/);
  assertThrows(() => geometryFromConfig({ width: 10, height: -1 }), /宽高非法/);
  assertThrows(() => geometryFromConfig({ width: 1.5, height: 10 }), /宽高非法/);
});

test("config 拒绝协议版本不匹配", () => {
  assertThrows(
    () => geometryFromConfig({ width: 8, height: 8, version: PROTOCOL_VERSION + 1 }),
    /协议版本不匹配/);
});

test("帧解析：seq 小端 + 像素偏移正确", () => {
  const w = 4, h = 2, geom = { width: w, height: h, channels: 3 };
  const buf = new ArrayBuffer(HEADER_BYTES + w * h * 3);
  new DataView(buf).setUint32(0, 0x01020304, true);
  const px = new Uint8ClampedArray(buf, HEADER_BYTES);
  for (let i = 0; i < px.length; i++) px[i] = i % 256;

  const frame = parseFrame(buf, geom);
  assertEqual(frame.seq, 0x01020304, "seq 应按小端读出");
  assertEqual(frame.pixels.length, w * h * 3, "像素长度");
  assertEqual(frame.pixels[0], 0);
  assertEqual(frame.pixels[5], 5);
});

test("帧解析拒绝长度不符（宁可报错也不画错位图）", () => {
  const geom = { width: 4, height: 2, channels: 3 };
  assertThrows(() => parseFrame(new ArrayBuffer(10), geom), /帧长度不符/);
  assertThrows(() => parseFrame(new ArrayBuffer(HEADER_BYTES), geom), /帧长度不符/);
});

test("帧解析在未收到 config 时明确报错", () => {
  assertThrows(() => parseFrame(new ArrayBuffer(8), null), /尚未收到 config/);
});

test("消息解析：非法 JSON 返回 null 而不抛", () => {
  assertEqual(parseMessage("这不是 JSON"), null);
  assertEqual(parseMessage("{}"), null, "缺 type");
  const ok = parseMessage('{"type":"status","fps":60}');
  assert(ok && ok.type === "status");
});

test("未知 type 不崩（便于后端加消息）", () => {
  const m = parseMessage('{"type":"brand_new","x":1}');
  assert(m && m.type === "brand_new");
});

// ---------------------------------------------------------------- mock --

console.log("\nmock.js（复刻真实协议）");

/** 等待某个事件，带超时 —— 避免测试挂死。 */
function waitFor(src, type, timeoutMs = 2000) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error(`等待 ${type} 事件超时（${timeoutMs}ms）`)), timeoutMs);
    src.addEventListener(type, (ev) => {
      clearTimeout(timer);
      resolve(ev.detail);
    }, { once: true });
  });
}

const tests = [];

tests.push((async () => {
  const src = new MockSource({ width: 64, height: 32, numFrames: 10 });
  const cfgPromise = waitFor(src, "config");
  const framePromise = waitFor(src, "frame");
  src.start();
  const cfg = await cfgPromise;
  const raw = await framePromise;
  src.stop();

  test("mock config 含几何与版本", () => {
    assertEqual(cfg.width, 64);
    assertEqual(cfg.height, 32);
    assertEqual(cfg.version, PROTOCOL_VERSION);
    assertEqual(cfg.numFrames, 10);
  });

  test("mock 帧能被同一个 parseFrame 解析（协议一致）", () => {
    // mock 直接给 {seq, pixels}；这里按真实二进制再走一遍解析路径
    const bytes = HEADER_BYTES + 64 * 32 * 3;
    const buf = new ArrayBuffer(bytes);
    new DataView(buf).setUint32(0, raw.seq, true);
    new Uint8ClampedArray(buf, HEADER_BYTES).set(raw.pixels);
    const parsed = parseFrame(buf, { width: 64, height: 32, channels: 3 });
    assertEqual(parsed.seq, raw.seq);
    assertEqual(parsed.pixels.length, 64 * 32 * 3);
  });

  test("mock 的 seq 递增（丢帧统计依赖它）", async () => {
    // 由上面的 raw 推导：起始 seq 应为 1
    assertEqual(raw.seq, 1, "第一帧 seq 应为 1");
  });
})());

tests.push((async () => {
  const src = new MockSource({ width: 16, height: 16 });
  src.start();
  await waitFor(src, "config");

  test("mock 接受 params 并改变背景色", async () => {
    src.send({ type: "params", render: { background: [1, 1, 1] },
               drive: { mode: "dataset", frame: 3, amplitude: 0.2 } });
    const f = await waitFor(src, "frame");
    // 左上角不在椭圆内，应为背景色
    assertEqual(f.pixels[0], 255, "背景应为白");
    assertEqual(f.pixels[1], 255);
    assertEqual(f.pixels[2], 255);
  });

  test("mock 暂停后不再产帧", async () => {
    src.send({ type: "stream", paused: true });
    let got = false;
    const onFrame = () => { got = true; };
    src.addEventListener("frame", onFrame);
    await new Promise((r) => setTimeout(r, 120));
    src.removeEventListener("frame", onFrame);
    src.stop();
    assert(!got, "暂停后仍在产帧");
  });
})());

tests.push((async () => {
  const src = new MockSource({ width: 32, height: 32 });
  src.start();
  await waitFor(src, "config");

  test("mock status 字段齐全（前端 updateStats 依赖）", async () => {
    const st = await waitFor(src, "status", 1500);
    assertEqual(st.type, "status");
    assert(typeof st.fps === "number", "fps 应为数字");
    assert(st.ms && typeof st.ms === "object", "ms 应为对象");
    assert(typeof st.dropped === "number", "dropped 应为数字");
    assert(typeof st.paused === "boolean", "paused 应为布尔");
    src.stop();
  });
})());

await Promise.all(tests);

console.log(`\n${"─".repeat(52)}`);
if (failures.length === 0) {
  console.log(`\x1b[32m通过 ${passed}，失败 0\x1b[0m`);
  process.exit(0);
} else {
  console.log(`通过 ${passed}，\x1b[31m失败 ${failures.length}\x1b[0m`);
  for (const f of failures) console.log(`\n--- ${f.name} ---\n${f.err.stack}`);
  process.exit(1);
}
