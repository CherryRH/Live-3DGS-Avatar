#!/usr/bin/env node
/**
 * 静态一致性检查：`index.html` 的元素 id ↔ JS 引用 ↔ DOM 测试桩。
 *
 * 为什么需要它：`querySelector` 拿到 null 之后再调 `.addEventListener`
 * 会抛 TypeError，**整个脚本在第一行就崩** —— 表现是"界面完全没反应"，
 * 与"代码没生效"极难区分。此前只能靠人工比对，容易漏。
 *
 * 三向核对：
 *   1. JS 里引用的 id 必须在 HTML 中定义
 *   2. HTML 中定义的 id 应在 JS 中被用到（否则是哑控件）
 *   3. DOM 测试桩的 id 清单必须与 HTML 完全一致（否则测试测的是另一套界面）
 */

import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const web = join(here, "..");

const html = readFileSync(join(web, "index.html"), "utf8");
const htmlIds = new Set([...html.matchAll(/id="([^"]+)"/g)].map((m) => m[1]));

// JS 里引用的 id
const jsIds = new Set();
for (const name of readdirSync(join(web, "js"))) {
  if (!name.endsWith(".js")) continue;
  const src = readFileSync(join(web, "js", name), "utf8");
  for (const m of src.matchAll(/querySelector\("#([^"]+)"\)/g)) jsIds.add(m[1]);
  for (const m of src.matchAll(/getElementById\("([^"]+)"\)/g)) jsIds.add(m[1]);
}

// DOM 测试桩的 id。
// 只要求它覆盖 **controls.js 用到的** id —— `#canvas` / `#stat-*` 等由
// main.js 使用，而 main.js 需要真实浏览器（它直接构造 FrameView），不入桩。
const controlsSrc = readFileSync(join(web, "js", "controls.js"), "utf8");
const controlsIds = new Set(
  [...controlsSrc.matchAll(/querySelector\("#([^"]+)"\)/g)].map((m) => m[1]));

const stub = readFileSync(join(here, "dom.mjs"), "utf8");
const stubBlock = stub.slice(stub.indexOf("function makeRoot()"));
const stubIds = new Set(
  [...stubBlock.matchAll(/"#([a-z0-9-]+)":/g)].map((m) => m[1]));

const problems = [];
for (const id of jsIds) {
  if (!htmlIds.has(id)) problems.push(`JS 引用了 HTML 中不存在的 id：#${id}`);
}
for (const id of htmlIds) {
  // `stats` 只是给样式用的容器，不参与逻辑
  if (!jsIds.has(id) && id !== "stats") {
    problems.push(`HTML 定义了但 JS 未使用（哑控件？）：#${id}`);
  }
}
for (const id of stubIds) {
  if (!htmlIds.has(id)) problems.push(`测试桩多出 HTML 里没有的 id：#${id}`);
}
for (const id of controlsIds) {
  if (!stubIds.has(id)) problems.push(`测试桩缺少 controls.js 用到的 id：#${id}`);
}
for (const id of stubIds) {
  if (!controlsIds.has(id)) problems.push(`测试桩多出 controls.js 未使用的 id：#${id}`);
}

if (problems.length === 0) {
  console.log(
    `\x1b[32m✓ id 一致性通过\x1b[0m（HTML ${htmlIds.size} 个；`
    + `JS 引用 ${jsIds.size} 个；controls 用到 ${controlsIds.size} 个，`
    + `测试桩 ${stubIds.size} 个）`);
  process.exit(0);
}
console.log(`\x1b[31m✗ id 一致性问题 ${problems.length} 处：\x1b[0m`);
for (const p of problems) console.log(`  ${p}`);
process.exit(1);
