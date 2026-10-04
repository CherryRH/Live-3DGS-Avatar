/**
 * 参数面板 ↔ 消息装配。
 *
 * 原则：控件只负责"读值"，消息的统一装配与节流放在这里，
 * 避免每个控件各写一遍 `send` 逻辑（那是重复与漂移的来源）。
 */

/** 面板上暴露的参数 → 默认值。与 docs/GUI_PROTOCOL.md §3.2 对应。 */
export const DEFAULTS = {
  background: [0, 0, 0],
  scalingModifier: 1.0,
  scale: 1.0,
  targetFps: 60,
  driveMode: "dataset",
  frame: 0,
  playing: false,
  amplitude: 0.1,
};

export class Controls {
  /**
   * @param {Document|HTMLElement} root
   * @param {{onParams:Function, onLoadModel:Function, onPause:Function,
   *          onCapture:Function, onReconnect:Function}} handlers
   */
  constructor(root, handlers) {
    this.root = root;
    this.handlers = handlers;
    this.el = {
      subject: root.querySelector("#f-subject"),
      workName: root.querySelector("#f-work-name"),
      loadModel: root.querySelector("#btn-load"),
      modelInfo: root.querySelector("#model-info"),
      bg: root.querySelector("#f-bg"),
      scaling: root.querySelector("#f-scaling"),
      scalingOut: root.querySelector("#o-scaling"),
      scale: root.querySelector("#f-scale"),
      scaleOut: root.querySelector("#o-scale"),
      fps: root.querySelector("#f-fps"),
      fpsOut: root.querySelector("#o-fps"),
      driveMode: root.querySelector("#f-drive-mode"),
      frame: root.querySelector("#f-frame"),
      frameOut: root.querySelector("#o-frame"),
      play: root.querySelector("#f-play"),
      amp: root.querySelector("#f-amp"),
      ampOut: root.querySelector("#o-amp"),
      pause: root.querySelector("#btn-pause"),
      capture: root.querySelector("#btn-capture"),
      reconnect: root.querySelector("#btn-reconnect"),
    };
    this.numFrames = null;
    this._playing = false;
    this._playTimer = null;
    this._wire();
    this._renderOutputs();
  }

  // ------------------------------------------------------------ 状态同步 --

  /** 用 `/api/state` 或 `config` 的返回值初始化控件。 */
  applyState(state) {
    if (!state) return;
    if (state.subject !== undefined) this.el.subject.value = state.subject;
    if (state.work_name !== undefined) this.el.workName.value = state.work_name;
    const model = state.model;
    if (model) {
      this.el.modelInfo.textContent =
        `${model.num_gaussians ?? "?"} 高斯 · K=${model.num_basis ?? "?"} · ` +
        `D=${model.num_basis_in ?? "?"} · tex=${model.tex_size ?? "?"}`;
    }
    const render = state.render;
    if (render) {
      if (Array.isArray(render.background)) {
        const key = render.background.join(",");
        if ([...this.el.bg.options].some((o) => o.value === key)) this.el.bg.value = key;
      }
      if (render.scaling_modifier !== undefined) this.el.scaling.value = render.scaling_modifier;
      if (render.scale !== undefined) this.el.scale.value = render.scale;
      if (render.target_fps !== undefined) this.el.fps.value = render.target_fps;
    }
    if (state.dataset?.num_frames) this.setFrameRange(state.dataset.num_frames);
    this._renderOutputs();
  }

  /** 数据集帧数确定后设置滑块范围（后端 config 里也会给一次）。 */
  setFrameRange(n) {
    this.numFrames = n;
    this.el.frame.max = String(Math.max(0, n - 1));
    if (Number(this.el.frame.value) > n - 1) this.el.frame.value = String(n - 1);
    this._renderOutputs();
  }

  setPaused(paused) {
    this.paused = paused;
    this.el.pause.textContent = paused ? "继续" : "暂停";
  }

  showModelError(text) {
    this.el.modelInfo.textContent = text;
  }

  // ---------------------------------------------------------------- 内部 --

  _wire() {
    const el = this.el;
    el.loadModel.addEventListener("click", () => {
      const subject = el.subject.value.trim();
      const workName = el.workName.value.trim();
      if (!subject || !workName) {
        this.showModelError("subject 与 work_name 都不能为空");
        return;
      }
      this.handlers.onLoadModel({ subject, work_name: workName });
    });

    // 参数类控件：值变即发（音量/滑块的节流交给 _emitParams）
    el.bg.addEventListener("change", () => this._emitParams());
    el.scaling.addEventListener("input", () => { this._renderOutputs(); this._emitParams(); });
    el.scale.addEventListener("input", () => { this._renderOutputs(); this._emitParams(); });
    el.fps.addEventListener("input", () => { this._renderOutputs(); this._emitParams(); });
    el.driveMode.addEventListener("change", () => this._emitParams());
    el.amp.addEventListener("input", () => { this._renderOutputs(); this._emitParams(); });

    el.frame.addEventListener("input", () => {
      this._renderOutputs();
      if (this._playing) return;   // 播放中由定时器推进，避免和手动拖动打架
      this._emitParams();
    });

    el.play.addEventListener("change", () => this._setPlaying(el.play.checked));

    el.pause.addEventListener("click", () => this.handlers.onPause(!this.paused));
    el.capture.addEventListener("click", () => this.handlers.onCapture());
    el.reconnect.addEventListener("click", () => this.handlers.onReconnect());
  }

  /** 循环播放：由前端推进帧号并周期发参数（后端不需要知道"播放"概念）。 */
  _setPlaying(on) {
    this._playing = on;
    if (this._playTimer !== null) {
      clearInterval(this._playTimer);
      this._playTimer = null;
    }
    if (!on) return;
    const step = () => {
      if (!this.numFrames) return;
      const next = (Number(this.el.frame.value) + 1) % this.numFrames;
      this.el.frame.value = String(next);
      this._renderOutputs();
      this._emitParams();
    };
    // 12 fps 的推进节奏：足够看出动作，又不至于把控制通道刷满
    this._playTimer = setInterval(step, 1000 / 12);
  }

  /** 组装并发送一条 `params` 消息（只发有变化的字段由后端合并，这里全发也无妨）。 */
  _emitParams() {
    const [r, g, b] = this.el.bg.value.split(",").map(Number);
    this.handlers.onParams({
      type: "params",
      render: {
        background: [r, g, b],
        scaling_modifier: Number(this.el.scaling.value),
        scale: Number(this.el.scale.value),
        target_fps: Number(this.el.fps.value),
      },
      drive: {
        mode: this.el.driveMode.value,
        frame: Number(this.el.frame.value),
        amplitude: Number(this.el.amp.value),
      },
    });
  }

  _renderOutputs() {
    const el = this.el;
    el.scalingOut.textContent = Number(el.scaling.value).toFixed(2);
    el.scaleOut.textContent = `${Math.round(Number(el.scale.value) * 100)}%`;
    el.fpsOut.textContent = el.fps.value;
    el.frameOut.textContent = el.frame.value;
    el.ampOut.textContent = Number(el.amp.value).toFixed(2);

    // 扰动幅度只在随机模式下有意义
    const random = el.driveMode.value === "random";
    el.amp.disabled = !random;
    el.frame.disabled = random;
  }

  destroy() {
    if (this._playTimer !== null) clearInterval(this._playTimer);
  }
}
