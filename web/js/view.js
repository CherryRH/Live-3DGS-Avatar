/**
 * canvas 绘制。
 *
 * 用 `putImageData` 直接写像素，不走 `Image`/`drawImage`：
 * 后端发的是原始 RGB8，没有任何编码，`putImageData` 是零解析代价的路径。
 *
 * 两个细节：
 * - 复用同一个 `ImageData` 与中间 RGB→RGBA 缓冲，避免每帧分配（60 FPS 下很可观）
 * - `putImageData` **忽略 canvas 的 CSS 缩放与 transform**，所以绘制前要临时
 *   重置 transform；否则高 DPI 或 CSS 缩放时会画错位置
 */

export class FrameView {
  /**
   * @param {HTMLCanvasElement} canvas
   */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: false });
    if (!this.ctx) throw new Error("无法取得 2D 上下文");

    this.width = 0;
    this.height = 0;
    /** @type {ImageData|null} */
    this._image = null;
    /** @type {Uint8ClampedArray|null} RGB→RGBA 的复用缓冲 */
    this._rgba = null;
    this.framesDrawn = 0;
  }

  /**
   * 按新几何重建缓冲。分辨率或通道数变化时调用。
   *
   * @param {number} width
   * @param {number} height
   * @param {number} [channels]
   */
  resize(width, height, channels = 3) {
    if (width === this.width && height === this.height && channels === this.channels) return;
    this.width = width;
    this.height = height;
    this.channels = channels;
    this.canvas.width = width;
    this.canvas.height = height;
    this._image = this.ctx.createImageData(width, height);
    this._rgba = this._image.data;
  }

  /**
   * 绘制一帧。
   *
   * @param {{seq:number, pixels:Uint8ClampedArray}} frame
   * @throws {Error} 尺寸与已建立的缓冲不符时抛出（宁可报错也不画出错位的图）
   */
  draw(frame) {
    if (!this._image) throw new Error("尚未 resize，无法绘制");

    const px = frame.pixels;
    const need = this.width * this.height * this.channels;
    if (px.length < need) {
      throw new Error(`帧像素不足：需要 ${need}，实际 ${px.length}`);
    }

    const dst = this._rgba;
    if (this.channels === 4) {
      dst.set(px.subarray(0, need));
    } else {
      // RGB → RGBA，alpha 恒为 255（背景色已由后端烘进图像）
      for (let i = 0, j = 0; i < need; i += 3, j += 4) {
        dst[j] = px[i];
        dst[j + 1] = px[i + 1];
        dst[j + 2] = px[i + 2];
        dst[j + 3] = 255;
      }
    }

    this.ctx.putImageData(this._image, 0, 0);
    this.framesDrawn += 1;
    this.lastSeq = frame.seq;
  }

  /** 把当前画面导出为 PNG blob（用于"截取当前帧"）。 */
  async toBlob(type = "image/png") {
    return new Promise((resolve) => this.canvas.toBlob(resolve, type));
  }

  /** 清空为背景色（断开连接时用，避免留下最后一帧让人误以为还在动）。 */
  clear(color = "#000") {
    this.ctx.save();
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.fillStyle = color;
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.restore();
  }
}
