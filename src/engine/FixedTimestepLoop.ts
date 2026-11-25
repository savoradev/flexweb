/**
 * FixedTimestepLoop
 * -----------------
 * Decouples rendering from physics. Physics always advances in `fixedDt`
 * increments (1/60 s, internally split into substeps by the solver) while the
 * renderer interpolates between the last two physics states using `alpha`.
 * Frame dt is clamped to 0.05 s so a background tab can't trigger a
 * spiral of death. Supports time scale (1.0x / 0.25x / 0.10x), pause and single-step.
 */
export interface LoopCallbacks {
  step(dt: number): void;
  render(alpha: number, frameDt: number): void;
}

export class FixedTimestepLoop {
  readonly fixedDt = 1 / 60;
  readonly maxFrameDt = 0.033;
  readonly maxStepsPerFrame = 2;

  timeScale = 1;
  paused = false;

  /** Rolling stats (ms). */
  physicsMs = 0;
  frameMs = 16.7;
  fps = 60;

  private acc = 0;
  private last = -1;
  private raf = 0;
  private pendingSteps = 0;
  private running = false;
  private fpsFrames = 0;
  private fpsTime = 0;

  constructor(private readonly cb: LoopCallbacks) {
    this.frame = this.frame.bind(this);
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.last = -1;
    this.raf = requestAnimationFrame(this.frame);
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.raf);
  }

  /** Advance exactly one physics step while paused. */
  requestStep(): void {
    this.pendingSteps++;
  }

  private frame(now: number): void {
    if (!this.running) return;
    this.raf = requestAnimationFrame(this.frame);

    if (this.last < 0) this.last = now;
    let dt = (now - this.last) / 1000;
    this.last = now;
    if (dt > this.maxFrameDt) dt = this.maxFrameDt;
    if (dt < 0) dt = 0;

    this.frameMs += (dt * 1000 - this.frameMs) * 0.1;
    this.fpsFrames++;
    this.fpsTime += dt;
    if (this.fpsTime >= 0.5) {
      this.fps = this.fpsFrames / this.fpsTime;
      this.fpsFrames = 0;
      this.fpsTime = 0;
    }

    const t0 = performance.now();
    if (!this.paused) {
      this.acc += dt * this.timeScale;
      let steps = 0;
      while (this.acc >= this.fixedDt && steps < this.maxStepsPerFrame) {
        this.cb.step(this.fixedDt);
        this.acc -= this.fixedDt;
        steps++;
      }
      if (steps === this.maxStepsPerFrame && this.acc > this.fixedDt) this.acc = 0;
    } else if (this.pendingSteps > 0) {
      this.pendingSteps--;
      this.cb.step(this.fixedDt);
      this.acc = 0;
    }
    const t1 = performance.now();
    this.physicsMs += (t1 - t0 - this.physicsMs) * 0.1;

    const alpha = this.paused ? 1 : Math.min(1, this.acc / this.fixedDt);
    this.cb.render(alpha, dt);
  }
}
