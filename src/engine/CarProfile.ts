/**
 * CarProfile
 * ----------
 * Shape description shared by the procedural sculpted car, the node-lattice
 * generator and the GLB auto-fitter. Car-local axes: +X = left, +Y = up,
 * +Z = forward. Y = 0 is the ground with the car at rest.
 *
 * Longitudinal curves are stored as lookup tables sampled on a uniform Z grid,
 * so analytic profiles (procedural sedan) and measured profiles (GLB) are
 * handled the same way.
 */
const SAMPLES = 96;

export class CarProfile {
  length = 4.65;
  width = 1.88;
  height = 1.38;
  frontAxleZ = 1.47;
  rearAxleZ = -1.40;
  trackHalf = 0.80;
  wheelRadius = 0.335;
  wheelWidth = 0.245;
  floorY = 0.2;
  cowlZ = 0.98;
  hoodFrontZ = 2.18;
  roofFrontZ = 0.15;
  roofRearZ = -0.80;
  deckZ = -1.58;
  trunkEndZ = -2.2;

  readonly top = new Float32Array(SAMPLES);
  readonly belt = new Float32Array(SAMPLES);
  readonly halfW = new Float32Array(SAMPLES);
  readonly roofHalfW = new Float32Array(SAMPLES);

  get halfL(): number { return this.length / 2; }

  private idx(z: number): number {
    const t = (z + this.halfL) / this.length;
    return Math.min(SAMPLES - 1.0001, Math.max(0, t * (SAMPLES - 1)));
  }
  private lerpTable(tab: Float32Array, z: number): number {
    const f = this.idx(z);
    const i = Math.floor(f);
    const t = f - i;
    return tab[i] + (tab[i + 1] - tab[i]) * t;
  }
  zAt(i: number): number { return -this.halfL + (i / (SAMPLES - 1)) * this.length; }

  topY(z: number): number { return this.lerpTable(this.top, z); }
  beltY(z: number): number { return this.lerpTable(this.belt, z); }
  /** Plan-view half width at shoulder height. */
  planHalfW(z: number): number { return this.lerpTable(this.halfW, z); }
  roofHalfWidth(z: number): number { return this.lerpTable(this.roofHalfW, z); }

  /** Body side surface half width at height y (tuck-under and shoulder). */
  sideHalfW(z: number, y: number): number {
    const hw = this.planHalfW(z);
    const belt = this.beltY(z);
    const s = this.height / 1.38;
    if (y < 0.36 * s) {
      const t = smoothstep(0.12 * s, 0.36 * s, y);
      return hw * (0.93 + 0.07 * t);
    }
    if (y <= belt) {
      const t = (y - 0.36 * s) / Math.max(0.05, belt - 0.36 * s);
      return hw * (1.0 - 0.018 * t * t);
    }
    return hw * 0.982;
  }

  isCabin(z: number): boolean { return z < this.cowlZ && z > this.deckZ; }

  static samples(): number { return SAMPLES; }
}

export function smoothstep(a: number, b: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

/** Monotone cubic (Fritsch-Carlson) interpolation through control points. */
export function monotoneCurve(pts: [number, number][]): (x: number) => number {
  const n = pts.length;
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const d: number[] = [];
  const m: number[] = new Array(n).fill(0);
  for (let i = 0; i < n - 1; i++) d.push((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]));
  m[0] = d[0];
  m[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) m[i] = d[i - 1] * d[i] <= 0 ? 0 : (d[i - 1] + d[i]) / 2;
  for (let i = 0; i < n - 1; i++) {
    if (d[i] === 0) { m[i] = 0; m[i + 1] = 0; continue; }
    const a = m[i] / d[i], b = m[i + 1] / d[i];
    const s = a * a + b * b;
    if (s > 9) { const t = 3 / Math.sqrt(s); m[i] = t * a * d[i]; m[i + 1] = t * b * d[i]; }
  }
  return (x: number) => {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n - 1]) return ys[n - 1];
    let i = 0;
    while (i < n - 2 && x > xs[i + 1]) i++;
    const h = xs[i + 1] - xs[i];
    const t = (x - xs[i]) / h;
    const t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h * m[i] + (-2 * t3 + 3 * t2) * ys[i + 1] + (t3 - t2) * h * m[i + 1];
  };
}

/** Mercedes-inspired 4-door sports sedan: 4.65 m x 1.88 m x 1.38 m. */
export function createSedanProfile(): CarProfile {
  const p = new CarProfile();
  const top = monotoneCurve([
    [-2.325, 0.70], [-2.29, 0.93], [-2.22, 1.005], [-2.0, 1.012], [-1.58, 1.02],
    [-1.25, 1.15], [-0.85, 1.315], [-0.45, 1.375], [-0.1, 1.38], [0.15, 1.355],
    [0.55, 1.17], [0.98, 0.935], [1.5, 0.865], [1.95, 0.795], [2.18, 0.735],
    [2.28, 0.68], [2.325, 0.58],
  ]);
  const belt = monotoneCurve([
    [-2.325, 0.70], [-2.25, 0.93], [-2.1, 0.975], [-1.58, 0.985], [-0.9, 0.955],
    [0.0, 0.935], [0.98, 0.905], [1.5, 0.835], [1.95, 0.77], [2.2, 0.71], [2.325, 0.58],
  ]);
  for (let i = 0; i < CarProfile.samples(); i++) {
    const z = p.zAt(i);
    p.top[i] = top(z);
    p.belt[i] = Math.min(belt(z), top(z) - 0.012);
    // plan view: rounded nose & tail
    let hw = 0.94;
    if (z > 1.72) hw -= 0.27 * Math.pow((z - 1.72) / 0.605, 2.3);
    if (z < -1.9) hw -= 0.2 * Math.pow((-1.9 - z) / 0.425, 2.3);
    p.halfW[i] = hw;
    // greenhouse half width at the roof
    const rz = Math.max(-1.0, Math.min(1.0, (z + 0.3) / 1.3));
    p.roofHalfW[i] = 0.70 - 0.06 * rz * rz;
  }
  return p;
}

/**
 * Low-slung mid-engine supercar profile (McLaren P1 silhouette:
 * ~4.55 m length, ~1.95 m width, ~1.20 m height, cab-forward teardrop cockpit,
 * low front splitter/frunk, and mid-rear engine deck).
 */
export function createSupercarProfile(): CarProfile {
  const p = new CarProfile();
  p.length = 4.55;
  p.width = 1.95;
  p.height = 1.20;
  p.frontAxleZ = 1.34;
  p.rearAxleZ = -1.34;
  p.trackHalf = 0.82;
  p.wheelRadius = 0.335;
  p.wheelWidth = 0.265;
  p.floorY = 0.19;
  p.cowlZ = 1.05;
  p.hoodFrontZ = 2.15;
  p.roofFrontZ = 0.40;
  p.roofRearZ = -0.58;
  p.deckZ = -1.28;
  p.trunkEndZ = -2.14;

  const top = monotoneCurve([
    [-2.275, 0.62], [-2.20, 0.80], [-1.92, 0.86], [-1.52, 0.89], [-1.28, 0.95],
    [-0.85, 1.11], [-0.42, 1.185], [0.02, 1.20], [0.38, 1.155], [0.76, 1.04],
    [1.05, 0.88], [1.48, 0.77], [1.88, 0.67], [2.15, 0.56], [2.275, 0.44],
  ]);
  const belt = monotoneCurve([
    [-2.275, 0.58], [-2.12, 0.76], [-1.60, 0.83], [-1.28, 0.85], [-0.50, 0.83],
    [0.30, 0.82], [1.05, 0.82], [1.50, 0.74], [1.90, 0.64], [2.15, 0.53], [2.275, 0.42],
  ]);
  for (let i = 0; i < CarProfile.samples(); i++) {
    const z = p.zAt(i);
    p.top[i] = top(z);
    p.belt[i] = Math.min(belt(z), top(z) - 0.015);
    let hw = 0.975;
    if (z > 1.55) hw -= 0.32 * Math.pow((z - 1.55) / 0.725, 2.1);
    if (z < -1.80) hw -= 0.22 * Math.pow((-1.80 - z) / 0.475, 2.1);
    // subtle Coke-bottle waist pinch in mid-door area
    hw -= 0.03 * Math.exp(-Math.pow((z - 0.1) / 0.55, 2));
    p.halfW[i] = hw;
    const rz = Math.max(-1.0, Math.min(1.0, (z - 0.05) / 1.15));
    p.roofHalfW[i] = 0.62 - 0.09 * rz * rz;
  }
  return p;
}

