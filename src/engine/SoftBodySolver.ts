import { TypedArrayPool } from './MemoryGuard';

/**
 * SoftBodySolver (JBeam High-Frequency Float32 SoA Architecture)
 * --------------------------------------------------------------
 * High-frequency (10–12 substeps/frame) node-and-beam soft-body physics solver
 * modeled after BeamNG.drive's JBeam architecture:
 *
 *  - Pure contiguous Float32Array Structure-of-Arrays (SoA) layout for L1/L2 CPU
 *    cache throughput and strictly zero heap allocations inside `step()`.
 *  - Symplectic Euler + stabilized XPBD/Verlet position projection with per-substep
 *    velocity and impulse clamping (`maxSubstepSpeed = 96 m/s`), eliminating numerical
 *    explosions or "jelly" oscillations even on 250–320 km/h impacts.
 *  - True 3-state JBeam beams (`BEAM_STATE_ELASTIC`, `BEAM_STATE_PLASTIC`,
 *    `BEAM_STATE_BROKEN`) parameterized by `beamSpring`, `beamDamp`, `beamDeform`,
 *    and `beamStrength` with accordion crumple-zone buckling.
 *  - Surface Collision Triangles (`coltris`) preventing thin poles, barrier corners,
 *    and detached parts from clipping between lattice nodes.
 *  - O(1) Spatial Hash Grid broadphase support + progressive crusher rollers & pendulums.
 */

export const enum Zone {
  Cabin = 0,
  Longeron = 1,
  Crumple = 2,
  Fender = 3,
  Hood = 4,
  Trunk = 5,
  Bumper = 6,
  Hinge = 7,
  Latch = 8,
  Suspension = 9,
  EngineSubframe = 10,
}
export const ZONE_COUNT = 11;
export const ZONE_NAMES = [
  'Cabin',
  'Longerons',
  'Crumple',
  'Fenders',
  'Hood',
  'Trunk',
  'Bumpers',
  'Hinges',
  'Latches',
  'Suspension',
  'EngineSubframe',
];

export const BEAM_STATE_ELASTIC = 0;
export const BEAM_STATE_PLASTIC = 1;
export const BEAM_STATE_BROKEN = 2;

export interface ZoneParams {
  spring: number;           // beamSpring (N/m)
  damp: number;             // beamDamp (N*s/m)
  plasticThreshold: number; // elastic strain before permanent plastic deformation (beamDeform = k * threshold * L0)
  hardening: number;        // work-hardening growth per unit plastic strain
  breakStrain: number;      // strain threshold for clean structural break (0 = unbreakable)
}

/**
 * Realistic 4-tier JBeam stiffness hierarchy:
 *  1. Survival Cell (Cabin) & Rigid Engine Subframe >>
 *  2. Longerons & Triangulated Suspension >>
 *  3. Front/Rear Accordion Crumple Zones & Bumpers >>
 *  4. Outer Sheet-Metal (Fenders, Doors, Hood) + Hinges & Latches.
 */
export const DEFAULT_ZONES: ZoneParams[] = [
  /* 0: Cabin          */ { spring: 5.2e6, damp: 9800, plasticThreshold: 0.019, hardening: 0.50, breakStrain: 0 },
  /* 1: Longeron       */ { spring: 3.5e6, damp: 8600, plasticThreshold: 0.015, hardening: 0.24, breakStrain: 0 },
  /* 2: Crumple        */ { spring: 1.75e6, damp: 7800, plasticThreshold: 0.0075, hardening: 0.09, breakStrain: 0 },
  /* 3: Fender         */ { spring: 4.8e5, damp: 2800, plasticThreshold: 0.0048, hardening: 0.07, breakStrain: 0 },
  /* 4: Hood           */ { spring: 5.4e5, damp: 5200, plasticThreshold: 0.0058, hardening: 0.12, breakStrain: 0 },
  /* 5: Trunk          */ { spring: 1.45e6, damp: 6500, plasticThreshold: 0.020, hardening: 0.22, breakStrain: 0 },
  /* 6: Bumper         */ { spring: 1.65e6, damp: 10800, plasticThreshold: 0.030, hardening: 0.42, breakStrain: 0 },
  /* 7: Hinge          */ { spring: 1.9e6, damp: 2800, plasticThreshold: 0.048, hardening: 0.0, breakStrain: 0.42 },
  /* 8: Latch          */ { spring: 6.0e5, damp: 1650, plasticThreshold: 0.025, hardening: 0.0, breakStrain: 0.14 },
  /* 9: Suspension     */ { spring: 3.4e6, damp: 3200, plasticThreshold: 0.032, hardening: 0.65, breakStrain: 0.28 },
  /* 10: EngineSubframe*/ { spring: 6.4e6, damp: 11500, plasticThreshold: 0.065, hardening: 0.90, breakStrain: 0 },
];

export const BEAM_BROKEN = 1;
export const BEAM_COLLIDE = 2;
export const BEAM_BREAKABLE = 4;
export const BEAM_HOOD_LONG = 8;
export const BEAM_UNBREAKABLE = 16;

export interface GroundSample { h: number; nx: number; ny: number; nz: number; mu: number; }

export const BOX_STRIDE = 13;
export const CYL_STRIDE = 6;
/** Roller stride: cx, cy, cz, halfW, radius, surfaceVz, minGapY */
export const ROLLER_STRIDE = 7;

/** World collision interface consumed by SoftBodySolver (implemented by CrashArena). */
export interface CollisionWorld {
  groundAt(x: number, z: number, out: GroundSample): void;
  readonly boxes: Float32Array | Float64Array;
  readonly boxCount: number;
  readonly cyls: Float32Array | Float64Array;
  readonly cylCount: number;
  readonly rollers?: Float32Array;
  readonly rollerCount?: number;
  readonly pressBoxIndex?: number;
  isPressBox?(boxIdx: number): boolean;
  /** Optional kinematic velocity query (vx, vy, vz) for pendulum hammers & moving obstacles. */
  getBoxVelocity?(boxIdx: number, outVel: Float32Array): boolean;
  /** O(1) Spatial Hash Grid broadphase hooks for 2 km × 2 km maps. */
  queryBoxes?(minX: number, minZ: number, maxX: number, maxZ: number, out: Int32Array, maxOut: number): number;
  queryCyls?(minX: number, minZ: number, maxX: number, maxZ: number, out: Int32Array, maxOut: number): number;
}

export interface SubstepForceProvider {
  /** Write external forces into solver.fx/fy/fz for this substep. */
  applySubstepForces(h: number): void;
}

export interface SolverBuildNode {
  x: number;
  y: number;
  z: number;
  mass: number;
  radius: number;
  group: number;
  friction: number;
}

export interface SolverBuildBeam {
  a: number;
  b: number;
  zone: Zone;
  flags: number;
  springScale?: number;
  dampScale?: number;
  plasticScale?: number;
  strengthScale?: number;
}

export interface SolverBuildColtri {
  a: number;
  b: number;
  c: number;
}

export class SoftBodySolver {
  readonly nodeCount: number;
  readonly beamCount: number;
  readonly coltriCount: number;
  readonly pool = new TypedArrayPool();

  /** High-frequency JBeam substep count (10 on 4 GB RAM tablets, 12 on desktop). */
  substeps = 12;
  iterations = 1;
  gravity = -9.81;
  restitution = 0.05;
  minRatio = 0.46;
  minRatioBumper = 0.80;
  minRatioEngine = 0.88;
  maxRatio = 1.32;
  airDamping = 0.02;
  /** Per-substep velocity ceiling (~345 km/h) preventing numerical blowup on extreme crashes. */
  maxSubstepSpeed = 96.0;
  zones: ZoneParams[];

  // ---- Contiguous Float32Array Node SoA ----
  readonly px: Float32Array; readonly py: Float32Array; readonly pz: Float32Array;
  readonly qx: Float32Array; readonly qy: Float32Array; readonly qz: Float32Array; // substep start
  readonly sx: Float32Array; readonly sy: Float32Array; readonly sz: Float32Array; // frame start (render interp)
  readonly vx: Float32Array; readonly vy: Float32Array; readonly vz: Float32Array;
  readonly fx: Float32Array; readonly fy: Float32Array; readonly fz: Float32Array;
  readonly mass: Float32Array; readonly invMass: Float32Array;
  readonly radius: Float32Array; readonly friction: Float32Array;
  readonly group: Uint8Array;

  // ---- Contacts (per node, per substep) ----
  readonly cnx: Float32Array; readonly cny: Float32Array; readonly cnz: Float32Array;
  readonly cvn: Float32Array; readonly cHit: Uint8Array; readonly cObstacle: Uint8Array;

  // ---- Contiguous Float32Array JBeam Beam SoA ----
  readonly beamA: Uint16Array; readonly beamB: Uint16Array;
  readonly restLength: Float32Array; readonly restLength0: Float32Array;
  /** Aliases kept for zero-copy compatibility with vehicle telemetry & snapshots. */
  readonly rest: Float32Array; readonly rest0: Float32Array;
  readonly beamSpring: Float32Array;
  readonly beamDamp: Float32Array; readonly damp: Float32Array;
  readonly beamDeform: Float32Array; readonly beamDeform0: Float32Array;
  readonly beamStrength: Float32Array;
  readonly compliance: Float32Array;
  readonly compDamp: Float32Array;
  readonly invRest0: Float32Array;
  readonly minLen: Float32Array;
  readonly maxLen: Float32Array;
  readonly yieldStrain: Float32Array; readonly yieldStrain0: Float32Array;
  readonly hardening: Float32Array; readonly breakStrain: Float32Array;
  readonly lambda: Float32Array;
  /** 3-state JBeam state per beam: 0 = ELASTIC, 1 = PLASTIC, 2 = BROKEN */
  readonly beamState: Uint8Array;
  readonly flags: Uint8Array; readonly flags0: Uint8Array;
  readonly zone: Uint8Array;

  // ---- Surface Collision Triangles (coltris — anti-clipping) ----
  readonly coltriA: Uint16Array;
  readonly coltriB: Uint16Array;
  readonly coltriC: Uint16Array;

  // ---- Telemetry ----
  readonly zoneDamage: Float32Array;
  readonly zoneCapacity: Float32Array;
  plasticWork = 0;
  peakImpact = 0;
  lastImpact = 0;
  brokenThisStep = 0;
  nanResets = 0;

  forceProvider: SubstepForceProvider | null = null;
  world: CollisionWorld | null = null;

  // Pre-allocated broadphase & scratch buffers
  private readonly activeBoxes: Int32Array;
  private readonly activeBoxIsPress: Uint8Array;
  private readonly activeCyls: Int32Array;
  private readonly boxCornersX: Float32Array;
  private readonly boxCornersZ: Float32Array;
  private readonly boxCornerY1: Float32Array;
  private boxCornerCount = 0;
  private activeBoxCount = 0;
  private activeCylCount = 0;
  private readonly kinVel = new Float32Array(3);
  private readonly ground: GroundSample = { h: 0, nx: 0, ny: 1, nz: 0, mu: 1 };
  private readonly collideBeams: Uint16Array;
  private collideBeamCount = 0;

  constructor(
    nodes: SolverBuildNode[],
    beams: SolverBuildBeam[],
    zones: ZoneParams[] = DEFAULT_ZONES,
    coltris: SolverBuildColtri[] = [],
  ) {
    const n = nodes.length;
    const m = beams.length;
    const ct = coltris.length;
    const P = this.pool;
    this.nodeCount = n;
    this.beamCount = m;
    this.coltriCount = ct;
    this.zones = zones;

    this.px = P.f32(n); this.py = P.f32(n); this.pz = P.f32(n);
    this.qx = P.f32(n); this.qy = P.f32(n); this.qz = P.f32(n);
    this.sx = P.f32(n); this.sy = P.f32(n); this.sz = P.f32(n);
    this.vx = P.f32(n); this.vy = P.f32(n); this.vz = P.f32(n);
    this.fx = P.f32(n); this.fy = P.f32(n); this.fz = P.f32(n);
    this.mass = P.f32(n); this.invMass = P.f32(n);
    this.radius = P.f32(n); this.friction = P.f32(n); this.group = P.u8(n);
    this.cnx = P.f32(n); this.cny = P.f32(n); this.cnz = P.f32(n);
    this.cvn = P.f32(n); this.cHit = P.u8(n); this.cObstacle = P.u8(n);

    this.beamA = P.u16(m); this.beamB = P.u16(m);
    this.restLength = P.f32(m); this.restLength0 = P.f32(m);
    this.rest = this.restLength; this.rest0 = this.restLength0;
    this.beamSpring = P.f32(m);
    this.beamDamp = P.f32(m); this.damp = this.beamDamp;
    this.beamDeform = P.f32(m); this.beamDeform0 = P.f32(m);
    this.beamStrength = P.f32(m);
    this.compliance = P.f32(m);
    this.compDamp = P.f32(m);
    this.invRest0 = P.f32(m);
    this.minLen = P.f32(m);
    this.maxLen = P.f32(m);
    this.yieldStrain = P.f32(m); this.yieldStrain0 = P.f32(m);
    this.hardening = P.f32(m); this.breakStrain = P.f32(m);
    this.lambda = P.f32(m);
    this.beamState = P.u8(m);
    this.flags = P.u8(m); this.flags0 = P.u8(m); this.zone = P.u8(m);

    this.coltriA = P.u16(Math.max(1, ct));
    this.coltriB = P.u16(Math.max(1, ct));
    this.coltriC = P.u16(Math.max(1, ct));
    for (let t = 0; t < ct; t++) {
      this.coltriA[t] = coltris[t].a;
      this.coltriB[t] = coltris[t].b;
      this.coltriC[t] = coltris[t].c;
    }

    this.zoneDamage = P.f32(ZONE_COUNT);
    this.zoneCapacity = P.f32(ZONE_COUNT);
    this.activeBoxes = P.i32(128);
    this.activeBoxIsPress = P.u8(128);
    this.activeCyls = P.i32(128);
    this.boxCornersX = P.f32(256);
    this.boxCornersZ = P.f32(256);
    this.boxCornerY1 = P.f32(256);
    this.collideBeams = P.u16(m);

    for (let i = 0; i < n; i++) {
      const nd = nodes[i];
      this.px[i] = nd.x; this.py[i] = nd.y; this.pz[i] = nd.z;
      this.mass[i] = nd.mass;
      this.invMass[i] = nd.mass > 0 ? 1 / nd.mass : 0;
      this.radius[i] = nd.radius;
      this.friction[i] = nd.friction;
      this.group[i] = nd.group;
    }

    for (let j = 0; j < m; j++) {
      const b = beams[j];
      const z = zones[b.zone] ?? zones[0];
      this.beamA[j] = b.a; this.beamB[j] = b.b;
      const dx = nodes[b.b].x - nodes[b.a].x;
      const dy = nodes[b.b].y - nodes[b.a].y;
      const dz = nodes[b.b].z - nodes[b.a].z;
      const L = Math.sqrt(dx * dx + dy * dy + dz * dz);
      this.restLength[j] = L;
      this.restLength0[j] = L;
      this.invRest0[j] = L > 1e-7 ? 1 / L : 0;
      const minR = b.zone === Zone.EngineSubframe ? this.minRatioEngine : b.zone === Zone.Bumper ? this.minRatioBumper : this.minRatio;
      this.minLen[j] = minR * L;
      this.maxLen[j] = this.maxRatio * L;

      const kSpring = z.spring * (b.springScale ?? 1);
      const cDamp = z.damp * (b.dampScale ?? 1);
      const yth = z.plasticThreshold * (b.plasticScale ?? 1);
      const brk = z.breakStrain * (b.strengthScale ?? 1);
      const isBreakable = brk > 0 && !(b.flags & BEAM_UNBREAKABLE);

      this.beamSpring[j] = kSpring;
      this.beamDamp[j] = cDamp;
      this.compliance[j] = 1 / kSpring;
      this.compDamp[j] = cDamp / kSpring;
      this.yieldStrain[j] = yth;
      this.yieldStrain0[j] = yth;
      this.beamDeform[j] = kSpring * yth * L;
      this.beamDeform0[j] = this.beamDeform[j];
      this.beamStrength[j] = isBreakable ? kSpring * brk * L : 0;
      this.hardening[j] = z.hardening;
      this.breakStrain[j] = isBreakable ? brk : 0;
      this.beamState[j] = BEAM_STATE_ELASTIC;

      const breakFlag = isBreakable ? BEAM_BREAKABLE : 0;
      this.flags[j] = b.flags | breakFlag;
      this.flags0[j] = this.flags[j];
      this.zone[j] = b.zone;
      this.zoneCapacity[b.zone] += L;
      if (b.flags & BEAM_COLLIDE) this.collideBeams[this.collideBeamCount++] = j;
    }
    this.snapshotStepStart();
  }

  /** Re-initialise all beams & telemetry to factory condition (used by fast < 2 ms Reset). */
  resetBeams(): void {
    this.restLength.set(this.restLength0);
    this.yieldStrain.set(this.yieldStrain0);
    this.beamDeform.set(this.beamDeform0);
    this.flags.set(this.flags0);
    this.beamState.fill(BEAM_STATE_ELASTIC);
    this.lambda.fill(0);
    this.zoneDamage.fill(0);
    this.plasticWork = 0;
    this.peakImpact = 0;
    this.lastImpact = 0;
  }

  snapshotStepStart(): void {
    this.sx.set(this.px); this.sy.set(this.py); this.sz.set(this.pz);
  }

  /** Advance one fixed step (dt = 1/60 s) split into 10–12 high-frequency substeps. */
  step(dt: number): void {
    this.snapshotStepStart();
    this.broadphase();
    this.brokenThisStep = 0;
    this.lastImpact = 0;
    const h = dt / this.substeps;
    for (let s = 0; s < this.substeps; s++) this.substep(h, (s & 1) === 0);
  }

  private substep(h: number, runColtris: boolean): void {
    const n = this.nodeCount;
    const { px, py, pz, qx, qy, qz, vx, vy, vz, fx, fy, fz, invMass } = this;
    if (this.forceProvider) this.forceProvider.applySubstepForces(h);

    const g = this.gravity * h;
    const damp = 1 - this.airDamping * h;
    const maxV = this.maxSubstepSpeed;
    const maxV2 = maxV * maxV;

    // Symplectic Euler prediction with velocity ceiling
    for (let i = 0; i < n; i++) {
      const w = invMass[i];
      qx[i] = px[i]; qy[i] = py[i]; qz[i] = pz[i];
      if (w > 0) {
        const wh = w * h;
        let nvx = (vx[i] + fx[i] * wh) * damp;
        let nvy = vy[i] + g + fy[i] * wh;
        let nvz = (vz[i] + fz[i] * wh) * damp;
        const v2 = nvx * nvx + nvy * nvy + nvz * nvz;
        if (v2 > maxV2) {
          const s = maxV / Math.sqrt(v2);
          nvx *= s; nvy *= s; nvz *= s;
        }
        vx[i] = nvx; vy[i] = nvy; vz[i] = nvz;
        px[i] += nvx * h; py[i] += nvy * h; pz[i] += nvz * h;
      }
      fx[i] = 0; fy[i] = 0; fz[i] = 0;
    }

    this.cHit.fill(0);
    this.cObstacle.fill(0);
    this.lambda.fill(0);
    for (let it = 0; it < this.iterations; it++) {
      this.solveBeams(h);
    }
    this.solveCollisions(h, runColtris);

    // Stabilized Verlet velocity recovery + per-substep speed clamp + NaN guard
    const inv = 1 / h;
    const maxDisp = maxV * h;
    const maxDisp2 = maxDisp * maxDisp;
    for (let i = 0; i < n; i++) {
      let dx = px[i] - qx[i];
      let dy = py[i] - qy[i];
      let dz = pz[i] - qz[i];
      if (dx !== dx || dy !== dy || dz !== dz || px[i] !== px[i]) {
        px[i] = qx[i]; py[i] = qy[i]; pz[i] = qz[i];
        vx[i] = 0; vy[i] = 0; vz[i] = 0;
        this.nanResets++;
        continue;
      }
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > maxDisp2) {
        const sc = maxDisp / Math.sqrt(d2);
        dx *= sc; dy *= sc; dz *= sc;
        px[i] = qx[i] + dx;
        py[i] = qy[i] + dy;
        pz[i] = qz[i] + dz;
      }
      vx[i] = dx * inv;
      vy[i] = dy * inv;
      vz[i] = dz * inv;
    }
    this.applyRestitution();

    // Strict floor clamp for every vehicle node so no body panel, roof, hood, trunk, bumper, door, or detached part can ever sink below ground
    const world = this.world;
    if (world) {
      const gs = this.ground;
      const rad = this.radius;
      for (let i = 0; i < n; i++) {
        if (invMass[i] === 0) continue;
        world.groundAt(px[i], pz[i], gs);
        const minY = gs.h + rad[i];
        if (py[i] < minY) {
          py[i] = minY;
          if (vy[i] < 0) vy[i] = 0;
        }
      }
    }
  }

  private solveBeams(h: number): void {
    const {
      px, py, pz, qx, qy, qz, invMass, beamA, beamB,
      restLength, restLength0, invRest0, minLen, maxLen,
      beamSpring, beamDeform, beamStrength,
      compliance, compDamp, flags, beamState, lambda, yieldStrain,
      hardening, breakStrain, cObstacle, zone, zoneDamage,
    } = this;
    const h2 = h * h;
    const invH = 1 / h;
    const invH2 = 1 / h2;
    const m = this.beamCount;

    for (let j = 0; j < m; j++) {
      const fl = flags[j];
      if (fl & BEAM_BROKEN) continue;
      const a = beamA[j], b = beamB[j];
      const wa = cObstacle[a] ? invMass[a] * 0.15 : invMass[a];
      const wb = cObstacle[b] ? invMass[b] * 0.15 : invMass[b];
      const w = wa + wb;
      if (w === 0) continue;

      const pxa = px[a], pya = py[a], pza = pz[a];
      const pxb = px[b], pyb = py[b], pzb = pz[b];
      let dx = pxb - pxa, dy = pyb - pya, dz = pzb - pza;
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (len < 1e-7) continue;
      const il = 1 / len;
      dx *= il; dy *= il; dz *= il;

      let L = restLength[j];
      const C = len - L;
      const alpha = compliance[j] * invH2;
      const gamma = compDamp[j] * invH;
      const vrel =
        ((pxb - qx[b]) - (pxa - qx[a])) * dx +
        ((pyb - qy[b]) - (pya - qy[a])) * dy +
        ((pzb - qz[b]) - (pza - qz[a])) * dz;

      // 1. Elastic trial force & JBeam 3-state check (ELASTIC -> PLASTIC -> BROKEN)
      const denomEl = w + alpha;
      const prevLam = lambda[j];
      let lamEl = prevLam + (-C - alpha * prevLam) / denomEl;
      const deformCap = beamDeform[j] * h2;
      let corr: number;

      if (lamEl <= deformCap && lamEl >= -deformCap) {
        // Fast elastic path (99% of beams on most substeps)
        lambda[j] = lamEl;
        const dlDamp = (-gamma * vrel) / ((1 + gamma) * w + alpha);
        corr = (lamEl - prevLam) + dlDamp;
      } else {
        const z = zone[j];
        const L0 = restLength0[j];
        const minL = minLen[j];
        const maxL = maxLen[j];
        const sign = lamEl > 0 ? 1 : -1;
        const excess = (Math.abs(lamEl) - deformCap) * denomEl;
        const oldL = L;
        // Accordion buckling in crumple zones absorbs extra compression with high plastic retention (minimal rubber-band bounce)
        const buckleRelax = (z === Zone.Crumple && sign > 0) ? 0.88 : 0.78;
        L -= sign * (excess + (deformCap * buckleRelax) * denomEl);
        if (L < minL) L = minL;
        else if (L > maxL) L = maxL;

        const plastic = Math.abs(L - oldL);
        if (plastic > 0) {
          restLength[j] = L;
          beamState[j] = BEAM_STATE_PLASTIC;
          const invL0 = invRest0[j];
          yieldStrain[j] += hardening[j] * (plastic * invL0);
          beamDeform[j] = beamSpring[j] * yieldStrain[j] * L0;
          zoneDamage[z] += plastic;
          this.plasticWork += (deformCap * invH2) * plastic;
        }
        lamEl = sign * deformCap * (1 - buckleRelax);

        if (fl & BEAM_BREAKABLE) {
          const invL0 = invRest0[j];
          const absDL0 = Math.abs(len - L0);
          const totalForce = beamSpring[j] * absDL0;
          const totalStrain = absDL0 * invL0;
          const plasticStrain = Math.abs(L - L0) * invL0;
          const strCap = beamStrength[j];
          const brkS = breakStrain[j];
          if (
            (strCap > 0 && totalForce > strCap) ||
            totalStrain > brkS ||
            plasticStrain > brkS * 0.55
          ) {
            flags[j] = fl | BEAM_BROKEN;
            beamState[j] = BEAM_STATE_BROKEN;
            this.brokenThisStep++;
            continue;
          }
        }

        lambda[j] = lamEl;
        const gYield = gamma * 2.6;
        const dlDamp = (-gYield * vrel) / ((1 + gYield) * w + alpha);
        corr = (lamEl - prevLam) + dlDamp;
      }

      const newLen = len + w * corr;
      const minL = minLen[j];
      const maxL = maxLen[j];
      if (newLen < minL) corr += (minL - newLen) / w;
      else if (newLen > maxL) corr += (maxL - newLen) / w;

      const ca = wa * corr, cb = wb * corr;
      px[a] = pxa - dx * ca; py[a] = pya - dy * ca; pz[a] = pza - dz * ca;
      px[b] = pxb + dx * cb; py[b] = pyb + dy * cb; pz[b] = pzb + dz * cb;
    }
  }

  /** O(1) Spatial Hash Grid broadphase (with linear fallback for custom test worlds). */
  private broadphase(): void {
    const world = this.world;
    this.activeBoxCount = 0;
    this.activeCylCount = 0;
    this.boxCornerCount = 0;
    if (!world) return;

    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (let i = 0; i < this.nodeCount; i++) {
      const x = this.px[i], z = this.pz[i];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
    const pad = 2.6;
    minX -= pad; maxX += pad; minZ -= pad; maxZ += pad;

    const B = world.boxes;
    if (world.queryBoxes) {
      this.activeBoxCount = world.queryBoxes(minX, minZ, maxX, maxZ, this.activeBoxes, 128);
    } else {
      for (let k = 0; k < world.boxCount && this.activeBoxCount < 128; k++) {
        const o = k * BOX_STRIDE;
        if (B[o + 10] < minX || B[o + 9] > maxX || B[o + 12] < minZ || B[o + 11] > maxZ) continue;
        this.activeBoxes[this.activeBoxCount++] = k;
      }
    }

    const Cy = world.cyls;
    if (world.queryCyls) {
      this.activeCylCount = world.queryCyls(minX, minZ, maxX, maxZ, this.activeCyls, 128);
    } else {
      for (let k = 0; k < world.cylCount && this.activeCylCount < 128; k++) {
        const o = k * CYL_STRIDE;
        const r = Cy[o + 2];
        if (Cy[o] + r < minX || Cy[o] - r > maxX || Cy[o + 1] + r < minZ || Cy[o + 1] - r > maxZ) continue;
        this.activeCyls[this.activeCylCount++] = k;
      }
    }

    // Cache isPress flag & precompute vertical corner edges of nearby non-press boxes once per frame
    for (let a = 0; a < this.activeBoxCount; a++) {
      const boxIdx = this.activeBoxes[a];
      const isPress = world.isPressBox
        ? world.isPressBox(boxIdx)
        : (world.pressBoxIndex !== undefined && boxIdx === world.pressBoxIndex);
      this.activeBoxIsPress[a] = isPress ? 1 : 0;
      if (isPress || this.boxCornerCount + 4 > 256) continue;
      const o = boxIdx * BOX_STRIDE;
      const hx = B[o + 3], hy = B[o + 4], hz = B[o + 5];
      if (hx > 40 || hz > 40) continue; // skip straight perimeter walls (no sharp corner poke)
      const cx = B[o], cy = B[o + 1], cz = B[o + 2];
      const c = B[o + 6], s = B[o + 7];
      const topY = cy + hy;
      for (let sx = -1; sx <= 1; sx += 2) {
        for (let sz = -1; sz <= 1; sz += 2) {
          const lx = sx * Math.max(0.05, hx - 0.06);
          const lz = sz * Math.max(0.05, hz - 0.06);
          const idx = this.boxCornerCount++;
          this.boxCornersX[idx] = cx + lx * c + lz * s;
          this.boxCornersZ[idx] = cz - lx * s + lz * c;
          this.boxCornerY1[idx] = topY;
        }
      }
    }

    // Check once per frame whether the vehicle is near any crusher roller
    this.nearRollers = false;
    const rollers = world.rollers;
    const rollerCount = world.rollerCount ?? 0;
    if (rollers && rollerCount > 0) {
      for (let rIdx = 0; rIdx < rollerCount; rIdx++) {
        const ro = rIdx * ROLLER_STRIDE;
        const rcx = rollers[ro], rcz = rollers[ro + 2];
        const hw = rollers[ro + 3] + 2.0, rad = rollers[ro + 4] + 2.0;
        if (rcx + hw >= minX && rcx - hw <= maxX && rcz + rad >= minZ && rcz - rad <= maxZ) {
          this.nearRollers = true;
          break;
        }
      }
    }
  }

  /** Records a contact and applies position-based Coulomb friction. */
  private contact(i: number, nx: number, ny: number, nz: number, pen: number, mu: number, obstacle: number): void {
    const { px, py, pz, qx, qy, qz, vx, vy, vz } = this;
    const preVn = vx[i] * nx + vy[i] * ny + vz[i] * nz;
    const dvx = px[i] - qx[i], dvy = py[i] - qy[i], dvz = pz[i] - qz[i];
    const dispN = dvx * nx + dvy * ny + dvz * nz;
    px[i] += nx * pen; py[i] += ny * pen; pz[i] += nz * pen;
    const tx = dvx - dispN * nx, ty = dvy - dispN * ny, tz = dvz - dispN * nz;
    const tl = Math.sqrt(tx * tx + ty * ty + tz * tz);
    if (tl > 1e-9) {
      const fmu = mu * this.friction[i];
      const maxStatic = fmu * 1.1 * pen;
      const k = tl < maxStatic ? 1 : Math.min(1, (fmu * pen) / tl);
      px[i] -= tx * k; py[i] -= ty * k; pz[i] -= tz * k;
    }
    if (!this.cHit[i] || (obstacle && !this.cObstacle[i]) || preVn < this.cvn[i]) {
      this.cnx[i] = nx; this.cny[i] = ny; this.cnz[i] = nz;
      this.cvn[i] = preVn;
    }
    this.cHit[i] = 1;
    if (obstacle) this.cObstacle[i] = 1;
  }

  private nearRollers = false;

  private solveCollisions(h: number, runColtris: boolean): void {
    const world = this.world;
    if (!world) return;
    const { px, py, pz, radius, invMass } = this;
    const gs = this.ground;
    const B = world.boxes, Cy = world.cyls;
    const rollers = this.nearRollers ? world.rollers : undefined;
    const rollerCount = rollers ? (world.rollerCount ?? 0) : 0;

    for (let i = 0; i < this.nodeCount; i++) {
      if (invMass[i] === 0) continue;
      const r = radius[i];

      // 1. Analytical + spatial-hashed ground (ramps, sine waves, cobbles, stairs, ditches, banked bowl)
      //    Full penetration resolution for EVERY car node (chassis, roof, hood, trunk, bumpers, doors, fenders, engine, detached parts)
      world.groundAt(px[i], pz[i], gs);
      const dg = (py[i] - gs.h) * gs.ny - r;
      if (dg < 0) {
        const pen = Math.min(-dg, 1.2);
        this.contact(i, gs.nx, gs.ny, gs.nz, pen, gs.mu, 0);
        if (-dg > 0.25 && this.vy[i] > 2.0) this.vy[i] = 2.0;
      }

      // 2. Oriented boxes (NCAP barriers, walls, hydraulic presses, pendulum hammers, guardrails, containers)
      for (let a = 0; a < this.activeBoxCount; a++) {
        const boxIdx = this.activeBoxes[a];
        const o = boxIdx * BOX_STRIDE;
        const c = B[o + 6], s = B[o + 7];
        const rx = px[i] - B[o], ry = py[i] - B[o + 1], rz = pz[i] - B[o + 2];
        const lx = rx * c - rz * s;
        const lz = rx * s + rz * c;
        const hx = B[o + 3] + r, hy = B[o + 4] + r, hz = B[o + 5] + r;
        if (lx > hx || lx < -hx || ry > hy || ry < -hy || lz > hz || lz < -hz) continue;

        const isPress = this.activeBoxIsPress[a] === 1;

        if (isPress && ry < hy * 0.65) {
          const minPressY = gs.h + 0.96;
          if (py[i] > minPressY) {
            const pBottom = Math.min(0.014, hy + ry, py[i] - minPressY);
            this.contact(i, 0, -1, 0, pBottom, 0.94, 1);
          }
          this.vx[i] *= 0.86;
          this.vz[i] *= 0.86;
          if (this.vy[i] > 0) this.vy[i] = 0;
          continue;
        }

        // Check if elevated moving obstacle (pendulum hammer) has a bottom face above ground
        const hasBottom = B[o + 1] - B[o + 4] > gs.h + 0.25;
        let pen = hx - lx, ax = 0, sg = 1;
        let p2 = hx + lx; if (p2 < pen) { pen = p2; ax = 0; sg = -1; }
        p2 = hz - lz; if (p2 < pen) { pen = p2; ax = 2; sg = 1; }
        p2 = hz + lz; if (p2 < pen) { pen = p2; ax = 2; sg = -1; }
        p2 = hy - ry; if (p2 < pen) { pen = p2; ax = 1; sg = 1; }
        if (hasBottom) {
          p2 = hy + ry;
          if (p2 < pen) { pen = p2; ax = 1; sg = -1; }
        }
        pen = Math.min(pen, 0.24);

        let nx = 0, ny = 0, nz = 0;
        if (ax === 0) { nx = sg * c; nz = -sg * s; }
        else if (ax === 2) { nx = sg * s; nz = sg * c; }
        else ny = sg;

        // Transfer kinematic velocity from swinging pendulum hammers smoothly
        if (world.getBoxVelocity && world.getBoxVelocity(boxIdx, this.kinVel)) {
          const kvx = this.kinVel[0], kvy = this.kinVel[1], kvz = this.kinVel[2];
          const relN = (kvx - this.vx[i]) * nx + (kvy - this.vy[i]) * ny + (kvz - this.vz[i]) * nz;
          if (relN > 0) {
            const push = Math.min(relN, 18.0) * h;
            px[i] += nx * push; py[i] += ny * push; pz[i] += nz * push;
          }
        }
        this.contact(i, nx, ny, nz, pen, B[o + 8], 1);
      }

      // 3. Slow Progressive Crusher Rollers (horizontal steel drums rotating above ground)
      if (rollers && rollerCount > 0) {
        for (let rIdx = 0; rIdx < rollerCount; rIdx++) {
          const ro = rIdx * ROLLER_STRIDE;
          const rcx = rollers[ro], rcy = rollers[ro + 1], rcz = rollers[ro + 2];
          const halfW = rollers[ro + 3], R = rollers[ro + 4] + r;
          if (Math.abs(px[i] - rcx) > halfW + r) continue;
          const dy = py[i] - rcy;
          const dz = pz[i] - rcz;
          const d2 = dy * dy + dz * dz;
          if (d2 >= R * R) continue;
          const minFloor = gs.h + rollers[ro + 6];
          const d = Math.sqrt(d2) || 1e-5;
          const ny = dy / d, nz = dz / d;
          const rawPen = R - d;
          if (ny < -0.15 && py[i] <= minFloor) {
            continue;
          }
          const pen = Math.min(0.016, rawPen);
          this.contact(i, 0, ny, nz, pen, 0.72, 1);
          const feedVz = rollers[ro + 5];
          if (feedVz !== 0) {
            pz[i] += (feedVz - this.vz[i]) * 0.12 * h;
          }
        }
      }

      // 4. Vertical cylinders (high-strength steel poles, bollards, lamp posts, tyres)
      for (let a = 0; a < this.activeCylCount; a++) {
        const o = this.activeCyls[a] * CYL_STRIDE;
        const y0 = Cy[o + 3], y1 = Cy[o + 4];
        if (py[i] < y0 - r || py[i] > y1 + r) continue;
        const dx = px[i] - Cy[o], dz = pz[i] - Cy[o + 1];
        const R = Cy[o + 2] + r;
        const d2 = dx * dx + dz * dz;
        if (d2 >= R * R) continue;
        const d = Math.sqrt(d2) || 1e-6;
        const penR = Math.min(0.22, R - d);
        const penTop = y1 + r - py[i];
        if (penTop < penR) this.contact(i, 0, 1, 0, penTop, Cy[o + 5], 1);
        else this.contact(i, dx / d, 0, dz / d, penR, Cy[o + 5], 1);
      }

      // 5. Post-obstacle ground floor guarantee (only needed when an obstacle pushed this node)
      if (this.cObstacle[i]) {
        const dgPost = (py[i] - gs.h) * gs.ny - r;
        if (dgPost < 0) {
          const penPost = -dgPost > 0.35 ? 0.055 : -dgPost;
          this.contact(i, gs.nx, gs.ny, gs.nz, penPost, gs.mu, 0);
          if (this.vy[i] < 0) this.vy[i] *= 0.2;
        }
      }
    }

    // 6. Beam-vs-Cylinder edge catch
    if (this.activeCylCount > 0) {
      const { beamA, beamB, flags } = this;
      for (let k = 0; k < this.collideBeamCount; k++) {
        const j = this.collideBeams[k];
        if (flags[j] & BEAM_BROKEN) continue;
        const a = beamA[j], b = beamB[j];
        for (let q = 0; q < this.activeCylCount; q++) {
          const o = this.activeCyls[q] * CYL_STRIDE;
          const cx = Cy[o], cz = Cy[o + 1], R = Cy[o + 2] + 0.06;
          const ex = px[b] - px[a], ez = pz[b] - pz[a];
          const ee = ex * ex + ez * ez;
          if (ee < 1e-8) continue;
          const t = ((cx - px[a]) * ex + (cz - pz[a]) * ez) / ee;
          if (t <= 0.02 || t >= 0.98) continue;
          const yy = py[a] + (py[b] - py[a]) * t;
          if (yy < Cy[o + 3] || yy > Cy[o + 4]) continue;
          const rx = px[a] + ex * t - cx, rz = pz[a] + ez * t - cz;
          const d2 = rx * rx + rz * rz;
          if (d2 >= R * R) continue;
          const d = Math.sqrt(d2) || 1e-6;
          const pen = Math.min(0.16, R - d);
          const nx = rx / d, nz = rz / d;
          const wa = invMass[a] * (1 - t), wb = invMass[b] * t;
          const s = pen / ((1 - t) * wa + t * wb + 1e-12);
          px[a] += nx * s * wa; pz[a] += nz * s * wa;
          px[b] += nx * s * wb; pz[b] += nz * s * wb;
          this.markContact(a, nx, nz); this.markContact(b, nx, nz);
        }
      }
    }

    // 7. Surface Collision Triangles (`coltris`) vs Poles & Sharp Barrier Corners (Anti-Clipping)
    if (runColtris && this.coltriCount > 0 && (this.activeCylCount > 0 || this.boxCornerCount > 0)) {
      this.solveColtris(Cy);
    }
  }

  /**
   * Resolves surface triangle (`coltri`) penetration against vertical cylinders and
   * sharp offset-barrier vertical corners using exact 2D Voronoi barycentric closest-point
   * projection. Prevents poles and barrier edges from poking between lattice nodes.
   */
  private solveColtris(Cy: Float32Array | Float64Array): void {
    const { px, py, pz, qx, qz, invMass, coltriA, coltriB, coltriC, coltriCount } = this;
    const skinPad = 0.055;

    for (let t = 0; t < coltriCount; t++) {
      const a = coltriA[t], b = coltriB[t], c = coltriC[t];
      const wa = invMass[a], wb = invMass[b], wc = invMass[c];
      if (wa + wb + wc === 0) continue;

      const ax = px[a], az = pz[a];
      const bx = px[b], bz = pz[b];
      const cxT = px[c], czT = pz[c];

      // Guard against stretched detached-part triangles
      const e1x = bx - ax, e1z = bz - az;
      const e2x = cxT - ax, e2z = czT - az;
      if (e1x * e1x + e1z * e1z > 3.2 || e2x * e2x + e2z * e2z > 3.2) continue;

      const minTx = ax < bx ? (ax < cxT ? ax : cxT) : (bx < cxT ? bx : cxT);
      const maxTx = ax > bx ? (ax > cxT ? ax : cxT) : (bx > cxT ? bx : cxT);
      const minTz = az < bz ? (az < czT ? az : czT) : (bz < czT ? bz : czT);
      const maxTz = az > bz ? (az > czT ? az : czT) : (bz > czT ? bz : czT);

      // A) Poles / Bollards vs Coltri
      for (let q = 0; q < this.activeCylCount; q++) {
        const o = this.activeCyls[q] * CYL_STRIDE;
        const pcx = Cy[o], pcz = Cy[o + 1];
        const Reff = Cy[o + 2] + skinPad;
        if (pcx + Reff < minTx || pcx - Reff > maxTx || pcz + Reff < minTz || pcz - Reff > maxTz) continue;
        this.resolvePointVsColtri2D(a, b, c, pcx, pcz, Cy[o + 3], Cy[o + 4], Reff, px, py, pz, qx, qz, wa, wb, wc);
      }

      // B) Sharp Offset-Barrier Corners vs Coltri
      for (let q = 0; q < this.boxCornerCount; q++) {
        const bcx = this.boxCornersX[q], bcz = this.boxCornersZ[q];
        const Reff = 0.085;
        if (bcx + Reff < minTx || bcx - Reff > maxTx || bcz + Reff < minTz || bcz - Reff > maxTz) continue;
        this.resolvePointVsColtri2D(a, b, c, bcx, bcz, 0, this.boxCornerY1[q], Reff, px, py, pz, qx, qz, wa, wb, wc);
      }
    }
  }

  private resolvePointVsColtri2D(
    a: number, b: number, c: number,
    ox: number, oz: number, y0: number, y1: number, Reff: number,
    px: Float32Array, py: Float32Array, pz: Float32Array,
    qx: Float32Array, qz: Float32Array,
    wa: number, wb: number, wc: number,
  ): void {
    const ax = px[a], az = pz[a];
    const bx = px[b], bz = pz[b];
    const cx = px[c], cz = pz[c];

    // Ericson closest-point on 2D triangle (ax,az)-(bx,bz)-(cx,cz) to (ox,oz)
    const abx = bx - ax, abz = bz - az;
    const acx = cx - ax, acz = cz - az;
    const apx = ox - ax, apz = oz - az;
    const d1 = abx * apx + abz * apz;
    const d2 = acx * apx + acz * apz;

    let u = 1, v = 0, w = 0;
    if (d1 <= 0 && d2 <= 0) {
      u = 1; v = 0; w = 0;
    } else {
      const bpx = ox - bx, bpz = oz - bz;
      const d3 = abx * bpx + abz * bpz;
      const d4 = acx * bpx + acz * bpz;
      if (d3 >= 0 && d4 <= d3) {
        u = 0; v = 1; w = 0;
      } else {
        const vc = d1 * d4 - d3 * d2;
        if (vc <= 0 && d1 >= 0 && d3 <= 0) {
          v = d1 / (d1 - d3 || 1); u = 1 - v; w = 0;
        } else {
          const cpx = ox - cx, cpz = oz - cz;
          const d5 = abx * cpx + abz * cpz;
          const d6 = acx * cpx + acz * cpz;
          if (d6 >= 0 && d5 <= d6) {
            u = 0; v = 0; w = 1;
          } else {
            const vb = d5 * d2 - d1 * d6;
            if (vb <= 0 && d2 >= 0 && d6 <= 0) {
              w = d2 / (d2 - d6 || 1); u = 1 - w; v = 0;
            } else {
              const va = d3 * d6 - d5 * d4;
              if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
                const tEdge = (d4 - d3) / ((d4 - d3) + (d5 - d6) || 1);
                u = 0; v = 1 - tEdge; w = tEdge;
              } else {
                const denom = 1 / (va + vb + vc || 1);
                v = vb * denom;
                w = vc * denom;
                u = 1 - v - w;
              }
            }
          }
        }
      }
    }

    const qy = u * py[a] + v * py[b] + w * py[c];
    if (qy < y0 - 0.04 || qy > y1 + 0.04) return;

    const nearX = u * ax + v * bx + w * cx;
    const nearZ = u * az + v * bz + w * cz;
    let rx = nearX - ox;
    let rz = nearZ - oz;
    const dist2 = rx * rx + rz * rz;
    if (dist2 >= Reff * Reff) return;

    let d = Math.sqrt(dist2);
    if (d < 1e-4) {
      // Fallback to substep-start barycentric point to prevent tunneling inversion
      rx = (u * qx[a] + v * qx[b] + w * qx[c]) - ox;
      rz = (u * qz[a] + v * qz[b] + w * qz[c]) - oz;
      d = Math.sqrt(rx * rx + rz * rz) || 1;
    }
    const nx = rx / d, nz = rz / d;
    const pen = Math.min(0.14, Reff - (dist2 < 1e-8 ? 0 : d));
    const denomW = u * u * wa + v * v * wb + w * w * wc;
    if (denomW < 1e-12) return;
    const s = pen / denomW;

    px[a] += nx * (s * u * wa); pz[a] += nz * (s * u * wa);
    px[b] += nx * (s * v * wb); pz[b] += nz * (s * v * wb);
    px[c] += nx * (s * w * wc); pz[c] += nz * (s * w * wc);
    if (u > 0.1) this.markContact(a, nx, nz);
    if (v > 0.1) this.markContact(b, nx, nz);
    if (w > 0.1) this.markContact(c, nx, nz);
  }

  private markContact(i: number, nx: number, nz: number): void {
    const preVn = this.vx[i] * nx + this.vz[i] * nz;
    if (!this.cHit[i] || preVn < this.cvn[i]) {
      this.cnx[i] = nx; this.cny[i] = 0; this.cnz[i] = nz;
      this.cvn[i] = Math.min(0, preVn);
    }
    this.cHit[i] = 1;
    this.cObstacle[i] = 1;
  }

  /** Clamp post-contact normal velocity to -e * v_pre: zero rubbery rebound. */
  private applyRestitution(): void {
    const e = this.restitution;
    const { vx, vy, vz, cnx, cny, cnz, cvn, cHit, cObstacle } = this;
    let impact = this.lastImpact;
    for (let i = 0; i < this.nodeCount; i++) {
      if (!cHit[i]) continue;
      const pre = cvn[i];
      const nx = cnx[i], ny = cny[i], nz = cnz[i];
      const vn = vx[i] * nx + vy[i] * ny + vz[i] * nz;
      const target = pre < -0.5 ? -e * pre : 0;
      if (vn > target) {
        const d = target - vn;
        vx[i] += nx * d; vy[i] += ny * d; vz[i] += nz * d;
      }
      if (cObstacle[i] && -pre > impact) impact = -pre;
    }
    this.lastImpact = impact;
    if (impact > this.peakImpact) this.peakImpact = impact;
  }

  /** Normalised 0..1 damage estimate per zone. */
  zoneDamageRatio(z: Zone): number {
    const cap = this.zoneCapacity[z];
    if (cap <= 0) return 0;
    return Math.min(1, this.zoneDamage[z] / (cap * 0.055));
  }
}
