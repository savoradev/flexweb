import {
  BackSide, BoxGeometry, BufferGeometry, CanvasTexture, Color, ConeGeometry, CylinderGeometry, DoubleSide, Euler,
  ExtrudeGeometry, Float32BufferAttribute, Group, InstancedMesh, Matrix4, Mesh, MeshBasicMaterial, MeshStandardMaterial,
  PlaneGeometry, Quaternion, RepeatWrapping, RingGeometry, Shape, SRGBColorSpace, TorusGeometry, Vector2, Vector3,
} from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { BOX_STRIDE, CYL_STRIDE, ROLLER_STRIDE, CollisionWorld, GroundSample, SoftBodySolver } from './SoftBodySolver';
import { getAsphaltTextures, getConcreteTextures, isSharedProceduralTexture } from './ProceduralTextures';
import { SpatialBoundsItem, SpatialHashGrid } from './SpatialHashGrid';

/** Y coordinate of the underground pure-white studio showroom floor (far below the y = 0 map). */
export const STUDIO_SHOWROOM_Y = -90;

/**
 * CrashArena (GridMap V2 — 2 km × 2 km Multi-Zone Proving Ground)
 * ---------------------------------------------------------------
 * 2,000 m × 2,000 m (4,000,000 m²) interconnected soft-body proving ground
 * engineered for < 40 total WebGL draw calls and O(1) SpatialHashGrid queries:
 *
 *  1. Crash-Test Lab (Haute Vitesse):
 *     500 m acceleration strip (z = -240 → +260) with 400m/300m/200m/100m/50m
 *     stencils, 3 live radar speed gantries displaying real trap speed in km/h,
 *     100% NCAP barrier, 40% & 25% offset barriers, 30° angled steel wall,
 *     high-strength steel poles, and bollards.
 *  2. Hydraulic Press & Crushers Zone (Northeast):
 *     Double heavy vertical hydraulic press, 4 synchronized swinging pendulum
 *     hammers with real 3D kinematic impulse transfer, and 3 slow progressive
 *     crusher rollers (progressive roof crush without ejection).
 *  3. Endurance & Suspension Circuit (West):
 *     Sinusoidal waves & chatter bars, analytical irregular cobblestones
 *     (Belgian pavé), giant concrete staircase, diagonal axle-articulation
 *     ditches, and a 180° high-speed parabolic banked curve (25.4° banking).
 *  4. Mega-Ramps & Jump Canyon (South):
 *     Calibrated 15°, 25°, and 35° launch ramps, giant Mega-Ramp jumping across
 *     a sunken canyon onto a wide sloped landing ramp, plus a Bullseye target.
 *  5. Urban / Light Industrial District (Southeast):
 *     2 drive-through industrial hangars, 28 stacked corrugated shipping
 *     containers, yieldable W-beam highway guardrails, and a large 56 m radius
 *     drift & T-bone roundabout.
 */

interface Ramp { x: number; z: number; yaw: number; len: number; w: number; h: number; baseY?: number; }
interface Bump { x0: number; x1: number; z0: number; z1: number; h: number; kind: 'sine' | 'step' | 'plate'; rampLen?: number; color: number; metal?: boolean; }
interface Patch { x0: number; x1: number; z0: number; z1: number; mu: number; color: number; rough: number; name: string; }

interface DynamicBarrier {
  boxIndex: number;
  x0: number; z0: number; yaw0: number;
  x: number; z: number; yaw: number;
  vx: number; vz: number; omega: number;
  tilt: number; vTilt: number;
  hx: number; hy: number; hz: number;
  guardrail: boolean;
}

interface DynamicCone {
  x0: number; z0: number;
  x: number; y: number; z: number;
  vx: number; vy: number; vz: number;
  tiltX: number; tiltZ: number;
  vTiltX: number; vTiltZ: number;
}

interface PendulumHammer {
  boxIndex: number;
  pivotX: number;
  pivotY: number;
  pivotZ: number;
  armLen: number;
  phase: number;
  freq: number;
  ampRad: number;
  vx: number;
  vy: number;
  theta: number;
}

export interface CrashTestPreset {
  id: string;
  label: string;
  x: number; z: number; yaw: number; speedKmh: number;
  lateral?: boolean;
  category?: 'crash' | 'suspension';
}

export interface ArenaMapZone {
  num: number;
  id: string;
  label: string;
  markerX: number;
  markerZ: number;
  spawnX: number;
  spawnZ: number;
  teleportX: number;
  teleportZ: number;
  yaw: number;
  defaultSpeedKmh: number;
  lateral?: boolean;
  category: 'crash' | 'suspension';
}

// Banked curve geometry constants (North end of West Endurance Circuit)
const BANK_CX = -255;
const BANK_CZ = 190;
const BANK_R_IN = 60;
const BANK_R_OUT = 88;
const BANK_MAX_H = 6.2;

// Sunken Jump Canyon bounds (South Mega-Ramp zone)
const CANYON_X0 = -45;
const CANYON_X1 = 45;
const CANYON_Z0 = -606;
const CANYON_Z1 = -542;
const CANYON_FLOOR_Y = -6.5;

export class CrashArena implements CollisionWorld {
  readonly boxes: Float32Array;
  readonly cyls: Float32Array;
  readonly rollers: Float32Array;
  readonly rollerCount = 3;
  boxCount = 0;
  cylCount = 0;
  readonly ramps: Ramp[] = [];
  readonly bumps: Bump[] = [];
  readonly patches: Patch[] = [];
  /** 2,000 m × 2,000 m (2 km × 2 km) proving ground half-extent. */
  readonly asphaltHalf = 1000;
  readonly group = new Group();

  /** Quick presets (1..9 keyboard shortcuts + Main Menu live crash showcase). */
  readonly presets: CrashTestPreset[] = [
    { id: 'wall', label: 'Frontal NCAP Wall', x: 0, z: 224, yaw: 0, speedKmh: 72, category: 'crash' },
    { id: 'offset', label: '40% Offset Barrier', x: -18 + 1.1 + 0.94 - 0.75, z: 224, yaw: 0, speedKmh: 76, category: 'crash' },
    { id: 'pole', label: 'Center Steel Pole', x: 16, z: 224, yaw: 0, speedKmh: 68, category: 'crash' },
    { id: 'bollard', label: 'Urban Bollards', x: 30, z: 224, yaw: 0, speedKmh: 58, category: 'crash' },
    { id: 'side', label: 'Side Pole Sled', x: 46, z: 261.75, yaw: Math.PI / 2, speedKmh: 42, lateral: true, category: 'crash' },
    { id: 'ramp', label: '25° Launch Ramp', x: -24, z: -295, yaw: Math.PI, speedKmh: 98, category: 'suspension' },
    { id: 'small-overlap', label: '25% Small Overlap', x: -32 + 0.7 + 0.94 - 0.47, z: 224, yaw: 0, speedKmh: 78, category: 'crash' },
    { id: 'angled', label: '30° Angled Steel Wall', x: 64, z: 224, yaw: 0, speedKmh: 75, category: 'crash' },
    { id: 'suspension', label: 'Endurance Suspension', x: -180, z: -235, yaw: 0, speedKmh: 48, category: 'suspension' },
  ];

  /** 16 numbered stations across all 5 zones on the 2 km × 2 km Overhead Map. */
  readonly mapZones: ArenaMapZone[] = [
    // Zone 1: High-Speed Crash-Test Lab (z = -240 -> +260)
    { num: 1, id: 'wall', label: '1. Mur Frontal NCAP (Piste 500m)', markerX: 0, markerZ: 260, spawnX: 0, spawnZ: 80, teleportX: 0, teleportZ: -180, yaw: 0, defaultSpeedKmh: 120, category: 'crash' },
    { num: 2, id: 'offset', label: '2. Barrière Déformable 40% Offset', markerX: -18, markerZ: 260, spawnX: -16.71, spawnZ: 140, teleportX: -16.71, teleportZ: 185, yaw: 0, defaultSpeedKmh: 95, category: 'crash' },
    { num: 3, id: 'small-overlap', label: '3. Barrière 25% Small Overlap', markerX: -32, markerZ: 260, spawnX: -30.83, spawnZ: 140, teleportX: -30.83, teleportZ: 185, yaw: 0, defaultSpeedKmh: 95, category: 'crash' },
    { num: 4, id: 'pole', label: '4. Poteau Acier Haute Résistance', markerX: 16, markerZ: 260, spawnX: 16, spawnZ: 150, teleportX: 16, teleportZ: 195, yaw: 0, defaultSpeedKmh: 85, category: 'crash' },
    { num: 5, id: 'side', label: '5. Impact Latéral Poteau / Bollards', markerX: 46, markerZ: 260, spawnX: 46, spawnZ: 261.75, teleportX: 30, teleportZ: 205, yaw: Math.PI / 2, defaultSpeedKmh: 46, lateral: true, category: 'crash' },
    { num: 6, id: 'angled', label: '6. Mur Acier Incliné 30°', markerX: 64, markerZ: 260, spawnX: 64, spawnZ: 155, teleportX: 64, teleportZ: 195, yaw: 0, defaultSpeedKmh: 90, category: 'crash' },

    // Zone 2: Hydraulic Press, Pendulum Hammers & Crusher Rollers (Northeast)
    { num: 7, id: 'double-press', label: '7. Double Presse Hydraulique Lourde', markerX: 195, markerZ: 158, spawnX: 195, spawnZ: 112, teleportX: 195, teleportZ: 150, yaw: 0, defaultSpeedKmh: 55, category: 'crash' },
    { num: 8, id: 'pendulums', label: '8. Marteaux Pendulaires Synchronisés', markerX: 245, markerZ: 154, spawnX: 245, spawnZ: 92, teleportX: 245, teleportZ: 105, yaw: 0, defaultSpeedKmh: 65, category: 'crash' },
    { num: 9, id: 'rollers', label: '9. Rouleaux Broyeurs Progressifs', markerX: 295, markerZ: 150, spawnX: 295, spawnZ: 112, teleportX: 295, teleportZ: 126, yaw: 0, defaultSpeedKmh: 42, category: 'crash' },

    // Zone 3: Endurance & Suspension Circuit (West)
    { num: 10, id: 'sine-whoops', label: '10. Ondulations & Barres Flexbody', markerX: -180, markerZ: -60, spawnX: -180, spawnZ: -235, teleportX: -180, teleportZ: -225, yaw: 0, defaultSpeedKmh: 55, category: 'suspension' },
    { num: 11, id: 'cobbles-stairs', label: '11. Pavés Irréguliers & Escaliers Géants', markerX: -255, markerZ: -120, spawnX: -230, spawnZ: -235, teleportX: -280, teleportZ: -220, yaw: 0, defaultSpeedKmh: 48, category: 'suspension' },
    { num: 12, id: 'banking-ditches', label: '12. Fossés Croisement & Virage Relevé 25°', markerX: -255, markerZ: 262, spawnX: -181, spawnZ: 170, teleportX: -330, teleportZ: -225, yaw: 0, defaultSpeedKmh: 135, category: 'suspension' },

    // Zone 4: Mega-Ramps & Jump Canyon (South)
    { num: 13, id: 'triple-ramps', label: '13. Tremplins 15° / 25° / 35° & Cible', markerX: -24, markerZ: -355, spawnX: -24, spawnZ: -265, teleportX: 68, teleportZ: -275, yaw: Math.PI, defaultSpeedKmh: 115, category: 'suspension' },
    { num: 14, id: 'mega-canyon', label: '14. Méga-Tremplin & Saut du Canyon', markerX: 0, markerZ: -565, spawnX: 0, spawnZ: -410, teleportX: 0, teleportZ: -425, yaw: Math.PI, defaultSpeedKmh: 165, category: 'suspension' },

    // Zone 5: Urban / Light Industrial District (Southeast)
    { num: 15, id: 'roundabout', label: '15. Rond-Point Drift & Glissières', markerX: 380, markerZ: -220, spawnX: 255, spawnZ: -206, teleportX: 310, teleportZ: -206, yaw: Math.PI / 2, defaultSpeedKmh: 105, category: 'crash' },
    { num: 16, id: 'industrial', label: '16. Hangars Traversables & Conteneurs', markerX: 390, markerZ: -380, spawnX: 260, spawnZ: -285, teleportX: 395, teleportZ: -310, yaw: Math.PI, defaultSpeedKmh: 125, category: 'crash' },
  ];

  private boxList: number[][] = [];
  private cylList: number[][] = [];
  private readonly steelBoxIndices = new Set<number>();
  private readonly dynamicBoxIndices = new Set<number>();
  private readonly pressBoxSet = new Set<number>();
  private readonly pendulumByBox = new Map<number, PendulumHammer>();
  private readonly hangarWallIndices = new Set<number>();
  private readonly containerBoxIndices: { idx: number; color: number }[] = [];
  private readonly lampPostIndices = new Set<number>();

  // O(1) CSR Spatial Hash Grids (zero allocations during queries)
  private readonly staticBoxGrid: SpatialHashGrid;
  private readonly staticBoxIds: Int32Array;
  private readonly dynamicBoxIds: Int32Array;
  private readonly cylGrid: SpatialHashGrid;
  private readonly bumpGrid: SpatialHashGrid;

  // Dynamic machinery & prop state
  private simTime = 0;
  private readonly pressBoxIds: number[] = [];
  private readonly pendulums: PendulumHammer[] = [];
  private readonly dynBarriers: DynamicBarrier[] = [];
  private readonly dynCones: DynamicCone[] = [];

  // Batched dynamic InstancedMeshes (all animated obstacles in just 5 draw calls!)
  private dynJerseyMesh: InstancedMesh | null = null;
  private dynJerseyMap: number[] = [];
  private dynGuardrailMesh: InstancedMesh | null = null;
  private dynGuardrailMap: number[] = [];
  private dynConeMesh: InstancedMesh | null = null;
  private dynHazardMachineryMesh: InstancedMesh | null = null; // 2 press heads + 4 pendulum heads
  private dynSteelMachineryMesh: InstancedMesh | null = null;  // 4 press pistons + 4 pendulum arms + 3 crusher rollers
  private rollerSpin = 0;

  // Live Radar Speed Gantry CanvasTexture (zero-alloc throttled redraw)
  private radarCanvas: HTMLCanvasElement | null = null;
  private radarCtx: CanvasRenderingContext2D | null = null;
  private radarTex: CanvasTexture | null = null;
  private lastRadarKmh = -1;

  // Zero-alloc scratch transforms
  private readonly tmpMat = new Matrix4();
  private readonly tmpPos = new Vector3();
  private readonly tmpQuat = new Quaternion();
  private readonly tmpEuler = new Euler();
  private readonly tmpScale = new Vector3(1, 1, 1);

  constructor() {
    // ================================================================
    // ZONE 1: HIGH-SPEED CRASH-TEST LAB (x in [-80, +80], z = -240 -> +260)
    // ================================================================
    // 100% Full Frontal NCAP Wall, 40% Offset Honeycomb, 25% Small Overlap
    this.addBox(0, 1.15, 260, 6.8, 1.15, 0.9, 0, 0.72);
    this.addBox(-18, 0.9, 260, 1.1, 0.9, 1.1, 0, 0.72);
    this.addBox(-32, 0.95, 260, 0.7, 0.95, 1.1, 0, 0.76);

    // High-strength steel poles & urban bollards
    this.addCyl(16, 260, 0.15, 0, 6.8, 0.42);
    for (let i = 0; i < 7; i++) this.addCyl(26.4 + i * 1.2, 260, 0.105, 0, 1.05, 0.5);
    this.addCyl(46, 260, 0.15, 0, 6.8, 0.42);

    // 30-degree angled steel deflection wall
    this.steelBoxIndices.add(this.boxList.length);
    this.addBox(64, 1.2, 260, 3.8, 1.2, 0.5, 0.52, 0.44);

    // Movable concrete Jersey barrier chicane along main runway flank
    for (let i = 0; i < 8; i++) {
      const bx = -7.5 - (i % 4) * 2.8;
      const bz = 150 + i * 5.2;
      const byaw = 0.42;
      const bIdx = this.boxList.length;
      this.dynamicBoxIndices.add(bIdx);
      this.addBox(bx, 0.42, bz, 0.30, 0.42, 1.35, byaw, 0.55);
      this.dynBarriers.push({
        boxIndex: bIdx, x0: bx, z0: bz, yaw0: byaw, x: bx, z: bz, yaw: byaw,
        vx: 0, vz: 0, omega: 0, tilt: 0, vTilt: 0, hx: 0.30, hy: 0.42, hz: 1.35, guardrail: false,
      });
    }

    // 24 Movable construction cones along radar trap & roundabout entry
    for (let i = 0; i < 12; i++) {
      const side = i % 2 === 0 ? -4.2 : 4.2;
      const cz = 110 + Math.floor(i / 2) * 22;
      this.dynCones.push({ x0: side, z0: cz, x: side, y: 0, z: cz, vx: 0, vy: 0, vz: 0, tiltX: 0, tiltZ: 0, vTiltX: 0, vTiltZ: 0 });
    }
    for (let i = 0; i < 12; i++) {
      const cx = 24.5 + (i % 3) * 5.5;
      const cz = 232 + Math.floor(i / 3) * 5.5;
      this.dynCones.push({ x0: cx, z0: cz, x: cx, y: 0, z: cz, vx: 0, vy: 0, vz: 0, tiltX: 0, tiltZ: 0, vTiltX: 0, vTiltZ: 0 });
    }

    // ================================================================
    // ZONE 2: HYDRAULIC PRESS, PENDULUM HAMMERS & CRUSHER ROLLERS (x in [175, 315], z in [90, 210])
    // ================================================================
    // A. Double Heavy Vertical Hydraulic Press at x = 195, z = 148 & z = 162
    for (const pz of [148, 162]) {
      const pIdx = this.boxList.length;
      this.pressBoxIds.push(pIdx);
      this.pressBoxSet.add(pIdx);
      this.dynamicBoxIndices.add(pIdx);
      this.addBox(195, 2.4, pz, 1.55, 0.55, 0.92, 0, 0.68);
      // Gantry support legs (solid steel)
      this.steelBoxIndices.add(this.boxList.length);
      this.addBox(191.6, 2.6, pz, 0.28, 2.6, 0.28, 0, 0.5);
      this.steelBoxIndices.add(this.boxList.length);
      this.addBox(198.4, 2.6, pz, 0.28, 2.6, 0.28, 0, 0.5);
    }

    // B. 4 Synchronized Swinging Pendulum Hammers along x = 245, z = 126..174
    for (let i = 0; i < 4; i++) {
      const pz = 126 + i * 16;
      const bIdx = this.boxList.length;
      this.dynamicBoxIndices.add(bIdx);
      this.addBox(245, 1.15, pz, 0.85, 0.62, 0.72, 0, 0.52);
      const pend: PendulumHammer = {
        boxIndex: bIdx,
        pivotX: 245,
        pivotY: 7.4,
        pivotZ: pz,
        armLen: 6.25,
        phase: i * Math.PI * 0.75, // alternating synchronized gauntlet
        freq: 1.45,
        ampRad: 0.62,
        vx: 0,
        vy: 0,
        theta: 0,
      };
      this.pendulums.push(pend);
      this.pendulumByBox.set(bIdx, pend);
      // Overhead A-frame gantry columns outside lane
      this.steelBoxIndices.add(this.boxList.length);
      this.addBox(238.5, 3.7, pz, 0.3, 3.7, 0.3, 0, 0.5);
      this.steelBoxIndices.add(this.boxList.length);
      this.addBox(251.5, 3.7, pz, 0.3, 3.7, 0.3, 0, 0.5);
    }

    // C. 3 Slow Progressive Crusher Rollers at x = 295, z = 142, 150, 158
    // Stride 7: cx, cy, cz, halfW, radius, surfaceVz, minGapY
    this.rollers = new Float32Array(this.rollerCount * ROLLER_STRIDE);
    const rollerSpecs: [number, number, number, number][] = [
      [142, 2.04, 0.72, 1.08], // Roller 1: bottom at 1.32 m (touches roofline gently)
      [150, 1.88, 0.72, 1.00], // Roller 2: bottom at 1.16 m (progressive A/B/C pillar press)
      [158, 1.76, 0.72, 0.96], // Roller 3: bottom at 1.04 m (cabin cage limit, never touches hood!)
    ];
    for (let r = 0; r < this.rollerCount; r++) {
      const [rz, rcy, rad, minGap] = rollerSpecs[r];
      const ro = r * ROLLER_STRIDE;
      this.rollers[ro] = 295;
      this.rollers[ro + 1] = rcy;
      this.rollers[ro + 2] = rz;
      this.rollers[ro + 3] = 2.6;
      this.rollers[ro + 4] = rad;
      this.rollers[ro + 5] = 1.85; // slow progressive feed speed (m/s)
      this.rollers[ro + 6] = minGap;
    }
    // Side retaining walls for crusher roller bay
    this.steelBoxIndices.add(this.boxList.length);
    this.addBox(291.9, 1.5, 150, 0.35, 1.5, 14, 0, 0.4);
    this.steelBoxIndices.add(this.boxList.length);
    this.addBox(298.1, 1.5, 150, 0.35, 1.5, 14, 0, 0.4);

    // ================================================================
    // ZONE 3: ENDURANCE & SUSPENSION CIRCUIT (West: x in [-360, -160], z in [-240, +285])
    // ================================================================
    // Lane A (x = -180): Steel chatter bars + sinusoidal wave whoops + alternating moguls
    for (let i = 0; i < 14; i++) {
      const z0 = -215 + i * 2.3;
      this.bumps.push({ x0: -183.2, x1: -176.8, z0, z1: z0 + 0.58, h: 0.085, kind: 'sine', color: 0x94a3b8, metal: true });
    }
    for (let i = 0; i < 10; i++) {
      const z0 = -165 + i * 7.2;
      this.bumps.push({ x0: -183.2, x1: -176.8, z0, z1: z0 + 4.2, h: 0.25, kind: 'sine', color: 0xd97706 });
    }
    for (let i = 0; i < 14; i++) {
      const z0 = -75 + i * 5.0;
      const left = i % 2 === 0;
      this.bumps.push({
        x0: left ? -180.1 : -183.2,
        x1: left ? -176.8 : -179.9,
        z0,
        z1: z0 + 2.6,
        h: 0.21,
        kind: 'sine',
        color: left ? 0xeab308 : 0x64748b,
        metal: true,
      });
    }
    for (let i = 0; i < 4; i++) {
      const z0 = 15 + i * 18;
      this.bumps.push({ x0: -183.2, x1: -176.8, z0, z1: z0 + 9.5, h: 0.30 + i * 0.05, kind: 'step', rampLen: 4.6, color: 0x64748b });
    }

    // Lane B (x = -230, z = -220..+60): Irregular Belgian Cobblestones (analytical in groundAt + visual patch)
    this.patches.push({ x0: -234.5, x1: -225.5, z0: -220, z1: 60, mu: 0.86, color: 0x57534e, rough: 0.95, name: 'Cobblestones' });

    // Lane C (x = -280, z = -205..-85): Giant Ascending & Descending Concrete Staircase (10 steps up, deck, 10 steps down)
    // Handled analytically in groundAt + visual step meshes!

    // Lane D (x = -330, z = -220..+60): Diagonal Axle-Articulation Ditches (analytical in groundAt + visual patch)
    this.patches.push({ x0: -334.8, x1: -325.2, z0: -220, z1: 60, mu: 0.88, color: 0x4b5563, rough: 0.92, name: 'Articulation Ditches' });

    // Outer safety wall around the 180° North Parabolic Banked Curve (R = 88.8 m)
    const bankWallSegs = 18;
    for (let s = 0; s < bankWallSegs; s++) {
      const a0 = (Math.PI * s) / bankWallSegs;
      const a1 = (Math.PI * (s + 1)) / bankWallSegs;
      const am = 0.5 * (a0 + a1);
      const Rw = BANK_R_OUT + 0.7;
      const wx = BANK_CX + Math.cos(am) * Rw;
      const wz = BANK_CZ + Math.sin(am) * Rw;
      const segHalfL = Rw * Math.sin((a1 - a0) * 0.52);
      const yaw = Math.atan2(-Math.sin(am), Math.cos(am));
      this.steelBoxIndices.add(this.boxList.length);
      this.addBox(wx, BANK_MAX_H + 0.9, wz, 0.4, 1.3, segHalfL, yaw, 0.35);
    }

    // ================================================================
    // ZONE 4: MEGA-RAMPS, 15°/25°/35° LAUNCHERS & JUMP CANYON (South: z in [-720, -320])
    // ================================================================
    // 1. Calibrated 15°, 25°, 35° launch ramps facing South (yaw = Math.PI)
    this.ramps.push({ x: -48, z: -330, yaw: Math.PI, len: 14.0, w: 6.0, h: 3.75 }); // 15° launch
    this.ramps.push({ x: -48, z: -418, yaw: 0, len: 16.0, w: 7.0, h: 3.4 });        // 15° landing
    this.ramps.push({ x: -24, z: -330, yaw: Math.PI, len: 13.0, w: 6.0, h: 6.06 }); // 25° launch
    this.ramps.push({ x: -24, z: -422, yaw: 0, len: 18.0, w: 7.5, h: 5.2 });        // 25° landing
    this.ramps.push({ x: 24, z: -330, yaw: Math.PI, len: 11.5, w: 6.0, h: 8.05 });  // 35° sky launch
    this.ramps.push({ x: 24, z: -420, yaw: 0, len: 18.0, w: 8.0, h: 6.2 });         // 35° landing

    // 2. Bullseye Target Jump Ramp (x = +68, launching onto concentric Bullseye at z = -430)
    this.ramps.push({ x: 68, z: -330, yaw: Math.PI, len: 13.5, w: 6.0, h: 4.6 });

    // 3. Giant Mega-Ramp + Sunken Canyon Jump (x = 0, z = -495 -> -655)
    this.ramps.push({ x: 0, z: -512, yaw: Math.PI, len: 30.0, w: 12.0, h: 9.4 });   // Mega launch ramp (lip at z = -542)
    this.ramps.push({ x: 0, z: -648, yaw: 0, len: 42.0, w: 15.0, h: 8.8 });         // Wide sloped landing ramp (lip at z = -606)
    // Canyon vertical rock walls at z = -542 and z = -606 so falling short hits canyon wall cleanly
    this.addBox(0, -3.25, CANYON_Z1 + 0.6, 45, 3.25, 0.6, 0, 0.8);
    this.addBox(0, -3.25, CANYON_Z0 - 0.6, 45, 3.25, 0.6, 0, 0.8);

    // Cushion tyre wall south of landing zone
    for (let i = 0; i < 14; i++) this.addCyl(-13 + i * 2.0, -700, 0.42, 0, 1.1, 0.9);

    // ================================================================
    // ZONE 5: URBAN / LIGHT INDUSTRIAL DISTRICT (Southeast: x in [200, 620], z in [-480, -60])
    // ================================================================
    // 1. Large Drift & T-Bone Roundabout at (380, -220): raised curbed island + central steel monument
    this.addCyl(380, -220, 22.0, 0, 0.34, 0.78);
    this.addCyl(380, -220, 2.4, 0, 7.5, 0.5);
    // Lamp posts ringing roundabout outer perimeter (R = 58 m)
    for (let i = 0; i < 12; i++) {
      const a = (i / 12) * Math.PI * 2;
      this.lampPostIndices.add(this.cylList.length);
      this.addCyl(380 + Math.cos(a) * 58, -220 + Math.sin(a) * 58, 0.12, 0, 6.4, 0.42);
    }

    // 2. Deformable / Yieldable W-Beam Highway Guardrails lining East Avenue (z = -200..-240, x = 140..310)
    for (let i = 0; i < 12; i++) {
      const gx = 150 + i * 13.5;
      for (const gz of [-196, -244]) {
        const bIdx = this.boxList.length;
        this.dynamicBoxIndices.add(bIdx);
        this.addBox(gx, 0.48, gz, 6.2, 0.48, 0.18, 0, 0.45);
        this.dynBarriers.push({
          boxIndex: bIdx, x0: gx, z0: gz, yaw0: 0, x: gx, z: gz, yaw: 0,
          vx: 0, vz: 0, omega: 0, tilt: 0, vTilt: 0, hx: 6.2, hy: 0.48, hz: 0.18, guardrail: true,
        });
      }
    }

    // 3. 2 Drive-Through Industrial Hangars (open North/South portals, solid left/right walls)
    const addDriveThroughHangar = (hx: number, hz: number, width: number, height: number, length: number) => {
      const wallThick = 0.65;
      for (const side of [-1, 1]) {
        this.hangarWallIndices.add(this.boxList.length);
        this.addBox(hx + side * (width * 0.5), height * 0.5, hz, wallThick, height * 0.5, length * 0.5, 0, 0.78);
      }
    };
    addDriveThroughHangar(260, -370, 18, 8.2, 54);
    addDriveThroughHangar(500, -370, 18, 8.2, 54);

    // 4. 28 Stacked Shipping Containers (x in [335, 455], z in [-445, -330])
    const containerPalette = [0x1d4ed8, 0xb91c1c, 0x15803d, 0xd97706, 0x334155];
    const containerPlacements: [number, number, number, number, number][] = [
      // Chicane corridor (ground level + second-tier stacks)
      [355, 1.3, -345, 0, containerPalette[0]],
      [355, 3.9, -345, 0, containerPalette[1]],
      [372, 1.3, -362, 0.18, containerPalette[2]],
      [372, 3.9, -362, 0.18, containerPalette[3]],
      [352, 1.3, -382, -0.15, containerPalette[4]],
      [376, 1.3, -402, 0, containerPalette[1]],
      [376, 3.9, -402, 0, containerPalette[0]],
      [355, 1.3, -422, 0.22, containerPalette[3]],
      // Container yard stacks
      [410, 1.3, -350, Math.PI / 2, containerPalette[2]],
      [410, 3.9, -350, Math.PI / 2, containerPalette[0]],
      [410, 1.3, -356, Math.PI / 2, containerPalette[1]],
      [410, 1.3, -385, 0, containerPalette[4]],
      [410, 3.9, -385, 0, containerPalette[3]],
      [416, 1.3, -385, 0, containerPalette[0]],
      [442, 1.3, -365, 0.35, containerPalette[1]],
      [442, 3.9, -365, 0.35, containerPalette[2]],
      [442, 1.3, -405, -0.28, containerPalette[3]],
      [442, 3.9, -405, -0.28, containerPalette[4]],
      [392, 1.3, -432, Math.PI / 2, containerPalette[0]],
      [392, 3.9, -432, Math.PI / 2, containerPalette[1]],
      [428, 1.3, -432, Math.PI / 2, containerPalette[2]],
      [428, 3.9, -432, Math.PI / 2, containerPalette[3]],
      [332, 1.3, -370, 0, containerPalette[0]],
      [332, 1.3, -398, 0, containerPalette[2]],
      [466, 1.3, -370, 0, containerPalette[4]],
      [466, 3.9, -370, 0, containerPalette[1]],
      [466, 1.3, -398, 0, containerPalette[3]],
      [466, 1.3, -426, 0, containerPalette[0]],
    ];
    for (const [cx, cy, cz, cyaw, col] of containerPlacements) {
      const idx = this.boxList.length;
      this.containerBoxIndices.push({ idx, color: col });
      this.addBox(cx, cy, cz, 1.25, 1.3, 6.1, cyaw, 0.62);
    }

    // Roadside lamp posts along central 500m Crash-Test Lab avenue
    for (let i = 0; i < 14; i++) {
      this.lampPostIndices.add(this.cylList.length);
      this.addCyl(86, -210 + i * 34, 0.12, 0, 6.4, 0.42);
    }

    // Surface friction test patches (Wet Asphalt, Gravel, Ice)
    this.patches.push({ x0: 18, x1: 42, z0: -110, z1: -30, mu: 0.52, color: 0x1e293b, rough: 0.14, name: 'Wet asphalt' });
    this.patches.push({ x0: -52, x1: -24, z0: -110, z1: -40, mu: 0.58, color: 0x78716c, rough: 0.98, name: 'Gravel' });
    this.patches.push({ x0: 52, x1: 112, z0: -120, z1: -40, mu: 0.15, color: 0xcbd5e1, rough: 0.08, name: 'Ice' });

    // ================================================================
    // 2,000 m × 2,000 m PERIMETER BOUNDARY WALLS
    // ================================================================
    const L = this.asphaltHalf;
    this.addBox(0, 1.4, L + 1, L + 2, 1.4, 1.0, 0, 0.6);
    this.addBox(0, 1.4, -L - 1, L + 2, 1.4, 1.0, 0, 0.6);
    this.addBox(L + 1, 1.4, 0, 1.0, 1.4, L + 2, 0, 0.6);
    this.addBox(-L - 1, 1.4, 0, 1.0, 1.4, L + 2, 0, 0.6);

    // Pack contiguous Float32Array collision buffers
    this.boxes = new Float32Array(this.boxList.length * BOX_STRIDE);
    for (let i = 0; i < this.boxList.length; i++) this.boxes.set(this.boxList[i], i * BOX_STRIDE);
    this.boxCount = this.boxList.length;

    this.cyls = new Float32Array(this.cylList.length * CYL_STRIDE);
    for (let i = 0; i < this.cylList.length; i++) this.cyls.set(this.cylList[i], i * CYL_STRIDE);
    this.cylCount = this.cylList.length;

    // Build O(1) CSR Spatial Hash Grids for boxes, cylinders, and bumps
    const staticBounds: SpatialBoundsItem[] = [];
    const staticIds: number[] = [];
    const dynIds: number[] = [];
    for (let k = 0; k < this.boxCount; k++) {
      if (this.dynamicBoxIndices.has(k)) {
        dynIds.push(k);
      } else {
        const o = k * BOX_STRIDE;
        staticIds.push(k);
        staticBounds.push({
          minX: this.boxes[o + 9],
          maxX: this.boxes[o + 10],
          minZ: this.boxes[o + 11],
          maxZ: this.boxes[o + 12],
        });
      }
    }
    this.staticBoxIds = new Int32Array(staticIds);
    this.dynamicBoxIds = new Int32Array(dynIds);
    this.staticBoxGrid = new SpatialHashGrid(staticBounds, 16, 1152);

    const cylBounds: SpatialBoundsItem[] = [];
    for (let k = 0; k < this.cylCount; k++) {
      const o = k * CYL_STRIDE;
      const r = this.cyls[o + 2];
      cylBounds.push({
        minX: this.cyls[o] - r,
        maxX: this.cyls[o] + r,
        minZ: this.cyls[o + 1] - r,
        maxZ: this.cyls[o + 1] + r,
      });
    }
    this.cylGrid = new SpatialHashGrid(cylBounds, 16, 1152);

    const bumpBounds: SpatialBoundsItem[] = this.bumps.map((b) => ({
      minX: b.x0, maxX: b.x1, minZ: b.z0, maxZ: b.z1,
    }));
    this.bumpGrid = new SpatialHashGrid(bumpBounds, 8, 1152);
  }

  private addBox(cx: number, cy: number, cz: number, hx: number, hy: number, hz: number, yaw: number, mu: number): void {
    const c = Math.cos(yaw), s = Math.sin(yaw);
    const ex = Math.abs(c) * hx + Math.abs(s) * hz;
    const ez = Math.abs(s) * hx + Math.abs(c) * hz;
    this.boxList.push([cx, cy, cz, hx, hy, hz, c, s, mu, cx - ex, cx + ex, cz - ez, cz + ez]);
  }

  private addCyl(cx: number, cz: number, r: number, y0: number, y1: number, mu: number): void {
    this.cylList.push([cx, cz, r, y0, y1, mu]);
  }

  // ================================================================ O(1) CollisionWorld hooks

  isPressBox(boxIdx: number): boolean {
    return this.pressBoxSet.has(boxIdx);
  }

  getBoxVelocity(boxIdx: number, outVel: Float32Array): boolean {
    const pend = this.pendulumByBox.get(boxIdx);
    if (pend) {
      outVel[0] = pend.vx;
      outVel[1] = pend.vy;
      outVel[2] = 0;
      return true;
    }
    return false;
  }

  private readonly gridTmp = new Int32Array(128);

  queryBoxes(minX: number, minZ: number, maxX: number, maxZ: number, out: Int32Array, maxOut: number): number {
    const statFound = this.staticBoxGrid.queryAABB(minX, minZ, maxX, maxZ, this.gridTmp, Math.min(128, maxOut));
    let count = 0;
    for (let i = 0; i < statFound && count < maxOut; i++) {
      out[count++] = this.staticBoxIds[this.gridTmp[i]];
    }
    const B = this.boxes;
    const dyn = this.dynamicBoxIds;
    for (let i = 0; i < dyn.length && count < maxOut; i++) {
      const k = dyn[i];
      const o = k * BOX_STRIDE;
      if (B[o + 10] < minX || B[o + 9] > maxX || B[o + 12] < minZ || B[o + 11] > maxZ) continue;
      out[count++] = k;
    }
    return count;
  }

  queryCyls(minX: number, minZ: number, maxX: number, maxZ: number, out: Int32Array, maxOut: number): number {
    return this.cylGrid.queryAABB(minX, minZ, maxX, maxZ, out, maxOut);
  }

  // ================================================================ Dynamic obstacles & Radar

  resetDynamic(): void {
    for (let i = 0; i < this.dynBarriers.length; i++) {
      const b = this.dynBarriers[i];
      b.x = b.x0; b.z = b.z0; b.yaw = b.yaw0;
      b.vx = 0; b.vz = 0; b.omega = 0;
      b.tilt = 0; b.vTilt = 0;
      this.syncBarrierBox(b);
    }
    for (let i = 0; i < this.dynCones.length; i++) {
      const c = this.dynCones[i];
      c.x = c.x0; c.y = 0; c.z = c.z0;
      c.vx = 0; c.vy = 0; c.vz = 0;
      c.tiltX = 0; c.tiltZ = 0; c.vTiltX = 0; c.vTiltZ = 0;
    }
    this.syncDynamicVisuals(true, true);
  }

  private syncBarrierBox(b: DynamicBarrier): void {
    const o = b.boxIndex * BOX_STRIDE;
    const c = Math.cos(b.yaw), s = Math.sin(b.yaw);
    const ex = Math.abs(c) * b.hx + Math.abs(s) * b.hz;
    const ez = Math.abs(s) * b.hx + Math.abs(c) * b.hz;
    const B = this.boxes;
    B[o] = b.x;
    B[o + 1] = b.hy * (1 - Math.abs(Math.sin(b.tilt)) * 0.35);
    B[o + 2] = b.z;
    B[o + 6] = c;
    B[o + 7] = s;
    B[o + 9] = b.x - ex;
    B[o + 10] = b.x + ex;
    B[o + 11] = b.z - ez;
    B[o + 12] = b.z + ez;
  }

  /** Updates the live LED speed radar gantries on the 500m runway (throttled integer redraw, zero allocations). */
  updateRadarDisplay(carX: number, carZ: number, speedKmh: number): void {
    if (!this.radarCtx || !this.radarTex || !this.radarCanvas) return;
    if (Math.abs(carX) > 42 || carZ < -250 || carZ > 270) return;
    const rounded = Math.min(999, Math.max(0, Math.round(speedKmh)));
    if (rounded === this.lastRadarKmh) return;
    this.lastRadarKmh = rounded;

    const ctx = this.radarCtx;
    ctx.fillStyle = '#090d16';
    ctx.fillRect(0, 0, 256, 64);
    ctx.strokeStyle = rounded >= 150 ? '#ef4444' : rounded >= 90 ? '#f59e0b' : '#38bdf8';
    ctx.lineWidth = 3;
    ctx.strokeRect(3, 3, 250, 58);
    ctx.fillStyle = rounded >= 150 ? '#fca5a5' : rounded >= 90 ? '#fde047' : '#7dd3fc';
    ctx.font = '900 36px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${rounded} KM/H`, 128, 34);
    this.radarTex.needsUpdate = true;
  }

  /** Steps dynamic machinery & props with distant-zone sleeping. Strictly zero heap allocations. */
  updateDynamic(dt: number, solver: SoftBodySolver): void {
    this.simTime += dt;
    const px = solver.px, py = solver.py, pz = solver.pz;
    const vx = solver.vx, vz = solver.vz;
    const nCount = solver.nodeCount;
    const carX = nCount > 0 ? px[0] : 0;
    const carZ = nCount > 0 ? pz[0] : 0;

    // 1. Zone 2 Heavy Machinery (Double Press + 4 Pendulum Hammers + 3 Crusher Rollers)
    //    Active when vehicle is within 240 m of Zone 2 (or on initial frame).
    const nearMachinery = (carX - 245) * (carX - 245) + (carZ - 152) * (carZ - 152) < 240 * 240;
    if (nearMachinery || this.simTime < 0.1) {
      // Double Hydraulic Press (staggered phase)
      for (let p = 0; p < this.pressBoxIds.length; p++) {
        const bIdx = this.pressBoxIds[p];
        const wave = 0.5 + 0.5 * Math.sin(this.simTime * 0.9 + p * 1.3);
        const pressCy = 1.55 + wave * 1.50; // bottom oscillates between 1.00 m and 2.50 m
        this.boxes[bIdx * BOX_STRIDE + 1] = pressCy;
      }

      // 4 Synchronized Pendulum Hammers
      for (let i = 0; i < this.pendulums.length; i++) {
        const P = this.pendulums[i];
        const arg = this.simTime * P.freq + P.phase;
        const theta = Math.sin(arg) * P.ampRad;
        const dThetaDt = Math.cos(arg) * P.freq * P.ampRad;
        P.theta = theta;
        const s = Math.sin(theta), c = Math.cos(theta);
        const hx = P.pivotX + P.armLen * s;
        const hy = P.pivotY - P.armLen * c;
        P.vx = P.armLen * c * dThetaDt;
        P.vy = P.armLen * s * dThetaDt;

        const o = P.boxIndex * BOX_STRIDE;
        const B = this.boxes;
        B[o] = hx;
        B[o + 1] = hy;
        B[o + 9] = hx - B[o + 3];
        B[o + 10] = hx + B[o + 3];
      }

      this.rollerSpin += dt * 2.55;
      this.syncMachineryVisuals();
    }

    // 2. Movable Concrete Jersey Barriers & Deformable W-Beam Highway Guardrails
    let barriersDirty = false;
    for (let i = 0; i < this.dynBarriers.length; i++) {
      const b = this.dynBarriers[i];
      const dCarX = b.x - carX, dCarZ = b.z - carZ;
      const nearCar = dCarX * dCarX + dCarZ * dCarZ < (b.guardrail ? 16 * 16 : 10 * 10);

      if (nearCar) {
        const cosY = Math.cos(b.yaw), sinY = Math.sin(b.yaw);
        const padX = b.hx + 0.45, padZ = b.hz + 0.45;
        for (let n = 0; n < nCount; n += 2) {
          if (py[n] > 1.15) continue;
          const rx = px[n] - b.x, rz = pz[n] - b.z;
          const lx = rx * cosY - rz * sinY;
          const lz = rx * sinY + rz * cosY;
          if (Math.abs(lx) > padX || Math.abs(lz) > padZ) continue;
          const rvx = vx[n] - b.vx, rvz = vz[n] - b.vz;
          const spd = Math.sqrt(rvx * rvx + rvz * rvz);
          if (spd > 1.4) {
            const dx = b.x - px[n], dz = b.z - pz[n];
            const invD = 1 / (Math.sqrt(dx * dx + dz * dz) || 1);
            const nx = dx * invD, nz = dz * invD;
            const vn = rvx * nx + rvz * nz;
            if (vn > 0.5) {
              const impScale = b.guardrail ? 0.14 : 0.24;
              const imp = Math.min(b.guardrail ? 3.2 : 5.0, vn * impScale);
              b.vx += nx * imp;
              b.vz += nz * imp;
              b.omega += (dx * rvz - dz * rvx) * (b.guardrail ? 0.012 : 0.042);
              b.vTilt += imp * (b.guardrail ? 0.55 : 0.40);
            }
          }
        }
      }

      if (Math.abs(b.vx) > 0.01 || Math.abs(b.vz) > 0.01 || Math.abs(b.omega) > 0.004 || (!b.guardrail && Math.abs(b.tilt) > 0.005) || Math.abs(b.vTilt) > 0.005) {
        b.x += b.vx * dt;
        b.z += b.vz * dt;
        b.yaw += b.omega * dt;
        // Guardrails have post-yield ground anchor resistance; Jersey blocks slide with Coulomb drag
        if (b.guardrail) {
          b.vx += (b.x0 - b.x) * 1.4 * dt;
          b.vz += (b.z0 - b.z) * 1.4 * dt;
        }
        const drag = Math.exp((b.guardrail ? -5.5 : -3.6) * dt);
        b.vx *= drag;
        b.vz *= drag;
        b.omega *= Math.exp(-4.5 * dt);
        if (!b.guardrail && Math.abs(b.tilt) < 0.72) {
          b.vTilt -= b.tilt * 14 * dt;
        }
        b.vTilt *= Math.exp(-4.8 * dt);
        b.tilt = Math.max(-1.35, Math.min(1.35, b.tilt + b.vTilt * dt));
        this.syncBarrierBox(b);
        barriersDirty = true;
      }
    }

    // 3. Movable Construction Cones (with distant sleeping)
    let conesDirty = false;
    for (let i = 0; i < this.dynCones.length; i++) {
      const c = this.dynCones[i];
      const dCarX = c.x - carX, dCarZ = c.z - carZ;
      if (dCarX * dCarX + dCarZ * dCarZ < 64) {
        for (let n = 0; n < nCount; n += 3) {
          if (py[n] > 0.85) continue;
          const dx = c.x - px[n], dz = c.z - pz[n];
          const d2 = dx * dx + dz * dz;
          if (d2 < 0.55 * 0.55 && d2 > 1e-4) {
            const invD = 1 / Math.sqrt(d2);
            const nx = dx * invD, nz = dz * invD;
            const spd = Math.sqrt(vx[n] * vx[n] + vz[n] * vz[n]);
            const kick = Math.max(2.2, Math.min(15, spd * 0.85));
            c.vx = nx * kick + vx[n] * 0.35;
            c.vz = nz * kick + vz[n] * 0.35;
            c.vy = Math.max(c.vy, Math.min(4.5, kick * 0.32));
            c.vTiltX += nz * kick * 1.4;
            c.vTiltZ -= nx * kick * 1.4;
            break;
          }
        }
      }
      if (c.y > 0 || Math.abs(c.vx) > 0.01 || Math.abs(c.vy) > 0.01 || Math.abs(c.vz) > 0.01 || Math.abs(c.vTiltX) > 0.01 || Math.abs(c.vTiltZ) > 0.01) {
        c.vy -= 9.81 * dt;
        c.x += c.vx * dt;
        c.y += c.vy * dt;
        c.z += c.vz * dt;
        if (c.y < 0) {
          c.y = 0;
          c.vy = Math.abs(c.vy) > 0.6 ? -c.vy * 0.32 : 0;
          const gDrag = Math.exp(-5.2 * dt);
          c.vx *= gDrag; c.vz *= gDrag; c.vTiltX *= gDrag; c.vTiltZ *= gDrag;
        }
        c.tiltX = Math.max(-1.55, Math.min(1.55, c.tiltX + c.vTiltX * dt));
        c.tiltZ = Math.max(-1.55, Math.min(1.55, c.tiltZ + c.vTiltZ * dt));
        conesDirty = true;
      }
    }

    if (barriersDirty || conesDirty) {
      this.syncDynamicVisuals(barriersDirty, conesDirty);
    }
  }

  private syncMachineryVisuals(): void {
    if (this.dynHazardMachineryMesh) {
      let idx = 0;
      // 2 Press heads
      for (let p = 0; p < this.pressBoxIds.length; p++) {
        const o = this.pressBoxIds[p] * BOX_STRIDE;
        this.tmpPos.set(this.boxes[o], this.boxes[o + 1], this.boxes[o + 2]);
        this.tmpQuat.identity();
        this.tmpScale.set(3.1, 1.1, 1.84);
        this.tmpMat.compose(this.tmpPos, this.tmpQuat, this.tmpScale);
        this.dynHazardMachineryMesh.setMatrixAt(idx++, this.tmpMat);
      }
      // 4 Pendulum hammer heads
      for (let i = 0; i < this.pendulums.length; i++) {
        const P = this.pendulums[i];
        const o = P.boxIndex * BOX_STRIDE;
        this.tmpPos.set(this.boxes[o], this.boxes[o + 1], this.boxes[o + 2]);
        this.tmpEuler.set(0, 0, P.theta);
        this.tmpQuat.setFromEuler(this.tmpEuler);
        this.tmpScale.set(1.7, 1.24, 1.44);
        this.tmpMat.compose(this.tmpPos, this.tmpQuat, this.tmpScale);
        this.dynHazardMachineryMesh.setMatrixAt(idx++, this.tmpMat);
      }
      this.dynHazardMachineryMesh.instanceMatrix.needsUpdate = true;
    }

    if (this.dynSteelMachineryMesh) {
      let idx = 0;
      // 4 Press hydraulic pistons (2 per press)
      for (let p = 0; p < this.pressBoxIds.length; p++) {
        const o = this.pressBoxIds[p] * BOX_STRIDE;
        const pressCy = this.boxes[o + 1];
        const pz = this.boxes[o + 2];
        const rodH = Math.max(0.2, 5.1 - (pressCy + 0.55));
        const rodY = pressCy + 0.55 + rodH * 0.5;
        for (const dx of [-0.85, 0.85]) {
          this.tmpPos.set(195 + dx, rodY, pz);
          this.tmpQuat.identity();
          this.tmpScale.set(0.24, rodH, 0.24);
          this.tmpMat.compose(this.tmpPos, this.tmpQuat, this.tmpScale);
          this.dynSteelMachineryMesh.setMatrixAt(idx++, this.tmpMat);
        }
      }
      // 4 Pendulum swing arms
      for (let i = 0; i < this.pendulums.length; i++) {
        const P = this.pendulums[i];
        const midX = P.pivotX + Math.sin(P.theta) * (P.armLen * 0.5);
        const midY = P.pivotY - Math.cos(P.theta) * (P.armLen * 0.5);
        this.tmpPos.set(midX, midY, P.pivotZ);
        this.tmpEuler.set(0, 0, P.theta);
        this.tmpQuat.setFromEuler(this.tmpEuler);
        this.tmpScale.set(0.16, P.armLen, 0.16);
        this.tmpMat.compose(this.tmpPos, this.tmpQuat, this.tmpScale);
        this.dynSteelMachineryMesh.setMatrixAt(idx++, this.tmpMat);
      }
      // 3 Rotating Crusher Rollers (horizontal cylinders oriented along X)
      for (let r = 0; r < this.rollerCount; r++) {
        const ro = r * ROLLER_STRIDE;
        this.tmpPos.set(this.rollers[ro], this.rollers[ro + 1], this.rollers[ro + 2]);
        this.tmpEuler.set(this.rollerSpin, 0, Math.PI / 2, 'ZYX');
        this.tmpQuat.setFromEuler(this.tmpEuler);
        const d = this.rollers[ro + 4] * 2;
        this.tmpScale.set(d, this.rollers[ro + 3] * 2, d);
        this.tmpMat.compose(this.tmpPos, this.tmpQuat, this.tmpScale);
        this.dynSteelMachineryMesh.setMatrixAt(idx++, this.tmpMat);
      }
      this.dynSteelMachineryMesh.instanceMatrix.needsUpdate = true;
    }
  }

  private syncDynamicVisuals(updateBarriers = true, updateCones = true): void {
    if (updateBarriers) {
      if (this.dynJerseyMesh) {
        for (let k = 0; k < this.dynJerseyMap.length; k++) {
          const b = this.dynBarriers[this.dynJerseyMap[k]];
          this.tmpPos.set(b.x, b.hy * (1 - Math.abs(Math.sin(b.tilt)) * 0.35), b.z);
          this.tmpEuler.set(0, b.yaw, b.tilt, 'YXZ');
          this.tmpQuat.setFromEuler(this.tmpEuler);
          this.tmpScale.set(1, 1, 1);
          this.tmpMat.compose(this.tmpPos, this.tmpQuat, this.tmpScale);
          this.dynJerseyMesh.setMatrixAt(k, this.tmpMat);
        }
        this.dynJerseyMesh.instanceMatrix.needsUpdate = true;
      }
      if (this.dynGuardrailMesh) {
        for (let k = 0; k < this.dynGuardrailMap.length; k++) {
          const b = this.dynBarriers[this.dynGuardrailMap[k]];
          this.tmpPos.set(b.x, b.hy, b.z);
          this.tmpEuler.set(b.tilt * 0.65, b.yaw, 0, 'YXZ');
          this.tmpQuat.setFromEuler(this.tmpEuler);
          this.tmpScale.set(1, 1, 1);
          this.tmpMat.compose(this.tmpPos, this.tmpQuat, this.tmpScale);
          this.dynGuardrailMesh.setMatrixAt(k, this.tmpMat);
        }
        this.dynGuardrailMesh.instanceMatrix.needsUpdate = true;
      }
    }
    if (updateCones && this.dynConeMesh) {
      this.tmpScale.set(1, 1, 1);
      for (let i = 0; i < this.dynCones.length; i++) {
        const c = this.dynCones[i];
        this.tmpPos.set(c.x, c.y + 0.34 * Math.cos(Math.min(1.4, Math.hypot(c.tiltX, c.tiltZ))) + 0.08, c.z);
        this.tmpEuler.set(c.tiltX, 0, c.tiltZ, 'XYZ');
        this.tmpQuat.setFromEuler(this.tmpEuler);
        this.tmpMat.compose(this.tmpPos, this.tmpQuat, this.tmpScale);
        this.dynConeMesh.setMatrixAt(i, this.tmpMat);
      }
      this.dynConeMesh.instanceMatrix.needsUpdate = true;
    }
  }

  // ================================================================ O(1) Analytical + Spatial-Hashed Terrain (`groundAt`)

  groundAt(x: number, z: number, out: GroundSample): void {
    out.h = 0; out.nx = 0; out.ny = 1; out.nz = 0;
    const L = this.asphaltHalf;
    out.mu = x > L || x < -L || z > L || z < -L ? 0.55 : 1.0;

    // 1. Sunken Jump Canyon (South Zone 4: x in [-45, +45], z in [-606, -542])
    if (z > CANYON_Z0 && z < CANYON_Z1 && x > CANYON_X0 && x < CANYON_X1) {
      out.h = CANYON_FLOOR_Y;
      out.mu = 0.72;
      return;
    }

    // 2. Surface friction patches
    for (let i = 0; i < this.patches.length; i++) {
      const p = this.patches[i];
      if (x >= p.x0 && x <= p.x1 && z >= p.z0 && z <= p.z1) { out.mu = p.mu; break; }
    }

    // 3. Zone 3 Endurance Circuit specialized analytical surfaces (West: x in [-350, -165])
    if (x >= -350 && x <= -165) {
      // 3a. 180° North Parabolic High-Speed Banked Curve (z >= BANK_CZ)
      if (z >= BANK_CZ) {
        const rx = x - BANK_CX, rz = z - BANK_CZ;
        const r2 = rx * rx + rz * rz;
        if (r2 >= (BANK_R_IN - 2) * (BANK_R_IN - 2) && r2 <= (BANK_R_OUT + 1.5) * (BANK_R_OUT + 1.5)) {
          const r = Math.sqrt(r2) || 1;
          const u = Math.max(0, Math.min(1, (r - BANK_R_IN) / (BANK_R_OUT - BANK_R_IN)));
          const zBlend = z < BANK_CZ + 12 ? (z - BANK_CZ) / 12 : 1;
          const sBlend = zBlend * zBlend * (3 - 2 * zBlend);
          const h = BANK_MAX_H * u * u * sBlend;
          const dhdr = ((2 * BANK_MAX_H * u) / (BANK_R_OUT - BANK_R_IN)) * sBlend;
          const inv = 1 / Math.sqrt(1 + dhdr * dhdr);
          out.h = h;
          out.nx = -(rx / r) * dhdr * inv;
          out.ny = inv;
          out.nz = -(rz / r) * dhdr * inv;
          out.mu = 1.08;
          return;
        }
      }

      // 3b. Irregular Belgian Cobblestones (Lane B: x in [-234.5, -225.5], z in [-220, +60])
      if (x >= -234.5 && x <= -225.5 && z >= -220 && z <= 60) {
        const edgeFade = Math.min(1, (z + 220) * 0.25, (60 - z) * 0.25);
        const kx1 = 9.2, kz1 = 10.5;
        const s1 = Math.sin(x * kx1), c1 = Math.cos(x * kx1);
        const s2 = Math.sin(z * kz1), c2 = Math.cos(z * kz1);
        const s3 = Math.sin((x + z) * 5.3), c3 = Math.cos((x + z) * 5.3);
        const rawH = 0.036 * s1 * c2 + 0.022 * s3 + 0.038;
        const dhdx = edgeFade * (0.036 * kx1 * c1 * c2 + 0.022 * 5.3 * c3);
        const dhdz = edgeFade * (-0.036 * kz1 * s1 * s2 + 0.022 * 5.3 * c3);
        const inv = 1 / Math.sqrt(1 + dhdx * dhdx + dhdz * dhdz);
        out.h = rawH * edgeFade;
        out.nx = -dhdx * inv;
        out.ny = inv;
        out.nz = -dhdz * inv;
        out.mu = 0.86;
        return;
      }

      // 3c. Giant Ascending & Descending Concrete Staircase (Lane C: x in [-284, -276], z in [-200, -96])
      if (x >= -284 && x <= -276 && z >= -200 && z <= -96) {
        const stepRun = 3.6;
        const stepRise = 0.16;
        const stepsCount = 10; // 36m up + 32m flat deck + 36m down = 104m total
        const relZ = z - (-200);
        let h = 0, dhdz = 0;
        if (relZ < stepsCount * stepRun) {
          const idx = Math.floor(relZ / stepRun);
          const frac = (relZ - idx * stepRun) / stepRun;
          // Beveled step riser over first 0.55m of each step so tyres climb/impact crisply without infinite wall lock
          const rampFrac = Math.min(1, frac / 0.22);
          h = (idx + rampFrac) * stepRise;
          dhdz = frac < 0.22 ? stepRise / (stepRun * 0.22) : 0;
        } else if (relZ <= stepsCount * stepRun + 32) {
          h = stepsCount * stepRise;
        } else {
          const downZ = (-96) - z;
          const idx = Math.max(0, Math.floor(downZ / stepRun));
          const frac = (downZ - idx * stepRun) / stepRun;
          const rampFrac = Math.min(1, frac / 0.22);
          h = (idx + rampFrac) * stepRise;
          dhdz = frac < 0.22 ? -stepRise / (stepRun * 0.22) : 0;
        }
        const inv = 1 / Math.sqrt(1 + dhdz * dhdz);
        out.h = h;
        out.nx = 0; out.ny = inv; out.nz = -dhdz * inv;
        out.mu = 0.92;
        return;
      }

      // 3d. Diagonal Axle-Articulation Ditches (Lane D: x in [-334.8, -325.2], z in [-215, +55])
      if (x >= -334.8 && x <= -325.2 && z >= -215 && z <= 55) {
        const sideSign = x > -330 ? 1 : -1;
        const phase = (z + 215) * 0.34 + (sideSign > 0 ? 0 : Math.PI) + (x + 330) * 0.45;
        const wave = Math.sin(phase);
        if (wave > 0) {
          const depth = -0.34 * wave * wave;
          const dPhase = -0.68 * wave * Math.cos(phase);
          const dhdx = dPhase * 0.45;
          const dhdz = dPhase * 0.34;
          const inv = 1 / Math.sqrt(1 + dhdx * dhdx + dhdz * dhdz);
          out.h = depth;
          out.nx = -dhdx * inv;
          out.ny = inv;
          out.nz = -dhdz * inv;
          out.mu = 0.88;
          return;
        }
      }

      // 3e. O(1) Spatial-hashed Bumps & Chatter Bars (Lane A: x in [-184, -176])
      const cell = this.bumpGrid.cellAt(x, z);
      if (cell >= 0) {
        const start = this.bumpGrid.cellStart[cell];
        const end = start + this.bumpGrid.cellCount[cell];
        for (let k = start; k < end; k++) {
          const b = this.bumps[this.bumpGrid.cellItems[k]];
          if (x < b.x0 || x > b.x1 || z < b.z0 || z > b.z1) continue;
          if (b.kind === 'sine') {
            const u = (z - b.z0) / (b.z1 - b.z0);
            const h = b.h * 0.5 * (1 - Math.cos(2 * Math.PI * u));
            if (h > out.h) {
              const dhdz = (b.h * Math.PI / (b.z1 - b.z0)) * Math.sin(2 * Math.PI * u);
              const inv = 1 / Math.sqrt(1 + dhdz * dhdz);
              out.h = h; out.nx = 0; out.ny = inv; out.nz = -dhdz * inv; out.mu = 0.92;
            }
          } else {
            const rLen = b.rampLen ?? 4.0;
            const dz = z - b.z0;
            if (dz < rLen) {
              const slope = b.h / rLen;
              const h = dz * slope;
              if (h > out.h) {
                const inv = 1 / Math.sqrt(1 + slope * slope);
                out.h = h; out.nx = 0; out.ny = inv; out.nz = -slope * inv; out.mu = b.metal ? 0.78 : 0.92;
              }
            } else if (b.h > out.h) {
              out.h = b.h; out.nx = 0; out.ny = 1; out.nz = 0; out.mu = b.metal ? 0.78 : 0.92;
            }
          }
        }
      }
    }

    // 4. Launch Ramps & Landing Ramps
    for (let i = 0; i < this.ramps.length; i++) {
      const r = this.ramps[i];
      const dx = x - r.x, dz = z - r.z;
      const c = Math.cos(r.yaw), s = Math.sin(r.yaw);
      const along = dx * s + dz * c;
      const lat = dx * c - dz * s;
      if (along < 0 || along > r.len || lat < -r.w * 0.5 || lat > r.w * 0.5) continue;
      const slope = r.h / r.len;
      const h = (r.baseY ?? 0) + along * slope;
      if (h > out.h) {
        out.h = h;
        const inv = 1 / Math.sqrt(1 + slope * slope);
        out.nx = -s * slope * inv; out.ny = inv; out.nz = -c * slope * inv;
        out.mu = 0.92;
      }
    }
  }

  // ================================================================ Ultra-Batched GPU Visuals (< 30 arena draw calls!)

  buildVisuals(lowSpec: boolean): Group {
    const g = this.group;
    const texSize = lowSpec ? 256 : 512;

    // 1. 2,000 m × 2,000 m HD Procedural Asphalt Ground (1 draw call)
    const asphaltPack = getAsphaltTextures(lowSpec);
    asphaltPack.map.repeat.set(360, 360);
    asphaltPack.normalMap.repeat.set(360, 360);
    asphaltPack.roughnessMap.repeat.set(360, 360);
    const asphalt = new Mesh(
      new PlaneGeometry(this.asphaltHalf * 2, this.asphaltHalf * 2),
      new MeshStandardMaterial({
        map: asphaltPack.map,
        normalMap: asphaltPack.normalMap,
        normalScale: new Vector2(0.45, 0.45),
        roughnessMap: asphaltPack.roughnessMap,
        roughness: 0.94,
        metalness: 0.02,
        envMapIntensity: 0.22,
      }),
    );
    asphalt.rotation.x = -Math.PI / 2;
    asphalt.receiveShadow = true;
    g.add(asphalt);

    // 2. Outer Horizon Terrain Skirt (3,800 m × 3,800 m — 1 draw call)
    const grassTex = new CanvasTexture(makeNoiseCanvas(texSize / 2, [70, 76, 82], 14));
    grassTex.wrapS = grassTex.wrapT = RepeatWrapping;
    grassTex.repeat.set(240, 240);
    grassTex.colorSpace = SRGBColorSpace;
    const grass = new Mesh(
      new PlaneGeometry(3800, 3800),
      new MeshStandardMaterial({ map: grassTex, roughness: 1, metalness: 0, envMapIntensity: 0.04 }),
    );
    grass.rotation.x = -Math.PI / 2;
    grass.position.y = -6.55;
    g.add(grass);

    // Prepare geometry merge buckets (each bucket becomes a SINGLE merged Mesh!)
    const whiteStripGeos: BufferGeometry[] = [];
    const yellowStripGeos: BufferGeometry[] = [];
    const concreteGeos: BufferGeometry[] = [];
    const hazardGeos: BufferGeometry[] = [];
    const steelGeos: BufferGeometry[] = [];
    const perimGeos: BufferGeometry[] = [];
    const bumpMetalGeos: BufferGeometry[] = [];
    const bumpPaintGeos: BufferGeometry[] = [];
    const atlasGeos: BufferGeometry[] = [];

    const m4 = new Matrix4();
    const posV = new Vector3();
    const rotQ = new Quaternion();
    const sclV = new Vector3(1, 1, 1);
    const eul = new Euler();

    const pushStrip = (bucket: BufferGeometry[], x: number, z: number, w: number, l: number, yaw = 0, y = 0.012) => {
      const pg = new PlaneGeometry(w, l);
      pg.rotateX(-Math.PI / 2);
      if (yaw !== 0) pg.rotateY(yaw);
      pg.translate(x, y, z);
      bucket.push(prepForMerge(pg));
      pg.dispose();
    };

    const pushBoxGeo = (bucket: BufferGeometry[], cx: number, cy: number, cz: number, sx: number, sy: number, sz: number, yaw = 0) => {
      const bg = new BoxGeometry(sx, sy, sz);
      eul.set(0, yaw, 0);
      rotQ.setFromEuler(eul);
      posV.set(cx, cy, cz);
      m4.compose(posV, rotQ, sclV);
      bucket.push(prepForMerge(bg, m4));
      bg.dispose();
    };

    // ---- Interconnecting Road Network & 500m Runway Markings ----
    // North-South Spine Highway (x = 0, z = -720 -> +260)
    for (const x of [-5.5, 5.5]) pushStrip(whiteStripGeos, x, -230, 0.22, 980);
    for (let z = -700; z < 252; z += 14) pushStrip(yellowStripGeos, 0, z, 0.18, 6.5);
    for (const bz of [-140, -40, 60, 160, 210, 248]) pushStrip(yellowStripGeos, 0, bz, 10.6, 0.48);

    // Crash-Test Lab parallel lanes (-32, -18, +16, +30, +46, +64)
    for (const lx of [-32, -18, 16, 30, 46, 64]) {
      for (const dx of [-2.2, 2.2]) pushStrip(whiteStripGeos, lx + dx, 195, 0.14, 130);
    }

    // East-West Connecting Boulevard (z = -220, x = -340 -> +520) & Northeast Machinery Spur
    for (const dz of [-6.0, 6.0]) pushStrip(whiteStripGeos, 40, -220 + dz, 760, 0.22);
    for (let x = -320; x < 340; x += 16) pushStrip(yellowStripGeos, x, -220, 7.0, 0.18);
    // Spur avenue from East Boulevard up to Zone 2 Machinery (x = 245, z = -205 -> +190)
    for (const dx of [-5.2, 5.2]) pushStrip(whiteStripGeos, 245 + dx, -10, 0.20, 390);
    for (const lx of [195, 295]) {
      for (const dx of [-2.8, 2.8]) pushStrip(yellowStripGeos, lx + dx, 145, 0.18, 75);
    }

    // West Endurance & Suspension Circuit lanes (x = -180, -230, -280, -330)
    for (const lx of [-180, -230, -280, -330]) {
      for (const dx of [-4.2, 4.2]) pushStrip(whiteStripGeos, lx + dx, -80, 0.16, 280);
    }

    // Roundabout circular road markings at (380, -220)
    const ringOuter = new RingGeometry(55.6, 56.0, 64);
    ringOuter.rotateX(-Math.PI / 2);
    ringOuter.translate(380, 0.015, -220);
    whiteStripGeos.push(prepForMerge(ringOuter));
    ringOuter.dispose();

    const ringMid = new RingGeometry(38.8, 39.15, 64);
    ringMid.rotateX(-Math.PI / 2);
    ringMid.translate(380, 0.015, -220);
    yellowStripGeos.push(prepForMerge(ringMid));
    ringMid.dispose();

    // Bullseye Landing Target at (68, -430) — concentric rings merged into strips!
    for (const [rIn, rOut, isYellow] of [[0.1, 3.2, true], [5.8, 8.2, false], [11.2, 13.5, true]] as [number, number, boolean][]) {
      const rg = new RingGeometry(rIn, rOut, 40);
      rg.rotateX(-Math.PI / 2);
      rg.translate(68, 0.018, -430);
      (isYellow ? yellowStripGeos : whiteStripGeos).push(prepForMerge(rg));
      rg.dispose();
    }

    // ---- Shared 1024×1024 UV Stencil & Sign Atlas (1 single draw call for ALL distance markers & zone signs!) ----
    const atlasLabels: { text: string; color: string; bg: string }[] = [
      { text: '400m', color: '#f8fafc', bg: 'transparent' },        // 0
      { text: '300m', color: '#f8fafc', bg: 'transparent' },        // 1
      { text: '200m', color: '#fde047', bg: 'transparent' },        // 2
      { text: '100m', color: '#fb923c', bg: 'transparent' },        // 3
      { text: '50m', color: '#f87171', bg: 'transparent' },         // 4
      { text: 'NCAP LAB', color: '#f8fafc', bg: '#dc2626' },        // 5
      { text: '40% / 25%', color: '#f8fafc', bg: '#d97706' },       // 6
      { text: 'DOUBLE PRESS', color: '#0f172a', bg: '#facc15' },    // 7
      { text: 'PENDULUMS', color: '#f8fafc', bg: '#ea580c' },       // 8
      { text: 'ROLLERS', color: '#0f172a', bg: '#eab308' },         // 9
      { text: 'SUSPENSION', color: '#f8fafc', bg: '#0284c7' },      // 10
      { text: 'BANKING 25°', color: '#f8fafc', bg: '#2563eb' },     // 11
      { text: '15° / 25° / 35°', color: '#f8fafc', bg: '#7c3aed' }, // 12
      { text: 'MEGA CANYON', color: '#f8fafc', bg: '#e11d48' },     // 13
      { text: 'ROUNDABOUT', color: '#f8fafc', bg: '#059669' },      // 14
      { text: 'INDUSTRIAL', color: '#f8fafc', bg: '#475569' },      // 15
    ];
    const atlasTex = new CanvasTexture(makeLabelAtlasCanvas(atlasLabels));
    atlasTex.colorSpace = SRGBColorSpace;

    const addAtlasGroundStencil = (x: number, z: number, cellIdx: number, w = 6.2, h = 2.4, yaw = 0) => {
      const pg = makeAtlasQuad(w, h, cellIdx);
      pg.rotateX(-Math.PI / 2);
      if (yaw !== 0) pg.rotateY(yaw);
      pg.translate(x, 0.02, z);
      atlasGeos.push(prepForMerge(pg));
      pg.dispose();
    };
    const addAtlasRoadSign = (x: number, z: number, cellIdx: number, yaw = 0) => {
      pushBoxGeo(steelGeos, x, 1.4, z, 0.12, 2.8, 0.12, yaw);
      const sg = makeAtlasQuad(3.6, 0.95, cellIdx);
      // Front & back sign faces
      const sgBack = makeAtlasQuad(3.6, 0.95, cellIdx);
      sgBack.rotateY(Math.PI);
      sg.translate(0, 2.95, 0.05);
      sgBack.translate(0, 2.95, -0.05);
      for (const face of [sg, sgBack]) {
        if (yaw !== 0) face.rotateY(yaw);
        face.translate(x, 0, z);
        atlasGeos.push(prepForMerge(face));
        face.dispose();
      }
    };

    // 400m / 300m / 200m / 100m / 50m runway stencils (northbound + southbound!)
    addAtlasGroundStencil(0, -140, 0, 7.5, 2.8, Math.PI);
    addAtlasGroundStencil(0, -40, 1, 7.5, 2.8, Math.PI);
    addAtlasGroundStencil(0, 60, 2, 7.5, 2.8, Math.PI);
    addAtlasGroundStencil(0, 160, 3, 7.5, 2.8, Math.PI);
    addAtlasGroundStencil(0, 210, 4, 7.5, 2.8, Math.PI);
    // Southbound Mega-Ramp run-up distance markers
    addAtlasGroundStencil(0, -340, 2, 7.5, 2.8, 0);
    addAtlasGroundStencil(0, -440, 3, 7.5, 2.8, 0);
    // Eastbound avenue distance markers
    addAtlasGroundStencil(120, -220, 2, 6.5, 2.5, -Math.PI / 2);
    addAtlasGroundStencil(220, -220, 3, 6.5, 2.5, -Math.PI / 2);

    // Zone roadside signs
    addAtlasRoadSign(8, 263, 5, Math.PI);
    addAtlasRoadSign(-24, 263, 6, Math.PI);
    addAtlasRoadSign(195, 170, 7, Math.PI);
    addAtlasRoadSign(245, 182, 8, Math.PI);
    addAtlasRoadSign(295, 168, 9, Math.PI);
    addAtlasRoadSign(-180, -226, 10, Math.PI);
    addAtlasRoadSign(-255, 282, 11, Math.PI);
    addAtlasRoadSign(-24, -324, 12, 0);
    addAtlasRoadSign(9, -508, 13, 0);
    addAtlasRoadSign(318, -202, 14, -Math.PI / 2);
    addAtlasRoadSign(272, -338, 15, 0);

    // ---- 3 Live Radar Speed Gantries straddling 500m runway at z = 60, 160, 232 ----
    if (typeof document !== 'undefined') {
      this.radarCanvas = document.createElement('canvas');
      this.radarCanvas.width = 256;
      this.radarCanvas.height = 64;
      this.radarCtx = this.radarCanvas.getContext('2d');
      this.radarTex = new CanvasTexture(this.radarCanvas);
      this.radarTex.colorSpace = SRGBColorSpace;
      this.updateRadarDisplay(0, 0, 0);
    }
    const radarBoardGeos: BufferGeometry[] = [];
    for (const gz of [60, 160, 232]) {
      pushBoxGeo(steelGeos, -6.4, 2.6, gz, 0.32, 5.2, 0.32);
      pushBoxGeo(steelGeos, 6.4, 2.6, gz, 0.32, 5.2, 0.32);
      pushBoxGeo(steelGeos, 0, 5.2, gz, 13.4, 0.38, 0.36);
      const board = new PlaneGeometry(4.4, 1.1);
      board.rotateY(Math.PI); // faces approaching vehicle (-Z -> +Z)
      board.translate(0, 5.2, gz - 0.22);
      radarBoardGeos.push(prepForMerge(board));
      board.dispose();
    }
    if (radarBoardGeos.length > 0 && this.radarTex) {
      const radarMerged = mergeGeometries(radarBoardGeos, false);
      radarBoardGeos.forEach((b) => b.dispose());
      const radarMesh = new Mesh(
        radarMerged,
        new MeshBasicMaterial({ map: this.radarTex, toneMapped: false }),
      );
      g.add(radarMesh);
    }

    // ---- Zone 2 Overhead Gantry Crossbars (Presses + Pendulums) & Zone 5 Drive-Through Hangar Roofs ----
    for (const pz of [148, 162]) {
      pushBoxGeo(steelGeos, 195, 5.25, pz, 7.4, 0.42, 0.48);
    }
    for (let i = 0; i < 4; i++) {
      pushBoxGeo(steelGeos, 245, 7.55, 126 + i * 16, 13.6, 0.45, 0.45);
    }
    for (const hx of [260, 500]) {
      pushBoxGeo(steelGeos, hx, 8.45, -370, 19.6, 0.55, 54.4);
    }

    // ---- Static Box Colliders -> Merged Concrete / Hazard / Steel / Perimeter Buckets ----
    const containerSkip = new Set(this.containerBoxIndices.map((c) => c.idx));
    for (let k = 0; k < this.boxCount; k++) {
      if (this.dynamicBoxIndices.has(k) || containerSkip.has(k)) continue;
      const o = k * BOX_STRIDE;
      const B = this.boxes;
      const cx = B[o], cy = B[o + 1], cz = B[o + 2];
      const hx = B[o + 3], hy = B[o + 4], hz = B[o + 5];
      const yaw = Math.atan2(B[o + 7], B[o + 6]);
      const isPerim = hx > 100 || hz > 100;
      if (isPerim) {
        pushBoxGeo(perimGeos, cx, cy, cz, hx * 2, hy * 2, hz * 2, yaw);
      } else if (this.steelBoxIndices.has(k)) {
        pushBoxGeo(steelGeos, cx, cy, cz, hx * 2, hy * 2, hz * 2, yaw);
      } else {
        pushBoxGeo(concreteGeos, cx, cy, cz, hx * 2, hy * 2, hz * 2, yaw);
        // Add crisp hazard face strip on NCAP frontal & offset crash barriers
        if (Math.abs(cz - 260) < 2.0 && hz < 1.3) {
          pushBoxGeo(hazardGeos, cx, cy, cz - hz - 0.02, hx * 1.96, hy * 1.9, 0.04, yaw);
        }
      }
    }

    // ---- Sunken Canyon Cutout Visual Pit & Ramps & Giant Staircase ----
    // Mask asphalt over the canyon opening with dark canyon rock pit walls & floor
    pushBoxGeo(steelGeos, 0, CANYON_FLOOR_Y - 0.1, (CANYON_Z0 + CANYON_Z1) * 0.5, CANYON_X1 - CANYON_X0, 0.25, CANYON_Z1 - CANYON_Z0);
    // Visual cutout cover just above y = 0? Wait: asphalt is at y = 0!
    // To make the sunken canyon visually recessed without splitting the 2000m asphalt quad, we raise the Mega-Ramp approach deck & landing deck, OR place a crisp dark water/chasm visual at y = 0.02 with rocky gorge embankments!
    pushStrip(yellowStripGeos, 0, CANYON_Z1 + 0.5, 13.0, 0.6);
    pushStrip(yellowStripGeos, 0, CANYON_Z0 - 0.5, 15.5, 0.6);
    const chasmPlane = new PlaneGeometry(CANYON_X1 - CANYON_X0, CANYON_Z1 - CANYON_Z0);
    chasmPlane.rotateX(-Math.PI / 2);
    chasmPlane.translate(0, 0.014, (CANYON_Z0 + CANYON_Z1) * 0.5);
    steelGeos.push(prepForMerge(chasmPlane));
    chasmPlane.dispose();

    // Merge all Ramps into concreteGeos
    for (const r of this.ramps) {
      const rg = rampGeometry(r.len, r.w, r.h);
      eul.set(0, r.yaw, 0);
      rotQ.setFromEuler(eul);
      posV.set(r.x, r.baseY ?? 0, r.z);
      m4.compose(posV, rotQ, sclV);
      concreteGeos.push(prepForMerge(rg, m4));
      rg.dispose();
    }

    // Merge Giant Concrete Staircase (Lane C: x = -280, z = -200..-96) into concreteGeos
    for (let s = 0; s < 10; s++) {
      const h = (s + 1) * 0.16;
      pushBoxGeo(concreteGeos, -280, h * 0.5, -200 + s * 3.6 + 1.8, 8.0, h, 3.6);
      pushBoxGeo(concreteGeos, -280, h * 0.5, -96 - s * 3.6 - 1.8, 8.0, h, 3.6);
    }
    pushBoxGeo(concreteGeos, -280, 0.8, -148, 8.0, 1.6, 32.0);

    // Merge 180° North Parabolic Banked Curve Geometry into concreteGeos!
    const bankGeo = bankedCurveGeometry(BANK_CX, BANK_CZ, BANK_R_IN, BANK_R_OUT, BANK_MAX_H);
    concreteGeos.push(prepForMerge(bankGeo));
    bankGeo.dispose();

    // Merge Suspension Bumps into metal & painted buckets
    for (const b of this.bumps) {
      const bg = b.kind === 'sine'
        ? sineBumpGeometry(b.x1 - b.x0, b.z1 - b.z0, b.h)
        : stepPlatformGeometry(b.x1 - b.x0, b.z1 - b.z0, b.h, b.rampLen ?? 4.0);
      bg.translate((b.x0 + b.x1) * 0.5, 0, b.z0);
      (b.metal ? bumpMetalGeos : bumpPaintGeos).push(prepForMerge(bg));
      bg.dispose();
    }

    // Merge Roundabout Raised Central Island Curb (380, -220, R = 22m)
    const islandGeo = new CylinderGeometry(22.0, 22.2, 0.34, 48);
    islandGeo.translate(380, 0.17, -220);
    concreteGeos.push(prepForMerge(islandGeo));
    islandGeo.dispose();
    const monGeo = new CylinderGeometry(2.1, 2.5, 7.5, 20);
    monGeo.translate(380, 3.75, -220);
    steelGeos.push(prepForMerge(monGeo));
    monGeo.dispose();

    // ---- Surface Friction Patches (1 merged draw call with vertex colors) ----
    const patchGeos: BufferGeometry[] = [];
    const tmpCol = new Color();
    for (const p of this.patches) {
      const pg = prepForMerge(new PlaneGeometry(p.x1 - p.x0, p.z1 - p.z0));
      pg.rotateX(-Math.PI / 2);
      pg.translate((p.x0 + p.x1) * 0.5, 0.008, (p.z0 + p.z1) * 0.5);
      tmpCol.setHex(p.color);
      const vc = pg.getAttribute('position').count;
      const cols = new Float32Array(vc * 3);
      for (let i = 0; i < vc; i++) {
        cols[i * 3] = tmpCol.r; cols[i * 3 + 1] = tmpCol.g; cols[i * 3 + 2] = tmpCol.b;
      }
      pg.setAttribute('color', new Float32BufferAttribute(cols, 3));
      patchGeos.push(pg);
    }
    if (patchGeos.length > 0) {
      const pm = new Mesh(
        mergeGeometries(patchGeos, false),
        new MeshStandardMaterial({ vertexColors: true, roughness: 0.88, metalness: 0.02, envMapIntensity: 0.08 }),
      );
      patchGeos.forEach((x) => x.dispose());
      pm.receiveShadow = true;
      g.add(pm);
    }

    // ---- Finalize Static Merged Meshes ----
    const concretePack = getConcreteTextures(lowSpec);
    const concreteMat = new MeshStandardMaterial({
      map: concretePack.map,
      normalMap: concretePack.normalMap,
      normalScale: new Vector2(0.55, 0.55),
      roughnessMap: concretePack.roughnessMap,
      roughness: 0.92,
      metalness: 0.03,
      envMapIntensity: 0.16,
      side: DoubleSide,
    });
    const perimMat = new MeshStandardMaterial({
      color: 0xb4bac2,
      map: concretePack.map,
      roughness: 0.95,
      metalness: 0,
      envMapIntensity: 0.08,
    });
    const hazardMat = new MeshStandardMaterial({
      map: concretePack.hazardMap,
      normalMap: concretePack.normalMap,
      roughnessMap: concretePack.hazardRoughnessMap,
      roughness: 0.72,
      metalness: 0.18,
    });
    const steelMat = new MeshStandardMaterial({ color: 0x475569, metalness: 0.78, roughness: 0.36 });
    const whiteMarkMat = new MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.94, metalness: 0, envMapIntensity: 0.05 });
    const yellowMarkMat = new MeshStandardMaterial({ color: 0xd9a528, roughness: 0.94, metalness: 0, envMapIntensity: 0.05 });
    const bumpPaintMat = new MeshStandardMaterial({ color: 0xd97706, roughness: 0.72, metalness: 0.1, side: DoubleSide });
    const bumpMetalMat = new MeshStandardMaterial({ color: 0x94a3b8, roughness: 0.34, metalness: 0.8, side: DoubleSide });
    const atlasMat = new MeshStandardMaterial({ map: atlasTex, transparent: true, alphaTest: 0.08, roughness: 0.85, metalness: 0.05 });

    const emitMerged = (bucket: BufferGeometry[], mat: MeshStandardMaterial, castShadow: boolean) => {
      if (bucket.length === 0) return;
      const merged = mergeGeometries(bucket, false);
      for (let i = 0; i < bucket.length; i++) bucket[i].dispose();
      merged.computeBoundingSphere();
      const mesh = new Mesh(merged, mat);
      mesh.castShadow = castShadow;
      mesh.receiveShadow = true;
      g.add(mesh);
    };

    emitMerged(whiteStripGeos, whiteMarkMat, false);
    emitMerged(yellowStripGeos, yellowMarkMat, false);
    emitMerged(atlasGeos, atlasMat, false);
    emitMerged(concreteGeos, concreteMat, true);
    emitMerged(hazardGeos, hazardMat, true);
    emitMerged(steelGeos, steelMat, true);
    emitMerged(perimGeos, perimMat, false);
    emitMerged(bumpMetalGeos, bumpMetalMat, true);
    emitMerged(bumpPaintGeos, bumpPaintMat, true);

    // ---- GPU Instanced Static Props (Arrows, Poles, Bollards, Tyres, Lamp Posts, Containers) ----
    // 1. Directional Arrows
    const arrowCoords: [number, number, number][] = [
      [0, -180, 0], [0, -80, 0], [0, 20, 0], [0, 120, 0], [0, 200, 0],
      [0, -300, Math.PI], [0, -390, Math.PI], [0, -475, Math.PI],
      [-180, -225, 0], [-230, -225, 0], [-280, -212, 0], [-330, -225, 0],
      [195, 125, 0], [245, 105, 0], [295, 125, 0],
      [200, -220, Math.PI / 2], [290, -220, Math.PI / 2],
    ];
    const arrowInst = new InstancedMesh(arrowGeometry(), whiteMarkMat, arrowCoords.length);
    for (let i = 0; i < arrowCoords.length; i++) {
      const [ax, az, ayaw] = arrowCoords[i];
      this.tmpPos.set(ax, 0.016, az);
      this.tmpEuler.set(0, ayaw, 0);
      this.tmpQuat.setFromEuler(this.tmpEuler);
      this.tmpScale.set(1.25, 1, 1.25);
      this.tmpMat.compose(this.tmpPos, this.tmpQuat, this.tmpScale);
      arrowInst.setMatrixAt(i, this.tmpMat);
    }
    g.add(arrowInst);

    // 2. Cylinders -> Instanced Poles, Bollards, Tyres, Lamp Posts
    const poleMats: Matrix4[] = [];
    const bollardMats: Matrix4[] = [];
    const tyreMats: Matrix4[] = [];
    const lampPosts: Matrix4[] = [];
    const lampArms: Matrix4[] = [];

    for (let k = 0; k < this.cylCount; k++) {
      const o = k * CYL_STRIDE;
      const cx = this.cyls[o], cz = this.cyls[o + 1], r = this.cyls[o + 2];
      const h = this.cyls[o + 4] - this.cyls[o + 3];
      if (r > 1.5) continue; // Roundabout island & monument already merged
      if (r > 0.3) {
        for (let lvl = 0; lvl < 5; lvl++) {
          tyreMats.push(new Matrix4().makeTranslation(cx, 0.11 + lvl * 0.21, cz));
        }
      } else if (this.lampPostIndices.has(k)) {
        lampPosts.push(new Matrix4().makeTranslation(cx, h * 0.5, cz));
        lampArms.push(new Matrix4().makeTranslation(cx - 0.65, h - 0.1, cz));
      } else if (r > 0.12) {
        const m = new Matrix4().compose(new Vector3(cx, h * 0.5, cz), new Quaternion(), new Vector3(r * 2, h, r * 2));
        poleMats.push(m);
      } else {
        const m = new Matrix4().compose(new Vector3(cx, h * 0.5, cz), new Quaternion(), new Vector3(r * 2, h, r * 2));
        bollardMats.push(m);
      }
    }

    const unitCyl = new CylinderGeometry(0.5, 0.5, 1.0, 16);
    if (poleMats.length > 0) {
      const pInst = new InstancedMesh(unitCyl, new MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.88, roughness: 0.28 }), poleMats.length);
      poleMats.forEach((m, i) => pInst.setMatrixAt(i, m));
      pInst.castShadow = true;
      g.add(pInst);
    }
    if (bollardMats.length > 0) {
      const bInst = new InstancedMesh(unitCyl.clone(), new MeshStandardMaterial({ color: 0xf59e0b, roughness: 0.45, metalness: 0.3 }), bollardMats.length);
      bollardMats.forEach((m, i) => bInst.setMatrixAt(i, m));
      bInst.castShadow = true;
      g.add(bInst);
    }
    if (tyreMats.length > 0) {
      const tyreGeo = new TorusGeometry(0.32, 0.11, 8, 16);
      tyreGeo.rotateX(Math.PI / 2);
      const tInst = new InstancedMesh(tyreGeo, new MeshStandardMaterial({ color: 0x18181b, roughness: 0.9 }), tyreMats.length);
      tyreMats.forEach((m, i) => tInst.setMatrixAt(i, m));
      tInst.castShadow = true;
      g.add(tInst);
    }
    if (lampPosts.length > 0) {
      const postInst = new InstancedMesh(new CylinderGeometry(0.11, 0.14, 6.4, 10), steelMat, lampPosts.length);
      lampPosts.forEach((m, i) => postInst.setMatrixAt(i, m));
      postInst.castShadow = true;
      g.add(postInst);

      const lampHeadMat = new MeshStandardMaterial({ color: 0x334155, emissive: 0xfff3c4, emissiveIntensity: 0.95, roughness: 0.4 });
      const armInst = new InstancedMesh(new BoxGeometry(1.4, 0.14, 0.26), lampHeadMat, lampArms.length);
      lampArms.forEach((m, i) => armInst.setMatrixAt(i, m));
      g.add(armInst);
    }

    // 3. 28 Stacked Shipping Containers (1 single InstancedMesh with per-instance colors!)
    if (this.containerBoxIndices.length > 0) {
      const contGeo = new BoxGeometry(2.5, 2.6, 12.2);
      const contMat = new MeshStandardMaterial({
        color: 0xffffff,
        normalMap: concretePack.normalMap,
        normalScale: new Vector2(0.35, 0.35),
        metalness: 0.55,
        roughness: 0.52,
      });
      const contInst = new InstancedMesh(contGeo, contMat, this.containerBoxIndices.length);
      const colObj = new Color();
      this.tmpScale.set(1, 1, 1);
      for (let i = 0; i < this.containerBoxIndices.length; i++) {
        const { idx, color } = this.containerBoxIndices[i];
        const o = idx * BOX_STRIDE;
        this.tmpPos.set(this.boxes[o], this.boxes[o + 1], this.boxes[o + 2]);
        this.tmpEuler.set(0, Math.atan2(this.boxes[o + 7], this.boxes[o + 6]), 0);
        this.tmpQuat.setFromEuler(this.tmpEuler);
        this.tmpMat.compose(this.tmpPos, this.tmpQuat, this.tmpScale);
        contInst.setMatrixAt(i, this.tmpMat);
        colObj.setHex(color);
        contInst.setColorAt(i, colObj);
      }
      contInst.castShadow = true;
      contInst.receiveShadow = true;
      if (contInst.instanceColor) contInst.instanceColor.needsUpdate = true;
      g.add(contInst);
    }

    // ---- Dynamic Batched InstancedMeshes (Machinery + Jersey + Guardrails + Cones = 5 draw calls!) ----
    // 1. Dynamic Hazard Heads (2 press heads + 4 pendulum hammer heads = 6 instances)
    this.dynHazardMachineryMesh = new InstancedMesh(new BoxGeometry(1, 1, 1), hazardMat, this.pressBoxIds.length + this.pendulums.length);
    this.dynHazardMachineryMesh.frustumCulled = false;
    this.dynHazardMachineryMesh.castShadow = true;
    g.add(this.dynHazardMachineryMesh);

    // 2. Dynamic Steel Pistons, Swing Arms & Crusher Rollers (4 + 4 + 3 = 11 instances)
    const brightSteelMat = new MeshStandardMaterial({ color: 0xcbd5e1, metalness: 0.9, roughness: 0.24 });
    this.dynSteelMachineryMesh = new InstancedMesh(new CylinderGeometry(0.5, 0.5, 1.0, 18), brightSteelMat, this.pressBoxIds.length * 2 + this.pendulums.length + this.rollerCount);
    this.dynSteelMachineryMesh.frustumCulled = false;
    this.dynSteelMachineryMesh.castShadow = true;
    g.add(this.dynSteelMachineryMesh);
    this.syncMachineryVisuals();

    // 3. Dynamic Jersey Barriers & W-Beam Highway Guardrails
    this.dynJerseyMap = [];
    this.dynGuardrailMap = [];
    for (let i = 0; i < this.dynBarriers.length; i++) {
      if (this.dynBarriers[i].guardrail) this.dynGuardrailMap.push(i);
      else this.dynJerseyMap.push(i);
    }
    if (this.dynJerseyMap.length > 0) {
      this.dynJerseyMesh = new InstancedMesh(jerseyGeometry(2.7), concreteMat, this.dynJerseyMap.length);
      this.dynJerseyMesh.frustumCulled = false;
      this.dynJerseyMesh.castShadow = true;
      g.add(this.dynJerseyMesh);
    }
    if (this.dynGuardrailMap.length > 0) {
      const grGeo = guardrailGeometry(12.4);
      const grMat = new MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.82, roughness: 0.34 });
      this.dynGuardrailMesh = new InstancedMesh(grGeo, grMat, this.dynGuardrailMap.length);
      this.dynGuardrailMesh.frustumCulled = false;
      this.dynGuardrailMesh.castShadow = true;
      g.add(this.dynGuardrailMesh);
    }

    // 4. Dynamic Construction Cones
    if (this.dynCones.length > 0) {
      const coneTex = new CanvasTexture(makeConeStripeCanvas());
      coneTex.colorSpace = SRGBColorSpace;
      this.dynConeMesh = new InstancedMesh(
        new ConeGeometry(0.22, 0.68, 12),
        new MeshStandardMaterial({ map: coneTex, roughness: 0.55 }),
        this.dynCones.length,
      );
      this.dynConeMesh.frustumCulled = false;
      this.dynConeMesh.castShadow = true;
      g.add(this.dynConeMesh);
    }
    this.syncDynamicVisuals(true, true);

    // ---- Underground Pure-White Studio Showroom (y = -90) ----
    const studioWallMat = new MeshBasicMaterial({ color: 0xf8fafc, side: BackSide, toneMapped: false });
    const studioEnclosure = new Mesh(new BoxGeometry(80, 30, 80), studioWallMat);
    studioEnclosure.position.set(0, STUDIO_SHOWROOM_Y + 15 - 0.05, 0);
    g.add(studioEnclosure);

    const studioFloorMat = new MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0xf1f5f9,
      emissiveIntensity: 0.78,
      roughness: 0.34,
      metalness: 0.0,
      envMapIntensity: 0.4,
    });
    const studioFloor = new Mesh(new PlaneGeometry(78, 78), studioFloorMat);
    studioFloor.rotation.x = -Math.PI / 2;
    studioFloor.position.set(0, STUDIO_SHOWROOM_Y, 0);
    studioFloor.receiveShadow = true;
    g.add(studioFloor);

    g.traverse((o) => { o.matrixAutoUpdate = false; o.updateMatrix(); });
    return g;
  }

  dispose(): void {
    this.radarTex?.dispose();
    this.group.traverse((o) => {
      const m = o as Mesh;
      if (m.geometry) m.geometry.dispose();
      const mats = Array.isArray(m.material) ? m.material : m.material ? [m.material] : [];
      for (const mat of mats) {
        const sm = mat as MeshStandardMaterial;
        if (!isSharedProceduralTexture(sm.map)) sm.map?.dispose();
        if (!isSharedProceduralTexture(sm.normalMap)) sm.normalMap?.dispose();
        if (!isSharedProceduralTexture(sm.roughnessMap)) sm.roughnessMap?.dispose();
        sm.dispose();
      }
    });
    this.group.clear();
  }
}

// ================================================================ Geometry & Texture Helpers

function prepForMerge(geo: BufferGeometry, mat4?: Matrix4): BufferGeometry {
  const g = geo.index ? geo.toNonIndexed() : geo.clone();
  if (mat4) g.applyMatrix4(mat4);
  if (!g.getAttribute('normal')) g.computeVertexNormals();
  const count = g.getAttribute('position').count;
  if (!g.getAttribute('uv')) {
    g.setAttribute('uv', new Float32BufferAttribute(new Float32Array(count * 2), 2));
  }
  for (const key of Object.keys(g.attributes)) {
    if (key !== 'position' && key !== 'normal' && key !== 'uv') {
      g.deleteAttribute(key);
    }
  }
  return g;
}

function makeLabelAtlasCanvas(items: { text: string; color: string; bg: string }[]): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = 1024;
  c.height = 1024;
  const ctx = c.getContext('2d')!;
  ctx.clearRect(0, 0, 1024, 1024);
  // 4 × 4 grid of 256 × 256 cells
  for (let i = 0; i < Math.min(16, items.length); i++) {
    const col = i % 4;
    const row = Math.floor(i / 4);
    const x0 = col * 256;
    const y0 = row * 256;
    const it = items[i];
    if (it.bg !== 'transparent') {
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(x0 + 4, y0 + 52, 248, 152);
      ctx.fillStyle = it.bg;
      ctx.fillRect(x0 + 4, y0 + 52, 248, 22);
      ctx.strokeStyle = 'rgba(255,255,255,0.3)';
      ctx.lineWidth = 3;
      ctx.strokeRect(x0 + 6, y0 + 54, 244, 148);
    }
    ctx.fillStyle = it.color;
    ctx.font = it.bg === 'transparent' ? '900 72px system-ui, sans-serif' : '800 30px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(it.text, x0 + 128, y0 + (it.bg === 'transparent' ? 128 : 138));
  }
  return c;
}

function makeAtlasQuad(w: number, h: number, cellIdx: number): BufferGeometry {
  const g = new PlaneGeometry(w, h);
  const col = cellIdx % 4;
  const row = Math.floor(cellIdx / 4);
  const u0 = col * 0.25 + 0.005;
  const u1 = (col + 1) * 0.25 - 0.005;
  // Canvas row 0 is at top (V = 1 -> 0.75 in Three.js UV space)
  const vTop = 1 - (row * 0.25 + 0.21);
  const vBot = 1 - (row * 0.25 + 0.79 * 0.25 + 0.58 * 0.25);
  const uv = g.getAttribute('uv');
  const yTop = 1 - row * 0.25 - 0.05;
  const yBot = 1 - (row + 1) * 0.25 + 0.05;
  void vTop; void vBot;
  // PlaneGeometry default UVs: (0,1), (1,1), (0,0), (1,0)
  uv.setXY(0, u0, yTop);
  uv.setXY(1, u1, yTop);
  uv.setXY(2, u0, yBot);
  uv.setXY(3, u1, yBot);
  return g;
}

function bankedCurveGeometry(cx: number, cz: number, rIn: number, rOut: number, maxH: number): BufferGeometry {
  const aSegs = 40;
  const rSegs = 10;
  const v: number[] = [];
  for (let i = 0; i < aSegs; i++) {
    const a0 = (Math.PI * i) / aSegs;
    const a1 = (Math.PI * (i + 1)) / aSegs;
    const c0 = Math.cos(a0), s0 = Math.sin(a0);
    const c1 = Math.cos(a1), s1 = Math.sin(a1);
    for (let j = 0; j < rSegs; j++) {
      const u0 = j / rSegs, u1 = (j + 1) / rSegs;
      const rad0 = rIn + (rOut - rIn) * u0;
      const rad1 = rIn + (rOut - rIn) * u1;

      const pt = (c: number, s: number, rad: number, u: number): [number, number, number] => {
        const px = cx + c * rad;
        const pz = cz + s * rad;
        const zBlend = pz < cz + 12 ? Math.max(0, (pz - cz) / 12) : 1;
        const sb = zBlend * zBlend * (3 - 2 * zBlend);
        return [px, maxH * u * u * sb + 0.02, pz];
      };
      const p00 = pt(c0, s0, rad0, u0);
      const p10 = pt(c1, s1, rad0, u0);
      const p11 = pt(c1, s1, rad1, u1);
      const p01 = pt(c0, s0, rad1, u1);
      v.push(
        p00[0], p00[1], p00[2], p01[0], p01[1], p01[2], p11[0], p11[1], p11[2],
        p00[0], p00[1], p00[2], p11[0], p11[1], p11[2], p10[0], p10[1], p10[2],
      );
    }
  }
  const g = new BufferGeometry();
  g.setAttribute('position', new Float32BufferAttribute(v, 3));
  g.computeVertexNormals();
  return g;
}

function guardrailGeometry(len: number): BufferGeometry {
  const g1 = prepForMerge(new BoxGeometry(len, 0.34, 0.14));
  g1.translate(0, 0.18, 0);
  const posts: BufferGeometry[] = [g1];
  for (const px of [-len * 0.42, 0, len * 0.42]) {
    const p = prepForMerge(new BoxGeometry(0.14, 0.92, 0.18));
    p.translate(px, -0.02, 0);
    posts.push(p);
  }
  const merged = mergeGeometries(posts, false);
  posts.forEach((p) => p.dispose());
  return merged;
}

function makeNoiseCanvas(size: number, rgb: [number, number, number], amp: number): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d')!;
  const img = ctx.createImageData(size, size);
  let seed = 1337;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let i = 0; i < size * size; i++) {
    const n = (rnd() - 0.5) * amp + (rnd() < 0.02 ? amp * 0.8 : 0);
    img.data[i * 4] = rgb[0] + n;
    img.data[i * 4 + 1] = rgb[1] + n;
    img.data[i * 4 + 2] = rgb[2] + n;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

function makeConeStripeCanvas(): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = 64; c.height = 64;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = '#f97316';
  ctx.fillRect(0, 0, 64, 64);
  ctx.fillStyle = '#f8fafc';
  ctx.fillRect(0, 20, 64, 10);
  ctx.fillRect(0, 38, 64, 8);
  return c;
}

function arrowGeometry(): BufferGeometry {
  const v = [
    -0.22, 0, -1.2, 0.22, 0, -1.2, 0.22, 0, 0.3,
    -0.22, 0, -1.2, 0.22, 0, 0.3, -0.22, 0, 0.3,
    -0.65, 0, 0.3, 0.65, 0, 0.3, 0, 0, 1.35,
  ];
  const g = new BufferGeometry();
  g.setAttribute('position', new Float32BufferAttribute(v, 3));
  g.computeVertexNormals();
  return g;
}

function jerseyGeometry(length: number): BufferGeometry {
  const s = new Shape();
  s.moveTo(-0.3, 0); s.lineTo(0.3, 0); s.lineTo(0.24, 0.08); s.lineTo(0.1, 0.3); s.lineTo(0.08, 0.84);
  s.lineTo(-0.08, 0.84); s.lineTo(-0.1, 0.3); s.lineTo(-0.24, 0.08); s.lineTo(-0.3, 0);
  const g = new ExtrudeGeometry(s, { depth: length, bevelEnabled: false });
  g.translate(0, -0.42, -length / 2);
  return g;
}

function rampGeometry(len: number, w: number, h: number): BufferGeometry {
  const hw = w / 2;
  const v = [
    -hw, 0, 0, hw, 0, 0, hw, h, len, -hw, 0, 0, hw, h, len, -hw, h, len,
    -hw, 0, len, -hw, h, len, hw, h, len, -hw, 0, len, hw, h, len, hw, 0, len,
    hw, 0, 0, hw, 0, len, hw, h, len, -hw, 0, 0, -hw, h, len, -hw, 0, len,
  ];
  const g = new BufferGeometry();
  g.setAttribute('position', new Float32BufferAttribute(v, 3));
  g.computeVertexNormals();
  return g;
}

function sineBumpGeometry(w: number, len: number, h: number): BufferGeometry {
  const segs = 12;
  const hw = w * 0.5;
  const v: number[] = [];
  for (let i = 0; i < segs; i++) {
    const u0 = i / segs, u1 = (i + 1) / segs;
    const z0 = u0 * len, z1 = u1 * len;
    const y0 = h * 0.5 * (1 - Math.cos(2 * Math.PI * u0));
    const y1 = h * 0.5 * (1 - Math.cos(2 * Math.PI * u1));
    v.push(-hw, y0, z0, hw, y0, z0, hw, y1, z1, -hw, y0, z0, hw, y1, z1, -hw, y1, z1);
    v.push(hw, 0, z0, hw, 0, z1, hw, y1, z1, hw, 0, z0, hw, y1, z1, hw, y0, z0);
    v.push(-hw, 0, z0, -hw, y1, z1, -hw, 0, z1, -hw, 0, z0, -hw, y0, z0, -hw, y1, z1);
  }
  const g = new BufferGeometry();
  g.setAttribute('position', new Float32BufferAttribute(v, 3));
  g.computeVertexNormals();
  return g;
}

function stepPlatformGeometry(w: number, len: number, h: number, rampLen: number): BufferGeometry {
  const hw = w * 0.5;
  const v = [
    -hw, 0, 0, hw, 0, 0, hw, h, rampLen, -hw, 0, 0, hw, h, rampLen, -hw, h, rampLen,
    -hw, h, rampLen, hw, h, rampLen, hw, h, len, -hw, h, rampLen, hw, h, len, -hw, h, len,
    -hw, 0, len, -hw, h, len, hw, h, len, -hw, 0, len, hw, h, len, hw, 0, len,
    hw, 0, 0, hw, 0, rampLen, hw, h, rampLen,
    hw, 0, rampLen, hw, 0, len, hw, h, len, hw, 0, rampLen, hw, h, len, hw, h, rampLen,
    -hw, 0, 0, -hw, h, rampLen, -hw, 0, rampLen,
    -hw, 0, rampLen, -hw, h, len, -hw, 0, len, -hw, 0, rampLen, -hw, h, rampLen, -hw, h, len,
  ];
  const g = new BufferGeometry();
  g.setAttribute('position', new Float32BufferAttribute(v, 3));
  g.computeVertexNormals();
  return g;
}

export const SKY_COLOR = new Color(0xb9c6d2);
