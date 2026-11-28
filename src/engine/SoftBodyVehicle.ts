import { Box3, BoxGeometry, BufferAttribute, BufferGeometry, CylinderGeometry, DoubleSide, Group, Material, Matrix4, Mesh, MeshPhysicalMaterial, MeshStandardMaterial, Object3D, PointLight, SpotLight, TorusGeometry, Vector2, Vector3 } from 'three';
import { CarProfile, createSedanProfile, createSupercarProfile } from './CarProfile';
import { MeshDeformer, ensurePlainAttributes } from './MeshDeformer';
import type { PrecomputedBindingData } from './MeshCacheDB';
import { BaselineSnapshot } from './MemoryGuard';
import {
  CarMaterials, CarPart, DEFORM_GROUP_COUNT, DeformGroup, WheelGeometrySet, buildDetailedEngineAssembly, buildProceduralCar, buildWheelVisual, createCarMaterials, createWheelGeometries,
} from './ProceduralCar';
import {
  getBrakeDiscTextures, getCarbonFiberTextures, isSharedProceduralTexture,
} from './ProceduralTextures';
import {
  BEAM_BROKEN, BEAM_COLLIDE, BEAM_HOOD_LONG, BEAM_UNBREAKABLE, CollisionWorld, DEFAULT_ZONES, GroundSample,
  SoftBodySolver, SolverBuildBeam, SolverBuildColtri, SolverBuildNode, SubstepForceProvider, Zone,
} from './SoftBodySolver';
import { GlassShatterSystem } from './GlassShatterSystem';
import { FluidEmitFrame, FluidLeakSystem } from './FluidLeakSystem';

const GEAR_RATIOS = [3.55, 2.42, 1.78, 1.36, 1.08, 0.88, 0.72] as const;

/**
 * SoftBodyVehicle (JBeam High-Frequency Architecture)
 * ---------------------------------------------------
 *  - Multi-zone JBeam node-and-beam lattice (~140 nodes, ~950 beams, ~140 coltris):
 *    1. Survival cell (`Zone.Cabin`) with diagonal cross-braced roll-cage.
 *    2. Rigid cross-braced engine cradle (`Zone.EngineSubframe`) that resists crushing
 *       and pushes into the cabin firewall (`tablier`) under extreme frontal impacts.
 *    3. Front & rear accordion crumple zones (`Zone.Crumple`) + longerons (`Zone.Longeron`).
 *    4. Triangulated wheel suspension knuckles (`Zone.Suspension`) supporting permanent
 *       camber/toe bending (`wheelBent`) and violent wheel detachment (`wheelDetached`).
 *  - Surface Collision Triangles (`coltris`) preventing thin poles, barrier corners,
 *    and detached parts from clipping between lattice nodes.
 *  - Cascading hinged & latched panels (Hood with V-fold + aerodynamic wind lift,
 *    Doors swinging open on B-pillar latch break or tearing off on hinge shear,
 *    Trunk, and Bumpers hanging from one corner before full detachment) + instanced
 *    3D glass shattering (`GlassShatterSystem`).
 */

export interface CustomWheelAssembly {
  spinner: Object3D;
  caliper?: Object3D;
  hubX: number;
  hubY: number;
  hubZ: number;
  radius: number;
}

export interface VehicleVisualSource {
  name: string;
  profile: CarProfile;
  parts: CarPart[];
  /** Optional rigid interior cockpit meshes that move with the cabin root frame without per-vertex skinning. */
  rigidMeshes?: Mesh[];
  /** Optional 3D engine block meshes that physically push back, tilt, and deform on frontal/engine impact. */
  engineMeshes?: Mesh[];
  /** Optional active rear spoiler / wing meshes animated by speed and airbrake braking. */
  spoilerMeshes?: Mesh[];
  /** Car-local pivot [x, y, z] for the active rear spoiler. */
  spoilerPivot?: [number, number, number];
  /** Optional custom wheel objects centred at their hub, order FL, FR, RL, RR. */
  wheels: CustomWheelAssembly[] | null;
  /** When true (imported .glb models), never spawn 4 extra procedural wheels if the GLB's wheels are embedded in body meshes. */
  suppressProceduralWheels?: boolean;
  materials: CarMaterials;
  paintMaterials?: MeshStandardMaterial[];
  taillightMaterials?: MeshStandardMaterial[];
  headlightMaterials?: MeshStandardMaterial[];
  dispose(): void;
}

export interface VehicleInput { throttle: number; brake: number; steer: number; handbrake: number; }

export interface PartState {
  name: string;
  label: string;
  group: DeformGroup;
  beams: number[];
  hingeBeams: number[];
  latchBeams: number[];
  leftBeams?: number[];
  rightBeams?: number[];
  state: 'attached' | 'unlatched' | 'detached';
}

export type TransmissionMode = 'auto' | 'semi' | 'manual';

export interface VehicleMechanicalState {
  engineHealth: number;
  engineStalled: boolean;
  engineBroken: boolean;
  engineTemp: number;
  overheating: boolean;
  radiatorDamaged: boolean;
  fuelLevel: number;
  coolantLevel: number;
  frontAxleDamage: number;
  rearAxleDamage: number;
  axleDamage: [number, number, number, number];
  flatTires: [boolean, boolean, boolean, boolean];
  headlightBrokenL: boolean;
  headlightBrokenR: boolean;
  steeringDamage: number;
  steeringLocked: boolean;
  steeringBiasDeg: number;
  oilLeak: boolean;
  coolantLeak: boolean;
  fuelLeak: boolean;
  coolantSeverity: number;
  oilSeverity: number;
  fuelSeverity: number;
}

export interface VehicleTelemetry {
  speedKmh: number;
  gear: string;
  transmissionMode: TransmissionMode;
  rpm: number;
  throttle: number;
  brake: number;
  steer: number;
  absActive: boolean;
  headlightsOn: boolean;
  maxTireSlip: number;
  shiftCount: number;
  gForce: number;
  peakGForce: number;
  impactKmh: number;
  peakImpactKmh: number;
  absorbedKJ: number;
  brokenBeams: number;
  suspCompression: [number, number, number, number];
  damage: { front: number; cabin: number; rear: number; hood: number; bumpers: number; fenders: number; engine: number };
  mech: VehicleMechanicalState;
  midRearEngine: boolean;
  parts: { label: string; state: string }[];
  nodes: number;
  beams: number;
  skinnedVerts: number;
  simBytes: number;
}

interface WheelState {
  left: boolean; front: boolean; driven: boolean;
  nodes: number[]; weights: number[];
  hubNode: number;
  suspBeams: number[]; // [lowerF, lowerR, upperF, upperR, tieF, tieR]
  hardLocal: [number, number, number];
  anchorRest: [number, number, number];
  radius: number;
  s: number; sPrev: number; contact: boolean; load: number;
  spin: number; omega: number; steer: number; slip: number;
  bentCamber: number; bentToe: number; detached: boolean;
  tyreFlat: number; tyreLeaking: number;
  detX: number; detY: number; detZ: number;
  detVx: number; detVy: number; detVz: number;
  detYaw: number;
  knuckle: Object3D; spinner: Object3D;
}

const NODE_BODY = 0;
const NODE_SUSPENSION = 250;
const NODE_ENGINE = 251;

export class SoftBodyVehicle implements SubstepForceProvider {
  readonly solver: SoftBodySolver;
  readonly profile: CarProfile;
  readonly root = new Group();
  readonly glass = new GlassShatterSystem();
  readonly fluids = new FluidLeakSystem();
  readonly deformer: MeshDeformer;
  readonly parts: PartState[] = [];
  readonly input: VehicleInput = { throttle: 0, brake: 0, steer: 0, handbrake: 0 };
  readonly telemetry: VehicleTelemetry;
  readonly totalMass: number;
  readonly name: string;

  /** Rest positions in car-local space (3 per node, contiguous Float32Array). */
  readonly restLocal: Float32Array;

  autopilotSpeed = 0; // m/s, crash-test cruise control (0 = off)
  /** Dynamic road wetness (0 = dry, 1 = heavy rain) subtly reducing peak tire friction. */
  wetness = 0;

  // Tuned independent spring-damper suspension & RWD drivetrain parameters
  springK = 48000;
  damperBump = 5400;
  damperRebound = 7600;
  restLen = 0.29;
  suspTop = 0.20;
  sMin = 0.045;
  sMax = 0.38;
  antiRoll = 22000;
  enginePower = 310000;
  maxDriveForce = 8800;
  maxBrakeForce = 11500;
  tireMu = 1.12;

  private readonly src: VehicleVisualSource;
  private readonly wheels: WheelState[] = [];
  private readonly wheelGeo: WheelGeometrySet | null;
  private readonly refNodes: number[]; // FL, FR, RL, RR cabin floor nodes
  private readonly refRestCentroid = [0, 0, 0];
  private readonly frame = new Float64Array(12); // O(3), X(3), Y(3), Z(3) world
  private readonly renderWorld: Float32Array;
  private readonly renderLocal: Float32Array;
  private readonly baseline = new BaselineSnapshot();
  private readonly ground: GroundSample = { h: 0, nx: 0, ny: 1, nz: 0, mu: 1 };
  private readonly rootMatrix = new Matrix4();
  private readonly hoodColumns: number[][] = [];
  private readonly hoodColumnRest: number[] = [];
  private readonly doorLNodes: { low: number[]; high: number[] };
  private readonly doorRNodes: { low: number[]; high: number[] };
  private readonly bodyNodeList: Uint16Array;
  private readonly isLatchBeam: Uint8Array;
  private readonly glassShattered = new Uint8Array(4); // 0: windshield, 1: left, 2: right, 3: rear
  private readonly engineMount = new Group();
  private readonly axleDmg = new Float32Array(4);
  private engineDmg = 0;
  private engineStalled = false;
  private frontAxleDmg = 0;
  private rearAxleDmg = 0;
  private steeringDmg = 0;
  private steeringBias = 0;
  private steeringLocked = false;
  private steeringLockAngle = 0;
  private oilLeak = false;
  private coolantLeak = false;
  private fuelLeak = false;
  private coolantSeverity = 0;
  private oilSeverity = 0;
  private fuelSeverity = 0;
  /** Headlights toggle (auto-enabled in Night Mode or manually toggled). */
  headlightsOn = false;
  private headlightBrokenL = false;
  private headlightBrokenR = false;
  private fuelLevel = 1.0;
  private coolantLevel = 1.0;
  private engineTemp = 88.0;
  private radiatorDamaged = false;
  private engineBroken = false;
  private readonly headlightL: SpotLight;
  private readonly headlightR: SpotLight;
  private readonly taillightL: PointLight;
  private readonly taillightR: PointLight;
  private readonly radiatorNode: number;
  private readonly oilPanNodeA: number;
  private readonly oilPanNodeB: number;
  private readonly fuelTankNode: number;
  private readonly fluidEmit: FluidEmitFrame = {
    radX: 0, radY: 0, radZ: 0,
    oilX: 0, oilY: 0, oilZ: 0,
    fuelX: 0, fuelY: 0, fuelZ: 0,
    carVx: 0, carVy: 0, carVz: 0,
    coolantSeverity: 0,
    oilSeverity: 0,
    fuelSeverity: 0,
    engineDamage: 0,
  };
  private misfirePhase = 0;
  private impactFxCooldown = 0;
  private steerAngle = 0;
  private gear: 1 | -1 = 1;
  private autoGear = 1;
  private engineRpm = 850;
  private speed = 0;
  private prevSpeed = 0;
  private gFiltered = 0;
  private peakG = 0;
  private maxBroken = 0;
  private lateralSledActive = false;
  private frontImpacted = false;
  private rearImpacted = false;
  private leftImpacted = false;
  private rightImpacted = false;
  private readonly doorAngle = new Float32Array(2);
  private readonly doorAngVel = new Float32Array(2);
  private readonly doorDent = new Float32Array(16);
  private spoilerGroup: Group | null = null;
  private spoilerPivot: [number, number, number] = [0, 0, 0];
  private spoilerDeploy = 0;
  private spoilerPitch = 0;
  private spoilerHoldTimer = 0;
  private absActive = false;
  private absPhase = 0;
  private shiftTimer = 0;
  private shiftKickTimer = 0;
  private shiftCount = 0;

  // Live settings controls (wired from Settings -> General)
  steeringSensitivity = 1.0;
  absEnabled = true;
  tcsEnabled = true;
  transmissionMode: TransmissionMode = 'auto';
  /** Current gear in semi/manual mode: -1 = R, 0 = N, 1..7 = forward gears. */
  manualGear = 1;

  /** Shift up one gear: Semi = R -> N -> D; Manual = R -> N -> 1..7. */
  shiftUp(): void {
    if (this.transmissionMode === 'auto') return;
    const maxG = this.transmissionMode === 'semi' ? 1 : 7;
    if (this.manualGear < maxG) {
      this.manualGear++;
      this.shiftCount++;
      this.shiftTimer = 0.11;
      if (this.manualGear > 0) {
        this.gear = 1;
        if (this.transmissionMode === 'manual') this.autoGear = this.manualGear;
        else this.autoGear = 1;
      } else if (this.manualGear === -1) {
        this.gear = -1;
      }
    }
  }

  /** Shift down one gear: Semi = D -> N -> R; Manual = 7..1 -> N -> R. */
  shiftDown(): void {
    if (this.transmissionMode === 'auto') return;
    if (this.transmissionMode === 'semi' && this.manualGear > 1) this.manualGear = 1;
    if (this.manualGear > -1) {
      // Prevent shifting directly into Reverse while rolling forward > 12 km/h
      if (this.manualGear === 0 && this.speed > 3.5) return;
      this.manualGear--;
      this.shiftCount++;
      this.shiftTimer = 0.09;
      if (this.manualGear > 0) {
        this.gear = 1;
        if (this.transmissionMode === 'manual') {
          this.autoGear = this.manualGear;
          this.engineRpm = Math.min(7200, this.engineRpm + 950);
        }
      } else if (this.manualGear === -1) {
        this.gear = -1;
      }
    }
  }

  constructor(src: VehicleVisualSource, world: CollisionWorld, precomputedBindings?: PrecomputedBindingData[] | 'defer') {
    this.src = src;
    this.name = src.name;
    this.profile = src.profile;
    const P = this.profile;
    const isMidRear = /mclaren|p1|ferrari|lamborghini|mid.?engine/i.test(src.name);

    // ------------------------------------------------------------ lattice
    const nodes: SolverBuildNode[] = [];
    const beams: SolverBuildBeam[] = [];
    const coltris: SolverBuildColtri[] = [];
    const nodeZone: Zone[] = [];
    const addNode = (x: number, y: number, z: number, mass: number, group: number, zone: Zone, radius = 0.12, friction = 1) => {
      nodes.push({ x, y, z, mass, radius: Math.max(0.115, radius), group, friction });
      nodeZone.push(zone);
      return nodes.length - 1;
    };
    const addColtri = (a: number, b: number, c: number) => {
      if (a < 0 || b < 0 || c < 0 || a === b || b === c || a === c) return;
      coltris.push({ a, b, c });
    };
    const addColQuad = (a: number, b: number, c: number, d: number) => {
      addColtri(a, b, c);
      addColtri(a, c, d);
    };
    const beamSet = new Set<string>();
    const addBeam = (
      a: number,
      b: number,
      zone?: Zone,
      flags = 0,
      springScale = 1,
      dampScale = 1,
      plasticScale = 1,
      strengthScale = 1,
    ) => {
      if (a === b || a < 0 || b < 0) return -1;
      const key = a < b ? `${a}_${b}` : `${b}_${a}`;
      if (beamSet.has(key)) return -1;
      beamSet.add(key);
      const z = zone ?? weakestZone(nodeZone[a], nodeZone[b]);
      beams.push({ a, b, zone: z, flags, springScale, dampScale, plasticScale, strengthScale });
      return beams.length - 1;
    };

    const S = 11;
    const zF = P.halfL - 0.13, zR = -P.halfL + 0.15;
    const hs = P.height / 1.38;
    const stations: number[][] = []; // lanes: 0 floorL,1 floorC,2 floorR,3 midR,4 beltR,5 roofR,6 roofL,7 beltL,8 midL
    const stationZ: number[] = [];
    const stationCoreZone: Zone[] = [];
    let firstCabin = -1, lastCabin = -1;
    for (let s = 0; s < S; s++) {
      const z = zF + (zR - zF) * (s / (S - 1));
      stationZ.push(z);
      let zone: Zone;
      if (z > P.frontAxleZ + 0.2) zone = Zone.Crumple;
      else if (z > P.cowlZ + 0.02) zone = Zone.Longeron;
      else if (z > P.deckZ + 0.02) zone = Zone.Cabin;
      else if (z > P.rearAxleZ - 0.3) zone = Zone.Longeron;
      else zone = Zone.Crumple;
      stationCoreZone.push(zone);
      if (zone === Zone.Cabin) { if (firstCabin < 0) firstCabin = s; lastCabin = s; }

      const fy = P.floorY + 0.02;
      // Pull inner cabin floor sill rails inboard (~22 cm inside the outer skin) so outer doors/flanks have a real lateral crush zone
      const fw = P.planHalfW(z) - (zone === Zone.Cabin ? 0.22 : 0.13);
      const my = 0.5 * hs;
      const mw = P.sideHalfW(z, my) - 0.03;
      const by = P.beltY(z) - 0.03;
      const bw = P.sideHalfW(z, by) - 0.045;
      const lanes = new Array<number>(9).fill(-1);
      // Front-heavy mass bias (~55% front / ~45% rear on front-engine cars) so nose naturally pitches forward during jumps & drops
      const massBias = isMidRear ? (z < 0 ? 1.08 : 0.92) : (z > 0 ? 1.16 : 0.86);
      // Inner structural core (floor pan, central tunnel, longerons, and cabin roll-cage) -> DeformGroup.Body
      lanes[0] = addNode(fw, fy, z, 20 * massBias, NODE_BODY, zone);
      lanes[1] = addNode(0, fy, z, 13 * massBias, NODE_BODY, zone);
      lanes[2] = addNode(-fw, fy, z, 20 * massBias, NODE_BODY, zone);
      // Outer side body panels (leftSide = SideL, rightSide = SideR) -> separated from inner roll-cage with lower plasticThreshold (Zone.Fender)
      lanes[3] = addNode(-mw, my, z, 7.5, DeformGroup.SideR, Zone.Fender, 0.085, 0.85);
      lanes[4] = addNode(-bw, by, z, 5.5, DeformGroup.SideR, Zone.Fender, 0.085, 0.85);
      lanes[7] = addNode(bw, by, z, 5.5, DeformGroup.SideL, Zone.Fender, 0.085, 0.85);
      lanes[8] = addNode(mw, my, z, 7.5, DeformGroup.SideL, Zone.Fender, 0.085, 0.85);
      const top = P.topY(z);
      if (zone === Zone.Cabin && top - P.beltY(z) > 0.16) {
        const ry = top - 0.045;
        const rw = Math.min(P.roofHalfWidth(z) * 0.86, bw - 0.05);
        lanes[5] = addNode(-rw, ry, z, 6.5, NODE_BODY, Zone.Cabin);
        lanes[6] = addNode(rw, ry, z, 6.5, NODE_BODY, Zone.Cabin);
      }
      stations.push(lanes);
    }

    // ------------------------------------------------------------ Rigid Engine Subframe / Cradle (Zone.EngineSubframe)
    // 6-node cross-braced engine block that does NOT pancake during frontal crashes and pushes into the firewall (tablier) on extreme hits.
    {
      const engZ0 = isMidRear ? P.rearAxleZ + 0.06 : P.cowlZ + 0.12;
      const engZ1 = isMidRear ? P.deckZ - 0.06 : P.frontAxleZ + 0.22;
      const engY0 = P.floorY + 0.08;
      const engY1 = P.floorY + 0.44;
      const engHW = 0.26;
      const engM = isMidRear ? 24 : 28;
      const engNodes = [
        addNode(engHW, engY0, engZ0, engM, NODE_ENGINE, Zone.EngineSubframe, 0.09, 0.9),
        addNode(-engHW, engY0, engZ0, engM, NODE_ENGINE, Zone.EngineSubframe, 0.09, 0.9),
        addNode(0, engY1, engZ0, engM - 2, NODE_ENGINE, Zone.EngineSubframe, 0.09, 0.9),
        addNode(engHW, engY0, engZ1, engM, NODE_ENGINE, Zone.EngineSubframe, 0.09, 0.9),
        addNode(-engHW, engY0, engZ1, engM, NODE_ENGINE, Zone.EngineSubframe, 0.09, 0.9),
        addNode(0, engY1, engZ1, engM - 2, NODE_ENGINE, Zone.EngineSubframe, 0.09, 0.9),
      ];
      this.radiatorNode = stations[0][1];
      this.oilPanNodeA = engNodes[0];
      this.oilPanNodeB = engNodes[4];
      this.fuelTankNode = stations[S - 2][1];
      // Ultra-rigid cross-bracing across all 15 pairs of the engine block
      for (let i = 0; i < engNodes.length; i++) {
        for (let j = i + 1; j < engNodes.length; j++) {
          addBeam(engNodes[i], engNodes[j], Zone.EngineSubframe, BEAM_COLLIDE | BEAM_UNBREAKABLE, 1.15, 1.2, 1.3);
        }
      }
      // Mount engine cradle to firewall (tablier) + subframe longerons
      const fwStation = isMidRear ? stations[lastCabin] : stations[firstCabin];
      for (const en of engNodes.slice(0, 3)) {
        addBeam(en, fwStation[0], Zone.EngineSubframe, BEAM_UNBREAKABLE, 0.95, 1.15, 1.1);
        addBeam(en, fwStation[1], Zone.EngineSubframe, BEAM_UNBREAKABLE, 1.05, 1.15, 1.1);
        addBeam(en, fwStation[2], Zone.EngineSubframe, BEAM_UNBREAKABLE, 0.95, 1.15, 1.1);
      }
      // Connect forward/outboard end to crumple/longeron rails so accordion crumple absorbs energy ahead of engine block
      for (let s = 0; s < S; s++) {
        const dz = Math.abs(stationZ[s] - engZ1);
        if (dz < 0.55) {
          const st = stations[s];
          addBeam(engNodes[3], st[0], Zone.Longeron, BEAM_UNBREAKABLE, 1.0, 1.1, 1.0);
          addBeam(engNodes[4], st[2], Zone.Longeron, BEAM_UNBREAKABLE, 1.0, 1.1, 1.0);
          addBeam(engNodes[5], st[1], Zone.Longeron, BEAM_UNBREAKABLE, 1.0, 1.1, 1.0);
        }
      }
    }

    // Secondary axle / differential / fuel tank mass distribution
    for (let s = 0; s < S; s++) {
      const z = stationZ[s];
      const L = stations[s];
      const otherAxleZ = isMidRear ? P.frontAxleZ : P.rearAxleZ;
      if (Math.abs(z - otherAxleZ) < 0.45) {
        nodes[L[1]].mass += 26;
        nodes[L[0]].mass += 12;
        nodes[L[2]].mass += 12;
      }
    }

    const coreRing = [0, 1, 2, 5, 6];
    const sideRing = [3, 4, 7, 8];
    for (let s = 0; s < S; s++) {
      const A = stations[s];
      const coreZone = stationCoreZone[s];
      const isCabin = coreZone === Zone.Cabin;
      const rearStation = s >= lastCabin;
      const dScale = rearStation ? 1.8 : 1.0;
      const pScale = rearStation ? 1.9 : 1.0;

      // 1. Inner Roll-Cage & Floor Core Bracing (ultra-rigid in Cabin so roof/pillars never collapse on side pole impacts)
      const coreNodes = coreRing.map((k) => A[k]).filter((v) => v >= 0);
      for (let i = 0; i < coreNodes.length; i++) {
        for (let j = i + 1; j < coreNodes.length; j++) {
          addBeam(coreNodes[i], coreNodes[j], coreZone, 0, isCabin ? 1.15 : 1.0, dScale, isCabin ? 1.25 : pScale);
        }
      }

      // 2. Outer Side Body Skins (SideL = [7,8], SideR = [3,4]) connected inward via deformable lateral crush struts
      const latSpring = isCabin ? 0.34 : rearStation ? 0.95 : 0.62;
      const latDamp = isCabin ? 1.45 : 1.25 * dScale;
      const latPlastic = isCabin ? 0.42 : rearStation ? 1.5 : 0.72;
      // vertical side skin ribs
      addBeam(A[3], A[4], Zone.Fender, BEAM_COLLIDE, 0.85, dScale, latPlastic * 1.15);
      addBeam(A[7], A[8], Zone.Fender, BEAM_COLLIDE, 0.85, dScale, latPlastic * 1.15);
      // right outer skin -> right inner sill (A[2]) & center tunnel (A[1])
      addBeam(A[3], A[2], Zone.Fender, 0, latSpring, latDamp, latPlastic);
      addBeam(A[4], A[2], Zone.Fender, 0, latSpring, latDamp, latPlastic);
      addBeam(A[3], A[1], Zone.Fender, 0, latSpring * 0.85, latDamp, latPlastic);
      addBeam(A[4], A[1], Zone.Fender, 0, latSpring * 0.75, latDamp, latPlastic);
      if (A[5] >= 0) addBeam(A[4], A[5], Zone.Fender, 0, 0.42, latDamp, 0.55);
      // left outer skin -> left inner sill (A[0]) & center tunnel (A[1])
      addBeam(A[8], A[0], Zone.Fender, 0, latSpring, latDamp, latPlastic);
      addBeam(A[7], A[0], Zone.Fender, 0, latSpring, latDamp, latPlastic);
      addBeam(A[8], A[1], Zone.Fender, 0, latSpring * 0.85, latDamp, latPlastic);
      addBeam(A[7], A[1], Zone.Fender, 0, latSpring * 0.75, latDamp, latPlastic);
      if (A[6] >= 0) addBeam(A[7], A[6], Zone.Fender, 0, 0.42, latDamp, 0.55);

      if (s + 1 < S) {
        const B = stations[s + 1];
        const nextCabin = stationCoreZone[s + 1] === Zone.Cabin;
        const segLatSpring = (isCabin || nextCabin) ? 0.32 : latSpring;
        const segLatPlastic = (isCabin || nextCabin) ? 0.42 : latPlastic;

        // Inner core longitudinal & diagonal shear bracing
        for (const k of coreRing) {
          if (A[k] >= 0 && B[k] >= 0) addBeam(A[k], B[k], undefined, BEAM_COLLIDE, 1.0, dScale, pScale);
        }
        const corePairs: [number, number][] = [[0, 1], [1, 2], [0, 2], [5, 6], [0, 6], [2, 5], [1, 5], [1, 6]];
        for (const [p, q] of corePairs) {
          if (A[p] >= 0 && B[q] >= 0) addBeam(A[p], B[q], undefined, 0, 1.0, dScale, pScale);
          if (A[q] >= 0 && B[p] >= 0) addBeam(A[q], B[p], undefined, 0, 1.0, dScale, pScale);
        }
        // Roof A-pillar / C-pillar transitions into inner floor longerons & belt shoulders
        for (const k of [5, 6]) {
          const sill = k === 5 ? 2 : 0;
          const shoulder = k === 5 ? 4 : 7;
          if (A[k] >= 0 && B[k] < 0) {
            addBeam(A[k], B[sill], Zone.Cabin, 0, 1.1, dScale, pScale);
            addBeam(A[k], B[1], Zone.Cabin, 0, 1.0, dScale, pScale);
            addBeam(A[k], B[shoulder], Zone.Fender, 0, 0.55, dScale, pScale);
          }
          if (B[k] >= 0 && A[k] < 0) {
            addBeam(B[k], A[sill], Zone.Cabin, 0, 1.1, dScale, pScale);
            addBeam(B[k], A[1], Zone.Cabin, 0, 1.0, dScale, pScale);
            addBeam(B[k], A[shoulder], Zone.Fender, 0, 0.55, dScale, pScale);
          }
        }

        // Outer side skin longitudinal & shear diagonals (with BEAM_COLLIDE so poles/bollards catch side beams)
        for (const k of sideRing) {
          addBeam(A[k], B[k], Zone.Fender, BEAM_COLLIDE, 0.65, dScale, segLatPlastic);
        }
        addBeam(A[3], B[4], Zone.Fender, BEAM_COLLIDE, 0.58, dScale, segLatPlastic);
        addBeam(A[4], B[3], Zone.Fender, BEAM_COLLIDE, 0.58, dScale, segLatPlastic);
        addBeam(A[7], B[8], Zone.Fender, BEAM_COLLIDE, 0.58, dScale, segLatPlastic);
        addBeam(A[8], B[7], Zone.Fender, BEAM_COLLIDE, 0.58, dScale, segLatPlastic);

        // Diagonal lateral crush struts from outer side skin to adjacent station inner floor sill
        addBeam(A[3], B[2], Zone.Fender, 0, segLatSpring, latDamp, segLatPlastic);
        addBeam(B[3], A[2], Zone.Fender, 0, segLatSpring, latDamp, segLatPlastic);
        addBeam(A[4], B[2], Zone.Fender, 0, segLatSpring * 0.85, latDamp, segLatPlastic);
        addBeam(B[4], A[2], Zone.Fender, 0, segLatSpring * 0.85, latDamp, segLatPlastic);

        addBeam(A[8], B[0], Zone.Fender, 0, segLatSpring, latDamp, segLatPlastic);
        addBeam(B[8], A[0], Zone.Fender, 0, segLatSpring, latDamp, segLatPlastic);
        addBeam(A[7], B[0], Zone.Fender, 0, segLatSpring * 0.85, latDamp, segLatPlastic);
        addBeam(B[7], A[0], Zone.Fender, 0, segLatSpring * 0.85, latDamp, segLatPlastic);

        // Surface Collision Triangles (`coltris`) along left/right flanks, sills, and roof
        addColQuad(A[8], A[7], B[7], B[8]); // Left flank outer skin
        addColQuad(A[0], A[8], B[8], B[0]); // Left rocker sill
        addColQuad(A[3], A[4], B[4], B[3]); // Right flank outer skin
        addColQuad(A[2], A[3], B[3], B[2]); // Right rocker sill
        if (A[5] >= 0 && A[6] >= 0 && B[5] >= 0 && B[6] >= 0) {
          addColQuad(A[5], A[6], B[6], B[5]); // Roof panel
        }
      }

      // Next-neighbour bending beams (softer in front/rear crumple zones so accordion folding progresses smoothly)
      if (s + 2 < S) {
        const C = stations[s + 2];
        const isCrumpleSpan = coreZone === Zone.Crumple || stationCoreZone[s + 2] === Zone.Crumple;
        const bendScale = isCrumpleSpan ? 0.36 : rearStation ? 0.85 : 0.58;
        for (const k of coreRing) {
          if (A[k] >= 0 && C[k] >= 0) addBeam(A[k], C[k], undefined, 0, bendScale, dScale, pScale);
        }
        for (const k of sideRing) {
          if (A[k] >= 0 && C[k] >= 0) addBeam(A[k], C[k], Zone.Fender, 0, rearStation ? 0.65 : 0.35, dScale, latPlastic);
        }
      }
    }
    // Cross-diagonal survival cell X-brace across cabin floor & roof corners
    addBeam(stations[firstCabin][0], stations[lastCabin][2], Zone.Cabin, BEAM_UNBREAKABLE, 1.25, 1.2, 1.4);
    addBeam(stations[firstCabin][2], stations[lastCabin][0], Zone.Cabin, BEAM_UNBREAKABLE, 1.25, 1.2, 1.4);
    if (stations[firstCabin][5] >= 0 && stations[lastCabin][6] >= 0) {
      addBeam(stations[firstCabin][5], stations[lastCabin][6], Zone.Cabin, BEAM_UNBREAKABLE, 1.15, 1.2, 1.4);
      addBeam(stations[firstCabin][6], stations[lastCabin][5], Zone.Cabin, BEAM_UNBREAKABLE, 1.15, 1.2, 1.4);
    }

    this.refNodes = [stations[firstCabin][0], stations[firstCabin][2], stations[lastCabin][0], stations[lastCabin][2]];

    // ------------------------------------------------------------ hood (4 x 5 grid: laterally rigid, longitudinally V-foldable)
    const hoodRows = 5, hoodCols = 4;
    const hoodGrid: number[][] = [];
    for (let r = 0; r < hoodRows; r++) {
      const z = (P.cowlZ + 0.05) + ((P.hoodFrontZ - 0.07) - (P.cowlZ + 0.05)) * (r / (hoodRows - 1));
      const belt = P.beltY(z), top = P.topY(z);
      const hw = P.sideHalfW(z, belt) - 0.06;
      const row: number[] = [];
      for (let c = 0; c < hoodCols; c++) {
        const v = -0.88 + (1.76 * c) / (hoodCols - 1);
        const crown = Math.pow(Math.max(0, 1 - v * v), 0.55);
        row.push(addNode(v * hw, belt + (top - belt) * crown - 0.012, z, 1.6, DeformGroup.Hood, Zone.Hood, 0.05, 0.8));
      }
      hoodGrid.push(row);
    }
    for (let r = 0; r < hoodRows; r++) for (let c = 0; c < hoodCols; c++) {
      const n = hoodGrid[r][c];
      if (c + 1 < hoodCols) addBeam(n, hoodGrid[r][c + 1], Zone.Hood, BEAM_COLLIDE, 2.6, 2.8, 3.2);
      if (r + 1 < hoodRows) addBeam(n, hoodGrid[r + 1][c], Zone.Hood, BEAM_COLLIDE | BEAM_HOOD_LONG, 0.92, 1.9, 0.95);
      if (r + 1 < hoodRows && c + 1 < hoodCols) {
        addBeam(n, hoodGrid[r + 1][c + 1], Zone.Hood, 0, 1.55, 2.4, 2.2);
        addBeam(hoodGrid[r][c + 1], hoodGrid[r + 1][c], Zone.Hood, 0, 1.55, 2.4, 2.2);
        addColQuad(n, hoodGrid[r][c + 1], hoodGrid[r + 1][c + 1], hoodGrid[r + 1][c]);
      }
      if (c + 2 < hoodCols) addBeam(n, hoodGrid[r][c + 2], Zone.Hood, 0, 2.2, 2.6, 3.0);
      if (r + 2 < hoodRows) addBeam(n, hoodGrid[r + 2][c], Zone.Hood, 0, 0.68, 1.9, 0.95);
    }
    for (let c = 0; c < hoodCols; c++) {
      const col: number[] = [];
      for (let r = 0; r < hoodRows; r++) col.push(hoodGrid[r][c]);
      this.hoodColumns.push(col);
      this.hoodColumnRest.push(dist(nodes[col[0]], nodes[col[hoodRows - 1]]));
    }

    // ------------------------------------------------------------ trunk lid (3 x 3, heavily cross-braced & damped)
    const trunkGrid: number[][] = [];
    for (let r = 0; r < 3; r++) {
      const z = (P.deckZ - 0.06) + ((P.trunkEndZ + 0.02) - (P.deckZ - 0.06)) * (r / 2);
      const belt = P.beltY(z), top = P.topY(z);
      const hw = P.sideHalfW(z, belt) - 0.08;
      const row: number[] = [];
      for (let c = 0; c < 3; c++) {
        const v = -0.85 + 0.85 * c;
        const crown = Math.pow(Math.max(0, 1 - v * v), 0.55);
        row.push(addNode(v * hw, belt + (top - belt) * crown - 0.015, z, 1.8, DeformGroup.Trunk, Zone.Trunk, 0.05, 0.8));
      }
      trunkGrid.push(row);
    }
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) {
      const n = trunkGrid[r][c];
      if (c + 1 < 3) addBeam(n, trunkGrid[r][c + 1], Zone.Trunk, BEAM_COLLIDE, 1.2, 1.8, 1.8);
      if (r + 1 < 3) addBeam(n, trunkGrid[r + 1][c], Zone.Trunk, BEAM_COLLIDE, 1.2, 1.8, 1.8);
      if (r + 1 < 3 && c + 1 < 3) {
        addBeam(n, trunkGrid[r + 1][c + 1], Zone.Trunk, 0, 1.15, 1.8, 1.8);
        addBeam(trunkGrid[r][c + 1], trunkGrid[r + 1][c], Zone.Trunk, 0, 1.15, 1.8, 1.8);
        addColQuad(n, trunkGrid[r][c + 1], trunkGrid[r + 1][c + 1], trunkGrid[r + 1][c]);
      }
    }
    addBeam(trunkGrid[0][0], trunkGrid[2][0], Zone.Trunk, 0, 1.0, 1.8, 1.8);
    addBeam(trunkGrid[0][1], trunkGrid[2][1], Zone.Trunk, 0, 1.0, 1.8, 1.8);
    addBeam(trunkGrid[0][2], trunkGrid[2][2], Zone.Trunk, 0, 1.0, 1.8, 1.8);
    addBeam(trunkGrid[0][0], trunkGrid[0][2], Zone.Trunk, 0, 1.0, 1.8, 1.8);
    addBeam(trunkGrid[1][0], trunkGrid[1][2], Zone.Trunk, 0, 1.0, 1.8, 1.8);
    addBeam(trunkGrid[2][0], trunkGrid[2][2], Zone.Trunk, 0, 1.0, 1.8, 1.8);
    addBeam(trunkGrid[0][0], trunkGrid[2][2], Zone.Trunk, 0, 1.0, 1.8, 1.8);
    addBeam(trunkGrid[0][2], trunkGrid[2][0], Zone.Trunk, 0, 1.0, 1.8, 1.8);

    // ------------------------------------------------------------ Front Bumper (2-tier 10-node box beam)
    const bumperFLow: number[] = [];
    const bumperFHigh: number[] = [];
    {
      const zc = P.halfL - 0.425;
      const depth = 0.425;
      const A = 0.935 * (P.width / 1.88);
      const yLow = Math.max(0.18, P.floorY + 0.02);
      const yHigh = Math.max(yLow + 0.16, Math.min(0.50 * hs, P.topY(P.halfL - 0.18) - 0.05));
      for (const phi of [-1.15, -0.55, 0, 0.55, 1.15]) {
        const sp = Math.sin(phi), cp = Math.cos(phi);
        const x = A * Math.sign(sp) * Math.sqrt(Math.abs(sp)) * 0.93;
        const zz = depth * Math.sqrt(Math.abs(cp)) * 0.93 - 0.05;
        bumperFLow.push(addNode(x, yLow, zc + zz, 2.2, DeformGroup.BumperF, Zone.Bumper, 0.165, 0.85));
        bumperFHigh.push(addNode(x, yHigh, zc + zz - 0.015, 2.2, DeformGroup.BumperF, Zone.Bumper, 0.125, 0.85));
      }
      for (let i = 0; i + 1 < 5; i++) {
        addBeam(bumperFLow[i], bumperFLow[i + 1], Zone.Bumper, BEAM_COLLIDE | BEAM_UNBREAKABLE, 1.4, 1.6, 1.15);
        addBeam(bumperFHigh[i], bumperFHigh[i + 1], Zone.Bumper, BEAM_COLLIDE | BEAM_UNBREAKABLE, 1.4, 1.6, 1.15);
        addColQuad(bumperFLow[i], bumperFLow[i + 1], bumperFHigh[i + 1], bumperFHigh[i]);
      }
      // Anti-clipping front nose coltris bridging station[0] to front bumper
      addColQuad(stations[0][2], stations[0][1], bumperFLow[2], bumperFLow[0]);
      addColQuad(stations[0][1], stations[0][0], bumperFLow[4], bumperFLow[2]);
      addColQuad(stations[0][4], stations[0][1], bumperFHigh[2], bumperFHigh[0]);
      addColQuad(stations[0][1], stations[0][7], bumperFHigh[4], bumperFHigh[2]);

      for (let i = 0; i < 5; i++) {
        addBeam(bumperFLow[i], bumperFHigh[i], Zone.Bumper, BEAM_COLLIDE | BEAM_UNBREAKABLE, 2.1, 2.0, 2.4);
      }
      for (let i = 0; i + 1 < 5; i++) {
        addBeam(bumperFLow[i], bumperFHigh[i + 1], Zone.Bumper, BEAM_COLLIDE | BEAM_UNBREAKABLE, 1.6, 1.8, 1.6);
        addBeam(bumperFHigh[i], bumperFLow[i + 1], Zone.Bumper, BEAM_COLLIDE | BEAM_UNBREAKABLE, 1.6, 1.8, 1.6);
      }
      for (let i = 0; i + 2 < 5; i++) {
        addBeam(bumperFLow[i], bumperFLow[i + 2], Zone.Bumper, BEAM_UNBREAKABLE, 1.45, 1.7, 1.4);
        addBeam(bumperFHigh[i], bumperFHigh[i + 2], Zone.Bumper, BEAM_UNBREAKABLE, 1.45, 1.7, 1.4);
        addBeam(bumperFLow[i], bumperFHigh[i + 2], Zone.Bumper, BEAM_UNBREAKABLE, 1.35, 1.7, 1.5);
        addBeam(bumperFHigh[i], bumperFLow[i + 2], Zone.Bumper, BEAM_UNBREAKABLE, 1.35, 1.7, 1.5);
      }
      addBeam(bumperFLow[0], bumperFLow[3], Zone.Bumper, BEAM_UNBREAKABLE, 1.3, 1.6, 1.4);
      addBeam(bumperFLow[1], bumperFLow[4], Zone.Bumper, BEAM_UNBREAKABLE, 1.3, 1.6, 1.4);
      addBeam(bumperFLow[0], bumperFLow[4], Zone.Bumper, BEAM_UNBREAKABLE, 1.25, 1.6, 1.4);
      addBeam(bumperFHigh[0], bumperFHigh[3], Zone.Bumper, BEAM_UNBREAKABLE, 1.3, 1.6, 1.4);
      addBeam(bumperFHigh[1], bumperFHigh[4], Zone.Bumper, BEAM_UNBREAKABLE, 1.3, 1.6, 1.4);
      addBeam(bumperFHigh[0], bumperFHigh[4], Zone.Bumper, BEAM_UNBREAKABLE, 1.25, 1.6, 1.4);
    }

    // ------------------------------------------------------------ Rear Bumper (5-node heavily damped beam)
    const bumperR: number[] = [];
    {
      const zc = -P.halfL + 0.375;
      const depth = 0.375;
      const A = 0.935 * (P.width / 1.88);
      for (const phi of [-1.15, -0.55, 0, 0.55, 1.15]) {
        const sp = Math.sin(phi), cp = Math.cos(phi);
        const x = A * Math.sign(sp) * Math.sqrt(Math.abs(sp)) * 0.93;
        const zz = depth * Math.sqrt(Math.abs(cp)) * 0.93 - 0.05;
        const y = 0.44 * hs;
        bumperR.push(addNode(x, y, zc - zz, 3.5, DeformGroup.BumperR, Zone.Bumper, 0.16, 0.8));
      }
      for (let i = 0; i + 1 < bumperR.length; i++) addBeam(bumperR[i], bumperR[i + 1], Zone.Bumper, BEAM_COLLIDE | BEAM_UNBREAKABLE, 1.6, 2.2, 2.5);
      for (let i = 0; i + 2 < bumperR.length; i++) addBeam(bumperR[i], bumperR[i + 2], Zone.Bumper, BEAM_UNBREAKABLE, 1.36, 2.2, 2.5);
      addBeam(bumperR[0], bumperR[3], Zone.Bumper, BEAM_UNBREAKABLE, 1.2, 2.2, 2.5);
      addBeam(bumperR[1], bumperR[4], Zone.Bumper, BEAM_UNBREAKABLE, 1.2, 2.2, 2.5);
      addBeam(bumperR[0], bumperR[4], Zone.Bumper, BEAM_UNBREAKABLE, 1.1, 2.2, 2.5);
      // Rear tail coltris
      const SR = stations[S - 1];
      addColQuad(SR[2], SR[1], bumperR[2], bumperR[0]);
      addColQuad(SR[1], SR[0], bumperR[4], bumperR[2]);
    }

    // ------------------------------------------------------------ Doors (4 x 2 = 8-node rigid stamped-steel door panel per side)
    const doorCols = 4;
    const buildDoor = (side: 1 | -1, grp: DeformGroup): { low: number[]; high: number[]; all: number[] } => {
      const zFront = P.cowlZ - 0.05;
      const zRear = P.rearAxleZ + 0.44;
      const yLow = P.floorY + 0.16;
      const low: number[] = [];
      const high: number[] = [];
      for (let c = 0; c < doorCols; c++) {
        const z = zFront + (zRear - zFront) * (c / (doorCols - 1));
        const yHigh = P.beltY(z) - 0.025;
        low.push(addNode(side * (P.sideHalfW(z, yLow) - 0.015), yLow, z, 2.2, grp, Zone.Longeron, 0.085, 0.85));
        high.push(addNode(side * (P.sideHalfW(z, yHigh) - 0.02), yHigh, z, 2.0, grp, Zone.Longeron, 0.085, 0.85));
      }
      for (let c = 0; c < doorCols; c++) {
        // Stiff stamped-steel door structure (firm, never floppy/rubbery)
        addBeam(low[c], high[c], Zone.Longeron, BEAM_COLLIDE | BEAM_UNBREAKABLE, 1.35, 1.9, 1.35);
        if (c + 1 < doorCols) {
          addBeam(low[c], low[c + 1], Zone.Longeron, BEAM_COLLIDE | BEAM_UNBREAKABLE, 1.25, 1.9, 1.2);
          addBeam(high[c], high[c + 1], Zone.Longeron, BEAM_COLLIDE | BEAM_UNBREAKABLE, 1.25, 1.9, 1.2);
          addBeam(low[c], high[c + 1], Zone.Longeron, BEAM_COLLIDE | BEAM_UNBREAKABLE, 1.15, 1.9, 1.2);
          addBeam(high[c], low[c + 1], Zone.Longeron, BEAM_COLLIDE | BEAM_UNBREAKABLE, 1.15, 1.9, 1.2);
          addColQuad(low[c], low[c + 1], high[c + 1], high[c]);
        }
        if (c + 2 < doorCols) {
          addBeam(low[c], low[c + 2], Zone.Longeron, BEAM_UNBREAKABLE, 1.05, 1.85, 1.15);
          addBeam(high[c], high[c + 2], Zone.Longeron, BEAM_UNBREAKABLE, 1.05, 1.85, 1.15);
        }
      }
      // Full-length door intrusion beam (keeps door frame rigid when swinging open)
      addBeam(low[0], low[doorCols - 1], Zone.Longeron, BEAM_UNBREAKABLE, 1.15, 1.9, 1.25);
      addBeam(high[0], high[doorCols - 1], Zone.Longeron, BEAM_UNBREAKABLE, 1.15, 1.9, 1.25);
      addBeam(low[0], high[doorCols - 1], Zone.Longeron, BEAM_UNBREAKABLE, 1.05, 1.9, 1.25);
      addBeam(high[0], low[doorCols - 1], Zone.Longeron, BEAM_UNBREAKABLE, 1.05, 1.9, 1.25);
      return { low, high, all: [...low, ...high] };
    };
    const doorL = buildDoor(1, DeformGroup.DoorL);
    const doorR = buildDoor(-1, DeformGroup.DoorR);
    this.doorLNodes = doorL;
    this.doorRNodes = doorR;

    // ------------------------------------------------------------ Cascading Hinged & Latched Panel Attachments
    const bodyIdx: number[] = [];
    for (let i = 0; i < nodes.length; i++) {
      const g = nodes[i].group;
      if (g === NODE_BODY || g === DeformGroup.SideL || g === DeformGroup.SideR) bodyIdx.push(i);
    }
    const coreBodyIdx: number[] = [];
    for (let i = 0; i < nodes.length; i++) if (nodes[i].group === NODE_BODY) coreBodyIdx.push(i);

    const nearestBody = (n: number, k: number, filter?: (i: number) => boolean): number[] => {
      return bodyIdx.filter((i) => !filter || filter(i)).map((i) => [i, dist(nodes[i], nodes[n])]).sort((a, b) => a[1] - b[1]).slice(0, k).map((p) => p[0]);
    };
    const attach = (
      part: PartState,
      n: number,
      k: number,
      zone: Zone,
      role: 'hinge' | 'latch' | 'mount',
      flags = 0,
      springScale = 1,
      dampScale = 1,
      plasticScale = 1,
      filter?: (i: number) => boolean,
      strengthScale = 1,
    ): number[] => {
      const out: number[] = [];
      for (const b of nearestBody(n, k, filter)) {
        const id = addBeam(n, b, zone, flags, springScale, dampScale, plasticScale, strengthScale);
        if (id < 0) continue;
        part.beams.push(id);
        if (role === 'hinge') part.hingeBeams.push(id);
        else if (role === 'latch') part.latchBeams.push(id);
        out.push(id);
      }
      return out;
    };
    const attachDirect = (
      part: PartState,
      a: number,
      b: number,
      zone: Zone,
      flags = 0,
      springScale = 1,
      dampScale = 1,
      plasticScale = 1,
      role: 'hinge' | 'latch' | 'mount' = 'mount',
      strengthScale = 1,
    ): number => {
      const id = addBeam(a, b, zone, flags, springScale, dampScale, plasticScale, strengthScale);
      if (id >= 0) {
        part.beams.push(id);
        if (role === 'hinge') part.hingeBeams.push(id);
        else if (role === 'latch') part.latchBeams.push(id);
      }
      return id;
    };

    const isCoreBody = (i: number) => nodes[i].group === NODE_BODY;

    // 1. Hood: hinged at LEFT & RIGHT cowl corners (Zone.Hinge, shears only on > 210 km/h extreme crash), front latches (Zone.Latch) pop open on medium crashes
    const hood: PartState = { name: 'hood', label: 'Hood', group: DeformGroup.Hood, beams: [], hingeBeams: [], latchBeams: [], state: 'attached' };
    attach(hood, hoodGrid[0][0], 3, Zone.Hinge, 'hinge', 0, 1.55, 1.8, 2.2, isCoreBody, 1.65);
    attach(hood, hoodGrid[0][hoodCols - 1], 3, Zone.Hinge, 'hinge', 0, 1.55, 1.8, 2.2, isCoreBody, 1.65);
    for (let c = 0; c < hoodCols; c++) {
      attach(hood, hoodGrid[hoodRows - 1][c], 2, Zone.Latch, 'latch', 0, 1.0, 1.4, 1.0, isCoreBody, 1.0);
    }
    for (let r = 1; r < hoodRows - 1; r++) {
      attach(hood, hoodGrid[r][0], 1, Zone.Latch, 'latch', 0, 0.75, 1.4, 1.0, isCoreBody, 1.0);
      attach(hood, hoodGrid[r][hoodCols - 1], 1, Zone.Latch, 'latch', 0, 0.75, 1.4, 1.0, isCoreBody, 1.0);
    }

    // 2. Trunk: hinged at front deck (Zone.Hinge), latched at rear tail (Zone.Latch)
    const trunk: PartState = { name: 'trunk', label: 'Trunk', group: DeformGroup.Trunk, beams: [], hingeBeams: [], latchBeams: [], state: 'attached' };
    for (let c = 0; c < 3; c++) {
      attach(trunk, trunkGrid[0][c], 3, Zone.Hinge, 'hinge', 0, 1.5, 2.2, 2.2, isCoreBody, 1.55);
    }
    for (let r = 1; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        attach(trunk, trunkGrid[r][c], 3, Zone.Latch, 'latch', 0, 1.35, 2.2, 1.8, isCoreBody, 1.35);
      }
    }

    // 3. Front Bumper: left (i=0,1), center (i=2), and right (i=3,4) mounts — hangs from one intact side before full tear-off
    const bf: PartState = {
      name: 'bumperF', label: 'Front bumper', group: DeformGroup.BumperF,
      beams: [], hingeBeams: [], latchBeams: [], leftBeams: [], rightBeams: [], state: 'attached',
    };
    {
      const F0 = stations[0];
      const floorTargets = [F0[2], F0[2], F0[1], F0[0], F0[0]];
      const midTargets = [F0[3], F0[3], F0[1], F0[8], F0[8]];
      const beltTargets = [F0[4], F0[4], F0[1], F0[7], F0[7]];
      for (let i = 0; i < 5; i++) {
        const nL = bumperFLow[i], nH = bumperFHigh[i];
        const isCenter = i === 2;
        const pScale = isCenter ? 1.30 : 1.15;
        const role = isCenter ? 'mount' : 'latch';
        const b1 = attachDirect(bf, nL, floorTargets[i], Zone.Bumper, 0, 1.55, 1.85, pScale, role);
        const b2 = attachDirect(bf, nL, midTargets[i], Zone.Bumper, 0, 1.45, 1.85, pScale, role);
        const b3 = attachDirect(bf, nH, midTargets[i], Zone.Bumper, 0, 1.55, 1.85, pScale, role);
        const b4 = attachDirect(bf, nH, beltTargets[i], Zone.Bumper, 0, 1.40, 1.85, pScale, role);
        for (const bid of [b1, b2, b3, b4]) {
          if (bid < 0) continue;
          if (i < 2) bf.leftBeams!.push(bid); // i=0,1 is right-side (-X), i=3,4 is left-side (+X)
          else if (i > 2) bf.rightBeams!.push(bid);
        }
      }
    }

    // 4. Rear Bumper: isolated on head-on, hangs from intact corner or detaches on rear crashes
    const br: PartState = {
      name: 'bumperR', label: 'Rear bumper', group: DeformGroup.BumperR,
      beams: [], hingeBeams: [], latchBeams: [], leftBeams: [], rightBeams: [], state: 'attached',
    };
    for (let i = 0; i < bumperR.length; i++) {
      const isCenter = i === 2;
      const ids = attach(br, bumperR[i], 3, Zone.Bumper, isCenter ? 'mount' : 'latch', 0, 1.8, 2.5, isCenter ? 1.6 : 1.35, isCoreBody);
      if (i < 2) br.leftBeams!.push(...ids);
      else if (i > 2) br.rightBeams!.push(...ids);
    }

    // 5. Left & Right Doors: front A-pillar hinges (c = 0, Zone.Hinge) + rear B-pillar latches (c = 3, Zone.Latch)
    //    Medium side/offset crash pops latches -> door swings open ('unlatched'); severe T-bone shears hinges -> door tears off ('detached')!
    const wireDoorAttachments = (part: PartState, door: { low: number[]; high: number[] }, sideGrp: DeformGroup, sideSign: 1 | -1) => {
      const isSideNode = (i: number) => nodes[i].group === sideGrp;
      const isInnerCabinFloor = (i: number) => nodes[i].group === NODE_BODY && nodes[i].y < P.floorY + 0.25 && nodes[i].x * sideSign >= -0.05;
      // Front A-pillar hinge column (c = 0) — welded solid (non-opening doors)
      attach(part, door.low[0], 2, Zone.Hinge, 'hinge', BEAM_UNBREAKABLE, 1.05, 1.5, 0.95, isSideNode, 1.35);
      attach(part, door.high[0], 2, Zone.Hinge, 'hinge', BEAM_UNBREAKABLE, 1.05, 1.5, 0.95, isSideNode, 1.35);
      attach(part, door.low[0], 2, Zone.Hinge, 'hinge', BEAM_UNBREAKABLE, 0.75, 1.5, 0.95, isInnerCabinFloor, 1.35);
      attach(part, door.high[0], 2, Zone.Hinge, 'hinge', BEAM_UNBREAKABLE, 0.70, 1.5, 0.95, isInnerCabinFloor, 1.35);
      // Rear B-pillar latch column (c = 3) — welded solid
      attach(part, door.low[doorCols - 1], 2, Zone.Latch, 'latch', BEAM_UNBREAKABLE, 0.95, 1.4, 0.85, isSideNode, 1.15);
      attach(part, door.high[doorCols - 1], 2, Zone.Latch, 'latch', BEAM_UNBREAKABLE, 0.95, 1.4, 0.85, isSideNode, 1.15);
      attach(part, door.low[doorCols - 1], 2, Zone.Latch, 'latch', BEAM_UNBREAKABLE, 0.55, 1.4, 0.85, isInnerCabinFloor, 1.15);
      // Middle door columns (c = 1, 2) — welded solid
      for (let c = 1; c < doorCols - 1; c++) {
        attach(part, door.low[c], 2, Zone.Latch, 'latch', BEAM_COLLIDE | BEAM_UNBREAKABLE, 0.75, 1.35, 0.80, isSideNode, 1.25);
        attach(part, door.high[c], 2, Zone.Latch, 'latch', BEAM_COLLIDE | BEAM_UNBREAKABLE, 0.75, 1.35, 0.80, isSideNode, 1.25);
        attach(part, door.low[c], 2, Zone.Latch, 'latch', BEAM_UNBREAKABLE, 0.42, 1.5, 0.80, isInnerCabinFloor, 1.25);
      }
    };
    const dl: PartState = { name: 'doorL', label: 'Left door', group: DeformGroup.DoorL, beams: [], hingeBeams: [], latchBeams: [], state: 'attached' };
    wireDoorAttachments(dl, doorL, DeformGroup.SideL, 1);
    const dr: PartState = { name: 'doorR', label: 'Right door', group: DeformGroup.DoorR, beams: [], hingeBeams: [], latchBeams: [], state: 'attached' };
    wireDoorAttachments(dr, doorR, DeformGroup.SideR, -1);
    this.parts.push(hood, trunk, bf, br, dl, dr);

    // ------------------------------------------------------------ Triangulated Wheel Suspension Knuckles (Zone.Suspension)
    const wheelZs = [P.frontAxleZ, P.frontAxleZ, P.rearAxleZ, P.rearAxleZ];
    const lefts = [true, false, true, false];
    const buildProceduralWheels = !src.wheels && !src.suppressProceduralWheels;
    this.wheelGeo = buildProceduralWheels ? createWheelGeometries(P.wheelRadius, P.wheelWidth) : null;

    interface WheelBuildPlan {
      left: boolean;
      z: number;
      x: number;
      wRadius: number;
      wn: number[];
      ww: number[];
      hubNode: number;
      suspBeams: number[];
      customW: CustomWheelAssembly | null;
    }
    const wheelPlans: WheelBuildPlan[] = [];
    for (let w = 0; w < 4; w++) {
      const customW = src.wheels ? src.wheels[w] : null;
      const z = customW ? customW.hubZ : wheelZs[w];
      const wRadius = customW ? customW.radius : P.wheelRadius;
      const x = customW ? customW.hubX : (lefts[w] ? 1 : -1) * P.trackHalf;
      let s0 = 0;
      while (s0 < S - 2 && stationZ[s0 + 1] > z) s0++;
      const t = Math.min(1, Math.max(0, (stationZ[s0] - z) / (stationZ[s0] - stationZ[s0 + 1])));
      const floorLane = lefts[w] ? 0 : 2, midLane = lefts[w] ? 8 : 3;
      const wn = [stations[s0][floorLane], stations[s0][midLane], stations[s0 + 1][floorLane], stations[s0 + 1][midLane]];
      const ww = [(1 - t) * 0.5, (1 - t) * 0.5, t * 0.5, t * 0.5];
      // Dedicated spindle/hub JBeam node at wheel outer track (catches offset barriers, curbs & poles)
      const hubNode = addNode(x, wRadius + 0.04, z, 11.5, NODE_SUSPENSION, Zone.Suspension, 0.135, 0.85);
      const suspBeams = [
        addBeam(hubNode, stations[s0][floorLane], Zone.Suspension, BEAM_COLLIDE, 1.0, 1.1, 1.0, 1.05),     // 0: lower front wishbone
        addBeam(hubNode, stations[s0 + 1][floorLane], Zone.Suspension, BEAM_COLLIDE, 1.0, 1.1, 1.0, 1.05), // 1: lower rear wishbone
        addBeam(hubNode, stations[s0][midLane], Zone.Suspension, 0, 0.85, 1.1, 0.95, 1.0),                 // 2: upper front strut
        addBeam(hubNode, stations[s0 + 1][midLane], Zone.Suspension, 0, 0.85, 1.1, 0.95, 1.0),             // 3: upper rear strut
        addBeam(hubNode, stations[s0][1], Zone.Suspension, 0, 0.95, 1.2, 1.05, 1.10),                      // 4: front tie-rod / subframe link
        addBeam(hubNode, stations[s0 + 1][1], Zone.Suspension, 0, 0.95, 1.2, 1.05, 1.10),                  // 5: rear track-rod / subframe link
      ].filter((id) => id >= 0);
      wheelPlans.push({ left: lefts[w], z, x, wRadius, wn, ww, hubNode, suspBeams, customW });
    }

    // ------------------------------------------------------------ normalise mass
    const target = 1650;
    let sum = 0, partMass = 0;
    for (const n of nodes) {
      if (n.group === NODE_BODY || n.group === NODE_ENGINE) sum += n.mass;
      else partMass += n.mass;
    }
    const scale = (target - partMass) / Math.max(1, sum);
    for (const n of nodes) {
      if (n.group === NODE_BODY || n.group === NODE_ENGINE) n.mass *= scale;
    }
    this.totalMass = target;

    this.solver = new SoftBodySolver(nodes, beams, DEFAULT_ZONES, coltris);
    this.solver.world = world;
    this.solver.forceProvider = this;

    // Precompute zero-allocation latch lookup table (`isLatchBeam`)
    this.isLatchBeam = new Uint8Array(beams.length);
    for (const p of this.parts) {
      for (const lb of p.latchBeams) {
        if (lb >= 0 && lb < beams.length) this.isLatchBeam[lb] = 1;
      }
    }

    const n = nodes.length;
    this.restLocal = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      this.restLocal[i * 3] = nodes[i].x;
      this.restLocal[i * 3 + 1] = nodes[i].y;
      this.restLocal[i * 3 + 2] = nodes[i].z;
    }
    for (const r of this.refNodes) for (let k = 0; k < 3; k++) this.refRestCentroid[k] += this.restLocal[r * 3 + k] / 4;
    this.bodyNodeList = new Uint16Array(coreBodyIdx);
    this.renderWorld = this.solver.pool.f32(n * 3);
    this.renderLocal = this.solver.pool.f32(n * 3);

    // ------------------------------------------------------------ build wheel visuals & state
    for (let w = 0; w < 4; w++) {
      const wp = wheelPlans[w];
      const anchor: [number, number, number] = [0, 0, 0];
      for (let k = 0; k < 4; k++) for (let c = 0; c < 3; c++) anchor[c] += wp.ww[k] * this.restLocal[wp.wn[k] * 3 + c];
      let knuckle: Object3D, spinner: Object3D;
      if (wp.customW) {
        knuckle = new Group();
        spinner = new Group();
        spinner.add(wp.customW.spinner);
        knuckle.add(spinner);
        if (wp.customW.caliper) knuckle.add(wp.customW.caliper);
      } else if (buildProceduralWheels && this.wheelGeo) {
        const v = buildWheelVisual(this.wheelGeo, src.materials, wp.left);
        knuckle = v.knuckle; spinner = v.spinner;
      } else {
        knuckle = new Group();
        spinner = new Group();
        knuckle.add(spinner);
      }
      this.root.add(knuckle);
      this.wheels.push({
        left: wp.left, front: w < 2, driven: w >= 2,
        nodes: wp.wn, weights: wp.ww,
        hubNode: wp.hubNode,
        suspBeams: wp.suspBeams,
        hardLocal: [wp.x, wp.wRadius + this.suspTop, wp.z],
        anchorRest: anchor,
        radius: wp.wRadius,
        s: this.suspTop, sPrev: this.suspTop, contact: false, load: 0,
        spin: 0, omega: 0, steer: 0, slip: 0,
        bentCamber: 0, bentToe: 0, detached: false,
        tyreFlat: 0, tyreLeaking: 0,
        detX: 0, detY: 0, detZ: 0, detVx: 0, detVy: 0, detVz: 0, detYaw: 0,
        knuckle, spinner,
      });
    }
    // static load balance: rest length so the car sits at design ride height
    this.restLen = this.suspTop + (this.totalMass * 9.81 / 4) / this.springK;

    // ------------------------------------------------------------ visuals & skinning
    this.root.matrixAutoUpdate = false;
    const neighbours: number[][] = Array.from({ length: n }, () => []);
    for (const b of beams) {
      if (nodes[b.a].group === nodes[b.b].group) { neighbours[b.a].push(b.b); neighbours[b.b].push(b.a); }
    }
    this.deformer = new MeshDeformer(this.restLocal, this.solver.group, DEFORM_GROUP_COUNT, neighbours);
    for (let i = 0; i < src.parts.length; i++) {
      const part = src.parts[i];
      if (Array.isArray(precomputedBindings) && precomputedBindings[i]) {
        this.deformer.bindPrecomputed(part.mesh, precomputedBindings[i]);
      } else if (precomputedBindings !== 'defer') {
        this.deformer.bind(part.mesh, part.group, this.restLocal);
      }
      part.mesh.matrixAutoUpdate = false;
      part.mesh.updateMatrix();
      this.root.add(part.mesh);
    }
    if (src.rigidMeshes) {
      for (const rm of src.rigidMeshes) {
        rm.matrixAutoUpdate = false;
        rm.updateMatrix();
        this.root.add(rm);
      }
    }
    this.engineMount.name = 'engineMount';
    if (src.engineMeshes && src.engineMeshes.length > 0) {
      for (const em of src.engineMeshes) {
        this.engineMount.add(em);
      }
    }
    this.root.add(this.engineMount);
    if (src.spoilerMeshes && src.spoilerMeshes.length > 0 && src.spoilerPivot) {
      this.spoilerPivot = [src.spoilerPivot[0], src.spoilerPivot[1], src.spoilerPivot[2]];
      this.spoilerGroup = new Group();
      this.spoilerGroup.name = 'activeSpoiler';
      this.spoilerGroup.position.set(this.spoilerPivot[0], this.spoilerPivot[1], this.spoilerPivot[2]);
      for (const sm of src.spoilerMeshes) {
        this.spoilerGroup.add(sm);
      }
      this.root.add(this.spoilerGroup);
    }
    if (precomputedBindings !== 'defer') {
      this.deformer.warmup();
    }

    // Twin real-time headlight beams (car-local, zero per-frame allocations) + high-intensity rear brake/ABS glow
    const hlY = Math.max(0.58, P.beltY(P.halfL - 0.15) - 0.14);
    const hlX = P.width * 0.34;
    const hlZ = P.halfL - 0.08;
    this.headlightL = new SpotLight(0xeef6ff, 0, 58, Math.PI * 0.24, 0.48, 1.35);
    this.headlightL.position.set(hlX, hlY, hlZ);
    this.headlightL.target.position.set(hlX * 0.7, Math.max(0.05, hlY - 0.42), hlZ + 22);
    this.headlightL.castShadow = false;
    this.root.add(this.headlightL);
    this.root.add(this.headlightL.target);

    this.headlightR = new SpotLight(0xeef6ff, 0, 58, Math.PI * 0.24, 0.48, 1.35);
    this.headlightR.position.set(-hlX, hlY, hlZ);
    this.headlightR.target.position.set(-hlX * 0.7, Math.max(0.05, hlY - 0.42), hlZ + 22);
    this.headlightR.castShadow = false;
    this.root.add(this.headlightR);
    this.root.add(this.headlightR.target);

    // Twin rear taillight PointLights positioned directly at the left & right rear light housings
    const tlX = P.width * 0.36;
    const tlY = Math.max(0.68, P.beltY(-P.halfL + 0.14) - 0.03);
    const tlZ = -P.halfL - 0.08;
    this.taillightL = new PointLight(0xff1022, 0, 12, 1.5);
    this.taillightL.position.set(tlX, tlY, tlZ);
    this.root.add(this.taillightL);

    this.taillightR = new PointLight(0xff1022, 0, 12, 1.5);
    this.taillightR.position.set(-tlX, tlY, tlZ);
    this.root.add(this.taillightR);

    this.baseline.track(this.solver.rest);
    this.telemetry = {
      speedKmh: 0, gear: 'N', transmissionMode: 'auto', rpm: 850, throttle: 0, brake: 0, steer: 0, absActive: false, headlightsOn: false, maxTireSlip: 0, shiftCount: 0, gForce: 0, peakGForce: 0,
      impactKmh: 0, peakImpactKmh: 0, absorbedKJ: 0, brokenBeams: 0,
      suspCompression: [0, 0, 0, 0],
      damage: { front: 0, cabin: 0, rear: 0, hood: 0, bumpers: 0, fenders: 0, engine: 0 },
      mech: {
        engineHealth: 1,
        engineStalled: false,
        engineBroken: false,
        engineTemp: 88,
        overheating: false,
        radiatorDamaged: false,
        fuelLevel: 1,
        coolantLevel: 1,
        frontAxleDamage: 0,
        rearAxleDamage: 0,
        axleDamage: [0, 0, 0, 0],
        flatTires: [false, false, false, false],
        headlightBrokenL: false,
        headlightBrokenR: false,
        steeringDamage: 0,
        steeringLocked: false,
        steeringBiasDeg: 0,
        oilLeak: false,
        coolantLeak: false,
        fuelLeak: false,
        coolantSeverity: 0,
        oilSeverity: 0,
        fuelSeverity: 0,
      },
      midRearEngine: isMidRear,
      parts: this.parts.map((p) => ({ label: p.label, state: p.state })),
      nodes: n, beams: beams.length, skinnedVerts: 0,
      simBytes: this.solver.pool.totalBytes + this.deformer.bytes + this.baseline.bytes,
    };
    this.spawn(0, 0, 0, 0);
  }

  /** Hydrates deferred deformer bindings produced by skinningWorker.ts or IndexedDB cache. */
  applyPrecomputedBindings(bindings: PrecomputedBindingData[]): void {
    for (let i = 0; i < this.src.parts.length; i++) {
      const part = this.src.parts[i];
      if (bindings[i]) {
        this.deformer.bindPrecomputed(part.mesh, bindings[i]);
      } else {
        this.deformer.bind(part.mesh, part.group, this.restLocal);
      }
    }
    this.deformer.warmup();
    this.telemetry.simBytes = this.solver.pool.totalBytes + this.deformer.bytes + this.baseline.bytes;
  }

  /** Returns the visual source parts list (used by EngineCore for Web Worker skinning input). */
  getVisualParts(): CarPart[] {
    return this.src.parts;
  }

  // ================================================================ spawn / reset

  /** Teleport to (x, z) with heading `yaw`, optional initial speed. Allocation free, well under 2 ms. */
  spawn(x: number, z: number, yaw: number, speed: number, lateral = false): void {
    const S = this.solver;
    const c = Math.cos(yaw), s = Math.sin(yaw);
    // forward = (sin, 0, cos); left = (cos, 0, -sin)
    const vx = lateral ? c * speed : s * speed;
    const vz = lateral ? -s * speed : c * speed;
    const R = this.restLocal;
    let groundY = 0;
    if (S.world) {
      S.world.groundAt(x, z, this.ground);
      groundY = this.ground.h;
    }
    for (let i = 0; i < S.nodeCount; i++) {
      const lx = R[i * 3], ly = R[i * 3 + 1], lz = R[i * 3 + 2];
      S.px[i] = x + lx * c + lz * s;
      S.py[i] = groundY + ly + 0.01;
      S.pz[i] = z - lx * s + lz * c;
      S.vx[i] = vx; S.vy[i] = 0; S.vz[i] = vz;
      S.fx[i] = 0; S.fy[i] = 0; S.fz[i] = 0;
    }
    S.qx.set(S.px); S.qy.set(S.py); S.qz.set(S.pz);
    S.snapshotStepStart();
    S.resetBeams();
    this.deformer.restore(this.restLocal);
    this.glass.reset();
    this.fluids.reset();
    this.glassShattered.fill(0);
    this.axleDmg.fill(0);
    this.engineDmg = 0;
    this.engineStalled = false;
    this.frontAxleDmg = 0;
    this.rearAxleDmg = 0;
    this.steeringDmg = 0;
    this.steeringBias = 0;
    this.steeringLocked = false;
    this.steeringLockAngle = 0;
    this.oilLeak = false;
    this.coolantLeak = false;
    this.fuelLeak = false;
    this.coolantSeverity = 0;
    this.oilSeverity = 0;
    this.fuelSeverity = 0;
    this.fuelLevel = 1.0;
    this.coolantLevel = 1.0;
    this.engineTemp = 88.0;
    this.radiatorDamaged = false;
    this.engineBroken = false;
    this.headlightBrokenL = false;
    this.headlightBrokenR = false;
    this.misfirePhase = 0;
    this.engineMount.position.set(0, 0, 0);
    this.engineMount.rotation.set(0, 0, 0);
    this.engineMount.scale.set(1, 1, 1);
    for (const w of this.wheels) {
      w.s = this.suspTop; w.sPrev = this.suspTop; w.spin = 0; w.omega = 0; w.steer = 0; w.contact = false;
      w.bentCamber = 0; w.bentToe = 0; w.detached = false;
      w.tyreFlat = 0; w.tyreLeaking = 0;
      w.knuckle.rotation.set(0, 0, 0);
      w.knuckle.scale.set(1, 1, 1);
    }
    for (const p of this.parts) p.state = 'attached';
    this.doorAngle.fill(0);
    this.doorAngVel.fill(0);
    this.doorDent.fill(0);
    this.steerAngle = 0;
    this.gear = 1;
    this.autoGear = 1;
    this.manualGear = 1;
    this.engineRpm = 850;
    this.speed = speed;
    this.prevSpeed = speed;
    this.gFiltered = 0;
    this.peakG = 0;
    this.maxBroken = 0;
    this.autopilotSpeed = 0;
    this.lateralSledActive = lateral && speed > 0;
    this.frontImpacted = false;
    this.rearImpacted = false;
    this.leftImpacted = false;
    this.rightImpacted = false;
    this.spoilerDeploy = 0;
    this.spoilerPitch = 0;
    this.spoilerHoldTimer = 0;
    this.shiftTimer = 0;
    this.shiftKickTimer = 0;
    if (this.spoilerGroup) {
      this.spoilerGroup.position.set(this.spoilerPivot[0], this.spoilerPivot[1], this.spoilerPivot[2]);
      this.spoilerGroup.rotation.x = 0;
    }
    this.updateFrame(S.px, S.py, S.pz);
  }

  // ================================================================ physics

  /** Body frame (origin + orthonormal axes) from the four cabin floor reference nodes. */
  private updateFrame(px: ArrayLike<number>, py: ArrayLike<number>, pz: ArrayLike<number>): void {
    const [a, b, c, d] = this.refNodes;
    const F = this.frame;
    let zx = (px[a] + px[b] - px[c] - px[d]) * 0.5, zy = (py[a] + py[b] - py[c] - py[d]) * 0.5, zz = (pz[a] + pz[b] - pz[c] - pz[d]) * 0.5;
    let l = Math.sqrt(zx * zx + zy * zy + zz * zz) || 1;
    zx /= l; zy /= l; zz /= l;
    let xx = px[a] - px[b] + px[c] - px[d], xy = py[a] - py[b] + py[c] - py[d], xz = pz[a] - pz[b] + pz[c] - pz[d];
    // Y = Z x X
    let yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
    l = Math.sqrt(yx * yx + yy * yy + yz * yz) || 1;
    yx /= l; yy /= l; yz /= l;
    // X = Y x Z
    xx = yy * zz - yz * zy; xy = yz * zx - yx * zz; xz = yx * zy - yy * zx;
    F[3] = xx; F[4] = xy; F[5] = xz;
    F[6] = yx; F[7] = yy; F[8] = yz;
    F[9] = zx; F[10] = zy; F[11] = zz;
    const cx = (px[a] + px[b] + px[c] + px[d]) * 0.25, cy = (py[a] + py[b] + py[c] + py[d]) * 0.25, cz = (pz[a] + pz[b] + pz[c] + pz[d]) * 0.25;
    const r = this.refRestCentroid;
    F[0] = cx - (xx * r[0] + yx * r[1] + zx * r[2]);
    F[1] = cy - (xy * r[0] + yy * r[1] + zy * r[2]);
    F[2] = cz - (xz * r[0] + yz * r[1] + zz * r[2]);
  }

  /** Called by the solver once per substep: suspension, tyres, drivetrain, aero, hood buckling bias. */
  applySubstepForces(h: number): void {
    const S = this.solver;
    const { px, py, pz, vx, vy, vz, fx, fy, fz } = S;
    this.updateFrame(px, py, pz);
    const F = this.frame;
    const Xx = F[3], Xy = F[4], Xz = F[5], Yx = F[6], Yy = F[7], Yz = F[8], Zx = F[9], Zy = F[10], Zz = F[11];

    // body velocity
    let bvx = 0, bvy = 0, bvz = 0;
    for (let k = 0; k < 4; k++) { const r = this.refNodes[k]; bvx += vx[r]; bvy += vy[r]; bvz += vz[r]; }
    bvx *= 0.25; bvy *= 0.25; bvz *= 0.25;
    const vForward = bvx * Zx + bvy * Zy + bvz * Zz;
    const vAbs = Math.sqrt(bvx * bvx + bvy * bvy + bvz * bvz);
    this.speed = this.lateralSledActive ? vAbs : vForward;

    // ---- driver / autopilot
    const inp = this.input;
    if (this.lateralSledActive && (S.lastImpact > 1.0 || S.peakImpact > 1.0 || inp.throttle > 0 || inp.brake > 0 || Math.abs(inp.steer) > 0.05)) {
      this.lateralSledActive = false;
    }
    let throttle = inp.throttle, brake = inp.brake;
    if (this.autopilotSpeed > 0) {
      if (S.lastImpact > 2 || inp.brake > 0 || inp.throttle > 0 || this.engineStalled) this.autopilotSpeed = 0;
      else { const err = this.autopilotSpeed - vForward; throttle = Math.max(0, Math.min(1, err * 0.8)); brake = err < -1 ? 0.3 : 0; }
    }

    let drive = 0, brakeIn = 0;
    if (this.engineStalled) {
      // Stalled engine: ZERO forward or reverse propulsion, brakes ALWAYS work at 100%!
      drive = 0;
      brakeIn = Math.max(brake, inp.handbrake);
    } else if (this.transmissionMode === 'auto') {
      if (this.gear === 1 && brake > 0.1 && vForward < 0.6 && throttle < 0.05) this.gear = -1;
      else if (this.gear === -1 && throttle > 0.1 && vForward > -0.6) this.gear = 1;
      if (this.gear === 1) { drive = throttle; brakeIn = brake; }
      else { drive = -brake; brakeIn = throttle; }
    } else {
      // Semi-Automatic & Full Manual: throttle drives in selected gear (R / N / 1..7), brake ALWAYS brakes!
      brakeIn = brake;
      if (this.manualGear === 0) {
        drive = 0; // Neutral: free rev, zero drive
      } else if (this.manualGear === -1) {
        this.gear = -1;
        drive = -throttle;
      } else {
        this.gear = 1;
        if (this.transmissionMode === 'manual') {
          this.autoGear = this.manualGear;
        }
        drive = throttle;
      }
    }
    const autoHold = !this.lateralSledActive && Math.abs(drive) < 0.02 && Math.abs(vForward) < 0.85 && this.autopilotSpeed === 0;

    const vSteer = Math.abs(vForward);
    // Progressive speed-sensitive steering curve + steering rack damage / lock after major front impacts
    const rackEfficiency = Math.max(0.28, 1 - this.steeringDmg * 0.72);
    const maxSteer = (0.64 / (1 + Math.pow(vSteer / 14.5, 1.35))) * rackEfficiency;
    const rawTarget = -inp.steer * maxSteer + this.steeringBias;
    if (this.steeringLocked) {
      // Steering rack jammed solid — player input can barely flex the bent column (±0.035 rad)
      const jammedTarget = this.steeringLockAngle - inp.steer * 0.035;
      this.steerAngle += (jammedTarget - this.steerAngle) * Math.min(1, 14 * h);
    } else {
      const isCentering = Math.abs(inp.steer) < 0.05 || Math.sign(rawTarget - this.steeringBias) !== Math.sign(this.steerAngle - this.steeringBias);
      const sens = Math.max(0.45, Math.min(1.6, this.steeringSensitivity)) * rackEfficiency;
      const steerRate = (isCentering
        ? (5.4 + Math.min(4.0, vSteer * 0.22)) * rackEfficiency
        : (3.6 + (1 - Math.min(1, vSteer / 24)) * 1.6) * sens) * h;
      this.steerAngle += Math.max(-steerRate, Math.min(steerRate, rawTarget - this.steerAngle));
    }

    // True Ackermann geometry: inner front wheel turns tighter than outer wheel around the turn center
    let steerFL = 0, steerFR = 0;
    if (Math.abs(this.steerAngle) > 1e-4) {
      const wb = Math.max(2.0, this.profile.frontAxleZ - this.profile.rearAxleZ);
      const halfT = this.profile.trackHalf;
      const turnR = wb / Math.tan(Math.abs(this.steerAngle));
      const inner = Math.atan(wb / Math.max(0.85, turnR - halfT));
      const outer = Math.atan(wb / (turnR + halfT));
      if (this.steerAngle > 0) {
        steerFL = inner;
        steerFR = outer;
      } else {
        steerFL = -outer;
        steerFR = -inner;
      }
    }

    // ---- 7-speed transmission, realistic redline rev-limiter fuel cut & engine damage power loss
    const gRatio = this.gear === -1 ? 3.1 : GEAR_RATIOS[Math.max(0, Math.min(6, this.autoGear - 1))];
    const isSupercar = this.profile.width > 1.98 && this.profile.height < 1.25;
    const isMuscle = this.totalMass > 1750;
    const baseRedline = isSupercar ? 8000 : isMuscle ? 6800 : 7600;
    const effectiveRedline = baseRedline * Math.max(0.45, 1 - this.engineDmg * 0.42);
    const rpmScale = isSupercar ? 3.2 : isMuscle ? 2.75 : 2.95;
    const wheelRpmNow = (Math.abs(vForward) / Math.max(0.25, this.profile.wheelRadius)) * (60 / (2 * Math.PI));
    const coupledRpmNow = 850 + wheelRpmNow * gRatio * rpmScale;
    const peakRpmNow = Math.max(this.engineRpm, coupledRpmNow);

    // Realistic hard rev-limiter: tapers fuel over the last 180 RPM and cuts to 0 drive force at redline!
    let revLimitFactor = 1.0;
    if (peakRpmNow >= effectiveRedline - 180) {
      revLimitFactor = Math.max(0, (effectiveRedline - peakRpmNow) / 180);
    }

    const rpmNorm = Math.max(0, Math.min(1, (this.engineRpm - 850) / Math.max(1000, effectiveRedline - 850)));
    const torqueCurve = 0.68 + 0.36 * Math.sin(rpmNorm * Math.PI * 0.92);
    const clutchFactor = this.shiftTimer > 0 ? 0.08 : this.shiftKickTimer > 0 ? 1.18 : 1.0;
    // Engine damage genuinely degrades acceleration, causes misfiring above 35%, and stalls completely at >= 82%
    let enginePowerFactor = 1.0;
    if (this.engineStalled) {
      enginePowerFactor = 0;
    } else if (this.engineDmg > 0.04) {
      enginePowerFactor = Math.max(0.14, 1 - this.engineDmg * 0.88);
      if (this.engineDmg > 0.35) {
        this.misfirePhase += h * (28 + rpmNorm * 42);
        const misfireDip = Math.sin(this.misfirePhase) > 0.45 ? (1 - this.engineDmg * 0.52) : 1.0;
        enginePowerFactor *= misfireDip;
      }
    }
    const rawDrive = Math.min(
      this.maxDriveForce * Math.max(0.15, enginePowerFactor),
      ((this.enginePower * enginePowerFactor) / Math.max(4.5, Math.abs(vForward))) * torqueCurve * Math.min(1.25, 0.62 + gRatio * 0.22),
    );
    let driveForce = drive * rawDrive * clutchFactor * revLimitFactor * (this.gear === -1 ? 0.48 : 1);
    // Over-rev engine braking if wheel speed exceeds current gear's redline (e.g. aggressive downshift)
    if (coupledRpmNow > effectiveRedline + 60 && !this.lateralSledActive) {
      driveForce -= Math.sign(vForward || this.gear) * Math.min(3600, (coupledRpmNow - effectiveRedline) * 4.5);
    }
    if (this.gear === 1 && drive < 0.02 && brakeIn < 0.02 && vForward > 2.5 && !this.lateralSledActive) {
      driveForce -= gRatio * (180 + vForward * 14);
    }
    const massShare = this.totalMass / 4;
    const gs = this.ground;

    // ---- independent spring-damper suspension + balanced Pacejka tyres + detached wheel physics
    let sFL = 0, sFR = 0, sRL = 0, sRR = 0;
    let totalLongForce = 0;
    let totalLatForce = 0;
    let groundedCount = 0;
    for (let w = 0; w < 4; w++) {
      const W = this.wheels[w];
      if (W.detached) {
        // Lightweight rigid-wheel rolling & ground bounce in world space (zero allocations)
        W.detVy -= 9.81 * h;
        W.detVx *= 1 - 0.18 * h;
        W.detVz *= 1 - 0.18 * h;
        W.detX += W.detVx * h;
        W.detY += W.detVy * h;
        W.detZ += W.detVz * h;
        if (S.world) {
          S.world.groundAt(W.detX, W.detZ, gs);
          const minY = gs.h + W.radius * 0.92;
          if (W.detY < minY) {
            W.detY = minY;
            if (W.detVy < -0.8) {
              W.detVy = -W.detVy * 0.36;
            } else {
              W.detVy = 0;
            }
            W.detVx *= 1 - 0.65 * h;
            W.detVz *= 1 - 0.65 * h;
            const fwdV = W.detVx * Math.sin(W.detYaw) + W.detVz * Math.cos(W.detYaw);
            W.omega = fwdV / Math.max(0.2, W.radius);
          }
        }
        W.contact = false;
        W.load = 0;
        continue;
      }

      const wn = W.nodes, ww = W.weights;
      let ax = 0, ay = 0, az = 0, avx = 0, avy = 0, avz = 0;
      for (let k = 0; k < 4; k++) {
        const i = wn[k], q = ww[k];
        ax += px[i] * q; ay += py[i] * q; az += pz[i] * q;
        avx += vx[i] * q; avy += vy[i] * q; avz += vz[i] * q;
      }
      // hard point = anchor + R * (hardLocal - anchorRest)
      const ox = W.hardLocal[0] - W.anchorRest[0], oy = W.hardLocal[1] - W.anchorRest[1], oz = W.hardLocal[2] - W.anchorRest[2];
      const hx = ax + Xx * ox + Yx * oy + Zx * oz;
      const hy = ay + Xy * ox + Yy * oy + Zy * oz;
      const hz = az + Xz * ox + Yz * oy + Zz * oz;
      W.sPrev = W.s;
      W.contact = false;
      W.load = 0;
      W.steer = (W.front ? (W.left ? steerFL : steerFR) : 0) + W.bentToe;
      if (Yy < 0.25) { W.s = this.sMax; W.omega *= 1 - 1.5 * h; continue; }
      // Progressive tire deflation when punctured (accelerates slightly while rolling)
      if (W.tyreLeaking > 0 && W.tyreFlat < 1) {
        const rollBoost = 1 + Math.min(1.5, Math.abs(W.omega) * 0.04);
        W.tyreFlat = Math.min(1, W.tyreFlat + W.tyreLeaking * rollBoost * h);
      }
      const effRadius = W.radius * (1 - W.tyreFlat * 0.22);
      S.world!.groundAt(hx, hz, gs);
      const t = (hy - gs.h) / Math.max(0.25, Yy); // distance along -Y to the ground
      let s = t - effRadius;
      if (s > this.sMax) {
        W.s = this.sMax;
        W.omega *= 1 - 1.2 * h;
        continue;
      }
      if (s < 0) s = 0;
      W.s = s;
      W.contact = true;
      groundedCount++;
      const sVel = (W.s - W.sPrev) / h;
      const dampC = sVel < 0 ? this.damperBump * 1.18 : this.damperRebound * 1.25;
      let Fs = this.springK * (this.restLen - s) - dampC * sVel;
      // Progressive heavy-duty bump stop + hard bottom-out impulse transfer into chassis/subframe (no trampoline bounce!)
      if (s < this.sMin + 0.05) {
        const bs = this.sMin + 0.05 - s;
        // Heavily damped bump stop absorbs energy instead of launching the car back upward
        const bumpDamp = sVel < 0 ? -sVel * 14500 : 0;
        Fs += Math.min(85000, 145000 * bs + bumpDamp);
        if (avy < -5.2) {
          const excessV = Math.min(22, -avy - 5.2);
          // Hard curb/landing slam can puncture or blow out the tire
          if (excessV > 1.2) {
            W.tyreLeaking = Math.max(W.tyreLeaking, 0.38 + excessV * 0.09);
            if (excessV > 8.0) W.tyreFlat = Math.min(1, W.tyreFlat + 0.45);
          }
          // Transfer excess vertical slam impulse directly into wheel arch / lower chassis / subframe soft-body nodes
          const slamForce = excessV * massShare * 18.5;
          if (W.hubNode >= 0) {
            fy[W.hubNode] += slamForce * 0.55;
            vy[W.hubNode] *= 0.82;
          }
          for (let k = 0; k < 4; k++) {
            const ni = wn[k];
            fy[ni] += slamForce * ww[k] * 0.45;
          }
          const slamDmg = Math.min(0.28, excessV * 0.012 * h * 60);
          this.axleDmg[w] = Math.min(1, this.axleDmg[w] + slamDmg);
          if (excessV + 5.2 > S.lastImpact) S.lastImpact = excessV + 5.2;
          if (S.lastImpact > S.peakImpact) S.peakImpact = S.lastImpact;
        }
      }
      if (Fs < 0) Fs = 0;
      if (w === 0) sFL = s; else if (w === 1) sFR = s; else if (w === 2) sRL = s; else sRR = s;
      W.load = Fs;

      // wheel heading projected on the ground plane
      const cs = Math.cos(W.steer), sn = Math.sin(W.steer);
      let fwx = Zx * cs + Xx * sn, fwy = Zy * cs + Xy * sn, fwz = Zz * cs + Xz * sn;
      const dn = fwx * gs.nx + fwy * gs.ny + fwz * gs.nz;
      fwx -= dn * gs.nx; fwy -= dn * gs.ny; fwz -= dn * gs.nz;
      const fl = Math.sqrt(fwx * fwx + fwy * fwy + fwz * fwz) || 1;
      fwx /= fl; fwy /= fl; fwz /= fl;
      // lateral = n x fwd (points left)
      const lx = gs.ny * fwz - gs.nz * fwy, ly = gs.nz * fwx - gs.nx * fwz, lz = gs.nx * fwy - gs.ny * fwx;
      const vLong = avx * fwx + avy * fwy + avz * fwz;
      const vLat = avx * lx + avy * ly + avz * lz;
      // Bent suspension camber & flat tire reduce effective tyre contact patch grip realistically
      const camberPenalty = 1 - Math.min(0.32, Math.abs(W.bentCamber) * 0.75);
      const flatPenalty = 1 - W.tyreFlat * 0.44;
      const hbActive = inp.handbrake > 0 && Math.abs(vForward) > 2.5;
      // Front tyres keep slightly stronger bite during handbrake/drift so the nose stays planted for counter-steering
      const driftFrontBoost = (W.front && hbActive) ? 1.10 : 1.0;
      const wetGrip = 1 - this.wetness * 0.14;
      const mu = this.tireMu * gs.mu * camberPenalty * flatPenalty * driftFrontBoost * wetGrip;
      const Fmax = mu * Fs;

      // Progressive Pacejka lateral slip curve (wider controllable drift sweet spot on rear tyres)
      const slip = Math.atan2(vLat, Math.max(Math.abs(vLong), 2.2));
      W.slip = slip;
      const pacejkaShape = !W.front && Math.abs(slip) > 0.28 ? 1.24 : 1.42;
      let Fy = -Fmax * Math.sin(pacejkaShape * Math.atan(7.8 * slip));
      const latCancel = massShare * Math.abs(vLat) / h * 0.5;
      if (autoHold && gs.ny > 0.90) {
        Fy = -Math.sign(vLat) * Math.min(Fmax * 0.95, latCancel);
      } else if (Math.abs(this.steerAngle) < 0.015 && !this.lateralSledActive && inp.handbrake === 0 && Math.abs(W.bentToe) < 0.02) {
        Fy -= vLat * Math.min(massShare * 3.2, Fmax * 0.18);
      }
      if (Math.abs(Fy) > latCancel) Fy = Math.sign(Fy) * latCancel;
      if (this.lateralSledActive) Fy *= 0.03;

      const axDmg = this.axleDmg[w];
      let Fx = 0;
      if (W.driven) {
        // Damaged or broken half-shaft/axle cuts drive torque on this wheel
        const halfShaftFactor = axDmg > 0.78 ? 0 : Math.max(0.15, 1 - axDmg * 0.88);
        let df = driveForce * 0.5 * halfShaftFactor;
        // Allow smooth RWD power-oversteer drift when steering hard + throttle (unless TCS is strictly clamping straight-line spin)
        if (this.tcsEnabled && !hbActive && Math.abs(this.steerAngle) < 0.14 && Math.abs(slip) > 0.24 && Math.abs(df) > Fmax * 0.68) {
          df *= 0.68;
        }
        // During RWD power-slide, slightly soften rear lateral grip so throttle smoothly sustains the drift angle
        if (!W.front && !hbActive && drive > 0.45 && Math.abs(slip) > 0.20) {
          Fy *= 0.76;
        }
        Fx += df;
      }
      const brakeBias = W.front ? 0.62 : 0.38;
      if (w === 0) {
        this.absActive = this.absEnabled && brakeIn > 0.35 && Math.abs(vForward) > 3.5 && !autoHold && inp.handbrake === 0;
        if (this.absActive) this.absPhase += h * 88;
      }
      const absPulse = this.absActive ? (0.76 + 0.24 * Math.sin(this.absPhase)) : 1.0;
      let brk = brakeIn * this.maxBrakeForce * brakeBias * absPulse;
      if (autoHold) brk = this.maxBrakeForce;
      const longCancel = massShare * Math.abs(vLong) / h * 0.5;
      if (brk > 0) Fx -= Math.sign(vLong) * Math.min(brk, longCancel);
      if (!W.front && inp.handbrake > 0) {
        // Crisp drift handbrake: locks rear wheels, drops rear lateral traction to 22%, and initiates clean yaw swing
        const hbBrk = Math.abs(this.steerAngle) > 0.08 ? Fmax * 0.54 : Fmax * 0.84;
        Fx = -Math.sign(vLong) * Math.min(hbBrk, longCancel);
        Fy *= Math.abs(vForward) > 5.0 ? 0.22 : 0.38;
        if (Math.abs(this.steerAngle) > 0.06 && Math.abs(vForward) > 5.5) {
          // Natural handbrake drift initiation kick opposite to turn center (rear kicks out cleanly)
          const kick = -Math.sign(this.steerAngle) * Math.min(1, Math.abs(this.steerAngle) * 2.4) * Math.min(1, Math.abs(vForward) / 14) * Fmax * 0.34;
          Fy += kick;
        }
      }
      // rolling resistance (dramatically higher if wheel spindle/axle is bent, crushed, or tire is flat!)
      Fx -= vLong * (11 + Math.abs(W.bentCamber) * 65 + axDmg * 140 + W.tyreFlat * 95);
      if (W.tyreFlat > 0.15 && Math.abs(vLong) > 0.4) {
        // Heavy flat-rim rolling drag naturally pulls the vehicle toward the punctured side
        Fx -= Math.sign(vLong) * W.tyreFlat * Fs * 0.12;
      }
      // combined-slip friction circle
      const tot = Math.sqrt(Fx * Fx + Fy * Fy);
      if (tot > Fmax && tot > 0) { const k = Fmax / tot; Fx *= k; Fy *= k; }
      totalLongForce += Fx * cs - Fy * sn;
      totalLatForce += Fy * cs + Fx * sn;
      W.omega = vLong / Math.max(0.18, effRadius);
      if (W.driven && Math.abs(driveForce) > Fmax * 0.9 && Math.abs(vLong) < 12) W.omega += Math.sign(driveForce) * 5.5;
      if (!W.front && inp.handbrake > 0) W.omega = 0;

      // Apply normal load strictly along ground normal gs.n* (eliminates pitched-chassis reverse drift & supports banked curves!)
      const Fxw = gs.nx * Fs + fwx * Fx + lx * Fy;
      const Fyw = gs.ny * Fs + fwy * Fx + ly * Fy;
      const Fzw = gs.nz * Fs + fwz * Fx + lz * Fy;
      for (let k = 0; k < 4; k++) {
        const i = wn[k], q = ww[k];
        fx[i] += Fxw * q; fy[i] += Fyw * q; fz[i] += Fzw * q;
      }
    }
    // anti-roll bars
    this.applyAntiRoll(0, 1, sFL, sFR, Yx, Yy, Yz);
    this.applyAntiRoll(2, 3, sRL, sRR, Yx, Yy, Yz);

    // Realistic G-Force Weight Transfer on Suspension (controlled brake dive without excessive rear lift)
    if (groundedCount >= 2 && Yy > 0.72 && !this.lateralSledActive) {
      const wb = Math.max(2.2, this.profile.frontAxleZ - this.profile.rearAxleZ);
      const cgH = 0.56;
      const pitchCouple = Math.max(-920, Math.min(1850, (totalLongForce * cgH) / wb));
      const rollCouple = Math.max(-1350, Math.min(1350, totalLatForce * 0.14));
      if (Math.abs(pitchCouple) > 25 || Math.abs(rollCouple) > 25) {
        for (let w = 0; w < 4; w++) {
          const W = this.wheels[w];
          if (!W.contact || W.detached) continue;
          // Under braking (pitchCouple < 0), compress front mildly and keep rear planted instead of lifting high
          const fPitch = W.front
            ? (pitchCouple < 0 ? pitchCouple * 0.85 : pitchCouple)
            : (pitchCouple < 0 ? -pitchCouple * 0.32 : -pitchCouple * 0.85);
          const fRoll = W.left ? rollCouple : -rollCouple;
          const fVert = fPitch + fRoll;
          const wn = W.nodes, ww = W.weights;
          for (let k = 0; k < 4; k++) {
            const i = wn[k], q = ww[k];
            fx[i] += Yx * fVert * q;
            fy[i] += Yy * fVert * q;
            fz[i] += Yz * fVert * q;
          }
        }
      }
    }

    // Horizontal-only aerodynamic drag
    const drag = 0.42 / this.totalMass;
    const vHoriz = Math.sqrt(bvx * bvx + bvz * bvz);
    if (vHoriz > 1) {
      const list = this.bodyNodeList;
      for (let k = 0; k < list.length; k++) {
        const i = list[k];
        const m = S.mass[i] * drag * vHoriz;
        fx[i] -= m * vx[i];
        fz[i] -= m * vz[i];
      }
    }

    // Zero-allocation anti-jelly sheet-metal & bumper settling damper (hood vibrates only during active impact, then settles cleanly!)
    {
      const hoodDetached = this.parts[0].state === 'detached';
      const bfDetached = this.parts[2].state === 'detached';
      const brDetached = this.parts[3].state === 'detached';
      const activeImpact = S.lastImpact > 1.8;
      const grp = S.group;
      for (let i = 0; i < S.nodeCount; i++) {
        const g = grp[i];
        let kDamp = 0;
        if (g === DeformGroup.Hood && !hoodDetached) {
          kDamp = activeImpact ? 11.0 : 34.0;
        } else if (g === DeformGroup.BumperF && !bfDetached) {
          kDamp = activeImpact ? 12.5 : 24.0;
        } else if (g === DeformGroup.BumperR && !brDetached) {
          kDamp = activeImpact ? 12.5 : 24.0;
        }
        if (kDamp > 0) {
          const rvx = vx[i] - bvx;
          const rvy = vy[i] - bvy;
          const rvz = vz[i] - bvz;
          if (!activeImpact && g === DeformGroup.Hood && rvx * rvx + rvy * rvy + rvz * rvz < 0.04) {
            // Crisp micro-vibration deadband once impact ends so hood rests completely still
            vx[i] = bvx + rvx * 0.55;
            vy[i] = bvy + rvy * 0.55;
            vz[i] = bvz + rvz * 0.55;
          } else {
            const m = S.mass[i] * kDamp;
            fx[i] -= m * rvx;
            fy[i] -= m * rvy;
            fz[i] -= m * rvz;
          }
        }
      }
    }

    // Post-crash stability fixes (tail-up settling & yaw-rate damping)
    if (S.peakImpact > 2.0 && !this.lateralSledActive) {
      const R = this.restLocal;
      const list = this.bodyNodeList;
      if (this.frontImpacted && Zy < -0.10 && vHoriz < 7.5) {
        const pitchSeverity = Math.min(1, (-Zy - 0.10) * 3.5);
        for (let k = 0; k < list.length; k++) {
          const i = list[k];
          const lz = R[i * 3 + 2];
          if (lz < 0) {
            const m = S.mass[i];
            fy[i] -= m * (6.5 + 14.0 * pitchSeverity);
            if (vy[i] > 0) fy[i] -= m * vy[i] * 14.0;
          }
        }
      }
      if (Math.abs(inp.steer) < 0.12) {
        const rFL = this.refNodes[0], rFR = this.refNodes[1], rRL = this.refNodes[2], rRR = this.refNodes[3];
        const fvx = (vx[rFL] + vx[rFR]) * 0.5, fvz = (vz[rFL] + vz[rFR]) * 0.5;
        const rvx = (vx[rRL] + vx[rRR]) * 0.5, rvz = (vz[rRL] + vz[rRR]) * 0.5;
        const vLatFront = fvx * Xx + fvz * Xz;
        const vLatRear = rvx * Xx + rvz * Xz;
        const wb = Math.max(1.8, this.profile.frontAxleZ - this.profile.rearAxleZ);
        const yawRate = (vLatFront - vLatRear) / wb;
        if (Math.abs(yawRate) > 0.25) {
          const dampK = Math.min(6.5, 2.8 + Math.abs(yawRate) * 1.4);
          for (let k = 0; k < list.length; k++) {
            const i = list[k];
            const lz = R[i * 3 + 2];
            const fLat = -S.mass[i] * yawRate * lz * dampK;
            fx[i] += Xx * fLat;
            fz[i] += Xz * fLat;
          }
        }
      }
    }

    // Anti-wiggle panel seat stabilization & momentum-conserving reaction
    {
      const ox = F[0], oy = F[1], oz = F[2];
      const R = this.restLocal;
      if (!this.frontImpacted && S.lastImpact > 2.2) {
        const frontZoneZ = this.profile.frontAxleZ + 0.15;
        for (let i = 0; i < S.nodeCount; i++) {
          if (!S.cObstacle[i]) continue;
          const g = S.group[i];
          if (g === DeformGroup.Hood || g === DeformGroup.BumperF || (g === NODE_BODY && R[i * 3 + 2] > frontZoneZ)) {
            this.frontImpacted = true;
            break;
          }
        }
      }

      for (let pIdx = 0; pIdx < this.parts.length; pIdx++) {
        const part = this.parts[pIdx];
        if (part.state !== 'attached') continue;
        const grp = part.group;
        const isRear = grp === DeformGroup.Trunk || grp === DeformGroup.BumperR;
        const isHood = grp === DeformGroup.Hood;
        if (isHood) {
          // After an impact ends, keep velocity-only seat damping active so the hood never rings or oscillates!
          if (this.frontImpacted && S.lastImpact > 1.8) continue;
        } else if (!isRear) {
          if (S.lastImpact >= 0.8) continue;
          const z = grp === DeformGroup.BumperF ? Zone.Bumper : Zone.Fender;
          if (S.zoneDamage[z] > 0.002) continue;
        }
        const pairedSide = grp === DeformGroup.DoorL ? DeformGroup.SideL : grp === DeformGroup.DoorR ? DeformGroup.SideR : -1;
        let directHit = false;
        for (let i = 0; i < S.nodeCount; i++) {
          if ((S.group[i] === grp || S.group[i] === pairedSide) && S.cObstacle[i]) { directHit = true; break; }
        }
        if (directHit) continue;

        const kSeat = isRear ? 32000 : isHood ? 24000 : 28000;
        const cSeat = isRear ? 340 : isHood ? 280 : 300;
        const hasCrashed = S.peakImpact > 1.5;
        let sumFx = 0, sumFy = 0, sumFz = 0;
        for (let i = 0; i < S.nodeCount; i++) {
          if (S.group[i] !== grp) continue;
          if (hasCrashed) {
            const dfx = -(vx[i] - bvx) * cSeat;
            const dfy = -(vy[i] - bvy) * cSeat;
            const dfz = -(vz[i] - bvz) * cSeat;
            fx[i] += dfx; fy[i] += dfy; fz[i] += dfz;
            sumFx += dfx; sumFy += dfy; sumFz += dfz;
            continue;
          }
          const lx = R[i * 3], ly = R[i * 3 + 1], lz = R[i * 3 + 2];
          const tx = ox + Xx * lx + Yx * ly + Zx * lz;
          const ty = oy + Xy * lx + Yy * ly + Zy * lz;
          const tz = oz + Xz * lx + Yz * ly + Zz * lz;
          const dfx = (tx - px[i]) * kSeat - (vx[i] - bvx) * cSeat;
          const dfy = (ty - py[i]) * kSeat - (vy[i] - bvy) * cSeat;
          const dfz = (tz - pz[i]) * kSeat - (vz[i] - bvz) * cSeat;
          fx[i] += dfx; fy[i] += dfy; fz[i] += dfz;
          sumFx += dfx; sumFy += dfy; sumFz += dfz;
        }
        const invRef = 0.25;
        for (let r = 0; r < 4; r++) {
          const ri = this.refNodes[r];
          fx[ri] -= sumFx * invRef;
          fy[ri] -= sumFy * invRef;
          fz[ri] -= sumFz * invRef;
        }
      }
    }

    // Hood frontal crash response: V-fold buckle ONLY during active impact + aerodynamic wind lift when unlatched!
    if (this.frontImpacted) {
      const hoodPart = this.parts[0];
      const hoodUnlatched = hoodPart.state === 'unlatched';
      const hoodDetached = hoodPart.state === 'detached';
      if (!hoodDetached) {
        const ox = F[0], oy = F[1], oz = F[2];
        const R = this.restLocal;
        const aeroWind = vForward > 3.0 ? Math.min(340, vForward * vForward * 0.42) : 0;
        for (let c = 0; c < this.hoodColumns.length; c++) {
          const col = this.hoodColumns[c];
          let len = 0;
          for (let r = 0; r + 1 < col.length; r++) {
            const a = col[r], b = col[r + 1];
            const dx = px[b] - px[a], dy = py[b] - py[a], dz = pz[b] - pz[a];
            len += Math.sqrt(dx * dx + dy * dy + dz * dz);
          }
          const chord = Math.sqrt((px[col[4]] - px[col[0]]) ** 2 + (py[col[4]] - py[col[0]]) ** 2 + (pz[col[4]] - pz[col[0]]) ** 2);
          const kappa = 1 - chord / this.hoodColumnRest[c];
          // Apply V-fold upward buckle ONLY during an active frontal impact (never continuously after the crash!)
          if (S.lastImpact > 3.5 && kappa > 0.065 && len > 0) {
            const f = Math.min(1, (kappa - 0.065) * 5.2) * 2600;
            fx[col[1]] += Yx * f * 0.6; fy[col[1]] += Yy * f * 0.6; fz[col[1]] += Yz * f * 0.6;
            fx[col[2]] += Yx * f; fy[col[2]] += Yy * f; fz[col[2]] += Yz * f;
            fx[col[3]] += Yx * f * 0.6; fy[col[3]] += Yy * f * 0.6; fz[col[3]] += Yz * f * 0.6;
            fx[col[0]] -= Yx * f * 1.1; fy[col[0]] -= Yy * f * 1.1; fz[col[0]] -= Yz * f * 1.1;
            fx[col[4]] -= Yx * f * 1.1; fy[col[4]] -= Yy * f * 1.1; fz[col[4]] -= Yz * f * 1.1;
          }
          if (hoodUnlatched) {
            const hingeZ = R[col[0] * 3 + 2];
            for (let r = 1; r < col.length; r++) {
              const ni = col[r];
              const distFromHinge = Math.max(0.05, R[ni * 3 + 2] - hingeZ);
              const restLocalY = R[ni * 3 + 1];
              const curLocalY = (px[ni] - ox) * Yx + (py[ni] - oy) * Yy + (pz[ni] - oz) * Yz;
              const relVy = (vx[ni] - bvx) * Yx + (vy[ni] - bvy) * Yy + (vz[ni] - bvz) * Yz;
              // Aerodynamic wind lift pushes unlatched hood open as car moves forward, damped smoothly when stationary
              let fY = aeroWind * (r / (col.length - 1)) - relVy * 28;
              if (curLocalY < restLocalY - 0.01) {
                const pen = (restLocalY - 0.01) - curLocalY;
                fY += pen * 4200 - Math.min(0, relVy) * 160;
              }
              const maxOpenY = restLocalY + 0.85 * distFromHinge;
              if (curLocalY > maxOpenY) {
                const over = curLocalY - maxOpenY;
                fY -= over * 3400 + Math.max(0, relVy) * 140;
              }
              const clampedF = Math.max(-440, Math.min(520, fY));
              fx[ni] += Yx * clampedF;
              fy[ni] += Yy * clampedF;
              fz[ni] += Yz * clampedF;
            }
          }
        }
      }
    }

    // Unlatched Door outward swing stops (prevents unlatched doors from folding inward through cabin sill, caps outward swing to ~58°)
    for (let dSide = 0; dSide < 2; dSide++) {
      const doorPart = this.parts[dSide === 0 ? 4 : 5];
      if (doorPart.state !== 'unlatched') continue;
      const sideSign = dSide === 0 ? 1 : -1;
      const dNodes = dSide === 0 ? this.doorLNodes : this.doorRNodes;
      const ox = F[0], oy = F[1], oz = F[2];
      const R = this.restLocal;
      const hingeZ = R[dNodes.low[0] * 3 + 2];
      for (let c = 1; c < dNodes.low.length; c++) {
        for (let row = 0; row < 2; row++) {
          const ni = row === 0 ? dNodes.low[c] : dNodes.high[c];
          const arm = Math.max(0.1, hingeZ - R[ni * 3 + 2]);
          const restOut = sideSign * R[ni * 3];
          const curOut = sideSign * ((px[ni] - ox) * Xx + (py[ni] - oy) * Xy + (pz[ni] - oz) * Xz);
          const relOutV = sideSign * ((vx[ni] - bvx) * Xx + (vy[ni] - bvy) * Xy + (vz[ni] - bvz) * Xz);
          let fOut = -relOutV * 10;
          if (curOut < restOut - 0.03) {
            fOut += ((restOut - 0.03) - curOut) * 3800 - Math.min(0, relOutV) * 120;
          }
          const maxOut = restOut + 0.85 * arm;
          if (curOut > maxOut) {
            fOut -= (curOut - maxOut) * 3200 + Math.max(0, relOutV) * 110;
          }
          const cf = sideSign * Math.max(-420, Math.min(420, fOut));
          fx[ni] += Xx * cf;
          fy[ni] += Xy * cf;
          fz[ni] += Xz * cf;
        }
      }
    }
  }

  private applyAntiRoll(a: number, b: number, sa: number, sb: number, Yx: number, Yy: number, Yz: number): void {
    const A = this.wheels[a], B = this.wheels[b];
    if (!A.contact || !B.contact || A.detached || B.detached) return;
    const f = this.antiRoll * (sb - sa);
    const fx = this.solver.fx, fy = this.solver.fy, fz = this.solver.fz;
    for (let k = 0; k < 4; k++) {
      const i = A.nodes[k], q = A.weights[k] * f;
      fx[i] += Yx * q; fy[i] += Yy * q; fz[i] += Yz * q;
      const j = B.nodes[k], r = B.weights[k] * f;
      fx[j] -= Yx * r; fy[j] -= Yy * r; fz[j] -= Yz * r;
    }
  }

  /** One fixed physics step (strictly zero heap allocations). */
  step(dt: number): void {
    this.prevSpeed = this.speed;
    this.solver.step(dt);
    this.stabilizeIsolatedPanels();
    const accelG = Math.abs((this.speed - this.prevSpeed) / dt) / 9.81;
    this.gFiltered += (accelG - this.gFiltered) * 0.25;
    if (accelG > this.peakG) this.peakG = accelG;
    if (this.shiftTimer > 0) {
      this.shiftTimer = Math.max(0, this.shiftTimer - dt);
      if (this.shiftTimer === 0) this.shiftKickTimer = 0.09;
    } else if (this.shiftKickTimer > 0) {
      this.shiftKickTimer = Math.max(0, this.shiftKickTimer - dt);
    }

    const isHighRev = this.telemetry.midRearEngine;
    const redlineRpm = isHighRev ? 8500 : 7200;
    const rpmScale = isHighRev ? 3.85 : 3.35;
    const kmh = Math.abs(this.speed) * 3.6;
    const gIdx = Math.max(0, Math.min(6, this.autoGear - 1));
    const wheelRpm = (kmh / 3.6) / Math.max(0.25, this.profile.wheelRadius) * (60 / (2 * Math.PI));
    const coupledRpm = 850 + wheelRpm * GEAR_RATIOS[gIdx] * rpmScale;
    const autoForwardActive =
      (this.transmissionMode === 'auto' && this.gear === 1) ||
      (this.transmissionMode === 'semi' && this.manualGear > 0);
    if (autoForwardActive && this.shiftTimer <= 0 && !this.engineStalled) {
      const upshiftRpm = this.input.throttle > 0.65 ? redlineRpm - 320 : redlineRpm * 0.74;
      const downshiftRpm = this.input.throttle > 0.75 ? redlineRpm * 0.56 : redlineRpm * 0.34;
      if (this.autoGear < 7 && coupledRpm > upshiftRpm) {
        this.autoGear++;
        this.shiftCount++;
        this.shiftTimer = 0.14;
      } else if (this.autoGear > 1 && (coupledRpm < downshiftRpm || kmh < 8)) {
        this.autoGear--;
        this.shiftCount++;
        this.shiftTimer = 0.11;
        this.engineRpm = Math.min(redlineRpm - 200, this.engineRpm + 1150);
      }
    }
    const activeGearIdx = this.transmissionMode === 'manual'
      ? Math.max(0, Math.min(6, Math.max(1, this.manualGear) - 1))
      : Math.max(0, Math.min(6, this.autoGear - 1));
    const newRatio = GEAR_RATIOS[activeGearIdx];
    const effectiveRedline = redlineRpm * Math.max(0.45, 1 - this.engineDmg * 0.42);
    const inNeutral = this.transmissionMode !== 'auto' && this.manualGear === 0;
    const targetRpm = this.engineStalled
      ? 0
      : this.shiftTimer > 0
        ? Math.max(1200, this.engineRpm * 0.92)
        : inNeutral
          ? 850 + this.input.throttle * (effectiveRedline - 950)
          : 850 + wheelRpm * newRatio * rpmScale + this.input.throttle * (kmh < 6 ? 1600 : 260) * Math.max(0.2, 1 - this.engineDmg * 0.75);
    this.engineRpm += (Math.min(effectiveRedline, targetRpm) - this.engineRpm) * 0.24;

    // Eliminate post-crash low-speed idle slides on flat ground when no drive is applied (or engine is stalled)
    const noDriveInput = this.engineStalled || (this.input.throttle < 0.02 && this.input.brake < 0.02);
    if (noDriveInput && !this.lateralSledActive && this.autopilotSpeed === 0 && Math.abs(this.speed) < 0.55) {
      const S = this.solver;
      const vx = S.vx, vy = S.vy, vz = S.vz;
      let v2Avg = 0;
      for (let k = 0; k < 4; k++) {
        const r = this.refNodes[k];
        v2Avg += vx[r] * vx[r] + vz[r] * vz[r];
      }
      if (v2Avg * 0.25 < 0.36 && this.frame[7] > 0.90) {
        for (let i = 0; i < S.nodeCount; i++) {
          vx[i] *= 0.86;
          vz[i] *= 0.86;
          if (Math.abs(vy[i]) < 0.15) vy[i] *= 0.86;
        }
      }
    }

    this.updateParts();
    this.updateMechanicalDamage(dt);
  }

  /** Updates engine pushback/stall, thermal overheat/blowout, fuel & coolant levels, axle damage, and fluid leak flags (60 Hz, zero GC). */
  private updateMechanicalDamage(dt: number): void {
    const S = this.solver;
    const frontDmg = Math.min(1, (S.zoneDamageRatio(Zone.Crumple) + S.zoneDamageRatio(Zone.Longeron)) * 0.75);
    const rearDmg = S.zoneDamageRatio(Zone.Trunk);
    const engSubDmg = S.zoneDamageRatio(Zone.EngineSubframe);
    const hoodDmg = S.zoneDamageRatio(Zone.Hood);
    const cabinDmg = S.zoneDamageRatio(Zone.Cabin) * 2;
    const isMidRear = this.telemetry.midRearEngine;

    const computedEngDmg = isMidRear
      ? Math.min(1, rearDmg * 0.75 + engSubDmg * 0.65 + cabinDmg * 0.22)
      : Math.min(1, frontDmg * 0.72 + engSubDmg * 0.70 + hoodDmg * 0.22);
    if (computedEngDmg > this.engineDmg) this.engineDmg = computedEngDmg;
    if (this.engineDmg >= 0.80) {
      this.engineBroken = true;
      this.engineStalled = true;
    }

    // Zone-based fluid leak triggers & dynamic severity scaling (0 = sealed, 0..1 = slow drip -> gushing stream)
    const bumperDmg = S.zoneDamageRatio(Zone.Bumper);
    const fenderDmg = S.zoneDamageRatio(Zone.Fender);
    // 1. Radiator / Front Grille & Bumper Nodes -> Coolant Leak & Radiator Damage
    const radTrigger = Math.max(frontDmg * 0.92, bumperDmg * 0.78);
    if (radTrigger > 0.16) {
      const sev = Math.min(1, (radTrigger - 0.16) / 0.64);
      if (sev > this.coolantSeverity) this.coolantSeverity = sev;
      this.coolantLeak = true;
      this.radiatorDamaged = true;
    }

    // 1b. Independent Left & Right Headlight breakage on frontal corner impacts
    if ((!this.headlightBrokenL || !this.headlightBrokenR) && S.lastImpact > 6.5) {
      const R = this.restLocal;
      const hlZMin = this.profile.frontAxleZ + 0.12;
      for (let i = 0; i < S.nodeCount; i++) {
        if (!S.cObstacle[i]) continue;
        const rz = R[i * 3 + 2];
        if (rz < hlZMin) continue;
        const rx = R[i * 3];
        if (rx > 0.18) this.headlightBrokenL = true;
        else if (rx < -0.18) this.headlightBrokenR = true;
        else if (S.lastImpact > 14.0) {
          this.headlightBrokenL = true;
          this.headlightBrokenR = true;
        }
      }
    }
    if (frontDmg > 0.42) {
      this.headlightBrokenL = true;
      this.headlightBrokenR = true;
    }

    // Fuel consumption (normal driving + accelerated drain if fuel tank is leaking)
    const rpmNorm = Math.max(0, Math.min(1, (this.engineRpm - 850) / 6650));
    if (!this.engineStalled) {
      const burnRate = 0.00018 + this.input.throttle * 0.0014 + rpmNorm * 0.0006;
      this.fuelLevel = Math.max(0, this.fuelLevel - burnRate * dt);
    }
    if (this.fuelLeak && this.fuelLevel > 0) {
      this.fuelLevel = Math.max(0, this.fuelLevel - (0.012 + this.fuelSeverity * 0.032) * dt);
    }
    if (this.fuelLevel <= 0.0001) {
      this.fuelLevel = 0;
      this.engineStalled = true;
    }

    // Coolant depletion when radiator is leaking
    if (this.coolantLeak && this.coolantLevel > 0) {
      this.coolantLevel = Math.max(0, this.coolantLevel - (0.020 + this.coolantSeverity * 0.055) * dt);
    }

    // Engine thermal model: healthy radiator keeps temp ~88–94°C; damaged radiator or empty coolant overheats & blows engine at 128°C!
    const radEfficiency = Math.max(0, 1 - Math.max(radTrigger, this.coolantSeverity) * 1.15) * Math.min(1, this.coolantLevel * 1.35);
    const heatGen = this.engineStalled
      ? 0
      : (0.85 + this.input.throttle * 3.8 + rpmNorm * 2.9 + this.engineDmg * 1.4);
    const speedAirflow = Math.min(1.8, Math.abs(this.speed) / 22);
    const coolRate = this.engineStalled
      ? (this.engineTemp - 65) * 0.045
      : (this.engineTemp - 86) * (0.04 + radEfficiency * (0.28 + speedAirflow * 0.14));
    const heatDelta = (heatGen * (1.0 - radEfficiency * 0.72) - coolRate) * dt;
    this.engineTemp = Math.max(65, Math.min(136, this.engineTemp + heatDelta));

    // Overheating emits extra steam and seizes/breaks the engine at >= 128°C
    if (this.engineTemp >= 112 && !this.engineBroken) {
      const heatSev = Math.min(1, (this.engineTemp - 110) / 18);
      if (heatSev * 0.45 > this.coolantSeverity && this.coolantLevel > 0.02) {
        this.coolantSeverity = heatSev * 0.45;
      }
      if (this.engineTemp >= 128) {
        this.engineBroken = true;
        this.engineStalled = true;
        this.engineDmg = Math.max(this.engineDmg, 0.92);
      }
    }
    // 2. Engine Block / Front Subframe / Oil Pan Nodes -> Engine Oil Leak (frontal OR undercarriage impact)
    const panImpact = Math.max(
      S.cHit[this.oilPanNodeA] ? Math.abs(S.cvn[this.oilPanNodeA]) : 0,
      S.cHit[this.oilPanNodeB] ? Math.abs(S.cvn[this.oilPanNodeB]) : 0,
    );
    const underCarriageSev = panImpact > 5.5 ? Math.min(1, (panImpact - 5.5) / 14.0) : 0;
    const oilTrigger = Math.max(this.engineDmg, engSubDmg * 1.15, underCarriageSev);
    if (oilTrigger > 0.20) {
      const sev = Math.min(1, (oilTrigger - 0.20) / 0.65);
      if (sev > this.oilSeverity) this.oilSeverity = sev;
      this.oilLeak = true;
    }
    // 3. Rear Subframe / Fuel Tank Nodes -> Fuel Leak (severe rear or side-rear impact)
    const rearTankImpact = S.cObstacle[this.fuelTankNode] ? Math.min(1, Math.abs(S.cvn[this.fuelTankNode]) / 16.0) : 0;
    const fuelTrigger = Math.max(rearDmg * 1.05, (this.rearImpacted ? fenderDmg * 0.72 : 0), rearTankImpact);
    if (fuelTrigger > 0.24) {
      const sev = Math.min(1, (fuelTrigger - 0.24) / 0.60);
      if (sev > this.fuelSeverity) this.fuelSeverity = sev;
      this.fuelLeak = true;
    }

    // Per-wheel axle / half-shaft damage (from suspension beam breakage & bent camber/toe)
    const flags = S.flags;
    for (let w = 0; w < 4; w++) {
      const W = this.wheels[w];
      if (W.detached) {
        this.axleDmg[w] = 1.0;
        continue;
      }
      let brokenSusp = 0;
      for (let k = 0; k < W.suspBeams.length; k++) {
        if (flags[W.suspBeams[k]] & BEAM_BROKEN) brokenSusp++;
      }
      const bendDmg = Math.min(1, (Math.abs(W.bentCamber) * 2.1 + Math.abs(W.bentToe) * 2.4));
      const zoneExtra = W.front ? frontDmg * 0.45 : rearDmg * 0.45;
      const d = Math.min(1, Math.max(bendDmg, brokenSusp / Math.max(1, W.suspBeams.length)) * 0.85 + zoneExtra);
      if (d > this.axleDmg[w]) this.axleDmg[w] = d;
    }
    this.frontAxleDmg = Math.max(this.axleDmg[0], this.axleDmg[1]);
    this.rearAxleDmg = Math.max(this.axleDmg[2], this.axleDmg[3]);

    // Steering rack damage, permanent tie-rod pull bias, and steering lock on severe front/wheel crash
    const fl = this.wheels[0], fr = this.wheels[1];
    const toeAsym = (fl.bentToe + fr.bentToe) * 0.55;
    const targetBias = Math.max(-0.34, Math.min(0.34, toeAsym + (this.axleDmg[0] - this.axleDmg[1]) * 0.22));
    if (Math.abs(targetBias) > Math.abs(this.steeringBias)) {
      this.steeringBias = targetBias;
    }
    const steerDmgNow = Math.min(1, this.frontAxleDmg * 0.72 + frontDmg * 0.55);
    if (steerDmgNow > this.steeringDmg) this.steeringDmg = steerDmgNow;
    if (!this.steeringLocked && (this.steeringDmg >= 0.70 || frontDmg >= 0.66 || fl.detached || fr.detached)) {
      this.steeringLocked = true;
      // Lock the steering rack at the current bent/impacted wheel angle (with minimum noticeable jam offset)
      const jamBase = Math.abs(this.steerAngle) > 0.12 ? this.steerAngle : (this.steeringBias >= 0 ? 0.24 : -0.24);
      this.steeringLockAngle = Math.max(-0.48, Math.min(0.48, jamBase));
    }
  }

  private stabilizeIsolatedPanels(): void {
    const S = this.solver;
    const px = S.px, py = S.py, pz = S.pz;
    const vx = S.vx, vy = S.vy, vz = S.vz;
    this.updateFrame(px, py, pz);
    const F = this.frame;
    const ox = F[0], oy = F[1], oz = F[2];
    const Xx = F[3], Xy = F[4], Xz = F[5], Yx = F[6], Yy = F[7], Yz = F[8], Zx = F[9], Zy = F[10], Zz = F[11];
    const R = this.restLocal;

    let bvx = 0, bvy = 0, bvz = 0;
    for (let k = 0; k < 4; k++) {
      const r = this.refNodes[k];
      bvx += vx[r]; bvy += vy[r]; bvz += vz[r];
    }
    bvx *= 0.25; bvy *= 0.25; bvz *= 0.25;

    const frontZoneZ = this.profile.frontAxleZ - 0.15;
    const rearZoneZ = this.profile.rearAxleZ + 0.15;
    for (let i = 0; i < S.nodeCount; i++) {
      if (!S.cObstacle[i]) continue;
      const g = S.group[i];
      if (!this.frontImpacted && (g === DeformGroup.Hood || g === DeformGroup.BumperF || (g === NODE_BODY && R[i * 3 + 2] > frontZoneZ))) {
        this.frontImpacted = true;
      }
      if (!this.rearImpacted && (g === DeformGroup.Trunk || g === DeformGroup.BumperR || (g === NODE_BODY && R[i * 3 + 2] < rearZoneZ))) {
        this.rearImpacted = true;
      }
      if (!this.leftImpacted && (g === DeformGroup.DoorL || g === DeformGroup.SideL || (g === NODE_SUSPENSION && R[i * 3] > 0))) {
        this.leftImpacted = true;
      }
      if (!this.rightImpacted && (g === DeformGroup.DoorR || g === DeformGroup.SideR || (g === NODE_SUSPENSION && R[i * 3] < 0))) {
        this.rightImpacted = true;
      }
    }

    for (let pIdx = 0; pIdx < this.parts.length; pIdx++) {
      const part = this.parts[pIdx];
      if (part.state !== 'attached') continue;
      const grp = part.group;
      const isRear = grp === DeformGroup.Trunk || grp === DeformGroup.BumperR;
      const isHood = grp === DeformGroup.Hood;
      if (!isRear && !isHood) continue;
      if (isRear && this.rearImpacted) continue;
      if (isHood && this.frontImpacted) continue;

      let directHit = false;
      for (let i = 0; i < S.nodeCount; i++) {
        if (S.group[i] === grp && S.cObstacle[i]) {
          directHit = true;
          break;
        }
      }
      if (directHit) continue;
      for (let i = 0; i < S.nodeCount; i++) {
        if (S.group[i] !== grp) continue;
        const lx = R[i * 3], ly = R[i * 3 + 1], lz = R[i * 3 + 2];
        const tx = ox + Xx * lx + Yx * ly + Zx * lz;
        const ty = oy + Xy * lx + Yy * ly + Zy * lz;
        const tz = oz + Xz * lx + Yz * ly + Zz * lz;
        px[i] += (tx - px[i]) * 0.92;
        py[i] += (ty - py[i]) * 0.92;
        pz[i] += (tz - pz[i]) * 0.92;
        vx[i] += (bvx - vx[i]) * 0.65;
        vy[i] += (bvy - vy[i]) * 0.65;
        vz[i] += (bvz - vz[i]) * 0.65;
      }
      if (isHood && S.zoneDamage[Zone.Hood] > 0) {
        S.zoneDamage[Zone.Hood] = 0;
        for (let b = 0; b < S.beamCount; b++) {
          if (S.zone[b] === Zone.Hood) S.rest[b] = S.rest0[b];
        }
      }
      if (grp === DeformGroup.Trunk && S.zoneDamage[Zone.Trunk] > 0) {
        S.zoneDamage[Zone.Trunk] = 0;
        for (let b = 0; b < S.beamCount; b++) {
          if (S.zone[b] === Zone.Trunk) S.rest[b] = S.rest0[b];
        }
      }
    }

    // Zero-jelly rigid stamped-steel door solver (closed = locked to body frame, unlatched = single rigid
    // hinge angle strictly clamped to 0°..58° around A-pillar pivot, preserving permanent inward side-impact dents).
    const MAX_DOOR_RAD = 1.012; // 58 degrees
    for (let dSide = 0; dSide < 2; dSide++) {
      const doorPart = this.parts[dSide === 0 ? 4 : 5];
      if (doorPart.state === 'detached') continue;
      const sideSign = dSide === 0 ? 1 : -1;
      const sideImpacted = dSide === 0 ? this.leftImpacted : this.rightImpacted;
      const dNodes = dSide === 0 ? this.doorLNodes : this.doorRNodes;
      const nCols = dNodes.low.length;
      const dentBase = dSide * 8;
      const hLow = dNodes.low[0];
      const hHigh = dNodes.high[0];
      const hx = 0.5 * (R[hLow * 3] + R[hHigh * 3]);
      const hz = 0.5 * (R[hLow * 3 + 2] + R[hHigh * 3 + 2]);

      if (doorPart.state === 'attached') {
        this.doorAngle[dSide] = 0;
        this.doorAngVel[dSide] = 0;
        // Closed door: lock rigidly to rest body frame (preserving any direct side-impact inward dent)
        for (let c = 0; c < nCols; c++) {
          for (let row = 0; row < 2; row++) {
            const ni = row === 0 ? dNodes.low[c] : dNodes.high[c];
            const dentSlot = dentBase + c * 2 + row;
            const rx = R[ni * 3], ry = R[ni * 3 + 1], rz = R[ni * 3 + 2];
            if (sideImpacted && S.cObstacle[ni]) {
              const dxw = px[ni] - ox, dyw = py[ni] - oy, dzw = pz[ni] - oz;
              const lx = dxw * Xx + dyw * Xy + dzw * Xz;
              const inward = sideSign * (lx - rx); // negative = dented inward toward cabin
              if (inward < this.doorDent[dentSlot]) {
                this.doorDent[dentSlot] = Math.max(-0.26, inward);
              }
            }
            const targetLx = rx + sideSign * this.doorDent[dentSlot];
            const tx = ox + Xx * targetLx + Yx * ry + Zx * rz;
            const ty = oy + Xy * targetLx + Yy * ry + Zy * rz;
            const tz = oz + Xz * targetLx + Yz * ry + Zz * rz;
            const blend = S.cObstacle[ni] ? 0.55 : 0.96;
            px[ni] += (tx - px[ni]) * blend;
            py[ni] += (ty - py[ni]) * blend;
            pz[ni] += (tz - pz[ni]) * blend;
            vx[ni] += (bvx - vx[ni]) * 0.85;
            vy[ni] += (bvy - vy[ni]) * 0.85;
            vz[ni] += (bvz - vz[ni]) * 0.85;
          }
        }
        continue;
      }

      // Unlatched door: compute ONE rigid hinge angle from rear latch nodes (no per-node twisting/jelly!)
      let sumOut = 0, sumBack = 0;
      const startCol = Math.max(1, nCols - 2);
      for (let c = startCol; c < nCols; c++) {
        for (let row = 0; row < 2; row++) {
          const ni = row === 0 ? dNodes.low[c] : dNodes.high[c];
          const dxw = px[ni] - ox, dyw = py[ni] - oy, dzw = pz[ni] - oz;
          const lx = dxw * Xx + dyw * Xy + dzw * Xz;
          const lz = dxw * Zx + dyw * Zy + dzw * Zz;
          sumOut += sideSign * (lx - hx);
          sumBack += hz - lz;
        }
      }
      let measuredTheta = Math.atan2(sumOut, Math.max(0.10, sumBack));
      if (measuredTheta < 0) measuredTheta = 0;
      else if (measuredTheta > MAX_DOOR_RAD) measuredTheta = MAX_DOOR_RAD;

      // Damped heavy-steel hinge dynamics with hard check-strap stops at 0° and 58°
      const curAngle = this.doorAngle[dSide];
      let angVel = this.doorAngVel[dSide] * 0.88 + (measuredTheta - curAngle) * 0.35;
      let nextAngle = curAngle + angVel;
      if (nextAngle <= 0) { nextAngle = 0; if (angVel < 0) angVel = 0; }
      else if (nextAngle >= MAX_DOOR_RAD) { nextAngle = MAX_DOOR_RAD; if (angVel > 0) angVel = 0; }
      this.doorAngle[dSide] = nextAngle;
      this.doorAngVel[dSide] = angVel;

      const sinT = Math.sin(nextAngle);
      const cosT = Math.cos(nextAngle);
      for (let c = 0; c < nCols; c++) {
        for (let row = 0; row < 2; row++) {
          const ni = row === 0 ? dNodes.low[c] : dNodes.high[c];
          const dentSlot = dentBase + c * 2 + row;
          const restDz = hz - R[ni * 3 + 2]; // positive rearward lever arm from hinge
          const restDx = sideSign * (R[ni * 3] - hx); // local thickness/contour offset from hinge plane
          const restY = R[ni * 3 + 1];
          const dent = this.doorDent[dentSlot]; // <= 0 permanent inward crease
          const effOut = restDx + dent;

          // Rigid rotation of (effOut, -restDz) around the vertical A-pillar hinge axis
          const rotOut = effOut * cosT + restDz * sinT;
          const rotBack = restDz * cosT - effOut * sinT;
          const targetLx = hx + sideSign * rotOut;
          const targetLz = hz - rotBack;

          const tx = ox + Xx * targetLx + Yx * restY + Zx * targetLz;
          const ty = oy + Xy * targetLx + Yy * restY + Zy * targetLz;
          const tz = oz + Xz * targetLx + Yz * restY + Zz * targetLz;

          // Strong rigid-steel projection (0.97) completely eliminates jelly wobble
          const blend = S.cObstacle[ni] ? 0.65 : 0.97;
          px[ni] += (tx - px[ni]) * blend;
          py[ni] += (ty - py[ni]) * blend;
          pz[ni] += (tz - pz[ni]) * blend;
          vx[ni] += (bvx - vx[ni]) * 0.80;
          vy[ni] += (bvy - vy[ni]) * 0.88;
          vz[ni] += (bvz - vz[ni]) * 0.80;
        }
      }
    }
  }

  private spawnWindowBurst(localX: number, localY: number, localZ: number, outNx: number, outNy: number, outNz: number, count: number): void {
    const F = this.frame;
    const S = this.solver;
    const wx = F[0] + F[3] * localX + F[6] * localY + F[9] * localZ;
    const wy = F[1] + F[4] * localX + F[7] * localY + F[10] * localZ;
    const wz = F[2] + F[5] * localX + F[8] * localY + F[11] * localZ;
    const wnx = F[3] * outNx + F[6] * outNy + F[9] * outNz;
    const wny = F[4] * outNx + F[7] * outNy + F[10] * outNz;
    const wnz = F[5] * outNx + F[8] * outNy + F[11] * outNz;
    let bvx = 0, bvy = 0, bvz = 0;
    for (let k = 0; k < 4; k++) {
      const r = this.refNodes[k];
      bvx += S.vx[r]; bvy += S.vy[r]; bvz += S.vz[r];
    }
    this.glass.spawnBurst(wx, wy, wz, bvx * 0.25, bvy * 0.25, bvz * 0.25, wnx, wny, wnz, count);
  }

  private updateParts(): void {
    const S = this.solver;
    const flags = S.flags;
    const rest = S.rest;
    const rest0 = S.rest0;
    const F = this.frame;

    // 1. Evaluate wheel spindle / wishbone bending & detachment (`Zone.Suspension`)
    for (let w = 0; w < 4; w++) {
      const W = this.wheels[w];
      if (W.detached) continue;
      const sb = W.suspBeams;
      let brokenSusp = 0;
      for (let k = 0; k < sb.length; k++) {
        if (flags[sb[k]] & BEAM_BROKEN) brokenSusp++;
      }
      if (brokenSusp >= 3 || (S.cObstacle[W.hubNode] && S.lastImpact > 32.0)) {
        for (let k = 0; k < sb.length; k++) flags[sb[k]] |= BEAM_BROKEN;
        W.detached = true;
        const hn = W.hubNode;
        const outSign = W.left ? 1 : -1;
        W.detX = S.px[hn];
        W.detY = Math.max(S.py[hn], W.radius);
        W.detZ = S.pz[hn];
        W.detVx = S.vx[hn] + F[3] * outSign * 3.2;
        W.detVy = Math.max(1.8, S.vy[hn] + 2.2);
        W.detVz = S.vz[hn];
        W.detYaw = Math.atan2(F[9], F[11]) + W.steer;
        continue;
      }
      if (sb.length >= 6) {
        const strain = (idx: number) => {
          const b = sb[idx];
          const r0 = rest0[b] || 1;
          return (rest[b] - r0) / r0;
        };
        const lowerStrain = (strain(0) + strain(1)) * 0.5;
        const upperStrain = (strain(2) + strain(3)) * 0.5;
        const frontStrain = (strain(0) + strain(4)) * 0.5;
        const rearStrain = (strain(1) + strain(5)) * 0.5;
        W.bentCamber = Math.max(-0.42, Math.min(0.42, (upperStrain - lowerStrain) * 4.5));
        W.bentToe = Math.max(-0.32, Math.min(0.32, (frontStrain - rearStrain) * 3.8 * (W.left ? 1 : -1)));
        // Wheel-hub impact or bent suspension strain punctures / blows out the tire
        const suspStrain = Math.abs(lowerStrain) + Math.abs(upperStrain);
        if ((S.cObstacle[W.hubNode] && S.lastImpact > 8.5) || suspStrain > 0.065 || this.axleDmg[w] > 0.22) {
          const leakRate = 0.32 + Math.min(1.8, suspStrain * 4.5 + this.axleDmg[w] * 0.9);
          if (leakRate > W.tyreLeaking) W.tyreLeaking = leakRate;
          if (S.cObstacle[W.hubNode] && S.lastImpact > 18.0) {
            W.tyreFlat = Math.min(1, W.tyreFlat + 0.55);
          }
        }
      }
    }

    // 2. Cascading Hinged & Detachable Body Panels
    for (let pIdx = 0; pIdx < this.parts.length; pIdx++) {
      const p = this.parts[pIdx];
      if (p.state === 'detached') continue;

      // Hood: V-folds, pops front latches -> lifts in wind ('unlatched'), or tears off if cowl hinges shear ('detached')
      if (p.group === DeformGroup.Hood) {
        if (!this.frontImpacted) {
          p.state = 'attached';
          continue;
        }
        let hingeBroken = 0;
        for (let k = 0; k < p.hingeBeams.length; k++) if (flags[p.hingeBeams[k]] & BEAM_BROKEN) hingeBroken++;
        if (p.hingeBeams.length > 0 && hingeBroken >= Math.ceil(p.hingeBeams.length * 0.65)) {
          for (let k = 0; k < p.beams.length; k++) flags[p.beams[k]] |= BEAM_BROKEN;
          p.state = 'detached';
          continue;
        }
        let latchBroken = 0;
        for (let k = 0; k < p.latchBeams.length; k++) if (flags[p.latchBeams[k]] & BEAM_BROKEN) latchBroken++;
        if (latchBroken > 0) {
          const wasAttached = p.state === 'attached';
          for (let k = 0; k < p.latchBeams.length; k++) flags[p.latchBeams[k]] |= BEAM_BROKEN;
          p.state = 'unlatched';
          if (wasAttached) {
            const Yx = F[6], Yy = F[7], Yz = F[8];
            for (let c = 0; c < this.hoodColumns.length; c++) {
              const col = this.hoodColumns[c];
              for (let r = 1; r < col.length; r++) {
                const ni = col[r];
                const dv = 1.9 * (r / (col.length - 1));
                S.vx[ni] += Yx * dv;
                S.vy[ni] += Yy * dv;
                S.vz[ni] += Yz * dv;
              }
            }
          }
        }
        continue;
      }

      // Doors remain permanently welded/closed (no opening doors)
      if (p.group === DeformGroup.DoorL || p.group === DeformGroup.DoorR) {
        p.state = 'attached';
        continue;
      }

      // Trunk isolation & detachment
      if (p.group === DeformGroup.Trunk) {
        if (!this.rearImpacted) {
          p.state = 'attached';
          continue;
        }
        let hingeBroken = 0;
        for (let k = 0; k < p.hingeBeams.length; k++) if (flags[p.hingeBeams[k]] & BEAM_BROKEN) hingeBroken++;
        let latchBroken = 0;
        for (let k = 0; k < p.latchBeams.length; k++) if (flags[p.latchBeams[k]] & BEAM_BROKEN) latchBroken++;
        if (p.hingeBeams.length > 0 && hingeBroken >= Math.ceil(p.hingeBeams.length * 0.65)) {
          for (let k = 0; k < p.beams.length; k++) flags[p.beams[k]] |= BEAM_BROKEN;
          p.state = 'detached';
        } else if (latchBroken >= 2) {
          for (let k = 0; k < p.latchBeams.length; k++) flags[p.latchBeams[k]] |= BEAM_BROKEN;
          p.state = 'unlatched';
        }
        continue;
      }

      // Bumpers: hang from one intact corner before full tear-off (strictly zero allocations via isLatchBeam)
      if (p.group === DeformGroup.BumperR && !this.rearImpacted) {
        p.state = 'attached';
        continue;
      }
      if (p.group === DeformGroup.BumperF && !this.frontImpacted) {
        p.state = 'attached';
        continue;
      }
      if (p.group === DeformGroup.BumperF || p.group === DeformGroup.BumperR) {
        for (let k = 0; k < p.beams.length; k++) {
          const b = p.beams[k];
          if (flags[b] & BEAM_BROKEN) continue;
          const r0 = rest0[b] || 1;
          const pDef = Math.abs(rest[b] - r0) / r0;
          const shearLimit = this.isLatchBeam[b] ? 0.024 : 0.048;
          if (pDef > shearLimit) flags[b] |= BEAM_BROKEN;
        }
        let intact = 0, leftIntact = 0, rightIntact = 0;
        for (let k = 0; k < p.beams.length; k++) if (!(flags[p.beams[k]] & BEAM_BROKEN)) intact++;
        if (p.leftBeams) {
          for (let k = 0; k < p.leftBeams.length; k++) if (!(flags[p.leftBeams[k]] & BEAM_BROKEN)) leftIntact++;
        }
        if (p.rightBeams) {
          for (let k = 0; k < p.rightBeams.length; k++) if (!(flags[p.rightBeams[k]] & BEAM_BROKEN)) rightIntact++;
        }
        if (intact === 0 || (leftIntact === 0 && rightIntact === 0) || intact <= Math.max(1, Math.floor(p.beams.length * 0.22))) {
          for (let k = 0; k < p.beams.length; k++) flags[p.beams[k]] |= BEAM_BROKEN;
          p.state = 'detached';
        } else if (leftIntact === 0 || rightIntact === 0 || intact < p.beams.length) {
          p.state = 'unlatched';
        }
      }
    }

    // 3. Instanced 3D Window Shattering (`GlassShatterSystem`)
    const P = this.profile;
    const beltMidY = P.beltY(0) + 0.18;
    // Windshield (0)
    if (!this.glassShattered[0] && (S.zoneDamageRatio(Zone.Cabin) > 0.07 || (this.frontImpacted && S.peakImpact > 14.5 && S.zoneDamageRatio(Zone.Longeron) > 0.14))) {
      this.glassShattered[0] = 1;
      this.spawnWindowBurst(0, beltMidY + 0.06, P.cowlZ - 0.22, 0, 0.72, 0.68, 36);
    }
    // Left side window (1)
    if (!this.glassShattered[1] && (this.parts[4].state !== 'attached' || (this.leftImpacted && S.lastImpact > 10.5))) {
      this.glassShattered[1] = 1;
      this.spawnWindowBurst(P.width * 0.44, beltMidY, (P.cowlZ + P.rearAxleZ) * 0.5, 0.92, 0.38, 0, 28);
    }
    // Right side window (2)
    if (!this.glassShattered[2] && (this.parts[5].state !== 'attached' || (this.rightImpacted && S.lastImpact > 10.5))) {
      this.glassShattered[2] = 1;
      this.spawnWindowBurst(-P.width * 0.44, beltMidY, (P.cowlZ + P.rearAxleZ) * 0.5, -0.92, 0.38, 0, 28);
    }
    // Rear window (3)
    if (!this.glassShattered[3] && (this.parts[1].state !== 'attached' || (this.rearImpacted && S.peakImpact > 13.5))) {
      this.glassShattered[3] = 1;
      this.spawnWindowBurst(0, beltMidY + 0.04, P.deckZ + 0.18, 0, 0.74, -0.66, 28);
    }
  }

  // ================================================================ rendering

  /** Interpolate physics state, update root transform, skin meshes and place wheels. */
  render(alpha: number, frameDt: number): void {
    const S = this.solver;
    const n = S.nodeCount;
    const W = this.renderWorld, L = this.renderLocal;
    for (let i = 0; i < n; i++) {
      W[i * 3] = S.sx[i] + (S.px[i] - S.sx[i]) * alpha;
      W[i * 3 + 1] = S.sy[i] + (S.py[i] - S.sy[i]) * alpha;
      W[i * 3 + 2] = S.sz[i] + (S.pz[i] - S.sz[i]) * alpha;
    }
    this.frameFromRender(W);
    const F = this.frame;
    const ox = F[0], oy = F[1], oz = F[2];
    for (let i = 0; i < n; i++) {
      const dx = W[i * 3] - ox, dy = W[i * 3 + 1] - oy, dz = W[i * 3 + 2] - oz;
      L[i * 3] = dx * F[3] + dy * F[4] + dz * F[5];
      L[i * 3 + 1] = dx * F[6] + dy * F[7] + dz * F[8];
      L[i * 3 + 2] = dx * F[9] + dy * F[10] + dz * F[11];
    }
    this.rootMatrix.set(
      F[3], F[6], F[9], ox,
      F[4], F[7], F[10], oy,
      F[5], F[8], F[11], oz,
      0, 0, 0, 1,
    );
    this.root.matrix.copy(this.rootMatrix);
    this.root.matrixWorldNeedsUpdate = true;

    this.deformer.update(L);
    this.glass.update(frameDt, S.world);

    // Engine stays anchored in place (no pushback into cabin) while visibly bending, tilting & crumpling on impact
    const engD = this.engineDmg;
    if (engD > 0.01) {
      const tiltDir = this.telemetry.midRearEngine ? 1 : -1;
      const asym = this.axleDmg[0] - this.axleDmg[1];
      this.engineMount.position.set(0, 0, 0);
      this.engineMount.rotation.set(tiltDir * engD * 0.16, asym * 0.12, asym * 0.18);
      this.engineMount.scale.set(1 + engD * 0.06, 1 + engD * 0.08, Math.max(0.70, 1 - engD * 0.28));
    }

    // Emit 3D fluid streams, falling droplets, easeOut puddles/trails & steam/vapor from exact deformed ruptured nodes (zero GC)
    const r3 = this.radiatorNode * 3;
    const oa3 = this.oilPanNodeA * 3;
    const ob3 = this.oilPanNodeB * 3;
    const f3 = this.fuelTankNode * 3;
    const fe = this.fluidEmit;
    fe.radX = W[r3];
    fe.radY = W[r3 + 1];
    fe.radZ = W[r3 + 2];
    fe.oilX = 0.5 * (W[oa3] + W[ob3]);
    fe.oilY = 0.5 * (W[oa3 + 1] + W[ob3 + 1]);
    fe.oilZ = 0.5 * (W[oa3 + 2] + W[ob3 + 2]);
    fe.fuelX = W[f3];
    fe.fuelY = W[f3 + 1];
    fe.fuelZ = W[f3 + 2];
    const ref0 = this.refNodes[0];
    fe.carVx = S.vx[ref0];
    fe.carVy = S.vy[ref0];
    fe.carVz = S.vz[ref0];
    // Trigger high-G metallic sparks & collision dust burst at active obstacle OR heavy ground/roof slam contact points
    if (this.impactFxCooldown > 0) this.impactFxCooldown -= frameDt;
    if (S.lastImpact > 5.8 && this.impactFxCooldown <= 0) {
      let sx = ox, sy = oy, sz = oz;
      let hitCount = 0;
      for (let i = 0; i < S.nodeCount; i++) {
        if (!S.cObstacle[i] && !S.cHit[i]) continue;
        if (hitCount === 0) { sx = 0; sy = 0; sz = 0; }
        sx += W[i * 3];
        sy += W[i * 3 + 1];
        sz += W[i * 3 + 2];
        hitCount++;
      }
      if (hitCount > 0) {
        const inv = 1 / hitCount;
        sx *= inv; sy *= inv; sz *= inv;
      }
      this.fluids.emitImpactBurst(sx, sy, sz, Math.min(1.5, S.lastImpact / 16));
      this.impactFxCooldown = 0.14;
    }

    fe.coolantSeverity = this.coolantSeverity;
    fe.oilSeverity = this.oilSeverity;
    fe.fuelSeverity = this.fuelSeverity;
    fe.engineDamage = engD;
    this.fluids.update(frameDt, S.world, fe);

    const chassisYaw = Math.atan2(F[9], F[11]);
    for (let w = 0; w < 4; w++) {
      const Wh = this.wheels[w];
      Wh.spin += Wh.omega * frameDt;
      if (Wh.spin > 1e4 || Wh.spin < -1e4) Wh.spin = 0;
      Wh.spinner.rotation.x = Wh.spin;

      if (Wh.detached) {
        // Transform world-space detached wheel coordinates into `this.root` local space (zero scene-graph churn!)
        const dx = Wh.detX - ox, dy = Wh.detY - oy, dz = Wh.detZ - oz;
        Wh.knuckle.position.set(
          dx * F[3] + dy * F[4] + dz * F[5],
          dx * F[6] + dy * F[7] + dz * F[8],
          dx * F[9] + dy * F[10] + dz * F[11],
        );
        Wh.knuckle.rotation.set(0, Wh.detYaw - chassisYaw, Wh.left ? -0.45 : 0.45);
        continue;
      }

      let ax = 0, ay = 0, az = 0;
      for (let k = 0; k < 4; k++) {
        const i = Wh.nodes[k], q = Wh.weights[k];
        ax += L[i * 3] * q; ay += L[i * 3 + 1] * q; az += L[i * 3 + 2] * q;
      }
      const s = Wh.sPrev + (Wh.s - Wh.sPrev) * alpha;
      const flatDrop = Wh.tyreFlat * Wh.radius * 0.14;
      Wh.knuckle.position.set(
        ax + Wh.hardLocal[0] - Wh.anchorRest[0],
        ay + Wh.hardLocal[1] - Wh.anchorRest[1] - s - flatDrop,
        az + Wh.hardLocal[2] - Wh.anchorRest[2],
      );
      // Visual deflated tire squash (flatter vertically, slightly wider sidewall bulge)
      if (Wh.tyreFlat > 0.005) {
        Wh.knuckle.scale.set(1 + Wh.tyreFlat * 0.07, 1 - Wh.tyreFlat * 0.15, 1 + Wh.tyreFlat * 0.05);
      } else {
        Wh.knuckle.scale.set(1, 1, 1);
      }
      // Visible bent-axle wheel wobble + rhythmic rim-thump wobble on flat tires
      const axleWobble = this.axleDmg[w] > 0.15 ? Math.sin(Wh.spin) * this.axleDmg[w] * 0.14 : 0;
      const flatWobble = Wh.tyreFlat > 0.12 ? Math.sin(Wh.spin * 2) * Wh.tyreFlat * 0.055 : 0;
      const wobble = axleWobble + flatWobble;
      Wh.knuckle.rotation.set(0, Wh.steer + wobble * 0.65, (Wh.left ? -Wh.bentCamber : Wh.bentCamber) + wobble);
    }
    if (this.spoilerGroup) {
      const speedKmh = Math.abs(this.speed) * 3.6;
      const airbrake = speedKmh > 35 && (this.input.brake > 0.25 || this.input.handbrake > 0.25);
      const rawDeploy = airbrake ? 1.0 : Math.max(0, Math.min(1, (speedKmh - 45) / 70));
      if (rawDeploy > 0.05) {
        this.spoilerHoldTimer = 1.6;
      } else if (this.spoilerHoldTimer > 0) {
        this.spoilerHoldTimer = Math.max(0, this.spoilerHoldTimer - frameDt);
      }
      const targetDeploy = rawDeploy > 0.05 ? rawDeploy : this.spoilerHoldTimer > 0 ? this.spoilerDeploy : 0;
      const targetPitch = airbrake ? 0.56 : targetDeploy * 0.15;
      const isRising = targetDeploy > this.spoilerDeploy;
      const deployRate = Math.min(1, frameDt * (airbrake ? 8.0 : isRising ? 4.2 : 2.2));
      const pitchRate = Math.min(1, frameDt * (airbrake ? 9.5 : 3.8));
      this.spoilerDeploy += (targetDeploy - this.spoilerDeploy) * deployRate;
      this.spoilerPitch += (targetPitch - this.spoilerPitch) * pitchRate;
      this.spoilerGroup.position.set(
        this.spoilerPivot[0],
        this.spoilerPivot[1] + this.spoilerDeploy * 0.22,
        this.spoilerPivot[2] - this.spoilerDeploy * 0.08,
      );
      this.spoilerGroup.rotation.x = this.spoilerPitch;
    }
    // Headlights (twin SpotLights that break independently on corner impacts + xenon emissive lenses)
    const hlOn = this.headlightsOn && !this.engineBroken;
    const hlLActive = hlOn && !this.headlightBrokenL;
    const hlRActive = hlOn && !this.headlightBrokenR;
    this.headlightL.intensity = hlLActive ? 38.0 : 0;
    this.headlightR.intensity = hlRActive ? 38.0 : 0;
    const anyHeadlightActive = hlLActive || hlRActive;
    if (this.src.headlightMaterials) {
      const hlEmissive = hlLActive && hlRActive ? 9.0 : anyHeadlightActive ? 4.5 : 0.0;
      for (let i = 0; i < this.src.headlightMaterials.length; i++) {
        const hm = this.src.headlightMaterials[i];
        hm.emissiveIntensity = hlEmissive;
        hm.toneMapped = !anyHeadlightActive;
      }
    }

    const brakeLit = this.input.brake > 0.05 || this.input.handbrake > 0;
    const speedKmhNow = Math.abs(this.speed) * 3.6;
    const absFlash = this.absActive && speedKmhNow > 25 && Math.sin(this.absPhase * 0.55) > 0;
    // Much stronger brake & ABS strobe directly from the left & right rear taillights
    const tailEmissive = absFlash ? 28.0 : brakeLit ? 14.0 : hlOn ? 3.0 : 0.9;
    this.src.materials.taillight.emissiveIntensity = tailEmissive;
    this.src.materials.taillight.toneMapped = !brakeLit && !absFlash && !hlOn;
    if (this.src.taillightMaterials) {
      for (let i = 0; i < this.src.taillightMaterials.length; i++) {
        const tm = this.src.taillightMaterials[i];
        tm.emissiveIntensity = tailEmissive;
        tm.toneMapped = !brakeLit && !absFlash && !hlOn;
      }
    }
    const rearGlow = absFlash ? 12.0 : brakeLit ? 5.2 : hlOn ? 1.1 : 0;
    const rearDist = absFlash ? 14 : brakeLit ? 9 : 5.5;
    this.taillightL.intensity = rearGlow;
    this.taillightL.distance = rearDist;
    this.taillightR.intensity = rearGlow;
    this.taillightR.distance = rearDist;
  }

  private frameFromRender(W: Float32Array): void {
    const [a, b, c, d] = this.refNodes;
    const sx = this.solver.qx, sy = this.solver.qy, sz = this.solver.qz;
    const keep = this.keep;
    for (let k = 0; k < 4; k++) {
      const r = this.refNodes[k];
      keep[k * 3] = sx[r]; keep[k * 3 + 1] = sy[r]; keep[k * 3 + 2] = sz[r];
      sx[r] = W[r * 3]; sy[r] = W[r * 3 + 1]; sz[r] = W[r * 3 + 2];
    }
    this.updateFrame(sx, sy, sz);
    for (let k = 0; k < 4; k++) {
      const r = this.refNodes[k];
      sx[r] = keep[k * 3]; sy[r] = keep[k * 3 + 1]; sz[r] = keep[k * 3 + 2];
    }
    void a; void b; void c; void d;
  }
  private readonly keep = new Float64Array(12);

  /** World-space body frame: origin, left (X), up (Y), forward (Z). */
  getFrame(): Float64Array { return this.frame; }

  updateTelemetry(): VehicleTelemetry {
    const S = this.solver;
    const T = this.telemetry;
    T.speedKmh = Math.abs(this.speed) * 3.6;
    T.transmissionMode = this.transmissionMode;
    if (this.transmissionMode === 'auto') {
      T.gear = Math.abs(this.speed) < 0.3 && this.input.throttle === 0 && this.input.brake === 0
        ? 'N'
        : this.gear === -1
          ? 'R'
          : 'D';
    } else if (this.transmissionMode === 'semi') {
      T.gear = this.manualGear === 0 ? 'N' : this.manualGear === -1 ? 'R' : 'D';
    } else {
      T.gear = this.manualGear === 0 ? 'N' : this.manualGear === -1 ? 'R' : String(this.manualGear);
    }
    T.rpm = Math.round(this.engineRpm);
    T.throttle = this.input.throttle; T.brake = this.input.brake; T.steer = this.input.steer;
    T.absActive = this.absActive;
    let maxSlip = 0;
    for (let w = 0; w < 4; w++) {
      const W = this.wheels[w];
      if (W.contact && !W.detached) {
        const s = Math.abs(W.slip);
        if (s > maxSlip) maxSlip = s;
      }
    }
    T.maxTireSlip = maxSlip;
    T.shiftCount = this.shiftCount;
    T.gForce = this.gFiltered;
    T.peakGForce = this.peakG;
    T.impactKmh = S.lastImpact * 3.6;
    T.peakImpactKmh = S.peakImpact * 3.6;
    T.absorbedKJ = S.plasticWork / 1000;
    const travel = Math.max(0.05, this.sMax - this.sMin);
    for (let w = 0; w < 4; w++) {
      T.suspCompression[w] = this.wheels[w].detached
        ? 0
        : Math.max(0, Math.min(1, (this.sMax - this.wheels[w].s) / travel));
    }
    let broken = 0;
    for (let j = 0; j < S.beamCount; j++) if (S.flags[j] & BEAM_BROKEN) broken++;
    this.maxBroken = Math.max(this.maxBroken, broken);
    T.brokenBeams = broken;
    T.damage.front = Math.min(1, (S.zoneDamageRatio(Zone.Crumple) + S.zoneDamageRatio(Zone.Longeron)) * 0.75);
    T.damage.cabin = S.zoneDamageRatio(Zone.Cabin) * 2;
    T.damage.rear = S.zoneDamageRatio(Zone.Trunk);
    T.damage.hood = S.zoneDamageRatio(Zone.Hood);
    T.damage.bumpers = S.zoneDamageRatio(Zone.Bumper);
    T.damage.fenders = S.zoneDamageRatio(Zone.Fender);
    T.damage.engine = this.engineDmg;
    T.headlightsOn = this.headlightsOn;
    T.mech.engineHealth = Math.max(0, 1 - this.engineDmg);
    T.mech.engineStalled = this.engineStalled;
    T.mech.engineBroken = this.engineBroken;
    T.mech.engineTemp = Math.round(this.engineTemp);
    T.mech.overheating = this.engineTemp >= 112;
    T.mech.radiatorDamaged = this.radiatorDamaged;
    T.mech.fuelLevel = this.fuelLevel;
    T.mech.coolantLevel = this.coolantLevel;
    T.mech.frontAxleDamage = this.frontAxleDmg;
    T.mech.rearAxleDamage = this.rearAxleDmg;
    T.mech.axleDamage[0] = this.axleDmg[0];
    T.mech.axleDamage[1] = this.axleDmg[1];
    T.mech.axleDamage[2] = this.axleDmg[2];
    T.mech.axleDamage[3] = this.axleDmg[3];
    T.mech.flatTires[0] = this.wheels[0].tyreFlat > 0.35;
    T.mech.flatTires[1] = this.wheels[1].tyreFlat > 0.35;
    T.mech.flatTires[2] = this.wheels[2].tyreFlat > 0.35;
    T.mech.flatTires[3] = this.wheels[3].tyreFlat > 0.35;
    T.mech.headlightBrokenL = this.headlightBrokenL;
    T.mech.headlightBrokenR = this.headlightBrokenR;
    T.mech.steeringDamage = this.steeringDmg;
    T.mech.steeringLocked = this.steeringLocked;
    T.mech.steeringBiasDeg = Math.round((this.steeringLocked ? this.steeringLockAngle : this.steeringBias) * (180 / Math.PI));
    T.mech.oilLeak = this.oilLeak;
    T.mech.coolantLeak = this.coolantLeak;
    T.mech.fuelLeak = this.fuelLeak;
    T.mech.coolantSeverity = this.coolantSeverity;
    T.mech.oilSeverity = this.oilSeverity;
    T.mech.fuelSeverity = this.fuelSeverity;
    for (let i = 0; i < this.parts.length; i++) T.parts[i].state = this.parts[i].state;
    T.skinnedVerts = this.deformer.lastSkinnedVerts;
    return T;
  }

  /** Zero-GC helper writing world contact-patch positions & grounded flags for all 4 wheels. */
  getWheelContactPositions(outPos: Float32Array, outGrounded: Uint8Array): void {
    const F = this.frame;
    const S = this.solver;
    const ox = F[0], oy = F[1], oz = F[2];
    const Xx = F[3], Xy = F[4], Xz = F[5];
    const Yx = F[6], Yy = F[7], Yz = F[8];
    const Zx = F[9], Zy = F[10], Zz = F[11];
    for (let w = 0; w < 4; w++) {
      const W = this.wheels[w];
      if (W.detached || !W.contact) {
        outGrounded[w] = 0;
        continue;
      }
      outGrounded[w] = 1;
      const hub = W.hubNode;
      if (hub >= 0 && hub < S.nodeCount) {
        outPos[w * 3] = S.px[hub];
        outPos[w * 3 + 1] = S.py[hub] - W.radius * 0.85;
        outPos[w * 3 + 2] = S.pz[hub];
      } else {
        const lx = W.hardLocal[0], ly = W.hardLocal[1] - W.s - W.radius, lz = W.hardLocal[2];
        outPos[w * 3] = ox + Xx * lx + Yx * ly + Zx * lz;
        outPos[w * 3 + 1] = oy + Xy * lx + Yy * ly + Zy * lz;
        outPos[w * 3 + 2] = oz + Xz * lx + Yz * ly + Zz * lz;
      }
    }
  }

  /** Adjusts body mesh transparency (0.15..1.0) for the X-Ray / Physics Inspector overlay. */
  setBodyOpacity(opacity: number): void {
    const clamped = Math.max(0.15, Math.min(1.0, opacity));
    const isTransparent = clamped < 0.99;
    const seen = new Set<Material>();
    for (const part of this.src.parts) {
      const mats = Array.isArray(part.mesh.material) ? part.mesh.material : [part.mesh.material];
      for (const m of mats) {
        if (!m || seen.has(m)) continue;
        seen.add(m);
        const sm = m as MeshStandardMaterial;
        if (sm.userData.__baseTransparent === undefined) {
          sm.userData.__baseTransparent = sm.transparent;
          sm.userData.__baseOpacity = sm.opacity;
        }
        const baseTrans = Boolean(sm.userData.__baseTransparent);
        const baseOp = typeof sm.userData.__baseOpacity === 'number' ? sm.userData.__baseOpacity : 1;
        const nextTrans = baseTrans || isTransparent;
        if (sm.transparent !== nextTrans) {
          sm.transparent = nextTrans;
          sm.needsUpdate = true;
        }
        sm.opacity = Math.min(baseOp, clamped);
      }
    }
  }

  dispose(): void {
    this.root.removeFromParent();
    this.glass.dispose();
    this.fluids.group.removeFromParent();
    this.fluids.dispose();
    this.src.dispose();
    if (this.wheelGeo) {
      this.wheelGeo.tire.dispose(); this.wheelGeo.rim.dispose(); this.wheelGeo.disc.dispose(); this.wheelGeo.caliper.dispose();
    }
    this.root.traverse((o) => {
      const m = o as Mesh;
      if (!m.isMesh) return;
      m.geometry?.dispose();
      const mats = Array.isArray(m.material) ? m.material : m.material ? [m.material] : [];
      for (const mat of mats) {
        const sm = mat as MeshStandardMaterial;
        if (!isSharedProceduralTexture(sm.map)) sm.map?.dispose();
        if (!isSharedProceduralTexture(sm.normalMap)) sm.normalMap?.dispose();
        if (!isSharedProceduralTexture(sm.roughnessMap)) sm.roughnessMap?.dispose();
        if (!isSharedProceduralTexture(sm.metalnessMap)) sm.metalnessMap?.dispose();
        sm.dispose();
      }
    });
    this.deformer.pool.release();
    this.solver.pool.release();
  }
}

function dist(a: SolverBuildNode, b: SolverBuildNode): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2);
}

/** Softer zone wins when a beam connects two zones. */
const ZONE_SOFTNESS: Record<number, number> = {
  [Zone.Cabin]: 0, [Zone.EngineSubframe]: 0, [Zone.Longeron]: 1, [Zone.Suspension]: 1,
  [Zone.Crumple]: 2, [Zone.Fender]: 3, [Zone.Bumper]: 4, [Zone.Hood]: 5, [Zone.Trunk]: 5,
  [Zone.Hinge]: 6, [Zone.Latch]: 7,
};
function weakestZone(a: Zone, b: Zone): Zone {
  return ZONE_SOFTNESS[a] >= ZONE_SOFTNESS[b] ? a : b;
}

// ================================================================ sources

/** Default sculpted Mercedes-inspired sports sedan source. */
export function createProceduralVehicleSource(
  name = 'FlexWeb GT-S 450 (Sculpted)',
  paintColor = 0x8a9299,
): VehicleVisualSource {
  const profile = createSedanProfile();
  const materials = createCarMaterials(paintColor);
  const parts = buildProceduralCar(profile, materials);
  return {
    name,
    profile,
    parts,
    wheels: null,
    materials,
    dispose() {
      for (const m of Object.values(materials) as Material[]) m.dispose();
    },
  };
}

/**
 * GLB Auto-Fitter
 * ---------------
 * Normalises an arbitrary GLTF/GLB car scene into car-local space (+Z forward,
 * +Y up, +X left, ground at Y = 0), auto-detects low-slung supercar proportions
 * (~4.55 m x 1.95 m x 1.20 m for McLaren P1) vs sports sedan (~4.65 m),
 * separates rotating wheel/disc meshes and steering calipers from MeshDeformer,
 * binds body panels (Hood/Frunk, Trunk/Engine cover, Front Bumper/Splitter,
 * Rear Bumper/Diffuser, Left/Right Doors, Fenders & Body) to DeformGroups,
 * and consolidates draw calls per (DeformGroup, Material) for 60 FPS tablet performance.
 */
export function fitGLBToVehicleSource(root: Object3D, name: string, paintColor?: number, overridePaint?: number): VehicleVisualSource {
  root.updateMatrixWorld(true);
  const rawMeshes: { mesh: Mesh; geo: BufferGeometry; path: string }[] = [];
  root.traverse((obj) => {
    const m = obj as Mesh;
    if (!m.isMesh || !m.geometry) return;
    // Build full ancestor path for accurate semantic classification
    const names: string[] = [];
    let cur: Object3D | null = obj;
    while (cur) {
      if (cur.name) names.push(cur.name);
      cur = cur.parent;
    }
    const path = names.reverse().join('/');

    const g = m.geometry.clone();
    ensurePlainAttributes(g);
    g.applyMatrix4(m.matrixWorld);
    // Fix inside-out triangle winding caused by negative-scale mirror transforms (e.g. BMW FR wheel)
    if (m.matrixWorld.determinant() < 0) {
      const idx = g.getIndex();
      if (idx) {
        const arr = idx.array as Uint16Array | Uint32Array;
        for (let i = 0; i + 2 < arr.length; i += 3) {
          const tmp = arr[i + 1];
          arr[i + 1] = arr[i + 2];
          arr[i + 2] = tmp;
        }
        idx.needsUpdate = true;
      } else {
        const pos = g.getAttribute('position') as BufferAttribute;
        for (let i = 0; i + 2 < pos.count; i += 3) {
          const x1 = pos.getX(i + 1), y1 = pos.getY(i + 1), z1 = pos.getZ(i + 1);
          pos.setXYZ(i + 1, pos.getX(i + 2), pos.getY(i + 2), pos.getZ(i + 2));
          pos.setXYZ(i + 2, x1, y1, z1);
        }
        pos.needsUpdate = true;
      }
      g.computeVertexNormals();
    }
    rawMeshes.push({ mesh: m, geo: g, path });
  });
  if (rawMeshes.length === 0) return createProceduralVehicleSource(name, paintColor ?? 0x8a9299);

  // Compute exact vertex bounding box & guard against zero / NaN dimensions
  const totalBox = new Box3();
  for (const r of rawMeshes) {
    const pos = r.geo.getAttribute('position');
    if (!pos || pos.count === 0) continue;
    r.geo.computeBoundingBox();
    if (r.geo.boundingBox && !r.geo.boundingBox.isEmpty()) {
      totalBox.union(r.geo.boundingBox);
    }
  }
  const size = new Vector3();
  const center = new Vector3();
  totalBox.getSize(size);
  totalBox.getCenter(center);
  if (
    totalBox.isEmpty() ||
    !Number.isFinite(size.x) || !Number.isFinite(size.y) || !Number.isFinite(size.z) ||
    !Number.isFinite(center.x) || !Number.isFinite(center.y) || !Number.isFinite(center.z) ||
    Math.max(size.x, size.y, size.z) < 1e-4
  ) {
    throw new Error(`Invalid or empty bounding box in GLB model "${name}"`);
  }

  // Orient longest horizontal axis along Z
  const rotY = size.x > size.z * 1.12 ? Math.PI / 2 : 0;
  const rawLen = Math.max(size.x, size.z, 0.01);
  const rawH = Math.max(size.y, 0.01);
  const rawAspect = rawH / rawLen;
  const isSupercar = /mclaren|p1|supercar|ferrari|lamborghini|porsche/i.test(name) || rawAspect < 0.275;

  const targetLen = isSupercar ? 4.55 : 4.65;
  const scaleZ = targetLen / rawLen;
  // For supercars with wide mirror bounds, scale X and Y to match ~1.95m width and ~1.20m height cleanly
  const rawWidth = Math.max(rotY !== 0 ? size.z : size.x, 0.01);
  const scaleX = isSupercar && rawWidth * scaleZ > 2.02 ? 1.95 / rawWidth : scaleZ;
  const scaleY = isSupercar ? Math.min(scaleZ, 1.20 / Math.max(rawH, 0.5)) : scaleZ;

  const normMat = new Matrix4()
    .makeScale(scaleX, scaleY, scaleZ)
    .multiply(new Matrix4().makeRotationY(rotY))
    .multiply(new Matrix4().makeTranslation(-center.x, -totalBox.min.y, -center.z));

  for (const r of rawMeshes) {
    r.geo.applyMatrix4(normMat);
    r.geo.computeBoundingBox();
  }

  // Check whether the nose is at -Z (front hood lower than rear deck/cabin)
  let maxYFront = 0, maxYRear = 0;
  for (const r of rawMeshes) {
    if (/wheel|tyre|tire|rim|disc|susp/i.test(r.path)) continue;
    const pos = r.geo.getAttribute('position') as BufferAttribute;
    for (let i = 0; i < pos.count; i += 4) {
      const y = pos.getY(i), z = pos.getZ(i);
      if (z > 1.2 && z < 2.1 && y > maxYFront) maxYFront = y;
      if (z < -1.2 && z > -2.1 && y > maxYRear) maxYRear = y;
    }
  }
  if (maxYFront > maxYRear + 0.14) {
    const flip = new Matrix4().makeRotationY(Math.PI);
    for (const r of rawMeshes) { r.geo.applyMatrix4(flip); r.geo.computeBoundingBox(); }
  }

  // Helper: split a combined multi-wheel mesh (e.g. Dodge Demon tire/rim/brakedisk spanning all 4 corners)
  // into up to 4 independent quadrant BufferGeometries (FL, FR, RL, RR) so every wheel spins, steers & suspends.
  const splitByWheelQuadrant = (geo: BufferGeometry): BufferGeometry[] => {
    const pos = geo.getAttribute('position') as BufferAttribute;
    if (!pos || pos.count < 3) return [];
    const norm = geo.getAttribute('normal') as BufferAttribute | undefined;
    const uv = geo.getAttribute('uv') as BufferAttribute | undefined;
    const index = geo.getIndex();
    const triCount = index ? Math.floor(index.count / 3) : Math.floor(pos.count / 3);
    const qPos: number[][] = [[], [], [], []];
    const qNorm: number[][] = [[], [], [], []];
    const qUv: number[][] = [[], [], [], []];

    for (let t = 0; t < triCount; t++) {
      const i0 = index ? index.getX(t * 3) : t * 3;
      const i1 = index ? index.getX(t * 3 + 1) : t * 3 + 1;
      const i2 = index ? index.getX(t * 3 + 2) : t * 3 + 2;
      const cx = (pos.getX(i0) + pos.getX(i1) + pos.getX(i2)) / 3;
      const cz = (pos.getZ(i0) + pos.getZ(i1) + pos.getZ(i2)) / 3;
      const q = (cz >= 0 ? 0 : 2) + (cx >= 0 ? 0 : 1);
      const qp = qPos[q];
      qp.push(
        pos.getX(i0), pos.getY(i0), pos.getZ(i0),
        pos.getX(i1), pos.getY(i1), pos.getZ(i1),
        pos.getX(i2), pos.getY(i2), pos.getZ(i2),
      );
      if (norm) {
        qNorm[q].push(
          norm.getX(i0), norm.getY(i0), norm.getZ(i0),
          norm.getX(i1), norm.getY(i1), norm.getZ(i1),
          norm.getX(i2), norm.getY(i2), norm.getZ(i2),
        );
      }
      if (uv) {
        qUv[q].push(
          uv.getX(i0), uv.getY(i0),
          uv.getX(i1), uv.getY(i1),
          uv.getX(i2), uv.getY(i2),
        );
      }
    }

    const out: BufferGeometry[] = [];
    for (let q = 0; q < 4; q++) {
      if (qPos[q].length < 9) continue;
      const g = new BufferGeometry();
      g.setAttribute('position', new BufferAttribute(new Float32Array(qPos[q]), 3));
      if (qNorm[q].length === qPos[q].length) {
        g.setAttribute('normal', new BufferAttribute(new Float32Array(qNorm[q]), 3));
      } else {
        g.computeVertexNormals();
      }
      if (qUv[q].length === (qPos[q].length / 3) * 2) {
        g.setAttribute('uv', new BufferAttribute(new Float32Array(qUv[q]), 2));
      }
      g.computeBoundingBox();
      out.push(g);
    }
    return out;
  };

  // Separate wheel/disc/hub/suspension meshes from deformable body meshes (matching both node path AND material name)
  const spinRe = /(?:^|\/)(WHEEL_[LR][FR]|RIM_[LR][FR]|Tyre_[LR][FR]|DISC_[LR][FR]|Brake_Disk_[LR][FR])(?:\/|$)|(?:^|\/|[_-\s])(wheel|tire|tyre|rim|rim_second|alloy|disc|disk|brakedisk|brake_?disk|rotor|pneu|roue|jante|llanta|felge)s?(?:_|\b)/i;
  const caliperRe = /(?:^|\/)(SUSP_[LR][FR]|g_Susp_Hub_[LR][FR])(?:\/|$)|caliper|brake_?cal|brembo|pinza|etrier/i;
  const wishboneRe = /g_Susp_ALever/i;

  const wheelSpinEntries: { geo: BufferGeometry; mat: Material | Material[]; c: Vector3; box: Box3 }[] = [];
  const wheelCaliperEntries: { geo: BufferGeometry; mat: Material | Material[]; c: Vector3; box: Box3 }[] = [];
  const bodyEntries: { geo: BufferGeometry; mat: Material | Material[]; path: string; box: Box3 }[] = [];

  for (const r of rawMeshes) {
    if (wishboneRe.test(r.path)) continue; // internal suspension wishbones hidden inside wheel wells
    if (!r.geo.boundingBox || r.geo.getAttribute('position').count === 0) continue;
    const b = r.geo.boundingBox;
    const c = new Vector3(); b.getCenter(c);
    const s = new Vector3(); b.getSize(s);
    const matNames = Array.isArray(r.mesh.material)
      ? r.mesh.material.map((m) => m?.name || '').join(' ')
      : (r.mesh.material?.name || '');
    const fullTag = `${r.path} ${matNames}`;
    const isMultiCorner = s.x > 0.95 || s.z > 1.25;

    if (caliperRe.test(fullTag)) {
      if (isMultiCorner) {
        for (const sub of splitByWheelQuadrant(r.geo)) {
          const sc = new Vector3(); sub.boundingBox!.getCenter(sc);
          wheelCaliperEntries.push({ geo: sub, mat: r.mesh.material, c: sc, box: sub.boundingBox! });
        }
      } else {
        wheelCaliperEntries.push({ geo: r.geo, mat: r.mesh.material, c, box: b });
      }
    } else if (
      spinRe.test(fullTag) ||
      (s.y > 0.25 && s.y < 0.88 && s.z > 0.25 && s.z < 0.88 && b.min.y < 0.18 && Math.abs(c.x) > 0.50 && Math.abs(c.z) > 0.75)
    ) {
      if (isMultiCorner) {
        for (const sub of splitByWheelQuadrant(r.geo)) {
          const sc = new Vector3(); sub.boundingBox!.getCenter(sc);
          wheelSpinEntries.push({ geo: sub, mat: r.mesh.material, c: sc, box: sub.boundingBox! });
        }
      } else {
        wheelSpinEntries.push({ geo: r.geo, mat: r.mesh.material, c, box: b });
      }
    } else {
      bodyEntries.push({ geo: r.geo, mat: r.mesh.material, path: r.path, box: b });
    }
  }

  // Ensure wheels rest cleanly on y = 0
  let wheelMinY = Infinity;
  for (const w of wheelSpinEntries) if (w.box.min.y < wheelMinY) wheelMinY = w.box.min.y;
  if (Number.isFinite(wheelMinY) && Math.abs(wheelMinY) > 1e-4) {
    const shiftY = new Matrix4().makeTranslation(0, -wheelMinY, 0);
    for (const w of wheelSpinEntries) { w.geo.applyMatrix4(shiftY); w.geo.computeBoundingBox(); w.box.getCenter(w.c); }
    for (const w of wheelCaliperEntries) { w.geo.applyMatrix4(shiftY); w.geo.computeBoundingBox(); w.box.getCenter(w.c); }
    for (const b of bodyEntries) { b.geo.applyMatrix4(shiftY); b.geo.computeBoundingBox(); }
  }

  // Recompute overall bounds
  const box = new Box3();
  for (const b of bodyEntries) box.union(b.box);
  for (const w of wheelSpinEntries) box.union(w.box);
  box.getSize(size);

  const profile = isSupercar ? createSupercarProfile() : createSedanProfile();
  const baseLen = isSupercar ? 4.55 : 4.65;
  const baseW = isSupercar ? 1.95 : 1.88;
  const baseH = isSupercar ? 1.20 : 1.38;
  profile.length = Math.max(3.6, Math.min(5.4, size.z || baseLen));
  profile.width = Math.max(1.55, Math.min(2.2, size.x || baseW));
  profile.height = Math.max(1.12, Math.min(1.85, size.y || baseH));
  const zS = profile.length / baseLen, xS = profile.width / baseW, yS = profile.height / baseH;
  profile.frontAxleZ *= zS; profile.rearAxleZ *= zS;
  profile.cowlZ *= zS; profile.hoodFrontZ *= zS;
  profile.roofFrontZ *= zS; profile.roofRearZ *= zS;
  profile.deckZ *= zS; profile.trunkEndZ *= zS;
  profile.trackHalf *= xS;
  for (let i = 0; i < CarProfile.samples(); i++) {
    profile.top[i] *= yS;
    profile.belt[i] *= yS;
    profile.halfW[i] *= xS;
    profile.roofHalfW[i] *= xS;
  }

  // Auto-measure BOTH front hood and rear trunk / engine-deck dimensions directly from the imported vertex envelope
  {
    const N = CarProfile.samples();
    const sliceMaxY = new Float32Array(N);
    const sliceMaxX = new Float32Array(N);
    const sliceCnt = new Uint32Array(N);
    const halfL = profile.halfL;
    const invLen = (N - 1) / Math.max(0.1, profile.length);
    for (const b of bodyEntries) {
      if (/wing|spoiler|antenna/i.test(b.path)) continue;
      const pos = b.geo.getAttribute('position') as BufferAttribute;
      const stride = pos.count > 20000 ? 4 : 1;
      for (let i = 0; i < pos.count; i += stride) {
        const x = Math.abs(pos.getX(i));
        const y = pos.getY(i);
        const z = pos.getZ(i);
        const si = Math.max(0, Math.min(N - 1, Math.round((z + halfL) * invLen)));
        sliceCnt[si]++;
        if (y > sliceMaxY[si]) sliceMaxY[si] = y;
        if (y > 0.32 * profile.height && x > sliceMaxX[si]) sliceMaxX[si] = x;
      }
    }
    // Find cabin roof peak height around center of car
    let roofPeakY = profile.height;
    for (let i = 0; i < N; i++) {
      const z = profile.zAt(i);
      if (z >= -0.6 && z <= 0.4 && sliceCnt[i] > 4 && sliceMaxY[i] > roofPeakY * 0.8) {
        if (sliceMaxY[i] > roofPeakY) roofPeakY = sliceMaxY[i];
      }
    }

    // 1) Rear trunk / engine-deck measurement: scan rearward from z = 0
    const deckDropY = roofPeakY - Math.max(0.11, 0.12 * profile.height);
    let measuredDeckZ = profile.deckZ;
    for (let i = Math.floor(N * 0.5); i >= 2; i--) {
      if (sliceCnt[i] < 4) continue;
      if (sliceMaxY[i] <= deckDropY) {
        measuredDeckZ = profile.zAt(i);
        break;
      }
    }
    measuredDeckZ = Math.max(-halfL + 0.55, Math.min(Math.min(profile.roofRearZ - 0.10, -1.05), measuredDeckZ));

    let measuredTrunkEndZ = profile.trunkEndZ;
    for (let i = 0; i < Math.floor(N * 0.45); i++) {
      if (sliceCnt[i] >= 4 && sliceMaxY[i] >= 0.42 * profile.height) {
        measuredTrunkEndZ = profile.zAt(i) + 0.04;
        break;
      }
    }
    measuredTrunkEndZ = Math.max(-halfL + 0.06, Math.min(measuredDeckZ - 0.30, measuredTrunkEndZ));
    profile.deckZ = measuredDeckZ;
    profile.trunkEndZ = measuredTrunkEndZ;

    // 2) Front hood measurement: scan forward from z = 0 to detect windshield-to-hood cowl transition
    const cowlDropY = roofPeakY - Math.max(0.14, 0.16 * profile.height);
    let measuredCowlZ = profile.cowlZ;
    for (let i = Math.floor(N * 0.5); i < N - 2; i++) {
      if (sliceCnt[i] < 4) continue;
      if (sliceMaxY[i] <= cowlDropY) {
        measuredCowlZ = profile.zAt(i);
        break;
      }
    }
    measuredCowlZ = Math.max(Math.max(profile.roofFrontZ + 0.10, 0.55), Math.min(halfL - 0.72, measuredCowlZ));

    let measuredHoodFrontZ = profile.hoodFrontZ;
    for (let i = N - 1; i > Math.floor(N * 0.55); i--) {
      if (sliceCnt[i] >= 4 && sliceMaxY[i] >= 0.36 * profile.height) {
        measuredHoodFrontZ = profile.zAt(i) - 0.04;
        break;
      }
    }
    measuredHoodFrontZ = Math.min(halfL - 0.06, Math.max(measuredCowlZ + 0.52, measuredHoodFrontZ));
    profile.cowlZ = measuredCowlZ;
    profile.hoodFrontZ = measuredHoodFrontZ;

    // Fit rear third & front hood top/belt/halfW profile tables to the smoothed measured GLB silhouette
    for (let i = 1; i < N - 1; i++) {
      const z = profile.zAt(i);
      const isRearZone = z <= measuredDeckZ + 0.12;
      const isFrontZone = z >= measuredCowlZ - 0.08;
      if (!isRearZone && !isFrontZone) continue;
      if (sliceCnt[i] < 4) continue;
      const smY = (
        (sliceCnt[i - 1] > 0 ? sliceMaxY[i - 1] : sliceMaxY[i]) +
        sliceMaxY[i] * 2 +
        (sliceCnt[i + 1] > 0 ? sliceMaxY[i + 1] : sliceMaxY[i])
      ) * 0.25;
      const smX = (
        (sliceCnt[i - 1] > 0 ? sliceMaxX[i - 1] : sliceMaxX[i]) +
        sliceMaxX[i] * 2 +
        (sliceCnt[i + 1] > 0 ? sliceMaxX[i + 1] : sliceMaxX[i])
      ) * 0.25;
      if (smY > 0.35 * profile.height && smY <= profile.height * 1.05) {
        profile.top[i] = smY;
        profile.belt[i] = Math.min(smY - 0.03, Math.max(0.45 * profile.height, smY * 0.88));
      }
      if (smX > 0.45 && smX <= profile.width * 0.56) {
        profile.halfW[i] = smX;
      }
    }
  }

  // Build 4 wheel assemblies (FL, FR, RL, RR) centred on their tire/rim bounding box hub
  let wheels: CustomWheelAssembly[] | null = null;
  if (wheelSpinEntries.length >= 4) {
    const spinBuckets: typeof wheelSpinEntries[] = [[], [], [], []]; // FL, FR, RL, RR
    const calBuckets: typeof wheelCaliperEntries[] = [[], [], [], []];
    for (const w of wheelSpinEntries) {
      const idx = (w.c.z >= 0 ? 0 : 2) + (w.c.x >= 0 ? 0 : 1);
      spinBuckets[idx].push(w);
    }
    for (const c of wheelCaliperEntries) {
      const idx = (c.c.z >= 0 ? 0 : 2) + (c.c.x >= 0 ? 0 : 1);
      calBuckets[idx].push(c);
    }
    if (spinBuckets.every((bk) => bk.length > 0)) {
      // Compute each wheel's hub center & radius so any unnamed brake caliper / hub mesh sitting strictly
      // inside a wheel cylinder (e.g. BMW red calipers or Dodge 4-corner calipers) is mounted onto the wheel knuckle
      // instead of staying floating in the fender when the car jumps!
      const wheelCenters = spinBuckets.map((bk) => {
        const wb = new Box3();
        for (const e of bk) wb.union(e.box);
        const wc = new Vector3(); wb.getCenter(wc);
        const ws = new Vector3(); wb.getSize(ws);
        return { c: wc, r: Math.max(0.28, Math.min(0.44, Math.max(ws.y, ws.z) * 0.5)) };
      });
      const skipCaliperRe = /paint|cars\b|glass|window|light|mirror|door|hood|capot|trunk|bumper|spoiler|wing|interior|seat|dash/i;
      const keptBody: typeof bodyEntries = [];
      for (const b of bodyEntries) {
        const matNames = Array.isArray(b.mat) ? b.mat.map((m) => m?.name || '').join(' ') : (b.mat?.name || '');
        const tag = `${b.path} ${matNames}`;
        const bs = new Vector3(); b.box.getSize(bs);
        const bc = new Vector3(); b.box.getCenter(bc);

        if (!skipCaliperRe.test(tag)) {
          // Case A: single compact caliper/hub mesh sitting inside one of the 4 wheel cylinders
          if (bs.x < 0.46 && bs.y < 0.48 && bs.z < 0.48) {
            let matchedQ = -1;
            for (let q = 0; q < 4; q++) {
              const wh = wheelCenters[q];
              const dyz = Math.hypot(bc.y - wh.c.y, bc.z - wh.c.z);
              if (dyz < wh.r * 0.90 && Math.abs(bc.x - wh.c.x) < 0.36) {
                matchedQ = q;
                break;
              }
            }
            if (matchedQ >= 0) {
              calBuckets[matchedQ].push({ geo: b.geo, mat: b.mat, c: bc, box: b.box });
              continue;
            }
          }
          // Case B: combined 4-corner caliper mesh whose low vertical span sits strictly within wheel height
          if ((bs.x > 0.95 || bs.z > 1.25) && bs.y < 0.48 && b.box.min.y >= 0.08 && b.box.max.y <= 0.72) {
            const subs = splitByWheelQuadrant(b.geo);
            const allInsideHubs = subs.length >= 2 && subs.every((sub) => {
              const sb = sub.boundingBox!;
              const ss = new Vector3(); sb.getSize(ss);
              const sc = new Vector3(); sb.getCenter(sc);
              if (ss.x > 0.46 || ss.y > 0.48 || ss.z > 0.48) return false;
              const q = (sc.z >= 0 ? 0 : 2) + (sc.x >= 0 ? 0 : 1);
              const wh = wheelCenters[q];
              return Math.hypot(sc.y - wh.c.y, sc.z - wh.c.z) < wh.r * 0.92 && Math.abs(sc.x - wh.c.x) < 0.38;
            });
            if (allInsideHubs) {
              for (const sub of subs) {
                const sc = new Vector3(); sub.boundingBox!.getCenter(sc);
                const q = (sc.z >= 0 ? 0 : 2) + (sc.x >= 0 ? 0 : 1);
                calBuckets[q].push({ geo: sub, mat: b.mat, c: sc, box: sub.boundingBox! });
              }
              continue;
            }
          }
        }
        keptBody.push(b);
      }
      bodyEntries.length = 0;
      for (const b of keptBody) bodyEntries.push(b);

      wheels = spinBuckets.map((bk, idx) => {
        const wb = new Box3();
        for (const e of bk) wb.union(e.box);
        const wc = new Vector3(); wb.getCenter(wc);
        const ws = new Vector3(); wb.getSize(ws);
        const radius = Math.max(0.28, Math.min(0.42, Math.max(ws.y, ws.z) * 0.5));
        const spinner = new Group();
        let normalizedTireMat: MeshStandardMaterial | null = null;
        for (const e of bk) {
          const cg = e.geo.clone();
          cg.translate(-wc.x, -wc.y, -wc.z);
          if (!cg.getAttribute('normal')) cg.computeVertexNormals();
          // Detect baked negative-scale inverted wheel geometry (e.g. BMW FR tire) and flip winding + normals outward
          {
            const pAttr = cg.getAttribute('position') as BufferAttribute;
            const nAttr = cg.getAttribute('normal') as BufferAttribute;
            let dotSum = 0, cnt = 0;
            const step = Math.max(1, Math.floor(pAttr.count / 250));
            for (let k = 0; k < pAttr.count; k += step) {
              dotSum += pAttr.getX(k) * nAttr.getX(k) + pAttr.getY(k) * nAttr.getY(k) + pAttr.getZ(k) * nAttr.getZ(k);
              cnt++;
            }
            if (cnt > 0 && dotSum / cnt < -0.015) {
              const idx = cg.getIndex();
              if (idx) {
                const arr = idx.array as Uint16Array | Uint32Array;
                for (let k = 0; k + 2 < arr.length; k += 3) {
                  const tmp = arr[k + 1]; arr[k + 1] = arr[k + 2]; arr[k + 2] = tmp;
                }
                idx.needsUpdate = true;
              }
              for (let k = 0; k < nAttr.count; k++) {
                nAttr.setXYZ(k, -nAttr.getX(k), -nAttr.getY(k), -nAttr.getZ(k));
              }
              nAttr.needsUpdate = true;
            }
          }
          cg.computeBoundingSphere();
          let useMat = e.mat;
          if (!Array.isArray(e.mat) && /^(Car_plastic_dark|Material_2125659471)$/i.test(e.mat.name || '')) {
            if (!normalizedTireMat) {
              normalizedTireMat = new MeshStandardMaterial({
                name: 'TireRubberNormalized',
                color: 0x131518,
                metalness: 0.04,
                roughness: 0.78,
                envMapIntensity: 0.45,
                side: DoubleSide,
              });
            }
            useMat = normalizedTireMat;
          }
          const m = new Mesh(cg, useMat);
          m.visible = true;
          m.castShadow = true;
          m.frustumCulled = true;
          spinner.add(m);
        }
        let caliper: Object3D | undefined;
        if (calBuckets[idx].length > 0) {
          caliper = new Group();
          for (const e of calBuckets[idx]) {
            const cg = e.geo.clone();
            cg.translate(-wc.x, -wc.y, -wc.z);
            if (!cg.getAttribute('normal')) cg.computeVertexNormals();
            cg.computeBoundingSphere();
            const m = new Mesh(cg, e.mat);
            m.visible = true;
            m.frustumCulled = true;
            caliper.add(m);
          }
        }
        return { spinner, caliper, hubX: wc.x, hubY: wc.y, hubZ: wc.z, radius };
      });
      profile.frontAxleZ = (wheels[0].hubZ + wheels[1].hubZ) * 0.5;
      profile.rearAxleZ = (wheels[2].hubZ + wheels[3].hubZ) * 0.5;
      profile.trackHalf = (Math.abs(wheels[0].hubX) + Math.abs(wheels[1].hubX) + Math.abs(wheels[2].hubX) + Math.abs(wheels[3].hubX)) * 0.25;
      profile.wheelRadius = (wheels[0].radius + wheels[2].radius) * 0.5;
    }
  }
  if (!wheels) {
    for (const w of wheelSpinEntries) bodyEntries.push({ geo: w.geo, mat: w.mat, path: 'body_part', box: w.box });
    for (const c of wheelCaliperEntries) bodyEntries.push({ geo: c.geo, mat: c.mat, path: 'body_part', box: c.box });
  }

  // Ensure every GLB material is visible, properly lit (envMapIntensity), and opaque unless glass/alpha-blended
  const isDodgeDemon = /dodge|demon|challenger/i.test(name);
  const defaultPaint = paintColor ?? (isSupercar ? 0xe06a10 : 0x8a9299);
  const materials = createCarMaterials(defaultPaint);
  const paintMaterials: MeshStandardMaterial[] = [materials.paint];
  const taillightMaterials: MeshStandardMaterial[] = [];
  const headlightMaterials: MeshStandardMaterial[] = [];
  const seenMats = new Set<Material>();
  const replacedMats = new Map<Material, Material>();
  const dodgeGlassMat = isDodgeDemon
    ? new MeshPhysicalMaterial({
        name: 'DodgeTintedGlass',
        color: 0x16202a,
        metalness: 0.14,
        roughness: 0.05,
        clearcoat: 1.0,
        clearcoatRoughness: 0.03,
        reflectivity: 1.0,
        transparent: true,
        opacity: 0.32,
        envMapIntensity: 1.85,
        depthWrite: false,
        side: DoubleSide,
      })
    : null;

  for (const e of bodyEntries) {
    const list = Array.isArray(e.mat) ? e.mat : [e.mat];

    for (let mi = 0; mi < list.length; mi++) {
      const m = list[mi];
      if (!m) continue;
      const existing = replacedMats.get(m);
      if (existing) {
        list[mi] = existing;
        continue;
      }
      const sm = m as MeshStandardMaterial;
      // On Dodge Challenger SRT Demon, ONLY replace the 3 exact greenhouse window panes (windshield, side windows,
      // rear window — mat "material" strictly inside upper greenhouse bounds) without polluting shared materials!
      const isDodgeGreenhousePane =
        isDodgeDemon &&
        dodgeGlassMat !== null &&
        sm.name === 'material' &&
        e.box.min.y > 0.65 * yS &&
        e.box.max.y > 0.86 * yS &&
        e.box.min.z > -0.74 * profile.halfL &&
        e.box.max.z < 0.50 * profile.halfL;

      if (isDodgeGreenhousePane) {
        list[mi] = dodgeGlassMat!;
        continue;
      }
      if (seenMats.has(m)) continue;
      seenMats.add(m);
      m.visible = true;

      const isGlassOrAlpha = /glass|glss|window|alpha|grid|detail|decal|light/i.test(sm.name) || Boolean(sm.alphaMap);
      if (!isGlassOrAlpha && sm.opacity < 0.99 && !sm.transparent) {
        sm.opacity = 1.0;
      }
      if (!isGlassOrAlpha && sm.transparent && sm.opacity <= 0.05) {
        sm.transparent = false;
        sm.opacity = 1.0;
      }
      if (sm.isMeshStandardMaterial && (sm.envMapIntensity === undefined || sm.envMapIntensity < 1.15)) {
        sm.envMapIntensity = 1.25;
      }
      if (isGlassOrAlpha && sm.isMeshStandardMaterial) {
        sm.roughness = Math.min(sm.roughness ?? 0.08, 0.06);
        sm.envMapIntensity = 1.9;
      }
      if (/^(Car_?paint|Car_?Paint_?\.?\d*|CARS|carpaint_second|ARm4_main)$/i.test(sm.name)) {
        const phys = new MeshPhysicalMaterial({
          name: sm.name,
          color: sm.color.clone(),
          map: sm.map,
          normalMap: sm.normalMap,
          normalScale: sm.normalScale ? sm.normalScale.clone() : undefined,
          roughnessMap: sm.roughnessMap,
          metalnessMap: sm.metalnessMap,
          aoMap: sm.aoMap,
          metalness: 0.72,
          roughness: 0.11,
          clearcoat: 1.0,
          clearcoatRoughness: 0.04,
          reflectivity: 1.0,
          envMapIntensity: 1.85,
          side: sm.side,
          transparent: false,
          opacity: 1.0,
        });
        // Apply user-picked color override to any car, or defaultPaint for untextured McLaren Carpaint
        if (overridePaint !== undefined) phys.color.setHex(overridePaint);
        else if (/^Carpaint$/i.test(sm.name)) phys.color.setHex(defaultPaint);
        replacedMats.set(m, phys);
        list[mi] = phys;
        paintMaterials.push(phys);
      } else if (/carbon|splitter|diffuser|spoiler|wing|lip|aero|skirt/i.test(`${sm.name} ${e.path}`) && !isGlassOrAlpha) {
        const carbonPack = getCarbonFiberTextures(false);
        const carbonMat = new MeshPhysicalMaterial({
          name: sm.name,
          color: 0xffffff,
          map: carbonPack.map,
          normalMap: carbonPack.normalMap,
          normalScale: new Vector2(0.55, 0.55),
          roughnessMap: carbonPack.roughnessMap,
          metalness: 0.25,
          roughness: 0.30,
          clearcoat: 0.94,
          clearcoatRoughness: 0.06,
          reflectivity: 0.9,
          envMapIntensity: 1.7,
          side: sm.side,
        });
        replacedMats.set(m, carbonMat);
        list[mi] = carbonMat;
      } else if (/disc|disk|rotor/i.test(`${sm.name} ${e.path}`) && !isGlassOrAlpha) {
        const discPack = getBrakeDiscTextures(false);
        sm.map = sm.map ?? discPack.map;
        sm.normalMap = sm.normalMap ?? discPack.normalMap;
        sm.normalScale = new Vector2(0.5, 0.5);
        sm.roughnessMap = sm.roughnessMap ?? discPack.roughnessMap;
        sm.metalness = 0.92;
        sm.roughness = 0.42;
        sm.envMapIntensity = 1.25;
      } else if (/Taillights|Red_Light|brake_?light|redglass|ARm4_taillight|ARm4_chmsl/i.test(sm.name)) {
        sm.emissive?.setHex(0xff1020);
        sm.emissiveIntensity = 0.85;
        taillightMaterials.push(sm);
      } else if (
        (/headlight|front_?light|drl|daytime|xenon|led_?front|ARm4_headlight|ARm4_drl|ARm4_beam/i.test(`${sm.name} ${e.path}`) ||
          (/Light_D|lamp|light/i.test(sm.name) && e.box.min.z > profile.halfL * 0.45)) &&
        !/tail|rear|brake|rev|turn|signal|glass|lens/i.test(sm.name)
      ) {
        sm.emissive?.setHex(0xeef6ff);
        sm.emissiveIntensity = 0.25;
        headlightMaterials.push(sm);
      }
      list[mi].needsUpdate = true;
    }
    if (Array.isArray(e.mat)) e.mat = list;
    else if (list.length === 1) e.mat = list[0];
  }

  // Auto-detect dedicated factory panel sub-meshes (doors, hood, trunk, bumpers) by geometry envelope
  // so models with unnamed Blender meshes (e.g. Circle053_Car_Paint_0, Object_101) follow exact factory seams!
  const hasHollowCenter = (geo: BufferGeometry, centerHalfX: number): boolean => {
    const pos = geo.getAttribute('position');
    if (!pos || pos.count === 0) return false;
    const step = Math.max(1, Math.floor(pos.count / 250));
    for (let i = 0; i < pos.count; i += step) {
      if (Math.abs(pos.getX(i)) < centerHalfX) return false;
    }
    return true;
  };

  const halfW = profile.width * 0.5;
  const factoryGroupByIdx = new Map<number, DeformGroup | 'DOOR_LR'>();
  let factoryDoorMinZ = Infinity, factoryDoorMaxZ = -Infinity;
  let factoryDoorMinY = Infinity, factoryDoorMaxY = -Infinity;
  let detectedFactoryDoors = false;
  let detectedFactoryHood = false;
  let detectedFactoryTrunk = false;
  let detectedFactoryFrontBumper = false;
  let detectedFactoryRearBumper = false;

  for (let idx = 0; idx < bodyEntries.length; idx++) {
    const e = bodyEntries[idx];
    const b = e.box;
    const xSpan = b.max.x - b.min.x;
    const zSpan = b.max.z - b.min.z;
    const maxAbsX = Math.max(Math.abs(b.min.x), Math.abs(b.max.x));
    const isOneSideDoor = (b.min.x > halfW * 0.35 || b.max.x < -halfW * 0.35);
    const isMirroredLR = b.min.x < -halfW * 0.72 && b.max.x > halfW * 0.72 && hasHollowCenter(e.geo, halfW * 0.42);

    // 1. Dedicated factory door panel mesh (single side or mirrored L+R with hollow center)
    if (
      b.min.z > -profile.halfL * 0.42 &&
      b.max.z < profile.halfL * 0.48 &&
      zSpan > profile.halfL * 0.28 &&
      b.min.y > 0.12 * yS &&
      b.max.y < 0.80 * yS &&
      maxAbsX > halfW * 0.72 &&
      (isOneSideDoor || isMirroredLR)
    ) {
      factoryGroupByIdx.set(idx, isMirroredLR ? 'DOOR_LR' : (b.min.x > 0 ? DeformGroup.DoorL : DeformGroup.DoorR));
      detectedFactoryDoors = true;
      if (b.min.z < factoryDoorMinZ) factoryDoorMinZ = b.min.z;
      if (b.max.z > factoryDoorMaxZ) factoryDoorMaxZ = b.max.z;
      if (b.min.y < factoryDoorMinY) factoryDoorMinY = b.min.y;
      if (b.max.y > factoryDoorMaxY) factoryDoorMaxY = b.max.y;
      continue;
    }

    // 2. Dedicated factory front bumper
    if (
      b.min.z > profile.halfL * 0.62 &&
      b.max.z > profile.halfL * 0.88 &&
      b.max.y < 0.60 * yS &&
      xSpan > profile.width * 0.55
    ) {
      factoryGroupByIdx.set(idx, DeformGroup.BumperF);
      detectedFactoryFrontBumper = true;
      continue;
    }

    // 3. Dedicated factory rear bumper
    if (
      b.max.z < -profile.halfL * 0.58 &&
      b.min.z < -profile.halfL * 0.88 &&
      b.max.y < 0.62 * yS &&
      xSpan > profile.width * 0.55
    ) {
      factoryGroupByIdx.set(idx, DeformGroup.BumperR);
      detectedFactoryRearBumper = true;
      continue;
    }

    // 4. Dedicated factory hood
    if (
      b.min.z > profile.halfL * 0.22 &&
      b.max.z > profile.halfL * 0.78 &&
      b.min.y > 0.42 * yS &&
      b.max.y < 0.78 * yS &&
      zSpan > profile.halfL * 0.42 &&
      xSpan > profile.width * 0.45
    ) {
      factoryGroupByIdx.set(idx, DeformGroup.Hood);
      detectedFactoryHood = true;
      continue;
    }

    // 5. Dedicated factory trunk lid
    if (
      b.max.z < -profile.halfL * 0.62 &&
      b.min.z < -profile.halfL * 0.86 &&
      b.min.y > 0.54 * yS &&
      b.max.y < 0.84 * yS &&
      xSpan > profile.width * 0.35
    ) {
      factoryGroupByIdx.set(idx, DeformGroup.Trunk);
      detectedFactoryTrunk = true;
    }
  }

  // Group door-mounted side mirrors & exterior door handles with their respective door so they swing open together
  if (detectedFactoryDoors) {
    for (let idx = 0; idx < bodyEntries.length; idx++) {
      if (factoryGroupByIdx.has(idx)) continue;
      const e = bodyEntries[idx];
      const b = e.box;
      const zSpan = b.max.z - b.min.z;
      const maxAbsX = Math.max(Math.abs(b.min.x), Math.abs(b.max.x));
      if (
        zSpan < profile.halfL * 0.38 &&
        b.min.z >= factoryDoorMinZ - 0.10 &&
        b.max.z <= factoryDoorMaxZ + 0.10 &&
        b.min.y >= factoryDoorMinY - 0.04 &&
        b.max.y <= factoryDoorMaxY + 0.22 &&
        maxAbsX > halfW * 0.78 &&
        hasHollowCenter(e.geo, halfW * 0.65)
      ) {
        const isLR = b.min.x < 0 && b.max.x > 0;
        factoryGroupByIdx.set(idx, isLR ? 'DOOR_LR' : (b.min.x > 0 ? DeformGroup.DoorL : DeformGroup.DoorR));
      }
    }
  }

  const hasExplicitFrontBumper = detectedFactoryFrontBumper || bodyEntries.some((e) => /(?:^|\/)(FRONT_BUMPER|g_Bumper_F|ARm4_bumper_F)(?:[/_.]|$)|(bumper|splitter).*(front|f\b)/i.test(e.path));
  const hasExplicitRearBumper = detectedFactoryRearBumper || bodyEntries.some((e) => /(?:^|\/)(REAR_BUMPER|g_Diffuser|ARm4_bumper_R)(?:[/_.]|$)|(bumper|diffuser|valance).*(rear|back|r\b)/i.test(e.path));
  const hasExplicitHood = detectedFactoryHood || bodyEntries.some((e) => /(?:^|\/)(HOOD|GEO_HOOD|bonnet|frunk|ARm4_hood)(?:[/_.]|$)/i.test(e.path));
  const hasExplicitTrunk = detectedFactoryTrunk || bodyEntries.some((e) => /(?:^|\/)(MOTORHOOD|trunk|boot|tailgate|decklid|ARm4_trunk)(?:[/_.]|$)/i.test(e.path));

  const classifyPath = (p: string): DeformGroup | null => {
    // BMW M4 (F82) exact factory panel seams / shutlines (ARm4_*)
    if (/(?:^|\/)ARm4_hood(?:[/_.]|$)/i.test(p)) return DeformGroup.Hood;
    if (/(?:^|\/)(ARm4_trunk|ARm4_lip_carbon|ARm4_trunklight)(?:[/_.]|$)/i.test(p)) return DeformGroup.Trunk;
    if (/(?:^|\/)(ARm4_bumper_F|ARm4_grille_carbon|ARm4_splitter_street|ARm4_canards_carbon)(?:[/_.]|$)/i.test(p)) return DeformGroup.BumperF;
    if (/(?:^|\/)(ARm4_bumper_R|ARm4_bumper_Support|ARm4_rear_diffuser|ARm4_mudflaps_R)(?:[/_.]|$)/i.test(p)) return DeformGroup.BumperR;
    if (/(?:^|\/)(ARm4_door_L|ARm4_doorpanel_L|ARm4_doorglass_L|ARm4_mirror_L|ARm4_fender_L|ARm4_dender_trim_L|ARm4_fender_trim_L|ARm4_mudflaps_F_L)(?:[/_.]|$)/i.test(p)) return DeformGroup.SideL;
    if (/(?:^|\/)(ARm4_door_R|ARm4_doorpanel_R|ARm4_doorglass_R|ARm4_mirror_R|ARm4_fender_R|ARm4_fender_trim_R|ARm4_mudflaps_F_R)(?:[/_.]|$)/i.test(p)) return DeformGroup.SideR;
    if (/(?:^|\/)(ARm4_roof|ARm4_ariel|ARm4_body|ARm4_interior|ARm4_gts_pad|ARm4_windshield|ARm4_backlight|ARm4_sideglass|ARm4_taillight|ARm4_headlight|ARm4_underbody|ARm4_exhaust|ARm4_subframe_R|ARm4_diff|ARm4_driveshaft|ARm4_halfshaft|ARm4_lowerarm|ARm4_upperarm|ARm4_spring_R|ARm4_shock_R|ARm4_hub_F|ARm4_tierod_F|ARm4_transfercase)(?:[/_.]|$)/i.test(p)) return DeformGroup.Body;

    if (/(?:^|\/)(HOOD|GEO_HOOD|bonnet|frunk)(?:\/|$)/i.test(p)) return DeformGroup.Hood;
    if (/(?:^|\/)(MOTORHOOD|trunk|boot|tailgate|decklid)(?:\/|$)/i.test(p)) return DeformGroup.Trunk;
    if (/(?:^|\/)(FRONT_BUMPER|g_Bumper_F)(?:\/|$)|(bumper|grille|splitter|lip).*(front|f\b)|(front|f\b).*(bumper|grille|splitter)/i.test(p)) return DeformGroup.BumperF;
    if (/(?:^|\/)(REAR_BUMPER|g_Diffuser)(?:\/|$)|(bumper|diffuser|valance).*(rear|back|r\b)|(rear|back|r\b).*(bumper|diffuser)/i.test(p)) return DeformGroup.BumperR;
    // Doors are welded seamlessly into SideL/SideR (non-opening doors)
    if (/DOOR_L1\/GEO_DOOR_RR/i.test(p)) return DeformGroup.SideR;
    if (/DOOR_R1\/polymsh5/i.test(p)) return DeformGroup.SideL;
    if (/(?:^|\/)(DOOR_L1|DOOR_L|GEO_DOOR_L|door_left|porte_g|GEO_frontFender_L|fender_l|quarter_l|side_l|sill_l)(?:\/|$)/i.test(p)) return DeformGroup.SideL;
    if (/(?:^|\/)(DOOR_R1|DOOR_R|GEO_DOOR_R1|door_right|porte_d|GEO_frontFender_R|fender_r|quarter_r|side_r|sill_r)(?:\/|$)/i.test(p)) return DeformGroup.SideR;
    return null;
  };
  const classifyPoint = (x: number, y: number, z: number): DeformGroup => {
    // Only slice unified body geometry when the GLB lacks dedicated factory meshes for that panel
    if (!hasExplicitFrontBumper) {
      const frontBumperMaxY = Math.abs(x) < 0.38 ? 0.66 * yS : 0.54 * yS;
      if (z > profile.halfL - 0.40 && y < frontBumperMaxY) return DeformGroup.BumperF;
    }
    if (!hasExplicitRearBumper) {
      if (z < -profile.halfL + 0.34 && y < 0.54 * yS && Math.abs(x) < profile.planHalfW(z) * 0.94) return DeformGroup.BumperR;
    }
    if (!hasExplicitHood && z > profile.cowlZ + 0.04 && z < profile.hoodFrontZ + 0.05 && y > profile.beltY(z) - 0.04 && Math.abs(x) < profile.planHalfW(z) * 0.82) return DeformGroup.Hood;
    if (!hasExplicitTrunk && z < profile.deckZ - 0.04 && z > profile.trunkEndZ - 0.06 && y > profile.beltY(z) - 0.05 && Math.abs(x) < profile.planHalfW(z) * 0.82) return DeformGroup.Trunk;
    const hw = profile.planHalfW(z);
    const belt = profile.beltY(z);
    if (y < belt + 0.04 && Math.abs(x) > hw * 0.62) {
      return x > 0 ? DeformGroup.SideL : DeformGroup.SideR;
    }
    return DeformGroup.Body;
  };

  // Group sub-geometries by (DeformGroup, Material) without destructive vertex welding so UVs, indices & shading stay 100% intact
  const rawParts: CarPart[] = [];
  const rawRigidParts: CarPart[] = [];
  const rawSpoilerParts: CarPart[] = [];
  const spoilerBox = new Box3();

  // Only keep small non-structural cabin fixtures (steering wheel, seats, displays, belts, pedals) rigid.
  // Structural carbon monocoque/cage, full-length interior tubs (ARm4_interior), chassis mechanicals, rear diffuser, and door panels MUST deform with the soft-body lattice!
  const isRigidInteriorEntry = (e: { mat: Material | Material[]; path: string; box: Box3 }): boolean => {
    if (/(?:^|\/)(REAR_BUMPER|FRONT_BUMPER|g_Diffuser|g_Bumper_F|DOOR_L|DOOR_R|g_Body|WING_0|ARm4_door|ARm4_engine|ARm4_hood|ARm4_trunk|ARm4_bumper|ARm4_fender|ARm4_body|ARm4_roof|ARm4_interior)(?:[/_.]|$)/i.test(e.path)) return false;
    if (/(?:^|\/)(STEERING_WHEEL|STEER_HR|GEO_Seats|seat|CINTURE_|LED_|DRS_Tag|IPAS_Tag|pedal|console|ARm4_seats|ARm4_buckedseat|ARm4_steer|ARm4_signalstalk|ARm4_brakepedal|ARm4_gaspedal|ARm4_needle|ARm4_sunvisor)(?:[/_.]|$)/i.test(e.path)) return true;
    const m = Array.isArray(e.mat) ? e.mat[0] : e.mat;
    const matName = m?.name || '';
    if (/^(INT_DISPLAY|IN_Cinture|INT_Skin|INT_Velvet|INT_Carpet|INT_Cuciture|seat|leather|carpet|alcantara)/i.test(matName)) return true;
    return false;
  };

  const isActiveSpoilerEntry = (p: string): boolean => {
    // McLaren P1 active aero wing blade & hydraulic pistons (leave fixed g_Wing_Guide on Body)
    return /(?:^|\/)(WINGAXLE_piston|WINGAXLE|g_Wing|g_Wing_Turn)(?:\/|$)/i.test(p) && !/g_Wing_Guide/i.test(p);
  };

  const isFrontHeadlightEntry = (e: { mat: Material | Material[]; path: string; box: Box3 }, c: Vector3): boolean => {
    if (/(?:^|\/)(FRONT_LIGHT|REAR_LIGHT)(?:\/|$)/i.test(e.path)) return true;
    if (c.z < profile.halfL - 0.72) return false;
    const m = Array.isArray(e.mat) ? e.mat[0] : e.mat;
    const matName = m?.name || '';
    if (/headlight|lamp|light|drl|lens|glass/i.test(e.path) || /headlight|lamp|light|drl|lens|glass/i.test(matName)) {
      if (e.box.min.y >= 0.45 * yS) return true;
    }
    if (e.box.min.y >= 0.55 * yS && e.box.max.y <= 0.76 * yS && e.box.max.z < profile.halfL - 0.08 && !/Car_?paint|logo/i.test(matName)) {
      return true;
    }
    return false;
  };

  for (let idx = 0; idx < bodyEntries.length; idx++) {
    const e = bodyEntries[idx];
    if (isActiveSpoilerEntry(e.path)) {
      spoilerBox.union(e.box);
      rawSpoilerParts.push(makePart(e.geo, e.mat, e.path || `spoiler_${idx}`, DeformGroup.Body));
      continue;
    }
    if (isRigidInteriorEntry(e)) {
      rawRigidParts.push(makePart(e.geo, e.mat, e.path || `rigid_${idx}`, DeformGroup.Body));
      continue;
    }
    const byName = classifyPath(e.path);
    const byFactory = factoryGroupByIdx.get(idx);
    const s = new Vector3(); e.box.getSize(s);
    const c = new Vector3(); e.box.getCenter(c);
    if (isFrontHeadlightEntry(e, c)) {
      rawParts.push(makePart(e.geo, e.mat, e.path || `headlight_${idx}`, DeformGroup.Body));
    } else if (byName !== null) {
      rawParts.push(makePart(e.geo, e.mat, e.path || `part_${idx}`, byName));
    } else if (byFactory === 'DOOR_LR') {
      // Welded seamlessly into SideL/SideR flanks (non-opening doors)
      const split = splitGeometryByRegion(e.geo, (x) => (x >= 0 ? DeformGroup.SideL : DeformGroup.SideR));
      for (const [grp, geo] of split) rawParts.push(makePart(geo, e.mat, `${e.path || 'side'}_${grp}`, grp));
    } else if (byFactory !== undefined) {
      const mapped = byFactory === DeformGroup.DoorL ? DeformGroup.SideL : byFactory === DeformGroup.DoorR ? DeformGroup.SideR : byFactory;
      rawParts.push(makePart(e.geo, e.mat, e.path || `factory_${idx}`, mapped));
    } else if ((s.z > profile.length * 0.35 || s.x > profile.width * 0.65) && !/COCKPIT|WIPER|Windows|g_Wing_Guide/i.test(e.path)) {
      const split = splitGeometryByRegion(e.geo, classifyPoint);
      for (const [grp, geo] of split) rawParts.push(makePart(geo, e.mat, `${e.path || 'body'}_${grp}`, grp));
    } else {
      const grp = /COCKPIT|WIPER|Windows|g_Wing_Guide/i.test(e.path)
        ? DeformGroup.Body
        : classifyPoint(c.x, c.y, c.z);
      rawParts.push(makePart(e.geo, e.mat, e.path || `part_${idx}`, grp));
    }
  }

  const parts = consolidatePartsByGroupAndMaterial(rawParts, 1).filter(
    (p) => p.mesh.geometry.getAttribute('position')?.count > 0,
  );
  const rigidMeshes = consolidatePartsByGroupAndMaterial(rawRigidParts, 1)
    .filter((p) => p.mesh.geometry.getAttribute('position')?.count > 0)
    .map((p) => p.mesh);

  let spoilerMeshes: Mesh[] | undefined;
  let spoilerPivot: [number, number, number] | undefined;
  if (rawSpoilerParts.length > 0 && !spoilerBox.isEmpty()) {
    const sc = new Vector3();
    spoilerBox.getCenter(sc);
    spoilerPivot = [sc.x, spoilerBox.min.y + (spoilerBox.max.y - spoilerBox.min.y) * 0.35, spoilerBox.max.z - (spoilerBox.max.z - spoilerBox.min.z) * 0.35];
    spoilerMeshes = consolidatePartsByGroupAndMaterial(rawSpoilerParts, 1)
      .filter((p) => p.mesh.geometry.getAttribute('position')?.count > 0)
      .map((p) => {
        p.mesh.geometry.translate(-spoilerPivot![0], -spoilerPivot![1], -spoilerPivot![2]);
        if (!p.mesh.geometry.getAttribute('normal')) p.mesh.geometry.computeVertexNormals();
        p.mesh.geometry.computeBoundingBox();
        p.mesh.geometry.computeBoundingSphere();
        p.mesh.visible = true;
        p.mesh.frustumCulled = true;
        return p.mesh;
      });
  }

  // Only mount our procedural 3D engine block if the GLB does NOT already include its own pre-modeled 3D engine (e.g., BMW M4 F82 S55)!
  const isMidRearEngine = /mclaren|p1|ferrari|lamborghini|mid.?engine/i.test(name);
  const hasPreModeledEngine = bodyEntries.some((e) =>
    /(?:^|\/)(?:ARm4_)?(engine_i6|enginebay|header_i6|engine_block|motor)(?:[/_.]|$)/i.test(e.path),
  );
  const engineMeshes: Mesh[] = [];
  if (!hasPreModeledEngine) {
    const engineAssembly = buildDetailedEngineAssembly(profile, isMidRearEngine ? 'rear' : 'front', materials);
    for (const em of engineAssembly.rigid) {
      const g = em.geometry;
      if (!g.getAttribute('normal')) g.computeVertexNormals();
      g.computeBoundingBox();
      g.computeBoundingSphere();
      em.visible = true;
      em.frustumCulled = true;
      engineMeshes.push(em);
    }
    for (const dp of engineAssembly.deformable) {
      parts.push(dp);
    }
  }

  // Add detailed 3D cabin interior for Dodge Challenger SRT Demon (visible through its semi-transparent tinted windows!)
  if (isDodgeDemon) {
    for (const im of buildDodgeCabinInterior(profile)) {
      rigidMeshes.push(im);
    }
  }

  if (parts.length === 0 && rigidMeshes.length === 0) {
    throw new Error(`No renderable meshes produced from GLB model "${name}"`);
  }

  // Ensure valid normals, bounding spheres, visibility, and frustumCulled = true on every mesh
  // Expand deformable part bounding spheres slightly (+2.2m) so hinged/crumpled/detached panels never cull prematurely.
  for (const p of parts) {
    const g = p.mesh.geometry;
    if (!g.getAttribute('normal')) g.computeVertexNormals();
    g.computeBoundingBox();
    g.computeBoundingSphere();
    if (g.boundingSphere) g.boundingSphere.radius += 2.2;
    p.mesh.visible = true;
    p.mesh.frustumCulled = true;
  }
  for (const rm of rigidMeshes) {
    const g = rm.geometry;
    if (!g.getAttribute('normal')) g.computeVertexNormals();
    g.computeBoundingBox();
    g.computeBoundingSphere();
    rm.visible = true;
    rm.frustumCulled = true;
  }

  return {
    name,
    profile,
    parts,
    rigidMeshes,
    engineMeshes,
    spoilerMeshes,
    spoilerPivot,
    wheels,
    suppressProceduralWheels: true,
    materials,
    paintMaterials,
    taillightMaterials,
    headlightMaterials,
    dispose() {
      for (const m of Object.values(materials) as Material[]) m.dispose();
    },
  };
}

/** Merges parts that share the same DeformGroup and single Material into one indexed Mesh. */
function consolidatePartsByGroupAndMaterial(parts: CarPart[], threshold = 42): CarPart[] {
  if (parts.length <= threshold) return parts;
  const buckets = new Map<string, { group: DeformGroup; mat: Material; geos: BufferGeometry[]; name: string }>();
  const passthrough: CarPart[] = [];
  let matIdCounter = 1;
  const matIds = new Map<Material, number>();

  for (const p of parts) {
    if (Array.isArray(p.mesh.material)) {
      passthrough.push(p);
      continue;
    }
    const mat = p.mesh.material;
    let mid = matIds.get(mat);
    if (!mid) { mid = matIdCounter++; matIds.set(mat, mid); }
    const key = `${p.group}_${mid}`;
    let b = buckets.get(key);
    if (!b) {
      b = { group: p.group, mat, geos: [], name: `group${p.group}_${mat.name || mid}` };
      buckets.set(key, b);
    }
    b.geos.push(p.mesh.geometry);
  }

  const out: CarPart[] = [...passthrough];
  for (const b of buckets.values()) {
    if (b.geos.length === 1) {
      out.push(makePart(b.geos[0], b.mat, b.name, b.group));
      continue;
    }
    const merged = mergePreservingIndex(b.geos);
    out.push(makePart(merged, b.mat, b.name, b.group));
  }
  return out;
}

/** Concatenates indexed or non-indexed geometries while keeping shared vertex indices compact. */
function mergePreservingIndex(geos: BufferGeometry[]): BufferGeometry {
  let totalVerts = 0;
  let totalIdx = 0;
  let hasUV = false;
  for (const g of geos) {
    const vCount = g.getAttribute('position').count;
    totalVerts += vCount;
    const idx = g.getIndex();
    totalIdx += idx ? idx.count : vCount;
    if (g.getAttribute('uv')) hasUV = true;
  }
  const pos = new Float32Array(totalVerts * 3);
  const nrm = new Float32Array(totalVerts * 3);
  const uv = hasUV ? new Float32Array(totalVerts * 2) : null;
  const indices = totalVerts < 65535 ? new Uint16Array(totalIdx) : new Uint32Array(totalIdx);

  let vOffset = 0;
  let iOffset = 0;
  for (const g of geos) {
    const p = g.getAttribute('position') as BufferAttribute;
    if (!g.getAttribute('normal')) g.computeVertexNormals();
    const n = g.getAttribute('normal') as BufferAttribute;
    const u = g.getAttribute('uv') as BufferAttribute | undefined;
    pos.set(p.array as Float32Array, vOffset * 3);
    nrm.set(n.array as Float32Array, vOffset * 3);
    if (uv && u && u.array instanceof Float32Array) {
      uv.set(u.array, vOffset * 2);
    }
    const idx = g.getIndex();
    if (idx) {
      const arr = idx.array;
      for (let k = 0; k < idx.count; k++) indices[iOffset + k] = arr[k] + vOffset;
      iOffset += idx.count;
    } else {
      for (let k = 0; k < p.count; k++) indices[iOffset + k] = vOffset + k;
      iOffset += p.count;
    }
    vOffset += p.count;
  }
  const out = new BufferGeometry();
  out.setAttribute('position', new BufferAttribute(pos, 3));
  out.setAttribute('normal', new BufferAttribute(nrm, 3));
  if (uv) out.setAttribute('uv', new BufferAttribute(uv, 2));
  out.setIndex(new BufferAttribute(indices, 1));
  return out;
}

function makePart(geo: BufferGeometry, mat: Material | Material[], name: string, group: DeformGroup): CarPart {
  if (!geo.getAttribute('normal')) geo.computeVertexNormals();
  geo.computeBoundingSphere();
  if (geo.boundingSphere) geo.boundingSphere.radius += 2.2;
  const mesh = new Mesh(geo, mat);
  mesh.name = name;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.frustumCulled = true;
  return { name, group, mesh };
}

function splitGeometryByRegion(
  geo: BufferGeometry,
  classify: (x: number, y: number, z: number) => DeformGroup,
): [DeformGroup, BufferGeometry][] {
  const src = geo.index ? geo.toNonIndexed() : geo;
  const pos = src.getAttribute('position') as BufferAttribute;
  const nrm = src.getAttribute('normal') as BufferAttribute | undefined;
  const uv = src.getAttribute('uv') as BufferAttribute | undefined;
  const triCount = Math.floor(pos.count / 3);
  const buckets = new Map<DeformGroup, { p: number[]; n: number[]; u: number[] }>();
  for (let t = 0; t < triCount; t++) {
    const i0 = t * 3, i1 = t * 3 + 1, i2 = t * 3 + 2;
    const cx = (pos.getX(i0) + pos.getX(i1) + pos.getX(i2)) / 3;
    const cy = (pos.getY(i0) + pos.getY(i1) + pos.getY(i2)) / 3;
    const cz = (pos.getZ(i0) + pos.getZ(i1) + pos.getZ(i2)) / 3;
    const g = classify(cx, cy, cz);
    let b = buckets.get(g);
    if (!b) { b = { p: [], n: [], u: [] }; buckets.set(g, b); }
    for (const i of [i0, i1, i2]) {
      b.p.push(pos.getX(i), pos.getY(i), pos.getZ(i));
      if (nrm) b.n.push(nrm.getX(i), nrm.getY(i), nrm.getZ(i));
      if (uv) b.u.push(uv.getX(i), uv.getY(i));
    }
  }
  const out: [DeformGroup, BufferGeometry][] = [];
  for (const [g, b] of buckets.entries()) {
    const sub = new BufferGeometry();
    sub.setAttribute('position', new BufferAttribute(new Float32Array(b.p), 3));
    if (b.n.length === b.p.length) sub.setAttribute('normal', new BufferAttribute(new Float32Array(b.n), 3));
    else sub.computeVertexNormals();
    if (b.u.length > 0) sub.setAttribute('uv', new BufferAttribute(new Float32Array(b.u), 2));
    out.push([g, sub]);
  }
  return out;
}

/**
 * Builds a clean, low-draw-call 3D cabin interior for the Dodge Challenger SRT Demon
 * (dashboard, instrument hood, steering wheel & column, center console & shifter,
 * two front sport bucket seats with headrests, and rear bench), strictly sized inside
 * the cabin envelope so zero geometry ever clips outside the bodywork.
 */
function buildDodgeCabinInterior(P: CarProfile): Mesh[] {
  const dashMat = new MeshStandardMaterial({ color: 0x16191f, roughness: 0.78, metalness: 0.08, envMapIntensity: 0.5 });
  const seatMat = new MeshStandardMaterial({ color: 0x20242c, roughness: 0.72, metalness: 0.06, envMapIntensity: 0.55 });
  const trimMat = new MeshStandardMaterial({ color: 0x991b24, roughness: 0.45, metalness: 0.35, envMapIntensity: 0.85 });

  const floorY = P.floorY + 0.08;
  const beltY = P.beltY(0);
  const cabinW = Math.min(1.32, P.width * 0.68);
  const dashZ = P.cowlZ - 0.14;
  const frontSeatZ = P.cowlZ - 0.86;
  const rearSeatZ = Math.max(P.deckZ + 0.24, frontSeatZ - 0.78);

  const raw: CarPart[] = [];
  const addBox = (
    w: number, h: number, d: number,
    x: number, y: number, z: number,
    mat: Material,
    rx = 0,
  ) => {
    const g = new BoxGeometry(w, h, d);
    if (rx !== 0) g.rotateX(rx);
    g.translate(x, y, z);
    raw.push(makePart(g, mat, 'dodge_int', DeformGroup.Body));
  };

  // 1. Cabin floor tub + dashboard + instrument cluster hood
  addBox(cabinW, 0.05, Math.max(1.2, dashZ - rearSeatZ + 0.35), 0, floorY, (dashZ + rearSeatZ) * 0.5, dashMat);
  addBox(cabinW, 0.28, 0.34, 0, beltY - 0.12, dashZ, dashMat);
  addBox(0.42, 0.12, 0.24, 0.34, beltY + 0.03, dashZ - 0.02, dashMat);
  addBox(0.28, 0.15, 0.04, 0, beltY - 0.08, dashZ - 0.17, trimMat);

  // 2. Center console + transmission tunnel + shifter
  addBox(0.24, 0.24, Math.max(0.8, dashZ - rearSeatZ - 0.1), 0, floorY + 0.14, (dashZ + rearSeatZ) * 0.5, dashMat);
  const shifterGeo = new CylinderGeometry(0.018, 0.022, 0.16, 8);
  shifterGeo.translate(0, floorY + 0.32, dashZ - 0.42);
  raw.push(makePart(shifterGeo, trimMat, 'dodge_shifter', DeformGroup.Body));

  // 3. Steering column + steering wheel rim (driver side +X)
  const colGeo = new CylinderGeometry(0.035, 0.042, 0.28, 10);
  colGeo.rotateX(Math.PI / 2 - 0.32);
  colGeo.translate(0.34, beltY - 0.04, dashZ - 0.22);
  raw.push(makePart(colGeo, dashMat, 'dodge_steer_col', DeformGroup.Body));

  const wheelGeo = new TorusGeometry(0.155, 0.02, 8, 20);
  wheelGeo.rotateX(0.32);
  wheelGeo.translate(0.34, beltY - 0.01, dashZ - 0.34);
  raw.push(makePart(wheelGeo, dashMat, 'dodge_steer_wheel', DeformGroup.Body));
  addBox(0.26, 0.032, 0.025, 0.34, beltY - 0.01, dashZ - 0.34, trimMat, 0.32);

  // 4. Two front sport bucket seats (driver +X, passenger -X)
  for (const sx of [0.34, -0.34]) {
    addBox(0.46, 0.11, 0.46, sx, floorY + 0.15, frontSeatZ + 0.06, seatMat);
    addBox(0.44, 0.54, 0.11, sx, floorY + 0.44, frontSeatZ - 0.16, seatMat, -0.22);
    addBox(0.06, 0.48, 0.14, sx + 0.21, floorY + 0.42, frontSeatZ - 0.12, trimMat, -0.22);
    addBox(0.06, 0.48, 0.14, sx - 0.21, floorY + 0.42, frontSeatZ - 0.12, trimMat, -0.22);
    addBox(0.22, 0.18, 0.09, sx, floorY + 0.77, frontSeatZ - 0.23, seatMat, -0.14);
  }

  // 5. Rear bench seat
  addBox(cabinW - 0.12, 0.12, 0.42, 0, floorY + 0.16, rearSeatZ + 0.08, seatMat);
  addBox(cabinW - 0.14, 0.48, 0.12, 0, floorY + 0.42, rearSeatZ - 0.12, seatMat, -0.20);

  return consolidatePartsByGroupAndMaterial(raw, 1).map((p) => {
    p.mesh.castShadow = false;
    p.mesh.receiveShadow = true;
    return p.mesh;
  });
}



