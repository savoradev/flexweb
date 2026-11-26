import {
  AdditiveBlending, BoxGeometry, BufferAttribute, BufferGeometry, CanvasTexture, Color, CylinderGeometry,
  Group, InstancedMesh, Matrix4, MeshBasicMaterial, MeshPhysicalMaterial, NormalBlending,
  PlaneGeometry, Points, Quaternion, ShaderMaterial, SphereGeometry, Vector2, Vector3,
} from 'three';
import { CollisionWorld, GroundSample } from './SoftBodySolver';

export const FLUID_COOLANT = 0;
export const FLUID_OIL = 1;
export const FLUID_FUEL = 2;

const MAX_STREAMS = 3; // 0: coolant, 1: oil, 2: fuel
const MAX_DROPS = 96;
const MAX_PUDDLES = 250;
const MAX_SMOKE = 96;
const MAX_SPARKS = 64;

// Distinct physical fluid colors
const COLOR_COOLANT = new Color(0x16e874); // bright fluorescent green radiator coolant
const COLOR_OIL = new Color(0x0b0a09);     // dark viscous black/amber motor oil
const COLOR_FUEL = new Color(0xc4d8ec);    // pale iridescent highly reflective fuel
const COLOR_STEAM = new Color(0xe0ebf5);   // hot radiator steam/mist
const COLOR_SMOKE = new Color(0x282c34);   // dark engine smoke
const COLOR_VAPOR = new Color(0xd8e8f8);   // faint fuel vapor shimmer
const COLOR_IMPACT_DUST = new Color(0xb8bcc4); // high-G collision smoke/dust burst
const COLOR_SPARK_HOT = new Color(0xfff1a8);
const COLOR_SPARK_ORANGE = new Color(0xff8418);

interface PuddleTexturePack {
  map: CanvasTexture;
  alphaMap: CanvasTexture;
  normalMap: CanvasTexture;
}

let cachedPuddlePack: PuddleTexturePack | null = null;
let cachedSmokeTex: CanvasTexture | null = null;

/**
 * Generates high-resolution (256×256) procedural liquid puddle textures:
 *  - map: subtle depth darkening in center + glossy wet meniscus highlight near rim
 *  - alphaMap: organic multi-lobe irregular liquid blotch with soft capillary edge
 *  - normalMap: subtle raised liquid meniscus rim + gentle surface ripples for realistic reflections
 */
function getPuddleTexturePack(): PuddleTexturePack | null {
  if (cachedPuddlePack) return cachedPuddlePack;
  if (typeof document === 'undefined') return null;
  const size = 256;

  // Height field (0..1) representing organic liquid thickness
  const height = new Float32Array(size * size);
  const blobs = [
    [0.00, 0.00, 0.38, 1.00],
    [0.11, -0.06, 0.25, 0.82],
    [-0.12, 0.08, 0.27, 0.84],
    [0.06, 0.14, 0.22, 0.72],
    [-0.08, -0.13, 0.23, 0.74],
    [0.18, 0.05, 0.17, 0.58],
    [-0.17, -0.05, 0.18, 0.60],
    [0.02, -0.17, 0.15, 0.50],
    [-0.04, 0.18, 0.16, 0.52],
  ];

  for (let y = 0; y < size; y++) {
    const ny = y / size - 0.5;
    for (let x = 0; x < size; x++) {
      const nx = x / size - 0.5;
      let h = 0;
      for (let b = 0; b < blobs.length; b++) {
        const dx = nx - blobs[b][0];
        const dy = ny - blobs[b][1];
        const r = blobs[b][2];
        const d2 = (dx * dx + dy * dy) / (r * r);
        if (d2 < 1) {
          const w = (1 - d2) * (1 - d2);
          h += w * blobs[b][3];
        }
      }
      // Subtle capillary micro-ripple along the puddle surface
      const ripple = 0.03 * Math.sin(nx * 44 + ny * 26) * Math.cos(ny * 38 - nx * 22);
      height[y * size + x] = Math.max(0, Math.min(1, h * 0.78 + (h > 0.12 ? ripple : 0)));
    }
  }

  // 1. Alpha map + 2. Color depth map + 3. Normal map
  const alphaCanvas = document.createElement('canvas');
  alphaCanvas.width = size;
  alphaCanvas.height = size;
  const aCtx = alphaCanvas.getContext('2d');

  const mapCanvas = document.createElement('canvas');
  mapCanvas.width = size;
  mapCanvas.height = size;
  const mCtx = mapCanvas.getContext('2d');

  const normCanvas = document.createElement('canvas');
  normCanvas.width = size;
  normCanvas.height = size;
  const nCtx = normCanvas.getContext('2d');

  if (!aCtx || !mCtx || !nCtx) return null;

  const aImg = aCtx.createImageData(size, size);
  const mImg = mCtx.createImageData(size, size);
  const nImg = nCtx.createImageData(size, size);

  for (let y = 0; y < size; y++) {
    const ym = Math.max(0, y - 1);
    const yp = Math.min(size - 1, y + 1);
    for (let x = 0; x < size; x++) {
      const xm = Math.max(0, x - 1);
      const xp = Math.min(size - 1, x + 1);
      const idx = y * size + x;
      const p4 = idx * 4;
      const h = height[idx];

      // Smooth capillary threshold for crisp yet soft wet-edge meniscus
      const edge = Math.max(0, Math.min(1, (h - 0.14) / 0.22));
      const alpha = edge * edge * (3 - 2 * edge); // smoothstep
      const aByte = Math.round(alpha * 255);
      aImg.data[p4] = aByte;
      aImg.data[p4 + 1] = aByte;
      aImg.data[p4 + 2] = aByte;
      aImg.data[p4 + 3] = 255;

      // Liquid depth shading: slightly richer/darker in thicker center, brighter wet sheen at meniscus rim
      const meniscus = edge * (1 - edge) * 4.0; // peaks right at the rim
      const shade = Math.round((0.82 + 0.18 * (1 - Math.min(1, h)) + meniscus * 0.12) * 255);
      const sClamped = Math.max(0, Math.min(255, shade));
      mImg.data[p4] = sClamped;
      mImg.data[p4 + 1] = sClamped;
      mImg.data[p4 + 2] = sClamped;
      mImg.data[p4 + 3] = 255;

      // Tangent-space normal from height gradient (gives realistic curved liquid meniscus & ripples)
      const dhx = (height[y * size + xp] - height[y * size + xm]) * 3.8;
      const dhy = (height[yp * size + x] - height[ym * size + x]) * 3.8;
      const nz = 1.0;
      const invLen = 1 / Math.sqrt(dhx * dhx + dhy * dhy + nz * nz);
      nImg.data[p4] = Math.round((-dhx * invLen * 0.5 + 0.5) * 255);
      nImg.data[p4 + 1] = Math.round((-dhy * invLen * 0.5 + 0.5) * 255);
      nImg.data[p4 + 2] = Math.round((nz * invLen * 0.5 + 0.5) * 255);
      nImg.data[p4 + 3] = 255;
    }
  }

  aCtx.putImageData(aImg, 0, 0);
  mCtx.putImageData(mImg, 0, 0);
  nCtx.putImageData(nImg, 0, 0);

  const map = new CanvasTexture(mapCanvas);
  const alphaMap = new CanvasTexture(alphaCanvas);
  const normalMap = new CanvasTexture(normCanvas);
  map.needsUpdate = true;
  alphaMap.needsUpdate = true;
  normalMap.needsUpdate = true;

  cachedPuddlePack = { map, alphaMap, normalMap };
  return cachedPuddlePack;
}

/** Generates a photorealistic wispy, soft volumetric smoke/steam/vapor sprite texture. */
function getSmokeTexture(): CanvasTexture | null {
  if (cachedSmokeTex) return cachedSmokeTex;
  if (typeof document === 'undefined') return null;
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  ctx.clearRect(0, 0, size, size);
  const puffs = [
    [0.50, 0.50, 0.44, 0.28],
    [0.40, 0.44, 0.34, 0.22],
    [0.61, 0.47, 0.32, 0.20],
    [0.47, 0.60, 0.31, 0.19],
    [0.55, 0.38, 0.29, 0.18],
    [0.36, 0.56, 0.26, 0.14],
    [0.64, 0.57, 0.25, 0.14],
    [0.48, 0.48, 0.22, 0.24],
  ];
  for (const [px, py, rad, alpha] of puffs) {
    const x = px * size;
    const y = py * size;
    const r = rad * size;
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, `rgba(255,255,255,${alpha})`);
    g.addColorStop(0.45, `rgba(255,255,255,${alpha * 0.55})`);
    g.addColorStop(0.80, `rgba(255,255,255,${alpha * 0.14})`);
    g.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
  const tex = new CanvasTexture(canvas);
  tex.needsUpdate = true;
  cachedSmokeTex = tex;
  return tex;
}

/** Pre-allocated zero-GC input struct passed from SoftBodyVehicle each frame. */
export interface FluidEmitFrame {
  radX: number; radY: number; radZ: number;
  oilX: number; oilY: number; oilZ: number;
  fuelX: number; fuelY: number; fuelZ: number;
  carVx: number; carVy: number; carVz: number;
  coolantSeverity: number;
  oilSeverity: number;
  fuelSeverity: number;
  engineDamage: number;
}

/**
 * FluidLeakSystem
 * ---------------
 * Physically driven, Zero-GC 3D Vehicle Fluid Leak System:
 *  - Slow realistic drop-by-drop dripping -> thin stream on catastrophic crush
 *  - Droplets landing in the same area merge & accumulate progressively into a growing puddle!
 *  - High-res 256×256 procedural liquid depth + meniscus normalMap puddles (250-slot ring buffer)
 *  - Soft volumetric billboard radiator steam, engine smoke & fuel vapor
 *  - Instant < 2ms reset reusing pre-allocated TypedArrays
 */
export class FluidLeakSystem {
  readonly group = new Group();
  private readonly streamMesh: InstancedMesh;
  private readonly dropMesh: InstancedMesh;
  private readonly puddleMesh: InstancedMesh;
  private readonly sparkMesh: InstancedMesh;
  private readonly smokePoints: Points;
  private readonly smokeGeo: BufferGeometry;
  private readonly smokeMat: ShaderMaterial;

  private readonly dripTimer = new Float32Array(3);
  private readonly streamActive = new Uint8Array(MAX_STREAMS);
  private readonly streamPuddleTimer = new Float32Array(MAX_STREAMS);

  // Falling 3D droplets
  private readonly dX = new Float32Array(MAX_DROPS);
  private readonly dY = new Float32Array(MAX_DROPS);
  private readonly dZ = new Float32Array(MAX_DROPS);
  private readonly dVx = new Float32Array(MAX_DROPS);
  private readonly dVy = new Float32Array(MAX_DROPS);
  private readonly dVz = new Float32Array(MAX_DROPS);
  private readonly dType = new Uint8Array(MAX_DROPS);
  private readonly dSev = new Float32Array(MAX_DROPS);
  private readonly dActive = new Uint8Array(MAX_DROPS);
  private dropCursor = 0;
  private activeDropCount = 0;

  // Persistent ground puddles & trails (250-slot ring buffer with progressive drop accumulation)
  private readonly pX = new Float32Array(MAX_PUDDLES);
  private readonly pY = new Float32Array(MAX_PUDDLES);
  private readonly pZ = new Float32Array(MAX_PUDDLES);
  private readonly pRot = new Float32Array(MAX_PUDDLES);
  private readonly pType = new Uint8Array(MAX_PUDDLES);
  private readonly pVolume = new Float32Array(MAX_PUDDLES);
  private readonly pCurScale = new Float32Array(MAX_PUDDLES);
  private readonly pTargetScale = new Float32Array(MAX_PUDDLES);
  private readonly pSpreadSpeed = new Float32Array(MAX_PUDDLES);
  private readonly pStretch = new Float32Array(MAX_PUDDLES);
  private readonly pActive = new Uint8Array(MAX_PUDDLES);
  private readonly pHidden = new Uint8Array(MAX_PUDDLES);
  private puddleCursor = 0;
  private puddleDirty = false;
  private hasActivePuddles = false;

  // High-G metallic collision sparks (64-slot zero-GC pool)
  private readonly spX = new Float32Array(MAX_SPARKS);
  private readonly spY = new Float32Array(MAX_SPARKS);
  private readonly spZ = new Float32Array(MAX_SPARKS);
  private readonly spVx = new Float32Array(MAX_SPARKS);
  private readonly spVy = new Float32Array(MAX_SPARKS);
  private readonly spVz = new Float32Array(MAX_SPARKS);
  private readonly spLife = new Float32Array(MAX_SPARKS);
  private readonly spMaxLife = new Float32Array(MAX_SPARKS);
  private readonly spActive = new Uint8Array(MAX_SPARKS);
  private sparkCursor = 0;
  private activeSparkCount = 0;

  // Soft volumetric billboard smoke / steam / vapor particles
  private readonly sPos = new Float32Array(MAX_SMOKE * 3);
  private readonly sCol = new Float32Array(MAX_SMOKE * 3);
  private readonly sSize = new Float32Array(MAX_SMOKE);
  private readonly sAlpha = new Float32Array(MAX_SMOKE);
  private readonly sRot = new Float32Array(MAX_SMOKE);
  private readonly sVx = new Float32Array(MAX_SMOKE);
  private readonly sVy = new Float32Array(MAX_SMOKE);
  private readonly sVz = new Float32Array(MAX_SMOKE);
  private readonly sSpin = new Float32Array(MAX_SMOKE);
  private readonly sBaseSize = new Float32Array(MAX_SMOKE);
  private readonly sMaxAlpha = new Float32Array(MAX_SMOKE);
  private readonly sLife = new Float32Array(MAX_SMOKE);
  private readonly sMaxLife = new Float32Array(MAX_SMOKE);
  private readonly sActive = new Uint8Array(MAX_SMOKE);
  private smokeCursor = 0;
  private smokeTimer = 0;
  private vaporTimer = 0;
  private activeSmokeCount = 0;

  // Pre-allocated scratch math (zero GC per frame)
  private readonly mat4 = new Matrix4();
  private readonly pos = new Vector3();
  private readonly quat = new Quaternion();
  private readonly scl = new Vector3();
  private readonly upAxis = new Vector3(0, 1, 0);
  private readonly zAxis = new Vector3(0, 0, 1);
  private readonly dirVec = new Vector3();
  private readonly ground: GroundSample = { h: 0, nx: 0, ny: 1, nz: 0, mu: 1 };

  constructor() {
    this.group.name = 'FluidLeakSystem';

    // 1. Continuous thin falling liquid streams (catastrophic crush only)
    const stGeo = new CylinderGeometry(0.007, 0.012, 1.0, 6, 1, false);
    const stMat = new MeshPhysicalMaterial({
      color: 0xffffff,
      roughness: 0.06,
      metalness: 0.12,
      clearcoat: 1.0,
      clearcoatRoughness: 0.03,
      transparent: true,
      opacity: 0.82,
      depthWrite: false,
    });
    this.streamMesh = new InstancedMesh(stGeo, stMat, MAX_STREAMS);
    this.streamMesh.frustumCulled = false;
    this.streamMesh.instanceMatrix.setUsage(35048);
    this.streamMesh.setColorAt(FLUID_COOLANT, COLOR_COOLANT);
    this.streamMesh.setColorAt(FLUID_OIL, COLOR_OIL);
    this.streamMesh.setColorAt(FLUID_FUEL, COLOR_FUEL);
    if (this.streamMesh.instanceColor) this.streamMesh.instanceColor.needsUpdate = true;
    this.streamMesh.visible = false;

    // 2. Falling liquid droplets
    const dGeo = new SphereGeometry(0.014, 6, 5);
    dGeo.scale(0.85, 2.0, 0.85);
    const dMat = new MeshPhysicalMaterial({
      color: 0xffffff,
      roughness: 0.05,
      metalness: 0.10,
      clearcoat: 1.0,
      transparent: true,
      opacity: 0.92,
    });
    this.dropMesh = new InstancedMesh(dGeo, dMat, MAX_DROPS);
    this.dropMesh.frustumCulled = false;
    this.dropMesh.instanceMatrix.setUsage(35048);
    this.dropMesh.visible = false;

    // 3. Persistent glossy organic ground puddles with meniscus normalMap & liquid depth
    const pGeo = new PlaneGeometry(1.0, 1.0);
    pGeo.rotateX(-Math.PI / 2);
    const pack = getPuddleTexturePack();
    const pMat = new MeshPhysicalMaterial({
      color: 0xffffff,
      map: pack?.map ?? undefined,
      alphaMap: pack?.alphaMap ?? undefined,
      normalMap: pack?.normalMap ?? undefined,
      normalScale: new Vector2(0.45, 0.45),
      roughness: 0.07,
      metalness: 0.12,
      clearcoat: 1.0,
      clearcoatRoughness: 0.03,
      reflectivity: 1.0,
      transparent: true,
      opacity: 0.90,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
    });
    this.puddleMesh = new InstancedMesh(pGeo, pMat, MAX_PUDDLES);
    this.puddleMesh.frustumCulled = false;
    this.puddleMesh.instanceMatrix.setUsage(35048);
    this.puddleMesh.visible = false;

    // 4. Photorealistic wispy billboard radiator steam, engine smoke & fuel vapor
    this.smokeGeo = new BufferGeometry();
    const posAttr = new BufferAttribute(this.sPos, 3);
    const colAttr = new BufferAttribute(this.sCol, 3);
    const sizeAttr = new BufferAttribute(this.sSize, 1);
    const alphaAttr = new BufferAttribute(this.sAlpha, 1);
    const rotAttr = new BufferAttribute(this.sRot, 1);
    posAttr.setUsage(35048);
    colAttr.setUsage(35048);
    sizeAttr.setUsage(35048);
    alphaAttr.setUsage(35048);
    rotAttr.setUsage(35048);
    this.smokeGeo.setAttribute('position', posAttr);
    this.smokeGeo.setAttribute('aColor', colAttr);
    this.smokeGeo.setAttribute('aSize', sizeAttr);
    this.smokeGeo.setAttribute('aAlpha', alphaAttr);
    this.smokeGeo.setAttribute('aRot', rotAttr);

    const smokeTex = getSmokeTexture();
    this.smokeMat = new ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: NormalBlending,
      uniforms: {
        uMap: { value: smokeTex },
        uHasMap: { value: smokeTex ? 1.0 : 0.0 },
        uScale: { value: 360.0 },
      },
      vertexShader: `
        attribute vec3 aColor;
        attribute float aSize;
        attribute float aAlpha;
        attribute float aRot;
        varying vec3 vColor;
        varying float vAlpha;
        varying float vRot;
        uniform float uScale;
        void main() {
          vColor = aColor;
          vAlpha = aAlpha;
          vRot = aRot;
          vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = clamp(aSize * (uScale / max(1.0, -mvPos.z)), 0.0, 220.0);
          gl_Position = projectionMatrix * mvPos;
        }
      `,
      fragmentShader: `
        uniform sampler2D uMap;
        uniform float uHasMap;
        varying vec3 vColor;
        varying float vAlpha;
        varying float vRot;
        void main() {
          if (vAlpha <= 0.004) discard;
          vec2 uv = gl_PointCoord - 0.5;
          float c = cos(vRot);
          float s = sin(vRot);
          vec2 ruv = vec2(uv.x * c - uv.y * s, uv.x * s + uv.y * c) + 0.5;
          float mask = 0.0;
          if (uHasMap > 0.5) {
            mask = texture2D(uMap, ruv).a;
          } else {
            float d = length(uv) * 2.0;
            mask = smoothstep(1.0, 0.0, d);
            mask *= mask;
          }
          float a = mask * vAlpha;
          if (a <= 0.004) discard;
          gl_FragColor = vec4(vColor, a);
        }
      `,
    });
    this.smokePoints = new Points(this.smokeGeo, this.smokeMat);
    this.smokePoints.frustumCulled = false;
    this.smokePoints.visible = false;

    // 5. High-G metallic collision sparks (additive stretched glow streaks, zero-GC)
    const spGeo = new BoxGeometry(0.015, 0.015, 0.24);
    const spMat = new MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.95,
      blending: AdditiveBlending,
      depthWrite: false,
    });
    this.sparkMesh = new InstancedMesh(spGeo, spMat, MAX_SPARKS);
    this.sparkMesh.frustumCulled = false;
    this.sparkMesh.instanceMatrix.setUsage(35048);
    for (let i = 0; i < MAX_SPARKS; i++) {
      this.sparkMesh.setColorAt(i, (i & 1) === 0 ? COLOR_SPARK_HOT : COLOR_SPARK_ORANGE);
    }
    if (this.sparkMesh.instanceColor) this.sparkMesh.instanceColor.needsUpdate = true;
    this.sparkMesh.visible = false;

    this.group.add(this.streamMesh, this.dropMesh, this.puddleMesh, this.smokePoints, this.sparkMesh);
    this.reset();
  }

  private fluidColor(type: number): Color {
    if (type === FLUID_COOLANT) return COLOR_COOLANT;
    if (type === FLUID_OIL) return COLOR_OIL;
    return COLOR_FUEL;
  }

  private spawnDroplet(
    type: number,
    ox: number,
    oy: number,
    oz: number,
    carVx: number,
    carVy: number,
    carVz: number,
    severity: number,
  ): void {
    const idx = this.dropCursor;
    this.dropCursor = (this.dropCursor + 1) % MAX_DROPS;

    const jx = ((idx * 37) % 13 - 6) * 0.016;
    const jz = ((idx * 53) % 13 - 6) * 0.016;
    this.dX[idx] = ox + jx;
    this.dY[idx] = oy;
    this.dZ[idx] = oz + jz;
    const spray = severity * 0.30;
    this.dVx[idx] = carVx * 0.72 + jx * 4.0 * spray;
    this.dVy[idx] = carVy * 0.3 - (0.32 + severity * 0.75);
    this.dVz[idx] = carVz * 0.72 + jz * 4.0 * spray;
    this.dType[idx] = type;
    this.dSev[idx] = severity;
    if (!this.dActive[idx]) this.activeDropCount++;
    this.dActive[idx] = 1;

    this.dropMesh.setColorAt(idx, this.fluidColor(type));
    if (this.dropMesh.instanceColor) this.dropMesh.instanceColor.needsUpdate = true;
    this.dropMesh.visible = true;
  }

  /**
   * Accumulates drops landing in the same spot into a progressively growing puddle,
   * or spawns a new elongated wet trail decal when the vehicle is driving/sliding.
   */
  private depositOnGround(
    wx: number,
    groundY: number,
    wz: number,
    type: number,
    severity: number,
    carVx: number,
    carVz: number,
  ): void {
    const speed = Math.hypot(carVx, carVz);
    const isMoving = speed > 1.2;

    // Merge into an existing nearby puddle of the same fluid when stationary/slow so drops build a real growing puddle!
    if (!isMoving) {
      const mergeRadius2 = 0.36 * 0.36;
      for (let i = 0; i < MAX_PUDDLES; i++) {
        if (!this.pActive[i] || this.pType[i] !== type) continue;
        const dx = this.pX[i] - wx;
        const dz = this.pZ[i] - wz;
        if (dx * dx + dz * dz <= mergeRadius2) {
          this.pVolume[i] = Math.min(36, this.pVolume[i] + 1);
          const maxCap = type === FLUID_FUEL ? 1.15 : type === FLUID_COOLANT ? 0.96 : 0.88;
          const growth = type === FLUID_FUEL ? 0.165 : type === FLUID_COOLANT ? 0.145 : 0.125;
          // Square-root area growth: realistic 2D puddle expansion proportional to accumulated drop volume
          this.pTargetScale[i] = Math.min(maxCap, 0.08 + growth * Math.sqrt(this.pVolume[i]));
          this.puddleDirty = true;
          return;
        }
      }
    }

    const idx = this.puddleCursor;
    this.puddleCursor = (this.puddleCursor + 1) % MAX_PUDDLES;

    this.pX[idx] = wx;
    this.pY[idx] = groundY + 0.004 + (idx % 6) * 0.0004;
    this.pZ[idx] = wz;
    this.pRot[idx] = isMoving
      ? Math.atan2(carVx, carVz)
      : ((idx * 1.6180339) % 1) * Math.PI * 2;
    this.pType[idx] = type;
    this.pVolume[idx] = 1;
    this.pCurScale[idx] = 0.038;

    if (type === FLUID_OIL) {
      this.pSpreadSpeed[idx] = 0.22; // thick slow viscous oil creep
      this.pTargetScale[idx] = isMoving ? 0.11 : 0.13 + severity * 0.06;
    } else if (type === FLUID_COOLANT) {
      this.pSpreadSpeed[idx] = 0.34; // medium coolant spread
      this.pTargetScale[idx] = isMoving ? 0.12 : 0.14 + severity * 0.07;
    } else {
      this.pSpreadSpeed[idx] = 0.56; // thin fast-spreading fuel
      this.pTargetScale[idx] = isMoving ? 0.14 : 0.16 + severity * 0.08;
    }

    this.pStretch[idx] = isMoving
      ? Math.min(3.4, 1.35 + speed * 0.14)
      : 0.90 + ((idx * 7) % 5) * 0.05;
    this.pActive[idx] = 1;
    this.pHidden[idx] = 0;
    this.hasActivePuddles = true;

    this.puddleMesh.setColorAt(idx, this.fluidColor(type));
    if (this.puddleMesh.instanceColor) this.puddleMesh.instanceColor.needsUpdate = true;
    this.puddleMesh.visible = true;
    this.puddleDirty = true;
  }

  private emitWispyParticle(
    wx: number,
    wy: number,
    wz: number,
    col: Color,
    maxAlpha: number,
    baseSize: number,
    riseSpeed: number,
    maxLife: number,
  ): void {
    const idx = this.smokeCursor;
    this.smokeCursor = (this.smokeCursor + 1) % MAX_SMOKE;
    const jx = ((idx * 29) % 17 - 8) * 0.036;
    const jz = ((idx * 41) % 17 - 8) * 0.036;

    this.sPos[idx * 3] = wx + jx;
    this.sPos[idx * 3 + 1] = wy;
    this.sPos[idx * 3 + 2] = wz + jz;
    this.sVx[idx] = jx * 0.65 + 0.16;
    this.sVy[idx] = riseSpeed + ((idx * 13) % 7) * 0.09;
    this.sVz[idx] = jz * 0.65;
    this.sRot[idx] = (idx * 2.39996) % 6.283;
    this.sSpin[idx] = ((idx & 1) === 0 ? 1 : -1) * (0.26 + ((idx * 5) % 4) * 0.08);
    this.sLife[idx] = 0;
    this.sMaxLife[idx] = maxLife;
    this.sBaseSize[idx] = baseSize;
    this.sCol[idx * 3] = col.r;
    this.sCol[idx * 3 + 1] = col.g;
    this.sCol[idx * 3 + 2] = col.b;
    this.sMaxAlpha[idx] = maxAlpha;
    this.sSize[idx] = baseSize;
    this.sAlpha[idx] = 0.01;

    if (!this.sActive[idx]) this.activeSmokeCount++;
    this.sActive[idx] = 1;
    this.smokePoints.visible = true;
  }

  /**
   * Advances streams, falling droplets, accumulating easeOut puddles/trails, and wispy steam/smoke/vapor.
   * Strictly zero heap allocations per frame.
   */
  update(dt: number, world: CollisionWorld | null, emit: FluidEmitFrame): void {
    const h = Math.min(dt, 0.05);
    const carSpeed = Math.hypot(emit.carVx, emit.carVz);

    let anyStreamVisible = false;
    for (let f = 0; f < 3; f++) {
      const sev = f === FLUID_COOLANT ? emit.coolantSeverity : f === FLUID_OIL ? emit.oilSeverity : emit.fuelSeverity;
      const ex = f === FLUID_COOLANT ? emit.radX : f === FLUID_OIL ? emit.oilX : emit.fuelX;
      const ey = f === FLUID_COOLANT ? emit.radY : f === FLUID_OIL ? emit.oilY : emit.fuelY;
      const ez = f === FLUID_COOLANT ? emit.radZ : f === FLUID_OIL ? emit.oilZ : emit.fuelZ;

      if (sev > 0.01 && world) {
        // Slow realistic drop-by-drop cadence: 0.4 drops/sec on minor dents -> ~2.2 drops/sec on heavy crush
        const dropsPerSec = 0.40 + sev * sev * 1.80;
        this.dripTimer[f] -= h;
        if (this.dripTimer[f] <= 0) {
          this.dripTimer[f] = (1 / dropsPerSec) * (0.85 + ((this.dropCursor * 11) % 7) * 0.05);
          this.spawnDroplet(f, ex, ey, ez, emit.carVx, emit.carVy, emit.carVz, sev);
        }

        // Thin continuous stream only on catastrophic crush (> 0.82)
        if (sev > 0.82) {
          world.groundAt(ex, ez, this.ground);
          const streamLen = ey - this.ground.h;
          if (streamLen > 0.06 && streamLen < 2.5) {
            const streamRatio = (sev - 0.82) / 0.18;
            const thickness = 0.45 + streamRatio * 0.65;
            const driftX = -Math.max(-0.22, Math.min(0.22, emit.carVx * 0.025));
            const driftZ = -Math.max(-0.22, Math.min(0.22, emit.carVz * 0.025));
            this.dirVec.set(driftX, -streamLen, driftZ);
            const actualLen = this.dirVec.length();
            this.dirVec.multiplyScalar(1 / Math.max(1e-4, actualLen));
            this.quat.setFromUnitVectors(this.upAxis, this.dirVec);
            this.pos.set(ex + driftX * 0.5, ey - streamLen * 0.5, ez + driftZ * 0.5);
            this.scl.set(thickness, actualLen, thickness);
            this.mat4.compose(this.pos, this.quat, this.scl);
            this.streamMesh.setMatrixAt(f, this.mat4);
            this.streamActive[f] = 1;
            anyStreamVisible = true;

            this.streamPuddleTimer[f] -= h;
            if (this.streamPuddleTimer[f] <= 0) {
              this.streamPuddleTimer[f] = carSpeed > 1.5 ? 0.24 : 1.15;
              this.depositOnGround(ex + driftX, this.ground.h, ez + driftZ, f, sev, emit.carVx, emit.carVz);
            }
          } else {
            this.streamActive[f] = 0;
          }
        } else {
          this.streamActive[f] = 0;
        }
      } else {
        this.streamActive[f] = 0;
      }

      if (!this.streamActive[f]) {
        this.quat.identity();
        this.pos.set(0, -500, 0);
        this.scl.set(0, 0, 0);
        this.mat4.compose(this.pos, this.quat, this.scl);
        this.streamMesh.setMatrixAt(f, this.mat4);
      }
    }
    this.streamMesh.visible = anyStreamVisible;
    if (anyStreamVisible) this.streamMesh.instanceMatrix.needsUpdate = true;

    // 2. Integrate falling droplets under gravity -> deposit/merge into ground puddle on impact
    if (this.activeDropCount > 0 && world) {
      let aliveDrops = 0;
      for (let i = 0; i < MAX_DROPS; i++) {
        if (!this.dActive[i]) continue;
        this.dVy[i] -= 9.81 * h;
        this.dX[i] += this.dVx[i] * h;
        this.dY[i] += this.dVy[i] * h;
        this.dZ[i] += this.dVz[i] * h;
        world.groundAt(this.dX[i], this.dZ[i], this.ground);
        if (this.dY[i] <= this.ground.h + 0.015) {
          this.dActive[i] = 0;
          this.quat.identity();
          this.pos.set(0, -500, 0);
          this.scl.set(0, 0, 0);
          this.mat4.compose(this.pos, this.quat, this.scl);
          this.dropMesh.setMatrixAt(i, this.mat4);
          this.depositOnGround(this.dX[i], this.ground.h, this.dZ[i], this.dType[i], this.dSev[i], emit.carVx, emit.carVz);
          continue;
        }
        aliveDrops++;
        const speedMag = Math.hypot(this.dVx[i], this.dVy[i], this.dVz[i]);
        if (speedMag > 0.2) {
          this.dirVec.set(this.dVx[i] / speedMag, this.dVy[i] / speedMag, this.dVz[i] / speedMag);
          this.quat.setFromUnitVectors(this.upAxis, this.dirVec);
        } else {
          this.quat.identity();
        }
        const stretchY = Math.min(2.4, 0.9 + speedMag * 0.22);
        this.pos.set(this.dX[i], this.dY[i], this.dZ[i]);
        this.scl.set(1, stretchY, 1);
        this.mat4.compose(this.pos, this.quat, this.scl);
        this.dropMesh.setMatrixAt(i, this.mat4);
      }
      this.activeDropCount = aliveDrops;
      this.dropMesh.instanceMatrix.needsUpdate = true;
      if (aliveDrops === 0) this.dropMesh.visible = false;
    }

    // 3. Smooth exponential ease-out viscous spreading & 80m spatial culling on ground puddles
    if (this.hasActivePuddles) {
      const PUDDLE_SLEEP_SQ = 80 * 80;
      let stillGrowing = false;
      let matrixDirty = false;
      for (let i = 0; i < MAX_PUDDLES; i++) {
        if (!this.pActive[i]) continue;
        const dx = this.pX[i] - emit.oilX;
        const dz = this.pZ[i] - emit.oilZ;
        if (dx * dx + dz * dz > PUDDLE_SLEEP_SQ) {
          if (!this.pHidden[i]) {
            this.pHidden[i] = 1;
            this.pos.set(0, -500, 0);
            this.quat.identity();
            this.scl.set(0, 0, 0);
            this.mat4.compose(this.pos, this.quat, this.scl);
            this.puddleMesh.setMatrixAt(i, this.mat4);
            matrixDirty = true;
          }
          continue;
        }

        const wokeUp = this.pHidden[i] === 1;
        if (wokeUp) this.pHidden[i] = 0;

        if (!this.puddleDirty && !wokeUp) continue;

        const cur = this.pCurScale[i];
        const tgt = this.pTargetScale[i];
        const diff = tgt - cur;
        if (diff > 0.0015) {
          // Viscous ease-out approach toward accumulated target diameter
          const step = diff * Math.min(1, this.pSpreadSpeed[i] * h);
          this.pCurScale[i] = cur + Math.max(0.0004, step);
          stillGrowing = true;
        } else {
          this.pCurScale[i] = tgt;
        }
        const sc = this.pCurScale[i];
        this.pos.set(this.pX[i], this.pY[i], this.pZ[i]);
        this.quat.setFromAxisAngle(this.upAxis, this.pRot[i]);
        this.scl.set(sc, 1, sc * this.pStretch[i]);
        this.mat4.compose(this.pos, this.quat, this.scl);
        this.puddleMesh.setMatrixAt(i, this.mat4);
        matrixDirty = true;
      }
      if (matrixDirty) this.puddleMesh.instanceMatrix.needsUpdate = true;
      this.puddleDirty = stillGrowing;
    }

    // 4. Soft volumetric wispy radiator steam, dark engine smoke, and subtle fuel vapor
    if (emit.coolantSeverity > 0.08 || emit.engineDamage > 0.42) {
      this.smokeTimer -= h;
      if (this.smokeTimer <= 0) {
        this.smokeTimer = emit.engineDamage > 0.65 ? 0.075 : 0.115;
        const isDarkSmoke = emit.engineDamage > 0.55 && (emit.coolantSeverity < 0.1 || (this.smokeCursor % 3) !== 0);
        if (isDarkSmoke) {
          this.emitWispyParticle(emit.oilX, emit.oilY + 0.34, emit.oilZ, COLOR_SMOKE, 0.34, 0.58, 0.90, 2.1);
        } else {
          this.emitWispyParticle(
            emit.radX,
            emit.radY + 0.32,
            emit.radZ,
            COLOR_STEAM,
            0.25 * Math.min(1, 0.45 + emit.coolantSeverity * 0.75),
            0.52,
            0.82,
            1.85,
          );
        }
      }
    }

    if (emit.fuelSeverity > 0.25) {
      this.vaporTimer -= h;
      if (this.vaporTimer <= 0) {
        this.vaporTimer = 0.24 - emit.fuelSeverity * 0.10;
        this.emitWispyParticle(emit.fuelX, emit.fuelY + 0.14, emit.fuelZ, COLOR_VAPOR, 0.11, 0.42, 0.52, 1.35);
      }
    }

    if (this.activeSmokeCount > 0) {
      let alive = 0;
      for (let i = 0; i < MAX_SMOKE; i++) {
        if (!this.sActive[i]) continue;
        this.sLife[i] += h;
        const t = this.sLife[i] / this.sMaxLife[i];
        if (t >= 1) {
          this.sActive[i] = 0;
          this.sAlpha[i] = 0;
          this.sSize[i] = 0;
          continue;
        }
        alive++;
        const drag = 1 - 0.55 * h;
        this.sVx[i] *= drag;
        this.sVz[i] *= drag;
        this.sVy[i] = Math.max(0.26, this.sVy[i] * (1 - 0.35 * h));
        this.sPos[i * 3] += this.sVx[i] * h;
        this.sPos[i * 3 + 1] += this.sVy[i] * h;
        this.sPos[i * 3 + 2] += this.sVz[i] * h;
        this.sRot[i] += this.sSpin[i] * h;

        const fadeIn = Math.min(1, t / 0.18);
        const fadeOut = Math.max(0, 1 - Math.max(0, (t - 0.22) / 0.78));
        this.sAlpha[i] = this.sMaxAlpha[i] * fadeIn * fadeOut * fadeOut;
        this.sSize[i] = this.sBaseSize[i] * (0.75 + t * 2.45);
      }
      this.activeSmokeCount = alive;
      (this.smokeGeo.getAttribute('position') as BufferAttribute).needsUpdate = true;
      (this.smokeGeo.getAttribute('aColor') as BufferAttribute).needsUpdate = true;
      (this.smokeGeo.getAttribute('aSize') as BufferAttribute).needsUpdate = true;
      (this.smokeGeo.getAttribute('aAlpha') as BufferAttribute).needsUpdate = true;
      (this.smokeGeo.getAttribute('aRot') as BufferAttribute).needsUpdate = true;
      if (alive === 0) this.smokePoints.visible = false;
    }

    // 5. High-G metallic collision sparks (gravity + ground bounce + velocity-aligned streaks)
    if (this.activeSparkCount > 0) {
      let aliveSparks = 0;
      for (let i = 0; i < MAX_SPARKS; i++) {
        if (!this.spActive[i]) continue;
        this.spLife[i] += h;
        const t = this.spLife[i] / this.spMaxLife[i];
        if (t >= 1) {
          this.spActive[i] = 0;
          this.pos.set(0, -500, 0);
          this.scl.set(0, 0, 0);
          this.quat.identity();
          this.mat4.compose(this.pos, this.quat, this.scl);
          this.sparkMesh.setMatrixAt(i, this.mat4);
          continue;
        }
        aliveSparks++;
        this.spVy[i] -= 13.5 * h;
        const drag = 1 - 1.4 * h;
        this.spVx[i] *= drag;
        this.spVz[i] *= drag;
        this.spX[i] += this.spVx[i] * h;
        this.spY[i] += this.spVy[i] * h;
        this.spZ[i] += this.spVz[i] * h;

        if (world) {
          world.groundAt(this.spX[i], this.spZ[i], this.ground);
        } else {
          this.ground.h = 0;
        }
        if (this.spY[i] <= this.ground.h + 0.02) {
          this.spY[i] = this.ground.h + 0.02;
          this.spVy[i] = Math.abs(this.spVy[i]) * 0.42;
          this.spVx[i] *= 0.72;
          this.spVz[i] *= 0.72;
        }

        const vx = this.spVx[i], vy = this.spVy[i], vz = this.spVz[i];
        const spd = Math.hypot(vx, vy, vz);
        if (spd > 0.05) {
          this.dirVec.set(vx / spd, vy / spd, vz / spd);
          this.quat.setFromUnitVectors(this.zAxis, this.dirVec);
        } else {
          this.quat.identity();
        }
        const fade = 1 - t;
        const streakLen = Math.max(0.25, Math.min(2.4, spd * 0.14)) * fade;
        this.pos.set(this.spX[i], this.spY[i], this.spZ[i]);
        this.scl.set(fade, fade, streakLen);
        this.mat4.compose(this.pos, this.quat, this.scl);
        this.sparkMesh.setMatrixAt(i, this.mat4);
      }
      this.activeSparkCount = aliveSparks;
      this.sparkMesh.instanceMatrix.needsUpdate = true;
      this.sparkMesh.visible = aliveSparks > 0;
    }
  }

  /**
   * Emits a high-G metallic impact burst of glowing sparks + collision dust/smoke puffs (zero GC).
   */
  emitImpactBurst(wx: number, wy: number, wz: number, intensity: number): void {
    const k = Math.max(0.25, Math.min(1.5, intensity));
    const sparkCount = Math.min(26, Math.round(10 + k * 12));
    for (let s = 0; s < sparkCount; s++) {
      const idx = this.sparkCursor;
      this.sparkCursor = (this.sparkCursor + 1) % MAX_SPARKS;
      const ang = Math.random() * Math.PI * 2;
      const spd = (3.5 + Math.random() * 9.5) * k;
      this.spX[idx] = wx + (Math.random() - 0.5) * 0.25;
      this.spY[idx] = wy + 0.08 + Math.random() * 0.22;
      this.spZ[idx] = wz + (Math.random() - 0.5) * 0.25;
      this.spVx[idx] = Math.cos(ang) * spd;
      this.spVy[idx] = 1.8 + Math.random() * 5.8 * k;
      this.spVz[idx] = Math.sin(ang) * spd;
      this.spLife[idx] = 0;
      this.spMaxLife[idx] = 0.26 + Math.random() * 0.34;
      this.spActive[idx] = 1;
    }
    this.activeSparkCount = Math.min(MAX_SPARKS, this.activeSparkCount + sparkCount);
    this.sparkMesh.visible = true;

    const smokePuffs = Math.min(6, Math.round(2 + k * 3));
    for (let p = 0; p < smokePuffs; p++) {
      this.emitWispyParticle(
        wx + (Math.random() - 0.5) * 0.55,
        wy + 0.15 + Math.random() * 0.35,
        wz + (Math.random() - 0.5) * 0.55,
        COLOR_IMPACT_DUST,
        0.24 * Math.min(1, k),
        0.55 + Math.random() * 0.4,
        0.72 + Math.random() * 0.45,
        1.15 + Math.random() * 0.55,
      );
    }
  }

  /** Instant < 2ms reset reusing all pre-allocated buffers (zero heap allocations). */
  reset(): void {
    this.dripTimer.fill(0);
    this.streamActive.fill(0);
    this.streamPuddleTimer.fill(0);
    this.dActive.fill(0);
    this.pActive.fill(0);
    this.pHidden.fill(0);
    this.pVolume.fill(0);
    this.pCurScale.fill(0);
    this.pTargetScale.fill(0);
    this.spActive.fill(0);
    this.sActive.fill(0);
    this.sAlpha.fill(0);
    this.sSize.fill(0);
    this.dropCursor = 0;
    this.activeDropCount = 0;
    this.puddleCursor = 0;
    this.sparkCursor = 0;
    this.activeSparkCount = 0;
    this.smokeCursor = 0;
    this.smokeTimer = 0;
    this.vaporTimer = 0;
    this.puddleDirty = false;
    this.hasActivePuddles = false;
    this.activeSmokeCount = 0;

    this.quat.identity();
    this.pos.set(0, -500, 0);
    this.scl.set(0, 0, 0);
    this.mat4.compose(this.pos, this.quat, this.scl);
    for (let i = 0; i < MAX_STREAMS; i++) this.streamMesh.setMatrixAt(i, this.mat4);
    for (let i = 0; i < MAX_DROPS; i++) this.dropMesh.setMatrixAt(i, this.mat4);
    for (let i = 0; i < MAX_PUDDLES; i++) this.puddleMesh.setMatrixAt(i, this.mat4);
    for (let i = 0; i < MAX_SPARKS; i++) this.sparkMesh.setMatrixAt(i, this.mat4);
    this.streamMesh.instanceMatrix.needsUpdate = true;
    this.dropMesh.instanceMatrix.needsUpdate = true;
    this.puddleMesh.instanceMatrix.needsUpdate = true;
    this.sparkMesh.instanceMatrix.needsUpdate = true;
    this.streamMesh.visible = false;
    this.dropMesh.visible = false;
    this.puddleMesh.visible = false;
    this.sparkMesh.visible = false;

    (this.smokeGeo.getAttribute('aAlpha') as BufferAttribute).needsUpdate = true;
    (this.smokeGeo.getAttribute('aSize') as BufferAttribute).needsUpdate = true;
    this.smokePoints.visible = false;
  }

  dispose(): void {
    this.streamMesh.geometry.dispose();
    (this.streamMesh.material as MeshPhysicalMaterial).dispose();
    this.dropMesh.geometry.dispose();
    (this.dropMesh.material as MeshPhysicalMaterial).dispose();
    this.puddleMesh.geometry.dispose();
    (this.puddleMesh.material as MeshPhysicalMaterial).dispose();
    this.sparkMesh.geometry.dispose();
    (this.sparkMesh.material as MeshBasicMaterial).dispose();
    this.smokeGeo.dispose();
    this.smokeMat.dispose();
  }
}
