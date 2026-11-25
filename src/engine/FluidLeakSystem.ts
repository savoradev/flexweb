import {
  AdditiveBlending, BufferAttribute, BufferGeometry, CanvasTexture, Color,
  Group, InstancedMesh, Matrix4, MeshPhysicalMaterial, NormalBlending,
  PlaneGeometry, Points, Quaternion, ShaderMaterial, SphereGeometry, Vector3,
} from 'three';
import { CollisionWorld, GroundSample } from './SoftBodySolver';

const MAX_DROPS = 28;
const MAX_PUDDLES = 96;
const MAX_SMOKE = 96;

const COLOR_OIL = new Color(0x0c0f15);
const COLOR_COOLANT = new Color(0x14c975);
const COLOR_STEAM = new Color(0xdce6f0);
const COLOR_SMOKE = new Color(0x2c3038);

let cachedPuddleAlpha: CanvasTexture | null = null;
let cachedSmokeTex: CanvasTexture | null = null;

/** Generates an organic, soft-edged wet-stain alpha texture (no hard geometric circles). */
function getPuddleAlphaTexture(): CanvasTexture | null {
  if (cachedPuddleAlpha) return cachedPuddleAlpha;
  if (typeof document === 'undefined') return null;
  const size = 128;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  ctx.clearRect(0, 0, size, size);
  const cx = size * 0.5;
  const cy = size * 0.5;
  // Layer 7 slightly offset soft radial blobs for an irregular organic oil/coolant puddle silhouette
  const blobs = [
    [0, 0, 0.44, 0.95],
    [0.10, -0.06, 0.30, 0.72],
    [-0.12, 0.08, 0.32, 0.75],
    [0.06, 0.14, 0.25, 0.65],
    [-0.08, -0.13, 0.26, 0.65],
    [0.17, 0.04, 0.20, 0.50],
    [-0.16, -0.04, 0.21, 0.52],
  ];
  for (const [ox, oy, rad, alpha] of blobs) {
    const bx = cx + ox * size;
    const by = cy + oy * size;
    const r = rad * size;
    const grad = ctx.createRadialGradient(bx, by, r * 0.15, bx, by, r);
    grad.addColorStop(0, `rgba(255,255,255,${alpha})`);
    grad.addColorStop(0.62, `rgba(255,255,255,${alpha * 0.72})`);
    grad.addColorStop(0.88, `rgba(255,255,255,${alpha * 0.18})`);
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(bx, by, r, 0, Math.PI * 2);
    ctx.fill();
  }
  const tex = new CanvasTexture(canvas);
  tex.needsUpdate = true;
  cachedPuddleAlpha = tex;
  return tex;
}

/** Generates a photorealistic wispy, soft volumetric smoke/steam sprite texture. */
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
  // Multi-lobe soft Gaussian puffs to mimic turbulent wisps instead of a flat circle
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

/**
 * FluidLeakSystem
 * ---------------
 * Drop-by-drop 3D oil & coolant leaks (falling droplets -> slowly spreading organic puddles)
 * + soft photorealistic wispy radiator steam / engine smoke billboards.
 * Strictly zero heap allocations per frame (3 draw calls total).
 */
export class FluidLeakSystem {
  readonly group = new Group();
  private readonly dropMesh: InstancedMesh;
  private readonly puddleMesh: InstancedMesh;
  private readonly smokePoints: Points;
  private readonly smokeGeo: BufferGeometry;
  private readonly smokeMat: ShaderMaterial;

  // Falling 3D droplets (drop-by-drop gravity physics)
  private readonly dX = new Float32Array(MAX_DROPS);
  private readonly dY = new Float32Array(MAX_DROPS);
  private readonly dZ = new Float32Array(MAX_DROPS);
  private readonly dVy = new Float32Array(MAX_DROPS);
  private readonly dCoolant = new Uint8Array(MAX_DROPS);
  private readonly dActive = new Uint8Array(MAX_DROPS);
  private dropCursor = 0;
  private dropTimer = 0;
  private activeDropCount = 0;

  // Slowly creeping organic asphalt puddles (spawned only when droplets land)
  private readonly pX = new Float32Array(MAX_PUDDLES);
  private readonly pY = new Float32Array(MAX_PUDDLES);
  private readonly pZ = new Float32Array(MAX_PUDDLES);
  private readonly pRot = new Float32Array(MAX_PUDDLES);
  private readonly pScale = new Float32Array(MAX_PUDDLES);
  private readonly pMaxScale = new Float32Array(MAX_PUDDLES);
  private readonly pActive = new Uint8Array(MAX_PUDDLES);
  private puddleCursor = 0;
  private puddleDirty = false;

  // Volumetric soft billboard smoke / steam particles
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
  private activeSmokeCount = 0;

  // Pre-allocated scratch math (zero GC)
  private readonly mat4 = new Matrix4();
  private readonly pos = new Vector3();
  private readonly quat = new Quaternion();
  private readonly scl = new Vector3();
  private readonly upAxis = new Vector3(0, 1, 0);
  private readonly ground: GroundSample = { h: 0, nx: 0, ny: 1, nz: 0, mu: 1 };

  constructor() {
    this.group.name = 'FluidLeakSystem';

    // 1. Falling liquid droplets (elongated tiny glossy teardrops)
    const dGeo = new SphereGeometry(0.016, 6, 5);
    dGeo.scale(0.85, 1.9, 0.85);
    const dMat = new MeshPhysicalMaterial({
      color: 0xffffff,
      roughness: 0.05,
      metalness: 0.1,
      clearcoat: 1.0,
      transparent: true,
      opacity: 0.92,
    });
    this.dropMesh = new InstancedMesh(dGeo, dMat, MAX_DROPS);
    this.dropMesh.frustumCulled = false;
    this.dropMesh.instanceMatrix.setUsage(35048);
    this.dropMesh.visible = false;

    // 2. Soft organic asphalt fluid puddles
    const pGeo = new PlaneGeometry(1.0, 1.0);
    pGeo.rotateX(-Math.PI / 2);
    const puddleAlpha = getPuddleAlphaTexture();
    const pMat = new MeshPhysicalMaterial({
      color: 0xffffff,
      alphaMap: puddleAlpha ?? undefined,
      roughness: 0.06,
      metalness: 0.18,
      clearcoat: 1.0,
      clearcoatRoughness: 0.04,
      transparent: true,
      opacity: 0.86,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -2,
    });
    this.puddleMesh = new InstancedMesh(pGeo, pMat, MAX_PUDDLES);
    this.puddleMesh.frustumCulled = false;
    this.puddleMesh.instanceMatrix.setUsage(35048);
    this.puddleMesh.visible = false;

    // 3. Soft realistic wispy billboard smoke & radiator steam (custom lightweight shader, 1 draw call)
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
    void AdditiveBlending;
    this.smokePoints = new Points(this.smokeGeo, this.smokeMat);
    this.smokePoints.frustumCulled = false;
    this.smokePoints.visible = false;

    this.group.add(this.dropMesh, this.puddleMesh, this.smokePoints);
    this.reset();
  }

  private spawnPuddle(wx: number, groundY: number, wz: number, isCoolant: boolean, carSpeed: number): void {
    const idx = this.puddleCursor;
    this.puddleCursor = (this.puddleCursor + 1) % MAX_PUDDLES;

    this.pX[idx] = wx;
    this.pY[idx] = groundY + 0.015;
    this.pZ[idx] = wz;
    this.pRot[idx] = ((idx * 1.6180339) % 1) * Math.PI * 2;
    this.pScale[idx] = 0.055;
    // Small realistic wet spots that creep slowly instead of instant huge pools
    this.pMaxScale[idx] = (carSpeed > 2.5 ? 0.14 : 0.28) + ((idx * 17) % 10) * 0.018;
    this.pActive[idx] = 1;

    this.puddleMesh.setColorAt(idx, isCoolant ? COLOR_COOLANT : COLOR_OIL);
    if (this.puddleMesh.instanceColor) this.puddleMesh.instanceColor.needsUpdate = true;
    this.puddleMesh.visible = true;
    this.puddleDirty = true;
  }

  /**
   * Emits drop-by-drop oil/coolant leaks and soft wispy radiator steam / engine smoke.
   * Call once per rendered frame (zero heap allocations).
   */
  update(
    dt: number,
    world: CollisionWorld | null,
    engineWorldX: number,
    engineWorldY: number,
    engineWorldZ: number,
    coolantLeak: boolean,
    oilLeak: boolean,
    engineDamage: number,
    carSpeed: number,
  ): void {
    const h = Math.min(dt, 0.05);

    // 1. Spawn falling liquid droplets drop-by-drop (0.38s – 0.68s cadence)
    if ((coolantLeak || oilLeak) && world) {
      this.dropTimer -= h;
      if (this.dropTimer <= 0) {
        const severity = Math.min(1, engineDamage);
        // Slower, natural drop-by-drop drip rate
        this.dropTimer = (0.64 - severity * 0.24) + ((this.dropCursor * 13) % 7) * 0.03;
        const idx = this.dropCursor;
        this.dropCursor = (this.dropCursor + 1) % MAX_DROPS;

        const jx = ((idx * 37) % 13 - 6) * 0.038;
        const jz = ((idx * 53) % 13 - 6) * 0.038;
        this.dX[idx] = engineWorldX + jx;
        this.dY[idx] = Math.max(engineWorldY - 0.08, 0.18);
        this.dZ[idx] = engineWorldZ + jz;
        this.dVy[idx] = -0.35;
        const isCoolant = coolantLeak && (!oilLeak || (idx & 1) === 0);
        this.dCoolant[idx] = isCoolant ? 1 : 0;
        if (!this.dActive[idx]) this.activeDropCount++;
        this.dActive[idx] = 1;

        this.dropMesh.setColorAt(idx, isCoolant ? COLOR_COOLANT : COLOR_OIL);
        if (this.dropMesh.instanceColor) this.dropMesh.instanceColor.needsUpdate = true;
        this.dropMesh.visible = true;
      }
    }

    // Integrate falling droplets under gravity -> spawn small wet spot on asphalt impact
    if (this.activeDropCount > 0 && world) {
      this.quat.identity();
      let aliveDrops = 0;
      for (let i = 0; i < MAX_DROPS; i++) {
        if (!this.dActive[i]) continue;
        this.dVy[i] -= 9.81 * h;
        this.dY[i] += this.dVy[i] * h;
        world.groundAt(this.dX[i], this.dZ[i], this.ground);
        if (this.dY[i] <= this.ground.h + 0.02) {
          this.dActive[i] = 0;
          this.pos.set(0, -500, 0);
          this.scl.set(0, 0, 0);
          this.mat4.compose(this.pos, this.quat, this.scl);
          this.dropMesh.setMatrixAt(i, this.mat4);
          this.spawnPuddle(this.dX[i], this.ground.h, this.dZ[i], this.dCoolant[i] === 1, carSpeed);
          continue;
        }
        aliveDrops++;
        this.pos.set(this.dX[i], this.dY[i], this.dZ[i]);
        this.scl.set(1, 1, 1);
        this.mat4.compose(this.pos, this.quat, this.scl);
        this.dropMesh.setMatrixAt(i, this.mat4);
      }
      this.activeDropCount = aliveDrops;
      this.dropMesh.instanceMatrix.needsUpdate = true;
      if (aliveDrops === 0) this.dropMesh.visible = false;
    }

    // 2. Slowly creep active asphalt puddles (very gradual spread)
    if (this.puddleDirty) {
      let stillGrowing = false;
      for (let i = 0; i < MAX_PUDDLES; i++) {
        if (!this.pActive[i]) continue;
        if (this.pScale[i] < this.pMaxScale[i]) {
          this.pScale[i] = Math.min(this.pMaxScale[i], this.pScale[i] + h * 0.065);
          stillGrowing = true;
        }
        const sc = this.pScale[i];
        this.pos.set(this.pX[i], this.pY[i], this.pZ[i]);
        this.quat.setFromAxisAngle(this.upAxis, this.pRot[i]);
        this.scl.set(sc, 1, sc * (0.88 + ((i * 7) % 5) * 0.06));
        this.mat4.compose(this.pos, this.quat, this.scl);
        this.puddleMesh.setMatrixAt(i, this.mat4);
      }
      this.puddleMesh.instanceMatrix.needsUpdate = true;
      this.puddleDirty = stillGrowing;
    }

    // 3. Spawn soft volumetric wispy steam (coolant leak) & dark engine smoke (severe engine damage)
    if (coolantLeak || engineDamage > 0.42) {
      this.smokeTimer -= h;
      if (this.smokeTimer <= 0) {
        this.smokeTimer = engineDamage > 0.65 ? 0.075 : 0.115;
        const idx = this.smokeCursor;
        this.smokeCursor = (this.smokeCursor + 1) % MAX_SMOKE;
        const jx = ((idx * 29) % 17 - 8) * 0.042;
        const jz = ((idx * 41) % 17 - 8) * 0.042;

        this.sPos[idx * 3] = engineWorldX + jx;
        this.sPos[idx * 3 + 1] = engineWorldY + 0.28;
        this.sPos[idx * 3 + 2] = engineWorldZ + jz;
        this.sVx[idx] = jx * 0.75 + 0.18; // gentle ambient breeze
        this.sVy[idx] = 0.85 + ((idx * 13) % 7) * 0.11;
        this.sVz[idx] = jz * 0.75;
        this.sRot[idx] = ((idx * 2.39996) % 6.283);
        this.sSpin[idx] = ((idx & 1) === 0 ? 1 : -1) * (0.28 + ((idx * 5) % 4) * 0.09);
        this.sLife[idx] = 0;
        this.sMaxLife[idx] = 1.9 + ((idx * 11) % 6) * 0.24;
        this.sBaseSize[idx] = 0.55 + ((idx * 7) % 5) * 0.09;

        const isDarkSmoke = engineDamage > 0.55 && (!coolantLeak || (idx % 3) !== 0);
        const col = isDarkSmoke ? COLOR_SMOKE : COLOR_STEAM;
        this.sCol[idx * 3] = col.r;
        this.sCol[idx * 3 + 1] = col.g;
        this.sCol[idx * 3 + 2] = col.b;
        this.sMaxAlpha[idx] = isDarkSmoke ? 0.34 : 0.26;
        this.sSize[idx] = this.sBaseSize[idx];
        this.sAlpha[idx] = 0.01;

        if (!this.sActive[idx]) this.activeSmokeCount++;
        this.sActive[idx] = 1;
        this.smokePoints.visible = true;
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
        // Soft turbulent dissipation as steam/smoke rises and slows
        const drag = 1 - 0.55 * h;
        this.sVx[i] *= drag;
        this.sVz[i] *= drag;
        this.sVy[i] = Math.max(0.28, this.sVy[i] * (1 - 0.35 * h));
        this.sPos[i * 3] += this.sVx[i] * h;
        this.sPos[i * 3 + 1] += this.sVy[i] * h;
        this.sPos[i * 3 + 2] += this.sVz[i] * h;
        this.sRot[i] += this.sSpin[i] * h;

        // Smooth fade-in (0..18%), gentle expansion, long natural fade-out (25%..100%)
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
  }

  reset(): void {
    this.dActive.fill(0);
    this.pActive.fill(0);
    this.sActive.fill(0);
    this.sAlpha.fill(0);
    this.sSize.fill(0);
    this.dropCursor = 0;
    this.dropTimer = 0;
    this.activeDropCount = 0;
    this.puddleCursor = 0;
    this.smokeCursor = 0;
    this.smokeTimer = 0;
    this.puddleDirty = false;
    this.activeSmokeCount = 0;

    this.quat.identity();
    this.pos.set(0, -500, 0);
    this.scl.set(0, 0, 0);
    this.mat4.compose(this.pos, this.quat, this.scl);
    for (let i = 0; i < MAX_DROPS; i++) this.dropMesh.setMatrixAt(i, this.mat4);
    for (let i = 0; i < MAX_PUDDLES; i++) this.puddleMesh.setMatrixAt(i, this.mat4);
    this.dropMesh.instanceMatrix.needsUpdate = true;
    this.puddleMesh.instanceMatrix.needsUpdate = true;
    this.dropMesh.visible = false;
    this.puddleMesh.visible = false;

    (this.smokeGeo.getAttribute('aAlpha') as BufferAttribute).needsUpdate = true;
    (this.smokeGeo.getAttribute('aSize') as BufferAttribute).needsUpdate = true;
    this.smokePoints.visible = false;
  }

  dispose(): void {
    this.dropMesh.geometry.dispose();
    (this.dropMesh.material as MeshPhysicalMaterial).dispose();
    this.puddleMesh.geometry.dispose();
    (this.puddleMesh.material as MeshPhysicalMaterial).dispose();
    this.smokeGeo.dispose();
    this.smokeMat.dispose();
  }
}
