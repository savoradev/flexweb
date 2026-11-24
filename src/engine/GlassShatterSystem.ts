import {
  BufferAttribute,
  BufferGeometry,
  DoubleSide,
  Euler,
  InstancedMesh,
  Matrix4,
  MeshStandardMaterial,
  Quaternion,
  Vector3,
} from 'three';
import type { CollisionWorld, GroundSample } from './SoftBodySolver';

/**
 * GlassShatterSystem
 * ------------------
 * Pre-allocated InstancedMesh (1 single WebGL draw call, 128 shards) with flat
 * Float32Array SoA physics state. Zero per-frame heap allocations.
 *
 * Triggered when windshield, side windows, or rear glass experience heavy crash
 * deformation — bursting reflective 3D glass shards that spin, fall under 9.81 m/s²,
 * and bounce/skitter on the arena ground.
 */

const MAX_SHARDS = 128;

function createShardGeometry(): BufferGeometry {
  // Crisp irregular triangular prism glass shard (6 vertices, 8 triangles)
  const verts = new Float32Array([
    0.0, 0.048, 0.004,
    -0.042, -0.032, 0.004,
    0.038, -0.024, 0.004,
    0.0, 0.048, -0.004,
    -0.042, -0.032, -0.004,
    0.038, -0.024, -0.004,
  ]);
  const idx = [
    0, 1, 2,
    3, 5, 4,
    0, 3, 1, 1, 3, 4,
    1, 4, 2, 2, 4, 5,
    2, 5, 0, 0, 5, 3,
  ];
  const g = new BufferGeometry();
  g.setAttribute('position', new BufferAttribute(verts, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

export class GlassShatterSystem {
  readonly mesh: InstancedMesh;
  private readonly px = new Float32Array(MAX_SHARDS);
  private readonly py = new Float32Array(MAX_SHARDS);
  private readonly pz = new Float32Array(MAX_SHARDS);
  private readonly vx = new Float32Array(MAX_SHARDS);
  private readonly vy = new Float32Array(MAX_SHARDS);
  private readonly vz = new Float32Array(MAX_SHARDS);
  private readonly rx = new Float32Array(MAX_SHARDS);
  private readonly ry = new Float32Array(MAX_SHARDS);
  private readonly rz = new Float32Array(MAX_SHARDS);
  private readonly vrx = new Float32Array(MAX_SHARDS);
  private readonly vry = new Float32Array(MAX_SHARDS);
  private readonly vrz = new Float32Array(MAX_SHARDS);
  private readonly scale = new Float32Array(MAX_SHARDS);
  private readonly life = new Float32Array(MAX_SHARDS);
  private cursor = 0;
  private activeCount = 0;
  private prngState = 0x13579bdf;

  private readonly tmpMat = new Matrix4();
  private readonly tmpPos = new Vector3();
  private readonly tmpEuler = new Euler();
  private readonly tmpQuat = new Quaternion();
  private readonly tmpScale = new Vector3();
  private readonly zeroMat = new Matrix4().makeScale(0, 0, 0);
  private readonly ground: GroundSample = { h: 0, nx: 0, ny: 1, nz: 0, mu: 0.8 };

  constructor() {
    const geo = createShardGeometry();
    const mat = new MeshStandardMaterial({
      color: 0xdbeafe,
      emissive: 0x38bdf8,
      emissiveIntensity: 0.14,
      metalness: 0.92,
      roughness: 0.06,
      transparent: true,
      opacity: 0.78,
      side: DoubleSide,
      envMapIntensity: 2.2,
    });
    this.mesh = new InstancedMesh(geo, mat, MAX_SHARDS);
    this.mesh.frustumCulled = false;
    this.mesh.visible = false;
    for (let i = 0; i < MAX_SHARDS; i++) {
      this.mesh.setMatrixAt(i, this.zeroMat);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
  }

  private rand(): number {
    let x = this.prngState;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.prngState = x;
    return ((x >>> 0) / 4294967296);
  }

  /** Spawns a burst of `count` 3D glass shards in world space. Zero heap allocations. */
  spawnBurst(
    wx: number,
    wy: number,
    wz: number,
    baseVx: number,
    baseVy: number,
    baseVz: number,
    outNx: number,
    outNy: number,
    outNz: number,
    count = 28,
  ): void {
    const n = Math.min(MAX_SHARDS, count);
    for (let k = 0; k < n; k++) {
      const i = this.cursor;
      this.cursor = (this.cursor + 1) % MAX_SHARDS;
      if (this.life[i] <= 0) this.activeCount++;

      const spreadX = (this.rand() - 0.5) * 0.65;
      const spreadY = (this.rand() - 0.5) * 0.32;
      const spreadZ = (this.rand() - 0.5) * 0.65;
      this.px[i] = wx + spreadX;
      this.py[i] = wy + spreadY;
      this.pz[i] = wz + spreadZ;

      const burst = 2.4 + this.rand() * 4.8;
      this.vx[i] = baseVx * 0.65 + outNx * burst + (this.rand() - 0.5) * 4.2;
      this.vy[i] = baseVy * 0.5 + outNy * burst + 1.4 + this.rand() * 3.2;
      this.vz[i] = baseVz * 0.65 + outNz * burst + (this.rand() - 0.5) * 4.2;

      this.rx[i] = this.rand() * 6.28;
      this.ry[i] = this.rand() * 6.28;
      this.rz[i] = this.rand() * 6.28;
      this.vrx[i] = (this.rand() - 0.5) * 22;
      this.vry[i] = (this.rand() - 0.5) * 22;
      this.vrz[i] = (this.rand() - 0.5) * 22;

      this.scale[i] = 0.65 + this.rand() * 1.15;
      this.life[i] = 3.8 + this.rand() * 2.2;
    }
    this.mesh.visible = this.activeCount > 0;
  }

  /** Steps shard physics and updates InstancedMesh transforms in place. Zero allocations. */
  update(dt: number, world: CollisionWorld | null): void {
    if (this.activeCount <= 0) return;
    const gs = this.ground;
    let alive = 0;

    for (let i = 0; i < MAX_SHARDS; i++) {
      let l = this.life[i];
      if (l <= 0) continue;
      l -= dt;
      if (l <= 0) {
        this.life[i] = 0;
        this.mesh.setMatrixAt(i, this.zeroMat);
        continue;
      }
      this.life[i] = l;
      alive++;

      this.vy[i] -= 9.81 * dt;
      this.vx[i] *= 1 - 0.35 * dt;
      this.vz[i] *= 1 - 0.35 * dt;

      const nx = this.px[i] + this.vx[i] * dt;
      let ny = this.py[i] + this.vy[i] * dt;
      const nz = this.pz[i] + this.vz[i] * dt;

      if (world) {
        world.groundAt(nx, nz, gs);
        const floorY = gs.h + 0.025;
        if (ny < floorY) {
          ny = floorY;
          if (this.vy[i] < -0.6) {
            this.vy[i] = -this.vy[i] * 0.32;
            this.vx[i] *= 0.68;
            this.vz[i] *= 0.68;
            this.vrx[i] *= 0.6;
            this.vry[i] *= 0.6;
            this.vrz[i] *= 0.6;
          } else {
            this.vy[i] = 0;
            this.vx[i] *= 0.82;
            this.vz[i] *= 0.82;
            this.vrx[i] *= 0.5;
            this.vry[i] *= 0.5;
            this.vrz[i] *= 0.5;
          }
        }
      }

      this.px[i] = nx;
      this.py[i] = ny;
      this.pz[i] = nz;
      this.rx[i] += this.vrx[i] * dt;
      this.ry[i] += this.vry[i] * dt;
      this.rz[i] += this.vrz[i] * dt;

      const fade = l < 0.75 ? l / 0.75 : 1.0;
      const s = this.scale[i] * fade;
      this.tmpPos.set(nx, ny, nz);
      this.tmpEuler.set(this.rx[i], this.ry[i], this.rz[i]);
      this.tmpQuat.setFromEuler(this.tmpEuler);
      this.tmpScale.set(s, s, s);
      this.tmpMat.compose(this.tmpPos, this.tmpQuat, this.tmpScale);
      this.mesh.setMatrixAt(i, this.tmpMat);
    }

    this.activeCount = alive;
    this.mesh.visible = alive > 0;
    this.mesh.instanceMatrix.needsUpdate = true;
  }

  reset(): void {
    if (this.activeCount === 0) return;
    this.life.fill(0);
    this.activeCount = 0;
    this.cursor = 0;
    for (let i = 0; i < MAX_SHARDS; i++) {
      this.mesh.setMatrixAt(i, this.zeroMat);
    }
    this.mesh.visible = false;
    this.mesh.instanceMatrix.needsUpdate = true;
  }

  dispose(): void {
    this.mesh.removeFromParent();
    this.mesh.geometry.dispose();
    (this.mesh.material as MeshStandardMaterial).dispose();
  }
}
