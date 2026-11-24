import { BufferAttribute, BufferGeometry, Mesh } from 'three';
import { TypedArrayPool } from './MemoryGuard';
import type { PrecomputedBindingData } from './MeshCacheDB';

/**
 * MeshDeformer
 * ------------
 * Skins high-resolution visual meshes to the coarse physics lattice at a locked
 * 60 FPS even on 70k–150k vertex imported `.glb` models:
 *
 *  - Precomputes and caches the top-4 nearest node indices (`Uint16Array`),
 *    modified-Shepard inverse-distance weights (`Float32Array`), and node-frame
 *    rest offsets + rest normals (`Float32Array`) once at load time.
 *  - Precomputes a compact fixed-stride (`MAX_NEIGHBORS = 6`) neighbor adjacency
 *    table (`Int32Array`) once at load time for small/medium meshes (`<= 15,000`
 *    total vehicle vertices), and automatically skips multi-pass Laplacian
 *    smoothing when total vehicle vertex count `> 15,000`.
 *  - Rotates vertex normals directly via the precomputed 4-node orthonormal
 *    frames (`F * n0`) on high-poly meshes (`> 3,500` verts) or falls back to
 *    bounded `computeVertexNormals()` on low-poly procedural panels, completely
 *    eliminating crash-impact frame drops.
 *  - Runs strictly once per rendered frame (never inside the 8 physics substeps)
 *    with per-group dirty tracking (`dirtyEpsilon`) and zero heap allocations.
 */

const MAX_NEIGHBORS = 6;
const EMPTY_I32 = new Int32Array(0);
const EMPTY_U8 = new Uint8Array(0);

export interface DeformBinding {
  mesh: Mesh;
  group: number;
  vertCount: number;
  uniqueCount: number;
  vertToUnique: Uint32Array;
  restPos: Float32Array;       // baseline positions (Float32 snapshot)
  restNormal: Float32Array;    // baseline normals (Float32 snapshot)
  uniqueRest: Float32Array;
  nodes: Uint16Array;          // 4 per unique vertex
  weights: Float32Array;       // 4 per unique vertex
  offsets: Float32Array;       // 12 per unique vertex (offset in node rest frame)
  normalOffsets: Float32Array; // 3 per vertex (rest normal in dominant node rest frame)
  usedNodes: Uint16Array;      // deduplicated list of node indices referenced by this mesh
  uniqueDirty: Uint8Array;     // 1 per unique vertex when any of its 4 bound nodes moved
  fixedAdj: Int32Array;        // stride MAX_NEIGHBORS (6) per unique vertex, -1 padded (EMPTY_I32 when disabled)
  fixedDeg: Uint8Array;        // neighbor count per unique vertex (EMPTY_U8 when disabled)
  pos: Float32Array;           // unique positions scratch
  disp: Float32Array;          // displacement scratch
  disp2: Float32Array;
  triU: Uint32Array;           // 3 welded unique indices per triangle (for targeted local computeVertexNormals)
  uniqueNrm: Float32Array;     // 3 per unique vertex normal accumulator
  uniqueCrease: Float32Array;  // 0..1 deformation crease blend per unique vertex
  maxDeformSq: number;         // peak squared displacement on this panel
  smoothIterations: number;
  fastAnalyticalNormals: boolean;
}

export class MeshDeformer {
  readonly bindings: DeformBinding[] = [];
  readonly pool = new TypedArrayPool();
  dirtyEpsilon = 0.0022;
  laplacianLambda = 0.45;
  lastSkinnedVerts = 0;
  totalVehicleVerts = 0;

  /** Current node frames (9 per node, column-major e1,e2,e3) in body space. */
  readonly frames: Float32Array;
  private readonly restFrames: Float32Array;
  private readonly frameA: Int32Array;
  private readonly frameB: Int32Array;
  private readonly skinnedLocal: Float32Array; // node positions at last skin, per node
  private readonly groupDirty: Uint8Array;
  private readonly nodeDirty: Uint8Array;
  private readonly groupNodes: Uint16Array[];
  private readonly nodeCount: number;

  constructor(
    restLocal: Float32Array | Float64Array,
    nodeGroup: Uint8Array,
    groupCount: number,
    neighbours: number[][],
  ) {
    const n = (this.nodeCount = nodeGroup.length);
    this.frames = this.pool.f32(n * 9);
    this.restFrames = this.pool.f32(n * 9);
    this.frameA = this.pool.i32(n);
    this.frameB = this.pool.i32(n);
    this.skinnedLocal = this.pool.f32(n * 3);
    this.groupDirty = this.pool.u8(groupCount);
    this.nodeDirty = this.pool.u8(n);
    this.groupNodes = [];
    for (let g = 0; g < groupCount; g++) {
      const list: number[] = [];
      for (let i = 0; i < n; i++) if (nodeGroup[i] === g) list.push(i);
      const arr = this.pool.u16(list.length);
      arr.set(list);
      this.groupNodes.push(arr);
    }
    // choose the two neighbours spanning the widest angle for each node frame
    for (let i = 0; i < n; i++) {
      let bestA = -1, bestB = -1, best = 0;
      const nb = neighbours[i];
      for (let p = 0; p < nb.length; p++) for (let q = p + 1; q < nb.length; q++) {
        const a = nb[p], b = nb[q];
        const ax = restLocal[a * 3] - restLocal[i * 3], ay = restLocal[a * 3 + 1] - restLocal[i * 3 + 1], az = restLocal[a * 3 + 2] - restLocal[i * 3 + 2];
        const bx = restLocal[b * 3] - restLocal[i * 3], by = restLocal[b * 3 + 1] - restLocal[i * 3 + 1], bz = restLocal[b * 3 + 2] - restLocal[i * 3 + 2];
        const la = Math.hypot(ax, ay, az), lb = Math.hypot(bx, by, bz);
        if (la < 1e-4 || lb < 1e-4) continue;
        const cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
        const s = Math.hypot(cx, cy, cz) / (la * lb);
        const score = s / Math.sqrt(la * lb + 0.05);
        if (score > best) { best = score; bestA = a; bestB = b; }
      }
      this.frameA[i] = bestA; this.frameB[i] = bestB;
    }
    for (let i = 0; i < n * 3; i++) this.skinnedLocal[i] = restLocal[i];
    this.computeFrames(this.skinnedLocal, this.restFrames);
    this.frames.set(this.restFrames);
  }

  /** Build an orthonormal frame per node from its two reference neighbours, with anti-inversion clamping against restFrames. */
  computeFrames(local: Float32Array, out: Float32Array): void {
    const isRest = out === this.restFrames;
    const R0 = this.restFrames;
    for (let i = 0; i < this.nodeCount; i++) {
      const o = i * 9;
      const a = this.frameA[i], b = this.frameB[i];
      if (a < 0) { out[o] = 1; out[o + 1] = 0; out[o + 2] = 0; out[o + 3] = 0; out[o + 4] = 1; out[o + 5] = 0; out[o + 6] = 0; out[o + 7] = 0; out[o + 8] = 1; continue; }
      const px = local[i * 3], py = local[i * 3 + 1], pz = local[i * 3 + 2];
      let e1x = local[a * 3] - px, e1y = local[a * 3 + 1] - py, e1z = local[a * 3 + 2] - pz;
      let l = Math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z);
      if (l < 1e-5) { this.copyRest(i, out); continue; }
      e1x /= l; e1y /= l; e1z /= l;

      if (!isRest) {
        const dot1 = e1x * R0[o] + e1y * R0[o + 1] + e1z * R0[o + 2];
        if (dot1 < 0.25) {
          // Prevent primary axis reversal under extreme axial fold
          e1x = e1x * 0.35 + R0[o] * 0.65;
          e1y = e1y * 0.35 + R0[o + 1] * 0.65;
          e1z = e1z * 0.35 + R0[o + 2] * 0.65;
          const l1 = Math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z) || 1;
          e1x /= l1; e1y /= l1; e1z /= l1;
        }
      }

      let bx = local[b * 3] - px, by = local[b * 3 + 1] - py, bz = local[b * 3 + 2] - pz;
      const d = bx * e1x + by * e1y + bz * e1z;
      bx -= d * e1x; by -= d * e1y; bz -= d * e1z;
      l = Math.sqrt(bx * bx + by * by + bz * bz);
      if (l < 1e-4) {
        if (isRest) { this.copyRest(i, out); continue; }
        bx = R0[o + 3]; by = R0[o + 4]; bz = R0[o + 5];
        const dr = bx * e1x + by * e1y + bz * e1z;
        bx -= dr * e1x; by -= dr * e1y; bz -= dr * e1z;
        l = Math.sqrt(bx * bx + by * by + bz * bz);
        if (l < 1e-5) { this.copyRest(i, out); continue; }
      }
      bx /= l; by /= l; bz /= l;

      if (!isRest) {
        const dot2 = bx * R0[o + 3] + by * R0[o + 4] + bz * R0[o + 5];
        let e3x = e1y * bz - e1z * by, e3y = e1z * bx - e1x * bz, e3z = e1x * by - e1y * bx;
        const dot3 = e3x * R0[o + 6] + e3y * R0[o + 7] + e3z * R0[o + 8];
        if (dot2 < 0.25 || dot3 < 0.25) {
          // Neighbor triangle flattened or crossed colinearity: re-anchor e2 to rest frame so the panel never flips upside-down
          bx = R0[o + 3]; by = R0[o + 4]; bz = R0[o + 5];
          const dr = bx * e1x + by * e1y + bz * e1z;
          bx -= dr * e1x; by -= dr * e1y; bz -= dr * e1z;
          const l2 = Math.sqrt(bx * bx + by * by + bz * bz);
          if (l2 < 1e-5) { this.copyRest(i, out); continue; }
          bx /= l2; by /= l2; bz /= l2;
        }
      }

      out[o] = e1x; out[o + 1] = e1y; out[o + 2] = e1z;
      out[o + 3] = bx; out[o + 4] = by; out[o + 5] = bz;
      out[o + 6] = e1y * bz - e1z * by; out[o + 7] = e1z * bx - e1x * bz; out[o + 8] = e1x * by - e1y * bx;
    }
  }

  private copyRest(i: number, out: Float32Array): void {
    const o = i * 9;
    if (out === this.restFrames) {
      out[o] = 1; out[o + 1] = 0; out[o + 2] = 0; out[o + 3] = 0; out[o + 4] = 1; out[o + 5] = 0; out[o + 6] = 0; out[o + 7] = 0; out[o + 8] = 1;
    } else {
      for (let k = 0; k < 9; k++) out[o + k] = this.restFrames[o + k];
    }
  }

  /** Bind a mesh (positions already in car-local rest space) to a deform group. */
  bind(mesh: Mesh, group: number, restLocal: Float32Array): DeformBinding {
    const geo = mesh.geometry as BufferGeometry;
    ensurePlainAttributes(geo);
    if (!geo.getAttribute('normal')) geo.computeVertexNormals();
    const posAttr = geo.getAttribute('position') as BufferAttribute;
    const nrmAttr = geo.getAttribute('normal') as BufferAttribute;
    const pos = posAttr.array as Float32Array;
    const nrm = nrmAttr.array as Float32Array;
    const vertCount = posAttr.count;
    this.totalVehicleVerts += vertCount;
    const P = this.pool;

    // Weld coincident vertices (seams, flat-shaded duplicates) using a 48-bit packed integer key (zero string heap allocations)
    const vertToUnique = P.u32(vertCount);
    const map = new Map<number, number>();
    const uniqueTemp = new Float32Array(vertCount * 3);
    let uniqueCount = 0;
    for (let v = 0; v < vertCount; v++) {
      const vx = pos[v * 3], vy = pos[v * 3 + 1], vz = pos[v * 3 + 2];
      const ix = (Math.round(vx * 2000) + 32768) & 0xffff;
      const iy = (Math.round(vy * 2000) + 32768) & 0xffff;
      const iz = (Math.round(vz * 2000) + 32768) & 0xffff;
      const key = ix + iy * 65536 + iz * 4294967296;
      let u = map.get(key);
      if (u === undefined) {
        u = uniqueCount++;
        map.set(key, u);
        const u3 = u * 3;
        uniqueTemp[u3] = vx;
        uniqueTemp[u3 + 1] = vy;
        uniqueTemp[u3 + 2] = vz;
      }
      vertToUnique[v] = u;
    }
    const uniqueRest = P.f32(uniqueCount * 3);
    uniqueRest.set(uniqueTemp.subarray(0, uniqueCount * 3));

    // Precompute & cache top-4 node indices, weights, and rest-frame offsets using O(V) 3D spatial grid hashing
    const nodesIdx = P.u16(uniqueCount * 4);
    const weights = P.f32(uniqueCount * 4);
    const offsets = P.f32(uniqueCount * 12);
    // Strict group isolation: each DeformGroup (especially Hood) binds exclusively to its own group's nodes
    const candidates = this.groupNodes[group].length >= 1 ? this.groupNodes[group] : this.groupNodes[0];
    const bestD = new Float64Array(5);
    const bestI = new Int32Array(5);
    const R0 = this.restFrames;
    const usedSet = new Set<number>();

    // Build compact 3D spatial hash grid over candidate nodes for O(1) per-vertex nearest-node queries
    const useGrid = candidates.length > 10;
    const invCell = 1 / 0.55;
    const grid = useGrid ? new Map<number, number[]>() : null;
    if (grid) {
      for (let c = 0; c < candidates.length; c++) {
        const ni = candidates[c];
        const cx = (Math.floor(restLocal[ni * 3] * invCell) + 512) & 0x3ff;
        const cy = (Math.floor(restLocal[ni * 3 + 1] * invCell) + 512) & 0x3ff;
        const cz = (Math.floor(restLocal[ni * 3 + 2] * invCell) + 512) & 0x3ff;
        const gKey = cx + cy * 1024 + cz * 1048576;
        let cell = grid.get(gKey);
        if (!cell) { cell = []; grid.set(gKey, cell); }
        cell.push(ni);
      }
    }

    for (let u = 0; u < uniqueCount; u++) {
      const x = uniqueRest[u * 3], y = uniqueRest[u * 3 + 1], z = uniqueRest[u * 3 + 2];
      bestD[0] = bestD[1] = bestD[2] = bestD[3] = bestD[4] = Infinity;
      bestI[0] = bestI[1] = bestI[2] = bestI[3] = bestI[4] = -1;

      let found = 0;
      if (grid) {
        const bx = Math.floor(x * invCell) + 512;
        const by = Math.floor(y * invCell) + 512;
        const bz = Math.floor(z * invCell) + 512;
        for (let ox = -1; ox <= 1; ox++) {
          const cx = (bx + ox) & 0x3ff;
          for (let oy = -1; oy <= 1; oy++) {
            const cy = ((by + oy) & 0x3ff) * 1024;
            for (let oz = -1; oz <= 1; oz++) {
              const cell = grid.get(cx + cy + ((bz + oz) & 0x3ff) * 1048576);
              if (!cell) continue;
              for (let m = 0; m < cell.length; m++) {
                const ni = cell[m];
                const dx = restLocal[ni * 3] - x, dy = restLocal[ni * 3 + 1] - y, dz = restLocal[ni * 3 + 2] - z;
                const d = dx * dx + dy * dy + dz * dz;
                found++;
                if (d >= bestD[4]) continue;
                let k = 4;
                while (k > 0 && bestD[k - 1] > d) { bestD[k] = bestD[k - 1]; bestI[k] = bestI[k - 1]; k--; }
                bestD[k] = d; bestI[k] = ni;
              }
            }
          }
        }
      }
      if (found < 5) {
        bestD[0] = bestD[1] = bestD[2] = bestD[3] = bestD[4] = Infinity;
        bestI[0] = bestI[1] = bestI[2] = bestI[3] = bestI[4] = -1;
        for (let c = 0; c < candidates.length; c++) {
          const ni = candidates[c];
          const dx = restLocal[ni * 3] - x, dy = restLocal[ni * 3 + 1] - y, dz = restLocal[ni * 3 + 2] - z;
          const d = dx * dx + dy * dy + dz * dz;
          if (d >= bestD[4]) continue;
          let k = 4;
          while (k > 0 && bestD[k - 1] > d) { bestD[k] = bestD[k - 1]; bestI[k] = bestI[k - 1]; k--; }
          bestD[k] = d; bestI[k] = ni;
        }
      }
      // Convert squared distances of top-5 to Euclidean distances
      for (let k = 0; k < 5; k++) if (Number.isFinite(bestD[k])) bestD[k] = Math.sqrt(bestD[k]);

      const R = Number.isFinite(bestD[4]) ? bestD[4] : (Number.isFinite(bestD[3]) ? bestD[3] * 1.5 + 0.05 : 1);
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        if (bestI[k] < 0) { weights[u * 4 + k] = 0; nodesIdx[u * 4 + k] = bestI[0] < 0 ? 0 : bestI[0]; continue; }
        const d = Math.max(bestD[k], 1e-4);
        const w = Math.pow(Math.max(0, R - d) / (R * d), 2) + 1e-9;
        weights[u * 4 + k] = w; nodesIdx[u * 4 + k] = bestI[k]; sum += w;
      }
      for (let k = 0; k < 4; k++) {
        weights[u * 4 + k] /= sum;
        const ni = nodesIdx[u * 4 + k];
        if (weights[u * 4 + k] > 0) usedSet.add(ni);
        const ox = x - restLocal[ni * 3], oy = y - restLocal[ni * 3 + 1], oz = z - restLocal[ni * 3 + 2];
        const f = ni * 9;
        offsets[u * 12 + k * 3] = R0[f] * ox + R0[f + 1] * oy + R0[f + 2] * oz;
        offsets[u * 12 + k * 3 + 1] = R0[f + 3] * ox + R0[f + 4] * oy + R0[f + 5] * oz;
        offsets[u * 12 + k * 3 + 2] = R0[f + 6] * ox + R0[f + 7] * oy + R0[f + 8] * oz;
      }
    }

    const usedArr = Array.from(usedSet);
    const usedNodes = P.u16(usedArr.length);
    usedNodes.set(usedArr);

    // Precompute per-vertex rest normal in its dominant bound node's rest frame (orthonormal)
    const normalOffsets = P.f32(vertCount * 3);
    for (let v = 0; v < vertCount; v++) {
      const u = vertToUnique[v];
      const nx = nrm[v * 3], ny = nrm[v * 3 + 1], nz = nrm[v * 3 + 2];
      const f = nodesIdx[u * 4] * 9;
      normalOffsets[v * 3] = R0[f] * nx + R0[f + 1] * ny + R0[f + 2] * nz;
      normalOffsets[v * 3 + 1] = R0[f + 3] * nx + R0[f + 4] * ny + R0[f + 5] * nz;
      normalOffsets[v * 3 + 2] = R0[f + 6] * nx + R0[f + 7] * ny + R0[f + 8] * nz;
    }

    // Only build fixed-stride Laplacian neighbor table when uniqueCount <= 15,000
    let fixedAdj: Int32Array = EMPTY_I32;
    let fixedDeg: Uint8Array = EMPTY_U8;
    let smoothIterations = 0;

    if (uniqueCount <= 15000 && this.totalVehicleVerts <= 15000) {
      fixedAdj = P.i32(uniqueCount * MAX_NEIGHBORS);
      fixedAdj.fill(-1);
      fixedDeg = P.u8(uniqueCount);
      const index = geo.getIndex();
      const triCount = index ? Math.floor(index.count / 3) : Math.floor(vertCount / 3);
      for (let t = 0; t < triCount; t++) {
        const a = vertToUnique[index ? index.getX(t * 3) : t * 3];
        const b = vertToUnique[index ? index.getX(t * 3 + 1) : t * 3 + 1];
        const c = vertToUnique[index ? index.getX(t * 3 + 2) : t * 3 + 2];
        MeshDeformer.addNeighborEdge(fixedAdj, fixedDeg, a, b);
        MeshDeformer.addNeighborEdge(fixedAdj, fixedDeg, b, a);
        MeshDeformer.addNeighborEdge(fixedAdj, fixedDeg, b, c);
        MeshDeformer.addNeighborEdge(fixedAdj, fixedDeg, c, b);
        MeshDeformer.addNeighborEdge(fixedAdj, fixedDeg, a, c);
        MeshDeformer.addNeighborEdge(fixedAdj, fixedDeg, c, a);
      }
      smoothIterations = 1;
    }

    // If total vehicle vertices exceed 15,000 (imported .glb), disable Laplacian passes across all bindings
    if (this.totalVehicleVerts > 15000) {
      for (const prev of this.bindings) {
        prev.smoothIterations = 0;
        prev.fastAnalyticalNormals = true;
      }
    }

    const restPos = P.f32(vertCount * 3); restPos.set(pos);
    const restNormal = P.f32(vertCount * 3); restNormal.set(nrm);
    posAttr.setUsage(35048); // DynamicDrawUsage
    nrmAttr.setUsage(35048);

    const triU = this.buildTriUniqueTable(geo, vertToUnique, vertCount);

    const binding: DeformBinding = {
      mesh,
      group,
      vertCount,
      uniqueCount,
      vertToUnique,
      restPos,
      restNormal,
      uniqueRest,
      nodes: nodesIdx,
      weights,
      offsets,
      normalOffsets,
      usedNodes,
      uniqueDirty: P.u8(uniqueCount),
      fixedAdj,
      fixedDeg,
      pos: P.f32(uniqueCount * 3),
      disp: P.f32(uniqueCount * 3),
      disp2: P.f32(uniqueCount * 3),
      triU,
      uniqueNrm: P.f32(uniqueCount * 3),
      uniqueCrease: P.f32(uniqueCount),
      maxDeformSq: 0,
      smoothIterations,
      fastAnalyticalNormals: this.totalVehicleVerts > 15000 || vertCount > 3500,
    };
    this.bindings.push(binding);
    return binding;
  }

  private buildTriUniqueTable(geo: BufferGeometry, vertToUnique: Uint32Array, vertCount: number): Uint32Array {
    const index = geo.getIndex();
    const triCount = index ? Math.floor(index.count / 3) : Math.floor(vertCount / 3);
    const triU = this.pool.u32(triCount * 3);
    if (index) {
      const idxArr = index.array;
      for (let t = 0; t < triCount; t++) {
        const t3 = t * 3;
        triU[t3] = vertToUnique[idxArr[t3]];
        triU[t3 + 1] = vertToUnique[idxArr[t3 + 1]];
        triU[t3 + 2] = vertToUnique[idxArr[t3 + 2]];
      }
    } else {
      for (let t = 0; t < triCount; t++) {
        const t3 = t * 3;
        triU[t3] = vertToUnique[t3];
        triU[t3 + 1] = vertToUnique[t3 + 1];
        triU[t3 + 2] = vertToUnique[t3 + 2];
      }
    }
    return triU;
  }

  getRestFrames(): Float32Array {
    return this.restFrames;
  }

  getGroupNodes(): Uint16Array[] {
    return this.groupNodes;
  }

  /**
   * Hydrates a DeformBinding directly from precomputed TypedArrays (either from the
   * IndexedDB binary cache or from skinningWorker.ts via Transferable buffers),
   * skipping all vertex welding, spatial hashing, and adjacency construction on the main thread.
   */
  bindPrecomputed(mesh: Mesh, data: PrecomputedBindingData): DeformBinding {
    const geo = mesh.geometry as BufferGeometry;
    ensurePlainAttributes(geo);
    if (!geo.getAttribute('normal')) geo.computeVertexNormals();
    const posAttr = geo.getAttribute('position') as BufferAttribute;
    const nrmAttr = geo.getAttribute('normal') as BufferAttribute;
    const pos = posAttr.array as Float32Array;
    const nrm = nrmAttr.array as Float32Array;
    const vertCount = posAttr.count;
    this.totalVehicleVerts += vertCount;
    const P = this.pool;

    const restPos = P.f32(vertCount * 3); restPos.set(pos);
    const restNormal = P.f32(vertCount * 3); restNormal.set(nrm);
    posAttr.setUsage(35048);
    nrmAttr.setUsage(35048);
    const triU = this.buildTriUniqueTable(geo, data.vertToUnique, vertCount);

    const binding: DeformBinding = {
      mesh,
      group: data.group,
      vertCount,
      uniqueCount: data.uniqueCount,
      vertToUnique: data.vertToUnique,
      restPos,
      restNormal,
      uniqueRest: data.uniqueRest,
      nodes: data.nodes,
      weights: data.weights,
      offsets: data.offsets,
      normalOffsets: data.normalOffsets,
      usedNodes: data.usedNodes,
      uniqueDirty: P.u8(data.uniqueCount),
      fixedAdj: data.fixedAdj.length > 0 ? data.fixedAdj : EMPTY_I32,
      fixedDeg: data.fixedDeg.length > 0 ? data.fixedDeg : EMPTY_U8,
      pos: P.f32(data.uniqueCount * 3),
      disp: P.f32(data.uniqueCount * 3),
      disp2: P.f32(data.uniqueCount * 3),
      triU,
      uniqueNrm: P.f32(data.uniqueCount * 3),
      uniqueCrease: P.f32(data.uniqueCount),
      maxDeformSq: 0,
      smoothIterations: data.smoothIterations,
      fastAnalyticalNormals: data.fastAnalyticalNormals,
    };
    this.bindings.push(binding);
    return binding;
  }

  /** Exports precomputed typed arrays for saving into IndexedDB (`FlexWebMeshCache`). */
  exportBindingsForCache(): PrecomputedBindingData[] {
    return this.bindings.map((b) => ({
      group: b.group,
      vertCount: b.vertCount,
      uniqueCount: b.uniqueCount,
      vertToUnique: b.vertToUnique,
      uniqueRest: b.uniqueRest,
      nodes: b.nodes,
      weights: b.weights,
      offsets: b.offsets,
      normalOffsets: b.normalOffsets,
      usedNodes: b.usedNodes,
      fixedAdj: b.fixedAdj,
      fixedDeg: b.fixedDeg,
      smoothIterations: b.smoothIterations,
      fastAnalyticalNormals: b.fastAnalyticalNormals,
    }));
  }

  private static addNeighborEdge(fixedAdj: Int32Array, fixedDeg: Uint8Array, a: number, b: number): void {
    if (a === b) return;
    const deg = fixedDeg[a];
    if (deg >= MAX_NEIGHBORS) return;
    const base = a * MAX_NEIGHBORS;
    for (let k = 0; k < deg; k++) if (fixedAdj[base + k] === b) return;
    fixedAdj[base + deg] = b;
    fixedDeg[a] = deg + 1;
  }

  markAllDirty(): void {
    this.groupDirty.fill(1);
    this.nodeDirty.fill(1);
  }

  /** Pre-warm V8 JIT on the skinning loop at vehicle load time so the first crash impact frame never stalls. */
  warmup(): void {
    this.markAllDirty();
    this.update(this.skinnedLocal);
    if (this.nodeCount > 1) {
      this.groupDirty.fill(1);
      this.nodeDirty.fill(0);
      this.nodeDirty[0] = 1;
      this.update(this.skinnedLocal);
    }
  }

  /**
   * Re-skin every group/sub-mesh whose bound nodes moved. `local` holds interpolated
   * node positions in body space (3 per node). Runs at most once per rendered frame.
   */
  update(local: Float32Array): void {
    const eps = this.dirtyEpsilon;
    let any = 0;
    for (let g = 0; g < this.groupNodes.length; g++) {
      const list = this.groupNodes[g];
      let grpMoved = this.groupDirty[g];
      for (let k = 0; k < list.length; k++) {
        const ni = list[k];
        const i = ni * 3;
        if (
          this.nodeDirty[ni] ||
          Math.abs(local[i] - this.skinnedLocal[i]) > eps ||
          Math.abs(local[i + 1] - this.skinnedLocal[i + 1]) > eps ||
          Math.abs(local[i + 2] - this.skinnedLocal[i + 2]) > eps
        ) {
          this.nodeDirty[ni] = 1;
          grpMoved = 1;
          this.skinnedLocal[i] = local[i];
          this.skinnedLocal[i + 1] = local[i + 1];
          this.skinnedLocal[i + 2] = local[i + 2];
        } else {
          this.nodeDirty[ni] = 0;
        }
      }
      this.groupDirty[g] = grpMoved;
      any |= grpMoved;
    }
    this.lastSkinnedVerts = 0;
    if (!any) return;

    this.computeFrames(local, this.frames);
    const F = this.frames;
    const nDirty = this.nodeDirty;

    for (let b = 0; b < this.bindings.length; b++) {
      const B = this.bindings[b];
      if (!this.groupDirty[B.group]) continue;

      const used = B.usedNodes;
      let meshMoved = 0;
      for (let i = 0; i < used.length; i++) {
        if (nDirty[used[i]]) { meshMoved = 1; break; }
      }
      if (!meshMoved) continue;

      if (B.fastAnalyticalNormals) {
        this.lastSkinnedVerts += this.skinBindingFast(B, local, F, nDirty);
      } else {
        this.lastSkinnedVerts += this.skinBindingSmooth(B, local, F);
      }
    }

    this.groupDirty.fill(0);
    this.nodeDirty.fill(0);
  }

  /**
   * Dedicated monomorphic TurboFan hot path for high-poly .glb meshes:
   * - Updates deformed vertex positions via 4-node frame skinning.
   * - Locally recomputes exact triangle face normals (`computeVertexNormals` targeted ONLY at
   *   affected deformed triangles) and blends them into crumpled regions so every dent, buckle,
   *   and metallic fold catches directional & sky light realistically at 60 FPS.
   */
  private skinBindingFast(B: DeformBinding, local: Float32Array, F: Float32Array, nDirty: Uint8Array): number {
    const nodes = B.nodes, weights = B.weights, offsets = B.offsets;
    const disp = B.disp, uniqueRest = B.uniqueRest, uniqueDirty = B.uniqueDirty;
    const crease = B.uniqueCrease;
    const uCount = B.uniqueCount;
    let dirtyUniqueCount = 0;
    let maxD2 = B.maxDeformSq;

    for (let u = 0; u < uCount; u++) {
      const u4 = u * 4;
      const n0 = nodes[u4], n1 = nodes[u4 + 1], n2 = nodes[u4 + 2], n3 = nodes[u4 + 3];
      const isDirty = nDirty[n0] | nDirty[n1] | nDirty[n2] | nDirty[n3];
      uniqueDirty[u] = isDirty;
      if (isDirty === 0) continue;
      dirtyUniqueCount++;

      const u12 = u * 12;
      const w0 = weights[u4], w1 = weights[u4 + 1], w2 = weights[u4 + 2], w3 = weights[u4 + 3];
      let x = 0, y = 0, z = 0;

      if (w0 > 0) {
        const f = n0 * 9, ni3 = n0 * 3;
        const ox = offsets[u12], oy = offsets[u12 + 1], oz = offsets[u12 + 2];
        x += w0 * (local[ni3] + F[f] * ox + F[f + 3] * oy + F[f + 6] * oz);
        y += w0 * (local[ni3 + 1] + F[f + 1] * ox + F[f + 4] * oy + F[f + 7] * oz);
        z += w0 * (local[ni3 + 2] + F[f + 2] * ox + F[f + 5] * oy + F[f + 8] * oz);
      }
      if (w1 > 0) {
        const f = n1 * 9, ni3 = n1 * 3;
        const ox = offsets[u12 + 3], oy = offsets[u12 + 4], oz = offsets[u12 + 5];
        x += w1 * (local[ni3] + F[f] * ox + F[f + 3] * oy + F[f + 6] * oz);
        y += w1 * (local[ni3 + 1] + F[f + 1] * ox + F[f + 4] * oy + F[f + 7] * oz);
        z += w1 * (local[ni3 + 2] + F[f + 2] * ox + F[f + 5] * oy + F[f + 8] * oz);
      }
      if (w2 > 0) {
        const f = n2 * 9, ni3 = n2 * 3;
        const ox = offsets[u12 + 6], oy = offsets[u12 + 7], oz = offsets[u12 + 8];
        x += w2 * (local[ni3] + F[f] * ox + F[f + 3] * oy + F[f + 6] * oz);
        y += w2 * (local[ni3 + 1] + F[f + 1] * ox + F[f + 4] * oy + F[f + 7] * oz);
        z += w2 * (local[ni3 + 2] + F[f + 2] * ox + F[f + 5] * oy + F[f + 8] * oz);
      }
      if (w3 > 0) {
        const f = n3 * 9, ni3 = n3 * 3;
        const ox = offsets[u12 + 9], oy = offsets[u12 + 10], oz = offsets[u12 + 11];
        x += w3 * (local[ni3] + F[f] * ox + F[f + 3] * oy + F[f + 6] * oz);
        y += w3 * (local[ni3 + 1] + F[f + 1] * ox + F[f + 4] * oy + F[f + 7] * oz);
        z += w3 * (local[ni3 + 2] + F[f + 2] * ox + F[f + 5] * oy + F[f + 8] * oz);
      }

      const u3 = u * 3;
      const dx = x - uniqueRest[u3];
      const dy = y - uniqueRest[u3 + 1];
      const dz = z - uniqueRest[u3 + 2];
      disp[u3] = dx;
      disp[u3 + 1] = dy;
      disp[u3 + 2] = dz;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > maxD2) maxD2 = d2;
      // Activate geometric crease normal blending once vertex displacement exceeds ~1.8 cm
      crease[u] = d2 > 0.00032 ? Math.min(0.88, (Math.sqrt(d2) - 0.018) * 7.5) : 0;
    }

    B.maxDeformSq = maxD2;
    if (dirtyUniqueCount === 0) return 0;

    // Targeted local computeVertexNormals strictly on triangles touching deformed vertices
    const recomputeLocalNormals = maxD2 > 0.00032 && B.triU.length > 0;
    const uNrm = B.uniqueNrm;
    if (recomputeLocalNormals) {
      for (let u = 0; u < uCount; u++) {
        if (crease[u] > 0) {
          const u3 = u * 3;
          uNrm[u3] = 0; uNrm[u3 + 1] = 0; uNrm[u3 + 2] = 0;
        }
      }
      const triU = B.triU;
      const tLen = triU.length;
      for (let t = 0; t < tLen; t += 3) {
        const a = triU[t], b = triU[t + 1], c = triU[t + 2];
        if (crease[a] === 0 && crease[b] === 0 && crease[c] === 0) continue;
        const a3 = a * 3, b3 = b * 3, c3 = c * 3;
        const ax = uniqueRest[a3] + disp[a3], ay = uniqueRest[a3 + 1] + disp[a3 + 1], az = uniqueRest[a3 + 2] + disp[a3 + 2];
        const e1x = (uniqueRest[b3] + disp[b3]) - ax;
        const e1y = (uniqueRest[b3 + 1] + disp[b3 + 1]) - ay;
        const e1z = (uniqueRest[b3 + 2] + disp[b3 + 2]) - az;
        const e2x = (uniqueRest[c3] + disp[c3]) - ax;
        const e2y = (uniqueRest[c3 + 1] + disp[c3 + 1]) - ay;
        const e2z = (uniqueRest[c3 + 2] + disp[c3 + 2]) - az;
        const fnx = e1y * e2z - e1z * e2y;
        const fny = e1z * e2x - e1x * e2z;
        const fnz = e1x * e2y - e1y * e2x;
        if (crease[a] > 0) { uNrm[a3] += fnx; uNrm[a3 + 1] += fny; uNrm[a3 + 2] += fnz; }
        if (crease[b] > 0) { uNrm[b3] += fnx; uNrm[b3 + 1] += fny; uNrm[b3 + 2] += fnz; }
        if (crease[c] > 0) { uNrm[c3] += fnx; uNrm[c3 + 1] += fny; uNrm[c3 + 2] += fnz; }
      }
      for (let u = 0; u < uCount; u++) {
        if (crease[u] > 0) {
          const u3 = u * 3;
          const l = Math.sqrt(uNrm[u3] * uNrm[u3] + uNrm[u3 + 1] * uNrm[u3 + 1] + uNrm[u3 + 2] * uNrm[u3 + 2]);
          if (l > 1e-8) {
            const il = 1 / l;
            uNrm[u3] *= il; uNrm[u3 + 1] *= il; uNrm[u3 + 2] *= il;
          }
        }
      }
    }

    const posAttr = B.mesh.geometry.getAttribute('position') as BufferAttribute;
    const nrmAttr = B.mesh.geometry.getAttribute('normal') as BufferAttribute;
    const out = posAttr.array as Float32Array;
    const nrmOut = nrmAttr.array as Float32Array;
    const map = B.vertToUnique;
    const nOff = B.normalOffsets;
    const vCount = B.vertCount;
    let touchedVerts = 0;

    for (let v = 0; v < vCount; v++) {
      const u = map[v];
      if (!uniqueDirty[u]) continue;
      touchedVerts++;
      const u3 = u * 3, v3 = v * 3;
      out[v3] = uniqueRest[u3] + disp[u3];
      out[v3 + 1] = uniqueRest[u3 + 1] + disp[u3 + 1];
      out[v3 + 2] = uniqueRest[u3 + 2] + disp[u3 + 2];

      const f = nodes[u * 4] * 9;
      const nox = nOff[v3], noy = nOff[v3 + 1], noz = nOff[v3 + 2];
      let nx = F[f] * nox + F[f + 3] * noy + F[f + 6] * noz;
      let ny = F[f + 1] * nox + F[f + 4] * noy + F[f + 7] * noz;
      let nz = F[f + 2] * nox + F[f + 5] * noy + F[f + 8] * noz;

      const cw = recomputeLocalNormals ? crease[u] : 0;
      if (cw > 0) {
        let gnx = uNrm[u3], gny = uNrm[u3 + 1], gnz = uNrm[u3 + 2];
        // Ensure geometric normal faces same hemisphere as analytical normal (handles double-sided/mirrored tris)
        if (gnx * nx + gny * ny + gnz * nz < 0) {
          gnx = -gnx; gny = -gny; gnz = -gnz;
        }
        nx = nx * (1 - cw) + gnx * cw;
        ny = ny * (1 - cw) + gny * cw;
        nz = nz * (1 - cw) + gnz * cw;
        const nl = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
        nx /= nl; ny /= nl; nz /= nl;
      }

      nrmOut[v3] = nx;
      nrmOut[v3 + 1] = ny;
      nrmOut[v3 + 2] = nz;
    }

    if (touchedVerts > 0) {
      posAttr.needsUpdate = true;
      nrmAttr.needsUpdate = true;
    }
    return touchedVerts;
  }

  /** Dedicated monomorphic TurboFan hot path for low-poly procedural panels (fixed-stride Laplacian smoothing + computeVertexNormals). */
  private skinBindingSmooth(B: DeformBinding, local: Float32Array, F: Float32Array): number {
    const nodes = B.nodes, weights = B.weights, offsets = B.offsets;
    const disp = B.disp, uniqueRest = B.uniqueRest;
    const uCount = B.uniqueCount;

    for (let u = 0; u < uCount; u++) {
      const u4 = u * 4, u12 = u * 12;
      let x = 0, y = 0, z = 0;
      for (let k = 0; k < 4; k++) {
        const w = weights[u4 + k];
        if (w === 0) continue;
        const ni = nodes[u4 + k];
        const f = ni * 9, ni3 = ni * 3, k3 = u12 + k * 3;
        const ox = offsets[k3], oy = offsets[k3 + 1], oz = offsets[k3 + 2];
        x += w * (local[ni3] + F[f] * ox + F[f + 3] * oy + F[f + 6] * oz);
        y += w * (local[ni3 + 1] + F[f + 1] * ox + F[f + 4] * oy + F[f + 7] * oz);
        z += w * (local[ni3 + 2] + F[f + 2] * ox + F[f + 5] * oy + F[f + 8] * oz);
      }
      const u3 = u * 3;
      disp[u3] = x - uniqueRest[u3];
      disp[u3 + 1] = y - uniqueRest[u3 + 1];
      disp[u3 + 2] = z - uniqueRest[u3 + 2];
    }

    let src = disp, dst = B.disp2;
    if (B.smoothIterations > 0 && B.fixedAdj.length > 0) {
      const lam = this.laplacianLambda;
      const adj = B.fixedAdj;
      const degArr = B.fixedDeg;
      for (let it = 0; it < B.smoothIterations; it++) {
        for (let u = 0; u < uCount; u++) {
          const deg = degArr[u];
          const u3 = u * 3;
          if (deg === 0) {
            dst[u3] = src[u3]; dst[u3 + 1] = src[u3 + 1]; dst[u3 + 2] = src[u3 + 2];
            continue;
          }
          let ax = 0, ay = 0, az = 0;
          const base = u * MAX_NEIGHBORS;
          for (let q = 0; q < deg; q++) {
            const nb = adj[base + q] * 3;
            ax += src[nb]; ay += src[nb + 1]; az += src[nb + 2];
          }
          const inv = 1 / deg;
          dst[u3] = src[u3] * (1 - lam) + ax * inv * lam;
          dst[u3 + 1] = src[u3 + 1] * (1 - lam) + ay * inv * lam;
          dst[u3 + 2] = src[u3 + 2] * (1 - lam) + az * inv * lam;
        }
        const t = src; src = dst; dst = t;
      }
    }

    const posAttr = B.mesh.geometry.getAttribute('position') as BufferAttribute;
    const out = posAttr.array as Float32Array;
    const map = B.vertToUnique;
    for (let v = 0; v < B.vertCount; v++) {
      const u3 = map[v] * 3, v3 = v * 3;
      out[v3] = uniqueRest[u3] + src[u3];
      out[v3 + 1] = uniqueRest[u3 + 1] + src[u3 + 1];
      out[v3 + 2] = uniqueRest[u3 + 2] + src[u3 + 2];
    }
    posAttr.needsUpdate = true;
    B.mesh.geometry.computeVertexNormals();
    return B.vertCount;
  }

  /** Restore every bound mesh to its baseline snapshot (Reset). */
  restore(restLocal: Float32Array | Float64Array): void {
    for (const B of this.bindings) {
      const geo = B.mesh.geometry;
      const p = geo.getAttribute('position') as BufferAttribute;
      const nrm = geo.getAttribute('normal') as BufferAttribute;
      (p.array as Float32Array).set(B.restPos);
      (nrm.array as Float32Array).set(B.restNormal);
      B.disp.fill(0);
      B.uniqueCrease.fill(0);
      B.maxDeformSq = 0;
      p.needsUpdate = true; nrm.needsUpdate = true;
    }
    for (let i = 0; i < this.nodeCount * 3; i++) this.skinnedLocal[i] = restLocal[i];
    this.frames.set(this.restFrames);
    this.groupDirty.fill(0);
    this.nodeDirty.fill(0);
  }

  get bytes(): number { return this.pool.totalBytes; }
}

/** Convert interleaved / quantised / indexed-uint8 attributes to plain Float32 so they can be rewritten. */
export function ensurePlainAttributes(geo: BufferGeometry): void {
  for (const name of ['position', 'normal']) {
    const attr = geo.getAttribute(name);
    if (!attr) continue;
    const plain = attr instanceof BufferAttribute && attr.array instanceof Float32Array && !attr.normalized;
    if (plain) continue;
    const arr = new Float32Array(attr.count * 3);
    for (let i = 0; i < attr.count; i++) { arr[i * 3] = attr.getX(i); arr[i * 3 + 1] = attr.getY(i); arr[i * 3 + 2] = attr.getZ(i); }
    geo.setAttribute(name, new BufferAttribute(arr, 3));
  }
}
