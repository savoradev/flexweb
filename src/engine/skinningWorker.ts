import type { PrecomputedBindingData } from './MeshCacheDB';

/**
 * skinningWorker
 * --------------
 * Dedicated Web Worker offloading MeshDeformer skinning weight computation
 * (O(V) 3D spatial grid hashing for top-4 nearest soft-body nodes, rest-frame
 * offsets, analytical normal offsets) and Laplacian neighbor adjacency table
 * construction off the main UI thread.
 *
 * Uses Transferable ArrayBuffers for zero-copy transfer back to the main thread.
 */

export interface SkinningWorkerPartInput {
  group: number;
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint16Array | Uint32Array | null;
}

export interface SkinningWorkerRequest {
  id: number;
  restLocal: Float32Array | Float64Array;
  restFrames: Float32Array;
  groupNodes: Uint16Array[];
  parts: SkinningWorkerPartInput[];
}

export interface SkinningWorkerResponse {
  id: number;
  totalVerts: number;
  bindings: PrecomputedBindingData[];
}

const MAX_NEIGHBORS = 6;

function addNeighborEdge(fixedAdj: Int32Array, fixedDeg: Uint8Array, a: number, b: number): void {
  if (a === b) return;
  const deg = fixedDeg[a];
  if (deg >= MAX_NEIGHBORS) return;
  const base = a * MAX_NEIGHBORS;
  for (let k = 0; k < deg; k++) if (fixedAdj[base + k] === b) return;
  fixedAdj[base + deg] = b;
  fixedDeg[a] = deg + 1;
}

function computePartBinding(
  part: SkinningWorkerPartInput,
  totalVehicleVerts: number,
  restLocal: Float32Array | Float64Array,
  restFrames: Float32Array,
  groupNodes: Uint16Array[],
): PrecomputedBindingData {
  const pos = part.positions;
  const nrm = part.normals;
  const group = part.group;
  const vertCount = Math.floor(pos.length / 3);

  // Weld coincident vertices using 48-bit packed integer key
  const vertToUnique = new Uint32Array(vertCount);
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
  const uniqueRest = uniqueTemp.slice(0, uniqueCount * 3);

  const nodesIdx = new Uint16Array(uniqueCount * 4);
  const weights = new Float32Array(uniqueCount * 4);
  const offsets = new Float32Array(uniqueCount * 12);
  const candidates = groupNodes[group].length >= 1 ? groupNodes[group] : groupNodes[0];
  const bestD = new Float64Array(5);
  const bestI = new Int32Array(5);
  const usedSet = new Set<number>();

  // O(V) 3D spatial hash grid over candidate nodes
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
      offsets[u * 12 + k * 3] = restFrames[f] * ox + restFrames[f + 1] * oy + restFrames[f + 2] * oz;
      offsets[u * 12 + k * 3 + 1] = restFrames[f + 3] * ox + restFrames[f + 4] * oy + restFrames[f + 5] * oz;
      offsets[u * 12 + k * 3 + 2] = restFrames[f + 6] * ox + restFrames[f + 7] * oy + restFrames[f + 8] * oz;
    }
  }

  const usedNodes = new Uint16Array(Array.from(usedSet));

  const normalOffsets = new Float32Array(vertCount * 3);
  for (let v = 0; v < vertCount; v++) {
    const u = vertToUnique[v];
    const nx = nrm[v * 3], ny = nrm[v * 3 + 1], nz = nrm[v * 3 + 2];
    const f = nodesIdx[u * 4] * 9;
    normalOffsets[v * 3] = restFrames[f] * nx + restFrames[f + 1] * ny + restFrames[f + 2] * nz;
    normalOffsets[v * 3 + 1] = restFrames[f + 3] * nx + restFrames[f + 4] * ny + restFrames[f + 5] * nz;
    normalOffsets[v * 3 + 2] = restFrames[f + 6] * nx + restFrames[f + 7] * ny + restFrames[f + 8] * nz;
  }

  let fixedAdj = new Int32Array(0);
  let fixedDeg = new Uint8Array(0);
  let smoothIterations = 0;
  if (uniqueCount <= 15000 && totalVehicleVerts <= 15000) {
    fixedAdj = new Int32Array(uniqueCount * MAX_NEIGHBORS);
    fixedAdj.fill(-1);
    fixedDeg = new Uint8Array(uniqueCount);
    const idx = part.indices;
    const triCount = idx ? Math.floor(idx.length / 3) : Math.floor(vertCount / 3);
    for (let t = 0; t < triCount; t++) {
      const a = vertToUnique[idx ? idx[t * 3] : t * 3];
      const b = vertToUnique[idx ? idx[t * 3 + 1] : t * 3 + 1];
      const c = vertToUnique[idx ? idx[t * 3 + 2] : t * 3 + 2];
      addNeighborEdge(fixedAdj, fixedDeg, a, b);
      addNeighborEdge(fixedAdj, fixedDeg, b, a);
      addNeighborEdge(fixedAdj, fixedDeg, b, c);
      addNeighborEdge(fixedAdj, fixedDeg, c, b);
      addNeighborEdge(fixedAdj, fixedDeg, a, c);
      addNeighborEdge(fixedAdj, fixedDeg, c, a);
    }
    smoothIterations = 1;
  }

  return {
    group,
    vertCount,
    uniqueCount,
    vertToUnique,
    uniqueRest,
    nodes: nodesIdx,
    weights,
    offsets,
    normalOffsets,
    usedNodes,
    fixedAdj,
    fixedDeg,
    smoothIterations,
    fastAnalyticalNormals: totalVehicleVerts > 15000 || vertCount > 3500,
  };
}

self.onmessage = (ev: MessageEvent<SkinningWorkerRequest>) => {
  const req = ev.data;
  let totalVerts = 0;
  for (let i = 0; i < req.parts.length; i++) {
    totalVerts += Math.floor(req.parts[i].positions.length / 3);
  }

  const bindings: PrecomputedBindingData[] = [];
  const transferables: Transferable[] = [];
  for (let i = 0; i < req.parts.length; i++) {
    const b = computePartBinding(req.parts[i], totalVerts, req.restLocal, req.restFrames, req.groupNodes);
    bindings.push(b);
    transferables.push(
      b.vertToUnique.buffer,
      b.uniqueRest.buffer,
      b.nodes.buffer,
      b.weights.buffer,
      b.offsets.buffer,
      b.normalOffsets.buffer,
      b.usedNodes.buffer,
    );
    if (b.fixedAdj.byteLength > 0) transferables.push(b.fixedAdj.buffer);
    if (b.fixedDeg.byteLength > 0) transferables.push(b.fixedDeg.buffer);
  }

  const resp: SkinningWorkerResponse = { id: req.id, totalVerts, bindings };
  (self as unknown as Worker).postMessage(resp, transferables);
};
