/**
 * SpatialHashGrid
 * ---------------
 * Zero-allocation 2D XZ Compressed-Sparse-Row (CSR) Spatial Hash Grid for the
 * 2 km × 2 km (4,000,000 m²) Proving Ground.
 *
 * - Partitioning [-1152 m, +1152 m] into 16 m × 16 m cells (144 × 144 = 20,736 cells).
 * - Broadphase queries run in O(1) per cell overlapped by the vehicle AABB
 *   (typically 1–4 cells) with integer stamp deduplication — zero heap allocations,
 *   making map size completely invisible to CPU frame time.
 */

export interface SpatialBoundsItem {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
}

export class SpatialHashGrid {
  readonly cellSize: number;
  readonly invCellSize: number;
  readonly halfExtent: number;
  readonly dim: number;
  readonly cellStart: Int32Array;
  readonly cellCount: Uint16Array;
  readonly cellItems: Uint16Array;
  private readonly itemStamp: Int32Array;
  private stampCounter = 1;

  constructor(items: SpatialBoundsItem[], cellSize = 16, halfExtent = 1152) {
    this.cellSize = cellSize;
    this.invCellSize = 1 / cellSize;
    this.halfExtent = halfExtent;
    const dim = Math.ceil((halfExtent * 2) / cellSize);
    this.dim = dim;
    const totalCells = dim * dim;

    this.cellStart = new Int32Array(totalCells);
    this.cellCount = new Uint16Array(totalCells);
    this.itemStamp = new Int32Array(Math.max(1, items.length));

    // Pass 1: count items per cell
    let totalRefs = 0;
    for (let i = 0; i < items.length; i++) {
      const it = items[i];
      const cx0 = this.clampCell(it.minX);
      const cx1 = this.clampCell(it.maxX);
      const cz0 = this.clampCell(it.minZ);
      const cz1 = this.clampCell(it.maxZ);
      for (let cz = cz0; cz <= cz1; cz++) {
        const row = cz * dim;
        for (let cx = cx0; cx <= cx1; cx++) {
          this.cellCount[row + cx]++;
          totalRefs++;
        }
      }
    }

    // Pass 2: prefix sum offsets
    let offset = 0;
    for (let c = 0; c < totalCells; c++) {
      this.cellStart[c] = offset;
      offset += this.cellCount[c];
    }

    // Pass 3: populate contiguous item indices
    this.cellItems = new Uint16Array(Math.max(1, totalRefs));
    const cursor = new Uint16Array(totalCells);
    for (let i = 0; i < items.length; i++) {
      const it = items[i];
      const cx0 = this.clampCell(it.minX);
      const cx1 = this.clampCell(it.maxX);
      const cz0 = this.clampCell(it.minZ);
      const cz1 = this.clampCell(it.maxZ);
      for (let cz = cz0; cz <= cz1; cz++) {
        const row = cz * dim;
        for (let cx = cx0; cx <= cx1; cx++) {
          const cell = row + cx;
          this.cellItems[this.cellStart[cell] + cursor[cell]++] = i;
        }
      }
    }
  }

  private clampCell(coord: number): number {
    const c = Math.floor((coord + this.halfExtent) * this.invCellSize);
    return c < 0 ? 0 : c >= this.dim ? this.dim - 1 : c;
  }

  /** Fast O(1) single-point cell lookup (returns -1 if outside grid bounds). */
  cellAt(x: number, z: number): number {
    const cx = Math.floor((x + this.halfExtent) * this.invCellSize);
    const cz = Math.floor((z + this.halfExtent) * this.invCellSize);
    if (cx < 0 || cx >= this.dim || cz < 0 || cz >= this.dim) return -1;
    return cz * this.dim + cx;
  }

  /**
   * Queries all unique item indices overlapping `[minX..maxX] × [minZ..maxZ]`
   * into pre-allocated `out` buffer. Zero heap allocations.
   */
  queryAABB(minX: number, minZ: number, maxX: number, maxZ: number, out: Int32Array, maxOut: number): number {
    const cx0 = this.clampCell(minX);
    const cx1 = this.clampCell(maxX);
    const cz0 = this.clampCell(minZ);
    const cz1 = this.clampCell(maxZ);

    let stamp = ++this.stampCounter;
    if (stamp >= 0x7ffffff0) {
      this.itemStamp.fill(0);
      this.stampCounter = 1;
      stamp = 1;
    }

    const dim = this.dim;
    const starts = this.cellStart;
    const counts = this.cellCount;
    const items = this.cellItems;
    const stamps = this.itemStamp;
    let found = 0;

    for (let cz = cz0; cz <= cz1; cz++) {
      const row = cz * dim;
      for (let cx = cx0; cx <= cx1; cx++) {
        const cell = row + cx;
        const start = starts[cell];
        const end = start + counts[cell];
        for (let k = start; k < end; k++) {
          const id = items[k];
          if (stamps[id] === stamp) continue;
          stamps[id] = stamp;
          out[found++] = id;
          if (found >= maxOut) return found;
        }
      }
    }
    return found;
  }
}
