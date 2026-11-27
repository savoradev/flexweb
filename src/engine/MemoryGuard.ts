import { Matrix4, Quaternion, Vector3 } from 'three';

/**
 * MemoryGuard
 * -----------
 * Central allocator for every typed array the simulation needs. All buffers are
 * allocated once (when a vehicle is built) and reused forever, so `step()` and
 * `render()` never touch the JS heap. Baseline snapshots let Reset/Respawn
 * restore state with bulk `TypedArray.set()` copies (well under 2 ms).
 */

type AnyTyped = Float64Array | Float32Array | Uint32Array | Uint16Array | Uint8Array | Int32Array;

export class TypedArrayPool {
  private bytes = 0;
  private readonly owned: AnyTyped[] = [];

  f64(n: number): Float64Array { return this.track(new Float64Array(n)); }
  f32(n: number): Float32Array { return this.track(new Float32Array(n)); }
  u32(n: number): Uint32Array { return this.track(new Uint32Array(n)); }
  u16(n: number): Uint16Array { return this.track(new Uint16Array(n)); }
  u8(n: number): Uint8Array { return this.track(new Uint8Array(n)); }
  i32(n: number): Int32Array { return this.track(new Int32Array(n)); }

  /** Index buffer sized for the vertex count (Uint16 when it fits). */
  index(n: number, maxIndex: number): Uint16Array | Uint32Array {
    return maxIndex < 65535 ? this.u16(n) : this.u32(n);
  }

  get totalBytes(): number { return this.bytes; }

  release(): void {
    this.owned.length = 0;
    this.bytes = 0;
  }

  private track<T extends AnyTyped>(a: T): T {
    this.bytes += a.byteLength;
    this.owned.push(a);
    return a;
  }
}

/** Ring of pre-built Vector3 objects for temporary math. Call `reset()` once per frame. */
export class Vec3Pool {
  private readonly items: Vector3[];
  private cursor = 0;
  constructor(size = 64) {
    this.items = new Array(size);
    for (let i = 0; i < size; i++) this.items[i] = new Vector3();
  }
  get(): Vector3 {
    const v = this.items[this.cursor];
    this.cursor = (this.cursor + 1) % this.items.length;
    return v;
  }
  reset(): void { this.cursor = 0; }
}

/** Pairs of (live, baseline) arrays restored together on Reset. */
export class BaselineSnapshot {
  private readonly live: AnyTyped[] = [];
  private readonly base: AnyTyped[] = [];

  /** Registers `arr` and stores a copy of its current contents as the baseline. */
  track(arr: AnyTyped): void {
    this.live.push(arr);
    this.base.push(arr.slice() as AnyTyped);
  }

  /** Re-captures the current values of every tracked array as the new baseline. */
  recapture(): void {
    for (let i = 0; i < this.live.length; i++) (this.base[i] as Float64Array).set(this.live[i] as Float64Array);
  }

  restore(): void {
    for (let i = 0; i < this.live.length; i++) (this.live[i] as Float64Array).set(this.base[i] as Float64Array);
  }

  get bytes(): number {
    let b = 0;
    for (let i = 0; i < this.base.length; i++) b += this.base[i].byteLength;
    return b;
  }
}

/** Shared scratch objects for code paths that need three.js math types. */
export const Scratch = {
  v0: new Vector3(), v1: new Vector3(), v2: new Vector3(), v3: new Vector3(),
  v4: new Vector3(), v5: new Vector3(), v6: new Vector3(), v7: new Vector3(),
  q0: new Quaternion(), q1: new Quaternion(),
  m0: new Matrix4(), m1: new Matrix4(),
  pool: new Vec3Pool(96),
};

/** Reports the JS heap when the browser exposes it (Chromium only). */
export function heapUsedMB(): number {
  const perf = performance as Performance & { memory?: { usedJSHeapSize: number } };
  return perf.memory ? perf.memory.usedJSHeapSize / 1048576 : -1;
}

/** Heuristic for low-memory / tablet class devices. */
export function isConstrainedDevice(): boolean {
  if (typeof navigator === 'undefined') return false;
  const nav = navigator as Navigator & { deviceMemory?: number };
  const touch = (navigator.maxTouchPoints ?? 0) > 1;
  const lowMem = nav.deviceMemory !== undefined && nav.deviceMemory <= 4;
  return touch || lowMem;
}

/**
 * Bounded LRU in-memory vehicle template cache.
 * Keeps up to `maxEntries` (default: 2 on low-RAM mobile/tablets, 4 on desktop)
 * parsed & pre-bound vehicles ready for < 5 ms instant switching while calling
 * `.dispose()` on evicted entries so 4 GB RAM devices never hit OOM.
 */
export class BoundedVehicleTemplateCache<T extends { dispose(): void }> {
  private readonly entries = new Map<string, T>();
  readonly maxEntries: number;

  constructor(maxEntries?: number) {
    this.maxEntries = maxEntries ?? (isConstrainedDevice() ? 2 : 4);
  }

  get(key: string): T | undefined {
    const hit = this.entries.get(key);
    if (!hit) return undefined;
    // Refresh LRU recency
    this.entries.delete(key);
    this.entries.set(key, hit);
    return hit;
  }

  has(key: string): boolean {
    return this.entries.has(key);
  }

  /** Removes an entry from the cache WITHOUT disposing it (e.g. when activating it in the scene). */
  take(key: string): T | undefined {
    const hit = this.entries.get(key);
    if (hit) this.entries.delete(key);
    return hit;
  }

  /** Parks a vehicle template in the cache; evicts & disposes the least-recently-used entry if full. */
  set(key: string, value: T): void {
    const existing = this.entries.get(key);
    if (existing && existing !== value) {
      existing.dispose();
    }
    this.entries.delete(key);
    this.entries.set(key, value);
    while (this.entries.size > this.maxEntries) {
      const oldestKey = this.entries.keys().next().value;
      if (oldestKey === undefined) break;
      const oldest = this.entries.get(oldestKey);
      this.entries.delete(oldestKey);
      oldest?.dispose();
    }
  }

  clear(): void {
    for (const v of this.entries.values()) {
      try { v.dispose(); } catch { /* ignore */ }
    }
    this.entries.clear();
  }

  get size(): number {
    return this.entries.size;
  }
}
