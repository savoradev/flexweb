/**
 * MeshCacheDB
 * -----------
 * Lightweight IndexedDB binary cache (`FlexWebMeshCache`) storing precomputed
 * MeshDeformer typed arrays (skinning node indices, weights, rest-frame offsets,
 * normal offsets, and Laplacian adjacency tables) keyed by vehicle ID + size.
 *
 * Subsequent loads of any .GLB vehicle bypass all geometry/skinning math and
 * hydrate MeshDeformer directly from structured-cloned TypedArrays in < 200 ms.
 */

const DB_NAME = 'FlexWebMeshCache';
const DB_VERSION = 1;
const STORE_NAME = 'skinning';
export const SKINNING_CACHE_SCHEMA_VERSION = 3;

export interface PrecomputedBindingData {
  group: number;
  vertCount: number;
  uniqueCount: number;
  vertToUnique: Uint32Array;
  uniqueRest: Float32Array;
  nodes: Uint16Array;
  weights: Float32Array;
  offsets: Float32Array;
  normalOffsets: Float32Array;
  usedNodes: Uint16Array;
  fixedAdj: Int32Array;
  fixedDeg: Uint8Array;
  smoothIterations: number;
  fastAnalyticalNormals: boolean;
}

export interface CachedVehicleSkinning {
  key: string;
  schemaVersion: number;
  nodeCount: number;
  totalVerts: number;
  savedAt: number;
  bindings: PrecomputedBindingData[];
}

function openCacheDB(): Promise<IDBDatabase | null> {
  if (typeof indexedDB === 'undefined') return Promise.resolve(null);
  return new Promise((resolve) => {
    try {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME, { keyPath: 'key' });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

export async function loadCachedSkinning(key: string): Promise<CachedVehicleSkinning | null> {
  const db = await openCacheDB();
  if (!db) return null;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const req = tx.objectStore(STORE_NAME).get(key);
      req.onsuccess = () => {
        const rec = req.result as CachedVehicleSkinning | undefined;
        db.close();
        if (!rec || rec.schemaVersion !== SKINNING_CACHE_SCHEMA_VERSION) {
          resolve(null);
          return;
        }
        resolve(rec);
      };
      req.onerror = () => { db.close(); resolve(null); };
    } catch {
      db.close();
      resolve(null);
    }
  });
}

export async function saveCachedSkinning(record: CachedVehicleSkinning): Promise<void> {
  const db = await openCacheDB();
  if (!db) return;
  return new Promise((resolve) => {
    try {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).put(record);
      tx.oncomplete = () => { db.close(); resolve(); };
      tx.onerror = () => { db.close(); resolve(); };
    } catch {
      db.close();
      resolve();
    }
  });
}
