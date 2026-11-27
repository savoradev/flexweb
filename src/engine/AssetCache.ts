/**
 * AssetCache.ts
 * -------------
 * Versioned persistent local cache for heavy `.glb` vehicle models and 3D assets.
 * Uses the browser Cache Storage API (`caches.open('flexweb-vehicles-v1')`) as primary
 * storage with an automatic IndexedDB fallback when Cache Storage is unavailable
 * (e.g., non-HTTPS origins or restricted WebViews).
 *
 * All read/write operations are wrapped in try/catch blocks so storage quota limits
 * on mobile/tablet devices (such as 4 GB RAM tablets) never interrupt vehicle loading.
 */

export const MODEL_CACHE_VERSION = 'v2';
export const ASSET_CACHE_PREFIX = 'flexweb-vehicles-';
export const ASSET_CACHE_NAME = `${ASSET_CACHE_PREFIX}${MODEL_CACHE_VERSION}`;

const IDB_NAME = 'FlexWebAssetCache';
const IDB_VERSION = 1;
const IDB_STORE = 'assets';

interface CachedIdbEntry {
  key: string;
  version: string;
  buffer: ArrayBuffer;
  savedAt: number;
}

function isCacheStorageAvailable(): boolean {
  try {
    return typeof window !== 'undefined' && 'caches' in window && typeof window.caches?.open === 'function';
  } catch {
    return false;
  }
}

function buildCacheUrlKey(url: string): string {
  const sep = url.includes('?') ? '&' : '?';
  return `${url}${sep}fwv=${MODEL_CACHE_VERSION}`;
}

function buildIdbKey(url: string): string {
  return `${MODEL_CACHE_VERSION}:${url}`;
}

function openAssetDB(): Promise<IDBDatabase | null> {
  return new Promise((resolve) => {
    try {
      if (typeof indexedDB === 'undefined') {
        resolve(null);
        return;
      }
      const req = indexedDB.open(IDB_NAME, IDB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(IDB_STORE)) {
          db.createObjectStore(IDB_STORE, { keyPath: 'key' });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

async function readFromIdb(url: string): Promise<Uint8Array | null> {
  try {
    const db = await openAssetDB();
    if (!db) return null;
    return await new Promise<Uint8Array | null>((resolve) => {
      try {
        const tx = db.transaction(IDB_STORE, 'readonly');
        const req = tx.objectStore(IDB_STORE).get(buildIdbKey(url));
        req.onsuccess = () => {
          const entry = req.result as CachedIdbEntry | undefined;
          if (entry && entry.buffer instanceof ArrayBuffer && entry.buffer.byteLength >= 12) {
            resolve(new Uint8Array(entry.buffer));
          } else {
            resolve(null);
          }
        };
        req.onerror = () => resolve(null);
      } catch {
        resolve(null);
      }
    });
  } catch {
    return null;
  }
}

async function writeToIdb(url: string, bytes: Uint8Array): Promise<boolean> {
  try {
    const db = await openAssetDB();
    if (!db) return false;
    const copyBuf = bytes.slice().buffer;
    return await new Promise<boolean>((resolve) => {
      try {
        const tx = db.transaction(IDB_STORE, 'readwrite');
        const entry: CachedIdbEntry = {
          key: buildIdbKey(url),
          version: MODEL_CACHE_VERSION,
          buffer: copyBuf,
          savedAt: Date.now(),
        };
        tx.objectStore(IDB_STORE).put(entry);
        tx.oncomplete = () => resolve(true);
        tx.onerror = () => resolve(false);
        tx.onabort = () => resolve(false);
      } catch {
        resolve(false);
      }
    });
  } catch {
    return false;
  }
}

/**
 * Reads a cached `.glb` / 3D binary asset by URL + `MODEL_CACHE_VERSION`.
 * Returns `null` on cache miss or if storage is unavailable. Never throws.
 */
export async function getCachedAssetBytes(url: string): Promise<Uint8Array | null> {
  if (!url) return null;

  // 1. Primary: Browser Cache Storage API (`caches.open('flexweb-vehicles-v1')`)
  if (isCacheStorageAvailable()) {
    try {
      const cache = await window.caches.open(ASSET_CACHE_NAME);
      const match = await cache.match(buildCacheUrlKey(url));
      if (match && match.ok) {
        const ab = await match.arrayBuffer();
        if (ab.byteLength >= 12) {
          return new Uint8Array(ab);
        }
      }
    } catch {
      // Fall through cleanly to IndexedDB fallback
    }
  }

  // 2. Fallback: IndexedDB
  return readFromIdb(url);
}

/**
 * Stores a cloned copy of a fetched `.glb` / 3D binary asset in persistent local cache.
 * Gracefully absorbs quota errors on mobile/tablet devices without throwing.
 */
export async function putCachedAssetBytes(
  url: string,
  bytes: Uint8Array,
  contentType = 'model/gltf-binary',
): Promise<boolean> {
  if (!url || !bytes || bytes.byteLength < 12) return false;

  // 1. Primary: Browser Cache Storage API
  if (isCacheStorageAvailable()) {
    try {
      const cache = await window.caches.open(ASSET_CACHE_NAME);
      const clonedCopy = bytes.slice();
      const response = new Response(clonedCopy, {
        status: 200,
        statusText: 'OK',
        headers: {
          'Content-Type': contentType,
          'Content-Length': String(clonedCopy.byteLength),
          'X-FlexWeb-Cache-Version': MODEL_CACHE_VERSION,
        },
      });
      await cache.put(buildCacheUrlKey(url), response);
      return true;
    } catch {
      // Storage quota exceeded or Cache Storage blocked — try IndexedDB fallback
    }
  }

  // 2. Fallback: IndexedDB
  return writeToIdb(url, bytes);
}

/**
 * Purges obsolete cache versions (e.g. `flexweb-vehicles-v1` when upgraded to `v2`)
 * across both Cache Storage and IndexedDB so stale vehicle assets never waste tablet storage.
 */
export async function clearOldAssetCaches(): Promise<number> {
  let removed = 0;

  if (isCacheStorageAvailable()) {
    try {
      const names = await window.caches.keys();
      for (let i = 0; i < names.length; i++) {
        const name = names[i];
        if (name.startsWith(ASSET_CACHE_PREFIX) && name !== ASSET_CACHE_NAME) {
          const ok = await window.caches.delete(name);
          if (ok) removed++;
        }
      }
    } catch {
      // ignore storage restrictions
    }
  }

  try {
    const db = await openAssetDB();
    if (db) {
      await new Promise<void>((resolve) => {
        try {
          const tx = db.transaction(IDB_STORE, 'readwrite');
          const store = tx.objectStore(IDB_STORE);
          const cursorReq = store.openCursor();
          cursorReq.onsuccess = () => {
            const cursor = cursorReq.result;
            if (!cursor) return;
            const val = cursor.value as CachedIdbEntry | undefined;
            if (!val || val.version !== MODEL_CACHE_VERSION) {
              cursor.delete();
              removed++;
            }
            cursor.continue();
          };
          tx.oncomplete = () => resolve();
          tx.onerror = () => resolve();
          tx.onabort = () => resolve();
        } catch {
          resolve();
        }
      });
    }
  } catch {
    // ignore storage restrictions
  }

  return removed;
}
