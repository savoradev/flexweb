/**
 * SavedCarsDB
 * -----------
 * Lightweight IndexedDB store (`FlexWebSavedCars`) that persists uploaded `.glb`
 * vehicle models (`ArrayBuffer` + metadata) across page reloads without keeping
 * inactive vehicle buffers in RAM (strict 4 GB RAM Samsung Galaxy Tab A9+ budget).
 */

const DB_NAME = 'FlexWebSavedCars';
const DB_VERSION = 1;
const STORE_CARS = 'cars';
const LAST_CAR_KEY = 'flexweb_selected_car';
const LEGACY_LAST_CAR_KEY = 'flexweb:lastSelectedCarId';

export interface SavedCarMetadata {
  id: string;
  name: string;
  bytes: number;
  createdAt: number;
  savedAt?: number;
}

interface SavedCarRecord extends SavedCarMetadata {
  buffer: ArrayBuffer;
}

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') {
      reject(new Error('IndexedDB is not available in this environment'));
      return;
    }
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_CARS)) {
        db.createObjectStore(STORE_CARS, { keyPath: 'id' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error('Failed to open FlexWebSavedCars IndexedDB'));
  });
}

function cleanCarName(fileName: string): string {
  return fileName
    .replace(/\.(glb|gltf)$/i, '')
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace(/\bMclaren\b/i, 'McLaren')
    .replace(/\bP1\b/i, 'P1')
    .replace(/\bAmg\b/i, 'AMG')
    .replace(/\bGt\b/i, 'GT');
}

/**
 * Persists a cloned copy of `rawBuffer` (`rawBuffer.slice(0)`) into `FlexWebSavedCars`
 * BEFORE anything touches GLTFLoader or Web Workers, guaranteeing the stored ArrayBuffer
 * is never detached and survives all browser reloads.
 */
export async function saveCarBufferToDB(
  fileName: string,
  rawBuffer: ArrayBuffer,
): Promise<{ meta: SavedCarMetadata; buffer: ArrayBuffer }> {
  const savedBuffer = rawBuffer.slice(0);
  const callerBuffer = rawBuffer.slice(0);
  const name = cleanCarName(fileName) || 'Custom Car';
  const slug = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  const id = `idb-car-${slug || Date.now()}`;
  const now = Date.now();
  const meta: SavedCarMetadata = {
    id,
    name,
    bytes: savedBuffer.byteLength,
    createdAt: now,
    savedAt: now,
  };
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_CARS, 'readwrite');
    const store = tx.objectStore(STORE_CARS);
    const record: SavedCarRecord = { ...meta, buffer: savedBuffer };
    store.put(record);
    tx.oncomplete = () => {
      db.close();
      resolve({ meta, buffer: callerBuffer });
    };
    tx.onerror = () => {
      db.close();
      reject(tx.error ?? new Error('Failed to store car in IndexedDB'));
    };
  });
}

/**
 * Saves an uploaded `.glb` file (`ArrayBuffer` + name) into `FlexWebSavedCars`.
 * Reuses the existing ID if a car with the same clean name was already uploaded.
 */
export async function saveUploadedCarGLB(file: File): Promise<{ meta: SavedCarMetadata; buffer: ArrayBuffer }> {
  const rawBuffer = await file.arrayBuffer();
  return saveCarBufferToDB(file.name, rawBuffer);
}

/**
 * Lists metadata (`id`, `name`, `bytes`, `createdAt`) for all saved cars in
 * `FlexWebSavedCars` without keeping their `ArrayBuffer` payloads in RAM.
 */
export async function listSavedCarsMetadata(): Promise<SavedCarMetadata[]> {
  try {
    const db = await openDB();
    return await new Promise<SavedCarMetadata[]>((resolve, reject) => {
      const tx = db.transaction(STORE_CARS, 'readonly');
      const store = tx.objectStore(STORE_CARS);
      const out: SavedCarMetadata[] = [];
      const cursorReq = store.openCursor();
      cursorReq.onsuccess = () => {
        const cursor = cursorReq.result;
        if (cursor) {
          const val = cursor.value as SavedCarRecord;
          out.push({
            id: val.id,
            name: val.name,
            bytes: val.bytes,
            createdAt: val.createdAt,
          });
          cursor.continue();
        } else {
          out.sort((a, b) => a.createdAt - b.createdAt);
          resolve(out);
        }
      };
      cursorReq.onerror = () => reject(cursorReq.error);
      tx.oncomplete = () => db.close();
    });
  } catch {
    return [];
  }
}

/** Loads the `ArrayBuffer` for a single saved car ID on demand. */
export async function loadSavedCarBuffer(id: string): Promise<ArrayBuffer | null> {
  try {
    const db = await openDB();
    return await new Promise<ArrayBuffer | null>((resolve, reject) => {
      const tx = db.transaction(STORE_CARS, 'readonly');
      const store = tx.objectStore(STORE_CARS);
      const req = store.get(id);
      req.onsuccess = () => {
        const rec = req.result as SavedCarRecord | undefined;
        resolve(rec?.buffer ?? null);
      };
      req.onerror = () => reject(req.error);
      tx.oncomplete = () => db.close();
    });
  } catch {
    return null;
  }
}

/** Deletes a saved car from `FlexWebSavedCars` by ID. */
export async function deleteSavedCarGLB(id: string): Promise<void> {
  try {
    const db = await openDB();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE_CARS, 'readwrite');
      const store = tx.objectStore(STORE_CARS);
      store.delete(id);
      tx.oncomplete = () => {
        db.close();
        resolve();
      };
      tx.onerror = () => {
        db.close();
        reject(tx.error);
      };
    });
    if (getLastSelectedCarId() === id) {
      setLastSelectedCarId(null);
    }
  } catch {
    // ignore if IndexedDB unavailable
  }
}

export function getLastSelectedCarId(): string | null {
  try {
    return localStorage.getItem(LAST_CAR_KEY) || localStorage.getItem(LEGACY_LAST_CAR_KEY);
  } catch {
    return null;
  }
}

export function setLastSelectedCarId(id: string | null): void {
  try {
    if (id) {
      localStorage.setItem(LAST_CAR_KEY, id);
      localStorage.removeItem(LEGACY_LAST_CAR_KEY);
    } else {
      localStorage.removeItem(LAST_CAR_KEY);
      localStorage.removeItem(LEGACY_LAST_CAR_KEY);
    }
  } catch {
    // ignore storage errors
  }
}
