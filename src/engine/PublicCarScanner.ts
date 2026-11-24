import virtualCars, { CarModelEntry } from 'virtual:flexweb-cars';

/**
 * PublicCarScanner
 * ----------------
 * Discovers `.glb` / `.gltf` vehicle models from `public/` and `public/models/`
 * via `virtual:flexweb-cars` plus live `GET /api/public-cars` & `GET /api/upload-car`
 * disk queries (without Vite `import.meta.glob` watching `public/models/`, so writing
 * a `.glb` to `public/models/` never triggers a dev-server restart or HMR reload).
 */

export function formatPublicCarName(relPath: string): string {
  const parts = relPath.split(/[\\/]/);
  const file = parts[parts.length - 1];
  const base = file.replace(/\.(glb|gltf)$/i, '');
  const raw = /^(scene|model|car|vehicle)$/i.test(base) && parts.length >= 2
    ? parts[parts.length - 2]
    : base;
  return raw
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace(/\bMclaren\b/i, 'McLaren')
    .replace(/\bP1\b/i, 'P1')
    .replace(/\bAmg\b/i, 'AMG')
    .replace(/\bGt\b/i, 'GT');
}

/** Ensures Vite public asset URLs never contain a `/public/` prefix (e.g. `/models/car.glb`). */
export function normalizePublicModelUrl(rawUrl: string): string {
  const cleaned = rawUrl.replace(/^\/?public\//i, '/');
  return cleaned.startsWith('/') ? cleaned : `/${cleaned}`;
}

export function discoverPublicCars(): CarModelEntry[] {
  return virtualCars.map((vc) => ({
    id: vc.id.replace(/^\/?public\//i, ''),
    name: vc.name || formatPublicCarName(vc.id),
    url: normalizePublicModelUrl(vc.url || vc.id),
    bytes: vc.bytes,
    mtimeMs: vc.mtimeMs,
  }));
}

/** Fetches the live list of `.glb` / `.gltf` files on disk inside `public/` and `public/models/`. */
export async function fetchPublicCarsFromServer(): Promise<CarModelEntry[]> {
  const byId = new Map<string, CarModelEntry>();
  for (const vc of discoverPublicCars()) byId.set(vc.id, vc);

  try {
    const [allRes, modelsRes] = await Promise.all([
      fetch('/api/public-cars', { cache: 'no-store' }).catch(() => null),
      fetch('/api/list-models', { cache: 'no-store' }).catch(() => null),
    ]);

    if (allRes && allRes.ok) {
      const live = (await allRes.json()) as CarModelEntry[];
      if (Array.isArray(live)) {
        byId.clear();
        for (const c of live) {
          const cleanId = c.id.replace(/^\/?public\//i, '');
          byId.set(cleanId, {
            id: cleanId,
            name: c.name || formatPublicCarName(cleanId),
            url: normalizePublicModelUrl(c.url || cleanId),
            bytes: c.bytes,
            mtimeMs: c.mtimeMs,
          });
        }
      }
    }

    if (modelsRes && modelsRes.ok) {
      const uploaded = (await modelsRes.json()) as CarModelEntry[];
      if (Array.isArray(uploaded)) {
        for (const c of uploaded) {
          const cleanId = c.id.replace(/^\/?public\//i, '');
          byId.set(cleanId, {
            id: cleanId,
            name: c.name || formatPublicCarName(cleanId),
            url: normalizePublicModelUrl(c.url || cleanId),
            bytes: c.bytes,
            mtimeMs: c.mtimeMs,
          });
        }
      }
    }
  } catch (err) {
    console.warn('[FlexWeb] Falling back to static public/ car list:', err);
  }
  return Array.from(byId.values());
}

/**
 * Persists an uploaded `.glb` / `.gltf` file directly to disk inside
 * `public/models/<filename>.glb` via `POST /api/upload-car?filename=...`.
 * Sends an ArrayBuffer so `Content-Length` is set explicitly through remote proxies.
 */
export async function persistCarToPublicFolder(
  file: File,
  rawBuffer?: ArrayBuffer,
): Promise<CarModelEntry | null> {
  try {
    const body = rawBuffer ? rawBuffer.slice(0) : await file.arrayBuffer();
    const url = `/api/upload-car?filename=${encodeURIComponent(file.name)}`;
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/octet-stream',
      },
      body,
    });
    if (!res.ok) {
      console.error('[FlexWeb] Failed to save car to public/models/:', res.status, await res.text());
      return null;
    }
    const data = (await res.json()) as { ok?: boolean; entry?: CarModelEntry };
    return data.ok && data.entry ? data.entry : null;
  } catch (err) {
    console.error('[FlexWeb] Error saving car to public/models/:', err);
    return null;
  }
}

export async function deleteCarFromPublicFolder(id: string): Promise<boolean> {
  try {
    const res = await fetch(`/api/public-cars?id=${encodeURIComponent(id)}`, { method: 'DELETE' });
    return res.ok;
  } catch {
    return false;
  }
}
