import { LoadingManager, Object3D } from 'three';
import { createManagedGltfLoader, getSharedGltfLoader } from './SharedGltfLoader';
import { unzipSync } from 'fflate';
import { getCachedAssetBytes, putCachedAssetBytes } from './AssetCache';

/**
 * ZipGltfLoader
 * -------------
 * Loads remote `.gltf` / `.glb` / `.zip` vehicle models (including GitHub raw URLs
 * with automatic branch/path fallbacks), reports real streaming download & extraction
 * progress, and persists the downloaded model bundle + resolved URL in IndexedDB
 * (`FlexWebRemoteModelCache`) so subsequent page loads skip GitHub completely.
 */

export interface ModelAttribution {
  title: string;
  modelUrl: string;
  author: string;
  licenseName: string;
  licenseUrl: string;
}

export interface RemoteCarPreset {
  id: string;
  name: string;
  remoteUrls: string[];
  attribution?: ModelAttribution;
}

export const GITHUB_CARS: RemoteCarPreset[] = [
  {
    id: 'github-bmw-m4-f82',
    name: 'BMW M4 F82',
    remoteUrls: [
      '/models/bmw_m4_f82.glb',
    ],
  },
  {
    id: 'github-dodge-demon',
    name: 'Dodge Challenger SRT Demon',
    remoteUrls: [
      '/models/dodge_challenger_srt_demon.glb',
    ],
    attribution: {
      title: 'Dodge Challenger SRT DEMON SDC',
      modelUrl: 'https://sketchfab.com/3d-models/dodge-challenger-srt-demon-sdc-54d229c2c0ff4eac8bc872f50b25e36c',
      author: 'SDC PERFORMANCE™️',
      licenseName: 'Creative Commons Attribution',
      licenseUrl: 'http://creativecommons.org/licenses/by/4.0/',
    },
  },
];

// ================================================================ Persistent IndexedDB + localStorage Cache

const REMOTE_DB_NAME = 'FlexWebRemoteModelCache';
const REMOTE_DB_VERSION = 1;
const REMOTE_STORE = 'models';
const RESOLVED_URL_PREFIX = 'flexweb_resolved_url:';

interface CachedGltfAsset {
  uri: string;
  mime: string;
  data: ArrayBuffer;
}

interface CachedRemoteModelRecord {
  carId: string;
  resolvedUrl: string;
  kind: 'zip' | 'glb' | 'gltf';
  bytes?: ArrayBuffer;
  gltfText?: string;
  assets?: CachedGltfAsset[];
  totalBytes: number;
  savedAt: number;
}

/** In-memory session cache for instant repeat switches within the same tab. */
const sessionModelCache = new Map<string, CachedRemoteModelRecord>();

function getSavedResolvedUrl(carId: string): string | null {
  try {
    return localStorage.getItem(RESOLVED_URL_PREFIX + carId);
  } catch {
    return null;
  }
}

function setSavedResolvedUrl(carId: string, url: string): void {
  try {
    localStorage.setItem(RESOLVED_URL_PREFIX + carId, url);
  } catch {
    // ignore storage quota / private mode
  }
}

function openRemoteDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') {
      reject(new Error('IndexedDB unavailable'));
      return;
    }
    const req = indexedDB.open(REMOTE_DB_NAME, REMOTE_DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(REMOTE_STORE)) {
        db.createObjectStore(REMOTE_STORE, { keyPath: 'carId' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error ?? new Error('Failed to open FlexWebRemoteModelCache'));
  });
}

async function loadCachedRemoteModel(carId: string): Promise<CachedRemoteModelRecord | null> {
  const mem = sessionModelCache.get(carId);
  if (mem) return mem;
  try {
    const db = await openRemoteDB();
    const rec = await new Promise<CachedRemoteModelRecord | null>((resolve, reject) => {
      const tx = db.transaction(REMOTE_STORE, 'readonly');
      const req = tx.objectStore(REMOTE_STORE).get(carId);
      req.onsuccess = () => resolve((req.result as CachedRemoteModelRecord | undefined) ?? null);
      req.onerror = () => reject(req.error);
      tx.oncomplete = () => db.close();
    });
    if (rec) sessionModelCache.set(carId, rec);
    return rec;
  } catch {
    return null;
  }
}

async function saveCachedRemoteModel(rec: CachedRemoteModelRecord): Promise<void> {
  sessionModelCache.set(rec.carId, rec);
  setSavedResolvedUrl(rec.carId, rec.resolvedUrl);
  try {
    const db = await openRemoteDB();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(REMOTE_STORE, 'readwrite');
      tx.objectStore(REMOTE_STORE).put(rec);
      tx.oncomplete = () => {
        db.close();
        resolve();
      };
      tx.onerror = () => {
        db.close();
        reject(tx.error);
      };
    });
  } catch {
    // non-fatal if browser storage quota is exceeded
  }
}

function yieldTick(): Promise<void> {
  if (typeof requestAnimationFrame !== 'undefined') {
    return new Promise((r) => requestAnimationFrame(() => r()));
  }
  return new Promise((r) => setTimeout(r, 0));
}

function formatMB(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function guessMime(uri: string): string {
  const clean = uri.split('?')[0].split('#')[0];
  const ext = clean.split('.').pop()?.toLowerCase() ?? '';
  if (ext === 'png') return 'image/png';
  if (ext === 'jpg' || ext === 'jpeg') return 'image/jpeg';
  if (ext === 'webp') return 'image/webp';
  return 'application/octet-stream';
}

/**
 * Fetches a local model from `public/` in safe 2 MB chunks via `/api/model-chunk`
 * so remote reverse proxies can never truncate large `.glb` binaries (which causes
 * `RangeError: Invalid typed array length` in `GLTFLoader`). Falls back to a normal
 * `fetch(url)` on static production hosts where `/api/model-chunk` is absent.
 */
export async function fetchLocalModelChunked(
  url: string,
  pctStart = 12,
  pctEnd = 42,
  onProgress?: (step: string, pct: number) => void,
): Promise<Uint8Array> {
  const fileName = url.split('/').pop() || 'model.glb';

  // 1. Check versioned persistent local cache (Cache Storage -> IndexedDB fallback) first
  const cached = await getCachedAssetBytes(url);
  if (cached && cached.byteLength >= 12 && cached[0] !== 0x3c) {
    console.info(`[FlexWeb Cache] HIT (${formatMB(cached.byteLength)}): ${url}`);
    onProgress?.(`Loaded ${fileName} from cache (${formatMB(cached.byteLength)})`, pctEnd);
    return cached;
  }

  const metaRes = await fetch(`/api/model-chunk?path=${encodeURIComponent(url)}&meta=1`, { cache: 'no-cache' }).catch(() => null);
  if (metaRes && metaRes.ok) {
    const meta = (await metaRes.json()) as { ok?: boolean; size?: number };
    if (meta.ok && typeof meta.size === 'number' && meta.size > 0) {
      const total = meta.size;
      const out = new Uint8Array(total);
      const CHUNK = 2 * 1024 * 1024; // 2 MB per request — safe under every proxy limit
      let offset = 0;
      while (offset < total) {
        const end = Math.min(total, offset + CHUNK);
        const chunkRes = await fetch(
          `/api/model-chunk?path=${encodeURIComponent(url)}&start=${offset}&end=${end}`,
          { cache: 'default' },
        );
        if (!chunkRes.ok) {
          throw new Error(`HTTP ${chunkRes.status} on chunk ${offset}-${end} of "${url}"`);
        }
        const chunkBuf = new Uint8Array(await chunkRes.arrayBuffer());
        if (chunkBuf.byteLength !== end - offset) {
          throw new Error(`Short chunk read (${chunkBuf.byteLength} != ${end - offset}) for "${url}"`);
        }
        out.set(chunkBuf, offset);
        offset = end;
        const frac = offset / total;
        const pct = Math.round(pctStart + frac * (pctEnd - pctStart));
        onProgress?.(`Loading ${fileName}… ${formatMB(offset)} / ${formatMB(total)}`, pct);
      }
      // Persist cloned binary in Cache Storage / IndexedDB for instant future reloads (quota-safe)
      if (out.byteLength >= 12 && out[0] !== 0x3c) {
        void putCachedAssetBytes(url, out);
      }
      return out;
    }
  }

  // Fallback for static hosting without dev/preview middleware
  const res = await fetch(url, { cache: 'default' });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} while fetching "${url}"`);
  }
  const bytes = await readResponseWithProgress(res, fileName, pctStart, pctEnd, onProgress);
  if (bytes.byteLength >= 12 && bytes[0] !== 0x3c) {
    void putCachedAssetBytes(url, bytes);
  }
  return bytes;
}

/**
 * Streams a `Response` body to an `ArrayBuffer` while reporting real byte progress.
 */
async function readResponseWithProgress(
  res: Response,
  label: string,
  pctStart: number,
  pctEnd: number,
  onProgress?: (step: string, pct: number) => void,
): Promise<Uint8Array> {
  const total = Number(res.headers.get('content-length') || 0);
  const reader = res.body?.getReader();
  if (!reader) {
    const buf = await res.arrayBuffer();
    onProgress?.(`Downloaded ${label} (${formatMB(buf.byteLength)})`, pctEnd);
    return new Uint8Array(buf);
  }

  const chunks: Uint8Array[] = [];
  let loaded = 0;
  let lastEmit = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      loaded += value.byteLength;
      const now = performance.now();
      if (now - lastEmit > 80) {
        lastEmit = now;
        if (total > 0) {
          const frac = Math.min(1, loaded / total);
          const pct = Math.round(pctStart + frac * (pctEnd - pctStart));
          onProgress?.(`Downloading ${label}… ${formatMB(loaded)} / ${formatMB(total)}`, pct);
        } else {
          const pct = Math.min(pctEnd - 1, Math.round(pctStart + Math.min(0.9, loaded / (25 * 1024 * 1024)) * (pctEnd - pctStart)));
          onProgress?.(`Downloading ${label}… ${formatMB(loaded)}`, pct);
        }
      }
    }
  }

  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.byteLength;
  }
  onProgress?.(`Downloaded ${label} (${formatMB(loaded)})`, pctEnd);
  return out;
}

// ================================================================ GLTF + Blob Bundle Parsing

async function parseGltfWithAssets(
  gltfText: string,
  assets: CachedGltfAsset[],
  gltfDir: string,
  onProgress?: (step: string, pct: number) => void,
): Promise<Object3D> {
  const blobUrls: string[] = [];
  const lookup = new Map<string, string>();

  const registerBlob = (key: string, data: ArrayBuffer, mime: string) => {
    const blob = new Blob([data.slice(0)], { type: mime });
    const url = URL.createObjectURL(blob);
    blobUrls.push(url);
    const norm = key.replace(/^\.?\//, '');
    lookup.set(norm, url);
    lookup.set(norm.toLowerCase(), url);
    if (gltfDir && norm.startsWith(gltfDir)) {
      const rel = norm.slice(gltfDir.length);
      lookup.set(rel, url);
      lookup.set(rel.toLowerCase(), url);
    }
    const base = norm.split('/').pop();
    if (base && !lookup.has(base)) {
      lookup.set(base, url);
      lookup.set(base.toLowerCase(), url);
    }
  };

  for (const a of assets) {
    registerBlob(a.uri, a.data, a.mime);
  }

  const manager = new LoadingManager();
  manager.setURLModifier((rawUrl: string) => {
    if (/^(blob:|data:)/i.test(rawUrl)) return rawUrl;
    let clean = rawUrl.replace(/^https?:\/\/[^/]+\//i, '').replace(/^\.?\//, '');
    try { clean = decodeURIComponent(clean); } catch { /* keep raw */ }
    const direct = lookup.get(clean) ?? lookup.get(clean.toLowerCase());
    if (direct) return direct;
    const base = clean.split('/').pop() ?? clean;
    return lookup.get(base) ?? lookup.get(base.toLowerCase()) ?? rawUrl;
  });

  onProgress?.('Parsing 3D Geometry & Textures…', 46);
  await yieldTick();
  try {
    const loader = createManagedGltfLoader(manager);
    const gltf = await loader.parseAsync(gltfText, '');
    return gltf.scene;
  } finally {
    setTimeout(() => {
      for (const u of blobUrls) URL.revokeObjectURL(u);
    }, 3500);
  }
}

/**
 * Packs a `.gltf` JSON + single external `.bin` directly into a standard binary GLB
 * in memory (zero `blob:` URL fetches, zero extra buffer copies, works under strict CSP).
 */
function packGltfAndBinToGlb(gltfText: string, binBytes: Uint8Array): ArrayBuffer {
  const json = JSON.parse(gltfText) as { buffers?: { uri?: string; byteLength?: number }[] };
  if (json.buffers && json.buffers[0]) {
    delete json.buffers[0].uri;
  }
  const jsonRaw = new TextEncoder().encode(JSON.stringify(json));
  const jsonPad = (4 - (jsonRaw.byteLength % 4)) % 4;
  const jsonChunkLen = jsonRaw.byteLength + jsonPad;
  const binPad = (4 - (binBytes.byteLength % 4)) % 4;
  const binChunkLen = binBytes.byteLength + binPad;
  const totalLen = 12 + 8 + jsonChunkLen + 8 + binChunkLen;

  const out = new ArrayBuffer(totalLen);
  const dv = new DataView(out);
  const u8 = new Uint8Array(out);
  dv.setUint32(0, 0x46546c67, true); // 'glTF'
  dv.setUint32(4, 2, true);          // version 2
  dv.setUint32(8, totalLen, true);
  // JSON chunk
  dv.setUint32(12, jsonChunkLen, true);
  dv.setUint32(16, 0x4e4f534a, true); // 'JSON'
  u8.set(jsonRaw, 20);
  for (let i = 0; i < jsonPad; i++) u8[20 + jsonRaw.byteLength + i] = 0x20;
  // BIN chunk
  const binOff = 20 + jsonChunkLen;
  dv.setUint32(binOff, binChunkLen, true);
  dv.setUint32(binOff + 4, 0x004e4942, true); // 'BIN\0'
  u8.set(binBytes, binOff + 8);
  return out;
}

/**
 * Extracts a `.zip` archive in memory using `fflate` and parses the contained
 * `.glb` or `.gltf` + `.bin` + textures.
 */
export async function parseGltfFromZipBytes(
  zipBytes: Uint8Array,
  onProgress?: (step: string, pct: number) => void,
): Promise<Object3D> {
  onProgress?.('Extracting .ZIP archive in memory…', 38);
  await yieldTick();
  const files = unzipSync(zipBytes);
  const entries = Object.entries(files).filter(([name]) => !name.endsWith('/') && !name.startsWith('__MACOSX/'));

  const glbEntry = entries.find(([name]) => /\.glb$/i.test(name));
  if (glbEntry) {
    onProgress?.('Parsing .GLB from archive…', 46);
    await yieldTick();
    const loader = getSharedGltfLoader();
    const copy = glbEntry[1].slice().buffer;
    const gltf = await loader.parseAsync(copy, '/');
    return gltf.scene;
  }

  const gltfEntry =
    entries.find(([name]) => /(?:^|\/)scene\.gltf$/i.test(name)) ??
    entries.find(([name]) => /\.gltf$/i.test(name));
  if (!gltfEntry) {
    throw new Error('ZIP archive does not contain a .gltf or .glb model');
  }

  const [gltfPath, gltfBytes] = gltfEntry;
  const gltfDir = gltfPath.includes('/') ? gltfPath.slice(0, gltfPath.lastIndexOf('/') + 1) : '';
  const text = new TextDecoder('utf-8').decode(gltfBytes);

  // Fast zero-blob path: if the GLTF has a single external .bin and no external image files,
  // pack directly into a binary GLB ArrayBuffer (avoids blob: fetch/CSP issues & cuts RAM 4x).
  try {
    const meta = JSON.parse(text) as {
      buffers?: { uri?: string }[];
      images?: { uri?: string }[];
    };
    const extImages = (meta.images ?? []).filter((img) => img.uri && !/^data:/i.test(img.uri));
    const extBuffers = (meta.buffers ?? []).filter((b) => b.uri && !/^data:/i.test(b.uri));
    if (extImages.length === 0 && extBuffers.length === 1 && extBuffers[0].uri) {
      const targetUri = extBuffers[0].uri.replace(/^\.?\//, '').toLowerCase();
      const binEntry = entries.find(([name]) => {
        const norm = name.replace(/^\.?\//, '').toLowerCase();
        return norm === targetUri || norm === (gltfDir + targetUri).toLowerCase() || norm.endsWith('/' + targetUri);
      });
      if (binEntry) {
        onProgress?.('Packing .GLTF + .BIN into 3D mesh…', 44);
        await yieldTick();
        const glbBuf = packGltfAndBinToGlb(text, binEntry[1]);
        onProgress?.('Parsing 3D Geometry…', 48);
        await yieldTick();
        const loader = getSharedGltfLoader();
        const gltf = await loader.parseAsync(glbBuf, '');
        return gltf.scene;
      }
    }
  } catch {
    // Fall through to LoadingManager blob path below
  }

  const assets: CachedGltfAsset[] = [];
  for (const [name, data] of entries) {
    if (name === gltfPath) continue;
    assets.push({ uri: name, mime: guessMime(name), data: data.slice().buffer });
  }
  return parseGltfWithAssets(text, assets, gltfDir, onProgress);
}

/**
 * Loads a remote car from an ordered list of GitHub raw `.gltf` / `.glb` / `.zip` URLs.
 * Persists the winning URL and full model payload in IndexedDB (`FlexWebRemoteModelCache`)
 * so future reloads skip network downloads and load nearly instantly.
 */
export async function loadRemoteCarWithFallbacks(
  carId: string,
  urls: string[],
  onProgress?: (step: string, pct: number) => void,
): Promise<{ scene: Object3D; resolvedUrl: string; bytes: number }> {
  const hasLocalGlbPrimary = urls.length > 0 && /^\/models\/.+\.glb(\?|$)/i.test(urls[0]);

  // 1. Instant local cache hit (IndexedDB or session RAM) — skip stale ZIP cache if a direct local .glb is now available
  if (!hasLocalGlbPrimary) {
    onProgress?.('Checking local model cache…', 10);
    const cached = await loadCachedRemoteModel(carId);
    if (cached) {
      console.info(`[FlexWeb Load] Cache HIT (${cached.kind}, ${formatMB(cached.totalBytes)}) for ${carId}: ${cached.resolvedUrl}`);
      onProgress?.(`Loaded from local cache (${formatMB(cached.totalBytes)})`, 34);
      await yieldTick();
      if (cached.kind === 'zip' && cached.bytes) {
        try {
          const scene = await parseGltfFromZipBytes(new Uint8Array(cached.bytes.slice(0)), onProgress);
          return { scene, resolvedUrl: cached.resolvedUrl, bytes: cached.totalBytes };
        } catch (cacheErr) {
          console.warn('[FlexWeb Load] Ignoring corrupt cached ZIP, re-fetching:', cacheErr);
        }
      }
      if (cached.kind === 'glb' && cached.bytes) {
        onProgress?.('Parsing cached .GLB mesh…', 44);
        await yieldTick();
        const loader = getSharedGltfLoader();
        const gltf = await loader.parseAsync(cached.bytes.slice(0), '/');
        return { scene: gltf.scene, resolvedUrl: cached.resolvedUrl, bytes: cached.totalBytes };
      }
      if (cached.kind === 'gltf' && cached.gltfText && cached.assets) {
        const scene = await parseGltfWithAssets(cached.gltfText, cached.assets, '', onProgress);
        return { scene, resolvedUrl: cached.resolvedUrl, bytes: cached.totalBytes };
      }
    }
  }

  // 2. Fetch — always prioritize local /models/*.glb when present; otherwise try previously resolved URL first
  const savedUrl = hasLocalGlbPrimary ? null : getSavedResolvedUrl(carId);
  const ordered = savedUrl ? [savedUrl, ...urls.filter((u) => u !== savedUrl)] : urls;

  const errors: string[] = [];
  for (let idx = 0; idx < ordered.length; idx++) {
    const url = ordered[idx];
    const fileName = url.split('/').pop() || 'model';
    const isLocalModel = url.startsWith('/models/');
    console.info(`[FlexWeb Load] Trying vehicle URL (${idx + 1}/${ordered.length}): ${url}`);
    onProgress?.(isLocalModel ? `Loading ${fileName}…` : `Connecting to GitHub (${fileName})…`, 12 + Math.min(6, idx));
    try {
      if (isLocalModel && /\.glb(\?|$)/i.test(url)) {
        const bytes = await fetchLocalModelChunked(url, 14, 42, onProgress);
        if (bytes.byteLength < 12 || new DataView(bytes.buffer, bytes.byteOffset, 4).getUint32(0, true) !== 0x46546c67) {
          errors.push(`${url} -> invalid GLB magic (${bytes.byteLength} bytes)`);
          continue;
        }
        setSavedResolvedUrl(carId, url);
        onProgress?.('Parsing 3D Mesh…', 46);
        await yieldTick();
        const loader = getSharedGltfLoader();
        const basePath = url.substring(0, url.lastIndexOf('/') + 1);
        const ab: ArrayBuffer =
          bytes.buffer instanceof ArrayBuffer && bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
            ? bytes.buffer
            : bytes.slice().buffer;
        const gltf = await loader.parseAsync(ab, basePath);
        return { scene: gltf.scene, resolvedUrl: url, bytes: bytes.byteLength };
      }

      const res = await fetch(url, isLocalModel ? { cache: 'default' } : { mode: 'cors', cache: 'default' });
      if (!res.ok) {
        errors.push(`${url} -> HTTP ${res.status}`);
        continue;
      }

      if (/\.zip(\?|$)/i.test(url)) {
        const bytes = await readResponseWithProgress(res, fileName, 16, 36, onProgress);
        if (bytes.byteLength < 4 || bytes[0] !== 0x50 || bytes[1] !== 0x4b) {
          errors.push(`${url} -> invalid ZIP header`);
          continue;
        }
        onProgress?.('Caching model locally for faster future loads…', 37);
        await saveCachedRemoteModel({
          carId,
          resolvedUrl: url,
          kind: 'zip',
          bytes: bytes.slice().buffer,
          totalBytes: bytes.byteLength,
          savedAt: Date.now(),
        });
        const scene = await parseGltfFromZipBytes(bytes, onProgress);
        return { scene, resolvedUrl: url, bytes: bytes.byteLength };
      }

      if (/\.glb(\?|$)/i.test(url)) {
        const bytes = await readResponseWithProgress(res, fileName, 16, 40, onProgress);
        if (bytes.byteLength < 12 || new DataView(bytes.buffer, bytes.byteOffset, 4).getUint32(0, true) !== 0x46546c67) {
          errors.push(`${url} -> invalid GLB magic`);
          continue;
        }
        if (!isLocalModel) {
          onProgress?.('Caching model locally for faster future loads…', 42);
          await saveCachedRemoteModel({
            carId,
            resolvedUrl: url,
            kind: 'glb',
            bytes: bytes.slice().buffer,
            totalBytes: bytes.byteLength,
            savedAt: Date.now(),
          });
        } else {
          setSavedResolvedUrl(carId, url);
        }
        onProgress?.('Parsing 3D Mesh…', 46);
        await yieldTick();
        const loader = getSharedGltfLoader();
        loader.setCrossOrigin('anonymous');
        const basePath = url.substring(0, url.lastIndexOf('/') + 1);
        const ab: ArrayBuffer =
          bytes.buffer instanceof ArrayBuffer && bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
            ? bytes.buffer
            : bytes.slice().buffer;
        const gltf = await loader.parseAsync(ab, basePath);
        return { scene: gltf.scene, resolvedUrl: url, bytes: bytes.byteLength };
      }

      // .gltf JSON + external .bin & textures hosted on GitHub raw
      const text = await res.text();
      if (!text.trim().startsWith('{')) {
        errors.push(`${url} -> non-JSON response`);
        continue;
      }

      const json = JSON.parse(text) as {
        buffers?: { uri?: string }[];
        images?: { uri?: string }[];
      };
      const basePath = url.substring(0, url.lastIndexOf('/') + 1);
      const uriSet = new Set<string>();
      for (const b of json.buffers ?? []) {
        if (b.uri && !/^data:/i.test(b.uri)) uriSet.add(b.uri);
      }
      for (const img of json.images ?? []) {
        if (img.uri && !/^data:/i.test(img.uri)) uriSet.add(img.uri);
      }
      const uris = Array.from(uriSet);
      const assets: CachedGltfAsset[] = [];
      let totalBytes = text.length;

      // Download all referenced .bin and texture files with live progress so we can cache the complete bundle
      for (let uIdx = 0; uIdx < uris.length; uIdx++) {
        const uri = uris[uIdx];
        const assetUrl = /^https?:\/\//i.test(uri) ? uri : basePath + uri;
        const assetLabel = `${uri.split('/').pop() ?? uri} (${uIdx + 1}/${uris.length})`;
        const pStart = Math.round(16 + (uIdx / Math.max(1, uris.length)) * 24);
        const pEnd = Math.round(16 + ((uIdx + 1) / Math.max(1, uris.length)) * 24);
        onProgress?.(`Downloading ${assetLabel}…`, pStart);
        const aRes = await fetch(assetUrl, { mode: 'cors', cache: 'default' });
        if (!aRes.ok) {
          throw new Error(`HTTP ${aRes.status} for asset "${assetUrl}"`);
        }
        const aBytes = await readResponseWithProgress(aRes, assetLabel, pStart, pEnd, onProgress);
        totalBytes += aBytes.byteLength;
        assets.push({ uri, mime: guessMime(uri), data: aBytes.slice().buffer });
      }

      onProgress?.('Caching model locally for faster future loads…', 42);
      await saveCachedRemoteModel({
        carId,
        resolvedUrl: url,
        kind: 'gltf',
        gltfText: text,
        assets,
        totalBytes,
        savedAt: Date.now(),
      });

      const scene = await parseGltfWithAssets(text, assets, '', onProgress);
      return { scene, resolvedUrl: url, bytes: totalBytes };
    } catch (err) {
      errors.push(`${url} -> ${String(err)}`);
    }
  }

  throw new Error(`Failed to load "${carId}": ${errors.join(' | ')}`);
}
