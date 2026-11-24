import fs from 'node:fs';
import path from 'node:path';
import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Plugin, PreviewServer, ViteDevServer } from 'vite';

/**
 * Scans `public/` (including `public/models/` and `public/cars/`) recursively
 * for `.glb` / `.gltf` vehicle models and exposes them through `virtual:flexweb-cars`.
 * Also provides `GET / POST / DELETE /api/public-cars` in both dev and preview servers
 * so any `.glb` uploaded in-game is streamed directly into `public/` on disk and stays
 * permanently saved across all future reloads.
 */
const VIRTUAL_ID = 'virtual:flexweb-cars';
const RESOLVED_ID = '\0' + VIRTUAL_ID;

export interface CarModelEntry {
  id: string;
  name: string;
  url: string;
  bytes: number;
  mtimeMs?: number;
}

export function prettyCarName(relPath: string): string {
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

function walkDir(dir: string, rootDir: string, out: CarModelEntry[]): void {
  if (!fs.existsSync(dir)) return;
  const entries = fs.readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name));
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walkDir(full, rootDir, out);
    } else if (entry.isFile() && /\.(glb|gltf)$/i.test(entry.name)) {
      const stat = fs.statSync(full);
      const rel = path.relative(rootDir, full).split(path.sep).join('/');
      let bytes = stat.size;
      const binPath = path.join(dir, 'scene.bin');
      if (/\.gltf$/i.test(entry.name) && fs.existsSync(binPath)) {
        bytes += fs.statSync(binPath).size;
      }
      out.push({
        id: rel,
        name: prettyCarName(rel),
        url: '/' + rel,
        bytes,
        mtimeMs: stat.mtimeMs,
      });
    }
  }
}

function scan(publicDir: string): CarModelEntry[] {
  const out: CarModelEntry[] = [];
  walkDir(publicDir, publicDir, out);
  return out;
}

function attachPublicCarsApi(
  middlewares: ViteDevServer['middlewares'] | PreviewServer['middlewares'],
  getPublicDir: () => string,
  invalidate?: () => void,
): void {
  middlewares.use('/api/public-cars', (req: IncomingMessage, res: ServerResponse) => {
    const publicDir = getPublicDir();
    const parsedUrl = new URL(req.url || '/', 'http://localhost');

    if (req.method === 'GET') {
      res.setHeader('Content-Type', 'application/json');
      res.setHeader('Cache-Control', 'no-store');
      res.end(JSON.stringify(scan(publicDir)));
      return;
    }

    if (req.method === 'POST') {
      const qName = parsedUrl.searchParams.get('filename');
      const hName = req.headers['x-car-filename'] as string | undefined;
      let rawName = 'custom_car.glb';
      try {
        rawName = decodeURIComponent(qName || hName || 'custom_car.glb');
      } catch {
        rawName = qName || hName || 'custom_car.glb';
      }
      const safeFile = path.basename(rawName).replace(/[^a-zA-Z0-9._-]/g, '_');
      const finalFile = /\.(glb|gltf)$/i.test(safeFile) ? safeFile : `${safeFile}.glb`;

      try {
        if (!fs.existsSync(publicDir)) fs.mkdirSync(publicDir, { recursive: true });
        const target = path.join(publicDir, finalFile);
        const tmpTarget = `${target}.uploading_${Date.now()}`;
        const ws = fs.createWriteStream(tmpTarget);
        let writtenBytes = 0;

        req.on('data', (chunk: Buffer) => {
          writtenBytes += chunk.length;
        });

        ws.on('finish', () => {
          try {
            fs.renameSync(tmpTarget, target);
            invalidate?.();
            const entry: CarModelEntry = {
              id: finalFile,
              name: prettyCarName(finalFile),
              url: '/' + finalFile,
              bytes: writtenBytes,
            };
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ ok: true, entry, cars: scan(publicDir) }));
          } catch (err) {
            try { fs.unlinkSync(tmpTarget); } catch { /* ignore */ }
            res.statusCode = 500;
            res.end(JSON.stringify({ ok: false, error: String(err) }));
          }
        });

        ws.on('error', (err) => {
          try { fs.unlinkSync(tmpTarget); } catch { /* ignore */ }
          res.statusCode = 500;
          res.end(JSON.stringify({ ok: false, error: String(err) }));
        });

        req.pipe(ws);
      } catch (err) {
        res.statusCode = 500;
        res.end(JSON.stringify({ ok: false, error: String(err) }));
      }
      return;
    }

    if (req.method === 'DELETE') {
      const id = parsedUrl.searchParams.get('id');
      if (!id) {
        res.statusCode = 400;
        res.end(JSON.stringify({ ok: false, error: 'Missing id' }));
        return;
      }
      const resolved = path.resolve(publicDir, id);
      if (!resolved.startsWith(path.resolve(publicDir) + path.sep)) {
        res.statusCode = 403;
        res.end(JSON.stringify({ ok: false, error: 'Forbidden path' }));
        return;
      }
      try {
        if (fs.existsSync(resolved)) fs.unlinkSync(resolved);
        invalidate?.();
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify({ ok: true, cars: scan(publicDir) }));
      } catch (err) {
        res.statusCode = 500;
        res.end(JSON.stringify({ ok: false, error: String(err) }));
      }
      return;
    }

    res.statusCode = 405;
    res.end();
  });
}

export default function flexwebCars(): Plugin {
  let publicDir = path.resolve('public');
  return {
    name: 'flexweb-cars',
    configResolved(cfg) {
      if (cfg.publicDir) publicDir = cfg.publicDir;
    },
    resolveId(id) {
      return id === VIRTUAL_ID ? RESOLVED_ID : null;
    },
    load(id) {
      if (id !== RESOLVED_ID) return null;
      return `export default ${JSON.stringify(scan(publicDir))};`;
    },
    configureServer(server: ViteDevServer) {
      const invalidate = () => {
        const mod = server.moduleGraph.getModuleById(RESOLVED_ID);
        if (mod) server.moduleGraph.invalidateModule(mod);
      };
      attachPublicCarsApi(server.middlewares, () => publicDir, invalidate);
    },
    configurePreviewServer(server: PreviewServer) {
      attachPublicCarsApi(server.middlewares, () => publicDir);
    },
  };
}
