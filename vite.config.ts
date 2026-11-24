import fs from 'node:fs';
import path from 'node:path';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { defineConfig, type Plugin, type PreviewServer, type ViteDevServer } from 'vite';
import react from '@vitejs/plugin-react';
import flexwebCars, { prettyCarName } from './vite-plugin-flexweb-cars.ts';

/**
 * Dev & preview server upload endpoint:
 * - `POST /api/upload-car?filename=<name>.glb` (and `/api/save-model` alias) streams
 *   the uploaded binary `.glb` directly into `public/models/<filename>.glb` on disk.
 * - `GET /api/upload-car` returns all `.glb`/`.gltf` files currently in `public/models/`
 *   sorted newest-first so startup & restarts immediately load the uploaded car.
 */
function uploadCarEndpoint(): Plugin {
  let publicDir = path.resolve('public');

  const listModels = () => {
    const modelsDir = path.join(publicDir, 'models');
    if (!fs.existsSync(modelsDir)) return [];
    return fs
      .readdirSync(modelsDir, { withFileTypes: true })
      .filter((e) => e.isFile() && /\.(glb|gltf)$/i.test(e.name))
      .map((e) => {
        const stat = fs.statSync(path.join(modelsDir, e.name));
        const relId = `models/${e.name}`;
        return {
          id: relId,
          filename: e.name,
          name: prettyCarName(relId),
          url: `/models/${e.name}`,
          bytes: stat.size,
          mtimeMs: stat.mtimeMs,
        };
      })
      .sort((a, b) => b.mtimeMs - a.mtimeMs);
  };

  const handler = (req: IncomingMessage, res: ServerResponse) => {
    if (req.method === 'GET') {
      res.setHeader('Content-Type', 'application/json');
      res.setHeader('Cache-Control', 'no-store');
      res.end(JSON.stringify(listModels()));
      return;
    }

    if (req.method !== 'POST') {
      res.statusCode = 405;
      res.end(JSON.stringify({ ok: false, error: 'Method Not Allowed' }));
      return;
    }

    const parsedUrl = new URL(req.url || '/', 'http://localhost');
    const qName = parsedUrl.searchParams.get('filename');
    const hName = req.headers['x-car-filename'] as string | undefined;
    let rawName = 'custom_car.glb';
    try {
      rawName = decodeURIComponent(qName || hName || 'custom_car.glb');
    } catch {
      rawName = qName || hName || 'custom_car.glb';
    }

    const safeBase = path.basename(rawName).replace(/[^a-zA-Z0-9._-]/g, '_');
    const finalFile = /\.(glb|gltf)$/i.test(safeBase) ? safeBase : `${safeBase}.glb`;
    const modelsDir = path.join(publicDir, 'models');
    const target = path.join(modelsDir, finalFile);
    const tmpTarget = `${target}.uploading_${Date.now()}`;

    try {
      if (!fs.existsSync(modelsDir)) fs.mkdirSync(modelsDir, { recursive: true });
      const ws = fs.createWriteStream(tmpTarget);
      let writtenBytes = 0;

      req.on('data', (chunk: Buffer) => {
        writtenBytes += chunk.length;
      });

      ws.on('finish', () => {
        try {
          fs.renameSync(tmpTarget, target);
          const stat = fs.statSync(target);
          const relId = `models/${finalFile}`;
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({
            ok: true,
            entry: {
              id: relId,
              filename: finalFile,
              name: prettyCarName(relId),
              url: `/models/${finalFile}`,
              bytes: writtenBytes,
              mtimeMs: stat.mtimeMs,
            },
          }));
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
  };

  const serveModels = (req: IncomingMessage, res: ServerResponse, next: () => void) => {
    if (req.method !== 'GET' && req.method !== 'HEAD') { next(); return; }
    const parsedUrl = new URL(req.url || '/', 'http://localhost');
    let relName = parsedUrl.pathname.replace(/^\/+/, '');
    try { relName = decodeURIComponent(relName); } catch { /* keep raw */ }
    if (!relName || relName.includes('..') || relName.includes('/') || relName.includes('\\')) {
      next();
      return;
    }
    const modelsDir = path.resolve(publicDir, 'models');
    const filePath = path.resolve(modelsDir, relName);
    if (!filePath.startsWith(modelsDir + path.sep) || !fs.existsSync(filePath)) {
      next();
      return;
    }
    const acceptEnc = String(req.headers['accept-encoding'] || '');
    const gzPath = `${filePath}.gz`;
    const useGzip = acceptEnc.includes('gzip') && fs.existsSync(gzPath);
    const targetPath = useGzip ? gzPath : filePath;
    const stat = fs.statSync(targetPath);
    if (!stat.isFile()) { next(); return; }
    res.setHeader('Content-Type', /\.gltf$/i.test(relName) ? 'model/gltf+json' : 'model/gltf-binary');
    if (useGzip) res.setHeader('Content-Encoding', 'gzip');
    res.setHeader('Vary', 'Accept-Encoding');
    res.setHeader('Content-Length', String(stat.size));
    res.setHeader('Cache-Control', 'public, max-age=3600');
    if (req.method === 'HEAD') { res.end(); return; }
    fs.createReadStream(targetPath).pipe(res);
  };

  const proxyGithubModel = async (req: IncomingMessage, res: ServerResponse) => {
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.statusCode = 405;
      res.end('Method Not Allowed');
      return;
    }
    const parsedUrl = new URL(req.url || '/', 'http://localhost');
    const relPath = parsedUrl.pathname.replace(/^\/+/, '');
    // Strictly allow only safe GitHub raw repo paths: <owner>/<repo>/<branch>/<file>
    if (!/^[a-zA-Z0-9._-]+\/[a-zA-Z0-9._-]+\/[a-zA-Z0-9._-]+\/[a-zA-Z0-9._/-]+$/.test(relPath) || relPath.includes('..')) {
      res.statusCode = 400;
      res.end('Invalid GitHub path');
      return;
    }

    const cacheDir = path.resolve('.cache', 'remote-models');
    const cacheFile = path.join(cacheDir, relPath.replace(/\//g, '__'));
    const mime = /\.zip$/i.test(relPath)
      ? 'application/zip'
      : /\.glb$/i.test(relPath)
        ? 'model/gltf-binary'
        : /\.gltf$/i.test(relPath)
          ? 'model/gltf+json'
          : 'application/octet-stream';

    if (fs.existsSync(cacheFile)) {
      const stat = fs.statSync(cacheFile);
      if (stat.isFile() && stat.size > 0) {
        res.setHeader('Content-Type', mime);
        res.setHeader('Content-Length', String(stat.size));
        res.setHeader('Cache-Control', 'public, max-age=86400');
        if (req.method === 'HEAD') { res.end(); return; }
        fs.createReadStream(cacheFile).pipe(res);
        return;
      }
    }

    const remoteUrl = `https://raw.githubusercontent.com/${relPath}`;
    try {
      const upstream = await fetch(remoteUrl);
      if (!upstream.ok) {
        res.statusCode = upstream.status;
        res.end(`Upstream HTTP ${upstream.status}`);
        return;
      }
      const arrBuf = await upstream.arrayBuffer();
      const buf = Buffer.from(arrBuf);
      try {
        if (!fs.existsSync(cacheDir)) fs.mkdirSync(cacheDir, { recursive: true });
        fs.writeFileSync(cacheFile, buf);
      } catch { /* non-fatal cache write */ }
      res.setHeader('Content-Type', mime);
      res.setHeader('Content-Length', String(buf.byteLength));
      res.setHeader('Cache-Control', 'public, max-age=86400');
      if (req.method === 'HEAD') { res.end(); return; }
      res.end(buf);
    } catch (err) {
      res.statusCode = 502;
      res.end(`Failed to fetch ${remoteUrl}: ${String(err)}`);
    }
  };

  const serveModelChunk = (req: IncomingMessage, res: ServerResponse) => {
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.statusCode = 405;
      res.end('Method Not Allowed');
      return;
    }
    const parsedUrl = new URL(req.url || '/', 'http://localhost');
    const rawPath = parsedUrl.searchParams.get('path') || '';
    const rel = rawPath.replace(/^\/+/, '').replace(/^public\//i, '');
    if (!rel || rel.includes('..') || rel.includes('\\')) {
      res.statusCode = 400;
      res.end('Invalid path');
      return;
    }
    const pubRoot = path.resolve(publicDir);
    const fullPath = path.resolve(pubRoot, rel);
    if (!fullPath.startsWith(pubRoot + path.sep) || !fs.existsSync(fullPath)) {
      res.statusCode = 404;
      res.end('Model not found');
      return;
    }
    const stat = fs.statSync(fullPath);
    if (!stat.isFile()) {
      res.statusCode = 404;
      res.end('Not a file');
      return;
    }

    if (parsedUrl.searchParams.get('meta') === '1') {
      res.setHeader('Content-Type', 'application/json');
      res.setHeader('Cache-Control', 'no-cache');
      res.end(JSON.stringify({ ok: true, size: stat.size }));
      return;
    }

    const start = Math.max(0, Math.min(stat.size, Number(parsedUrl.searchParams.get('start') || 0)));
    const rawEnd = Number(parsedUrl.searchParams.get('end') || stat.size);
    const end = Math.max(start, Math.min(stat.size, Number.isFinite(rawEnd) ? rawEnd : stat.size));
    const len = end - start;
    res.setHeader('Content-Type', 'application/octet-stream');
    res.setHeader('Content-Length', String(len));
    res.setHeader('Cache-Control', 'public, max-age=3600');
    if (req.method === 'HEAD' || len === 0) {
      res.end();
      return;
    }
    fs.createReadStream(fullPath, { start, end: end - 1 }).pipe(res);
  };

  const attach = (middlewares: ViteDevServer['middlewares'] | PreviewServer['middlewares']) => {
    middlewares.use('/models', serveModels);
    middlewares.use('/api/model-chunk', serveModelChunk);
    middlewares.use('/api/github-proxy', (req, res) => { void proxyGithubModel(req, res); });
    middlewares.use('/api/list-models', handler);
    middlewares.use('/api/upload-car', handler);
    middlewares.use('/api/save-model', handler);
  };

  return {
    name: 'flexweb-upload-car',
    configResolved(cfg) {
      if (cfg.publicDir) publicDir = cfg.publicDir;
    },
    configureServer(server) {
      attach(server.middlewares);
    },
    configurePreviewServer(server) {
      attach(server.middlewares);
    },
  };
}

export default defineConfig({
  plugins: [react(), flexwebCars(), uploadCarEndpoint()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    allowedHosts: true,
    watch: {
      // Never restart Vite or trigger HMR reloads when a .glb is written into public/models/
      ignored: ['**/public/models/**', '**/public/*.uploading_*'],
    },
  },
  preview: { host: '0.0.0.0', port: 4173, allowedHosts: true },
  build: {
    target: 'es2020',
    chunkSizeWarningLimit: 1200,
  },
});
