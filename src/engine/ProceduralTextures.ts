import {
  CanvasTexture,
  RepeatWrapping,
  SRGBColorSpace,
  Texture,
} from 'three';

/**
 * ProceduralTextures
 * ------------------
 * Generates and caches lightweight, high-definition PBR texture sets
 * (`map`, `normalMap`, `roughnessMap`) purely via HTML5 Canvas:
 *  - Zero external image files
 *  - Cached singleton per tier (`lowSpec` 256px vs `high` 512px) for 60 FPS on 4 GB RAM tablets
 *  - Marked `userData.sharedProcedural = true` so vehicle swaps never dispose shared VRAM textures
 */

export interface PBRTexturePack {
  map: CanvasTexture;
  normalMap: CanvasTexture;
  roughnessMap: CanvasTexture;
}

export interface ConcreteTexturePack extends PBRTexturePack {
  hazardMap: CanvasTexture;
  hazardRoughnessMap: CanvasTexture;
}

export interface CrashWearTexturePack {
  normalMap: CanvasTexture;
  roughnessMap: CanvasTexture;
}

interface TextureCache {
  asphalt?: PBRTexturePack;
  concrete?: ConcreteTexturePack;
  carbon?: PBRTexturePack;
  brushedMetal?: PBRTexturePack;
  brakeDisc?: PBRTexturePack;
  crashWear?: CrashWearTexturePack;
}

const caches: Record<'low' | 'high', TextureCache> = { low: {}, high: {} };

/** Fast deterministic PRNG (Mulberry32) so textures look consistent across sessions. */
function mulberry32(seed: number): () => number {
  let a = seed | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeCanvas(size: number): HTMLCanvasElement {
  if (typeof document === 'undefined') {
    // Node/headless test fallback stub
    return {
      width: size,
      height: size,
      getContext: () => null,
    } as unknown as HTMLCanvasElement;
  }
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  return c;
}

function configureTex(
  canvas: HTMLCanvasElement,
  srgb: boolean,
  repeatX = 1,
  repeatY = 1,
  anisotropy = 4,
): CanvasTexture {
  const tex = new CanvasTexture(canvas);
  tex.wrapS = RepeatWrapping;
  tex.wrapT = RepeatWrapping;
  tex.repeat.set(repeatX, repeatY);
  if (srgb) tex.colorSpace = SRGBColorSpace;
  tex.anisotropy = anisotropy;
  tex.userData.sharedProcedural = true;
  tex.needsUpdate = true;
  return tex;
}

/** Convert a tileable Float32 heightfield [0..1] into a tangent-space RGB normal map canvas. */
function heightToNormalCanvas(height: Float32Array, size: number, strength: number): HTMLCanvasElement {
  const c = makeCanvas(size);
  const ctx = c.getContext('2d');
  if (!ctx) return c;
  const img = ctx.createImageData(size, size);
  const d = img.data;
  const mask = size - 1;
  for (let y = 0; y < size; y++) {
    const ym = (y - 1) & mask;
    const yp = (y + 1) & mask;
    for (let x = 0; x < size; x++) {
      const xm = (x - 1) & mask;
      const xp = (x + 1) & mask;
      // 3x3 Sobel gradient with seamless wrap
      const tl = height[ym * size + xm];
      const t = height[ym * size + x];
      const tr = height[ym * size + xp];
      const l = height[y * size + xm];
      const r = height[y * size + xp];
      const bl = height[yp * size + xm];
      const b = height[yp * size + x];
      const br = height[yp * size + xp];

      const dx = (tr + 2 * r + br - (tl + 2 * l + bl)) * strength;
      const dy = (bl + 2 * b + br - (tl + 2 * t + tr)) * strength;
      const dz = 1.0;
      const inv = 1 / Math.sqrt(dx * dx + dy * dy + dz * dz);

      const idx = (y * size + x) * 4;
      d[idx] = Math.max(0, Math.min(255, Math.round((-dx * inv * 0.5 + 0.5) * 255)));
      d[idx + 1] = Math.max(0, Math.min(255, Math.round((-dy * inv * 0.5 + 0.5) * 255)));
      d[idx + 2] = Math.max(0, Math.min(255, Math.round((dz * inv * 0.5 + 0.5) * 255)));
      d[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

/** Check if a texture is managed by the shared procedural cache (should not be disposed per car). */
export function isSharedProceduralTexture(tex: Texture | null | undefined): boolean {
  return Boolean(tex?.userData?.sharedProcedural);
}

// ============================================================================
// 1. REALISTIC BITUMEN ASPHALT (GridMap Arena Ground)
// ============================================================================

export function getAsphaltTextures(lowSpec: boolean): PBRTexturePack {
  const tier = lowSpec ? 'low' : 'high';
  if (caches[tier].asphalt) return caches[tier].asphalt!;

  const size = lowSpec ? 256 : 512;
  const aniso = lowSpec ? 4 : 8;
  const rand = mulberry32(4109);

  const colorC = makeCanvas(size);
  const roughC = makeCanvas(size);
  const height = new Float32Array(size * size);

  const cCtx = colorC.getContext('2d');
  const rCtx = roughC.getContext('2d');
  if (cCtx && rCtx) {
    const cImg = cCtx.createImageData(size, size);
    const rImg = rCtx.createImageData(size, size);
    const cd = cImg.data;
    const rd = rImg.data;

    // Subtle tileable macro patch variation (low-frequency bitumen wear)
    const freq1 = (Math.PI * 2) / size;
    for (let y = 0; y < size; y++) {
      const sy = Math.sin(y * freq1);
      const cy = Math.cos(y * freq1 * 2);
      for (let x = 0; x < size; x++) {
        const sx = Math.sin(x * freq1);
        const cx = Math.cos(x * freq1 * 2);
        const macro = (sx * sy * 0.5 + cx * cy * 0.35 + Math.sin((x + y) * freq1) * 0.25);

        // Fine stone aggregate grain + micro bitumen pores
        const g1 = rand();
        const g2 = rand();
        const pebble = g1 > 0.92 ? 14 : g1 < 0.08 ? -12 : (g2 - 0.5) * 14;
        const base = 76 + macro * 6.5 + pebble;

        const i = y * size + x;
        height[i] = Math.max(0, Math.min(1, 0.5 + pebble * 0.025 + macro * 0.08));

        const p = i * 4;
        cd[p] = Math.max(38, Math.min(125, Math.round(base)));
        cd[p + 1] = Math.max(40, Math.min(128, Math.round(base + 2)));
        cd[p + 2] = Math.max(44, Math.min(134, Math.round(base + 6)));
        cd[p + 3] = 255;

        // Roughness: mostly matte (0.80–0.96) with subtle stone micro-glint
        const rv = Math.max(195, Math.min(248, Math.round(226 - macro * 10 - (g1 > 0.93 ? 22 : 0) + (g2 - 0.5) * 14)));
        rd[p] = rv;
        rd[p + 1] = rv;
        rd[p + 2] = rv;
        rd[p + 3] = 255;
      }
    }
    cCtx.putImageData(cImg, 0, 0);
    rCtx.putImageData(rImg, 0, 0);

    // Subtle GridMap 1m / 5m calibration lines over the realistic asphalt
    cCtx.strokeStyle = 'rgba(203, 213, 225, 0.13)';
    cCtx.lineWidth = Math.max(1, size / 256);
    const step = size / 4;
    for (let k = 1; k < 4; k++) {
      const p = Math.round(k * step) + 0.5;
      cCtx.beginPath(); cCtx.moveTo(p, 0); cCtx.lineTo(p, size); cCtx.stroke();
      cCtx.beginPath(); cCtx.moveTo(0, p); cCtx.lineTo(size, p); cCtx.stroke();
    }
    cCtx.strokeStyle = 'rgba(226, 232, 240, 0.24)';
    cCtx.lineWidth = Math.max(1.5, size / 170);
    cCtx.strokeRect(1, 1, size - 2, size - 2);
  }

  const normalC = heightToNormalCanvas(height, size, 1.35);
  const pack: PBRTexturePack = {
    map: configureTex(colorC, true, 80, 80, aniso),
    normalMap: configureTex(normalC, false, 80, 80, aniso),
    roughnessMap: configureTex(roughC, false, 80, 80, aniso),
  };
  caches[tier].asphalt = pack;
  return pack;
}

// ============================================================================
// 2. WEATHERED CONCRETE & CRISP HAZARD STRIPES (Blocks, Walls, Ramps)
// ============================================================================

export function getConcreteTextures(lowSpec: boolean): ConcreteTexturePack {
  const tier = lowSpec ? 'low' : 'high';
  if (caches[tier].concrete) return caches[tier].concrete!;

  const size = lowSpec ? 256 : 512;
  const aniso = lowSpec ? 4 : 8;
  const rand = mulberry32(7721);

  const colorC = makeCanvas(size);
  const roughC = makeCanvas(size);
  const height = new Float32Array(size * size);
  const cCtx = colorC.getContext('2d');
  const rCtx = roughC.getContext('2d');

  if (cCtx && rCtx) {
    const cImg = cCtx.createImageData(size, size);
    const rImg = rCtx.createImageData(size, size);
    const cd = cImg.data;
    const rd = rImg.data;
    const freq = (Math.PI * 2) / size;

    for (let y = 0; y < size; y++) {
      // Subtle vertical weathering / damp stain near base
      const damp = Math.max(0, (y / size - 0.65) / 0.35) * 14;
      for (let x = 0; x < size; x++) {
        const stain = Math.sin(x * freq * 2 + y * freq) * 7 + Math.cos(x * freq * 3 - y * freq * 2) * 5;
        const n = (rand() - 0.5) * 18;
        const pore = rand() < 0.015 ? -26 : 0;
        const val = 148 + stain + n + pore - damp;
        const i = y * size + x;
        height[i] = Math.max(0, Math.min(1, 0.52 + (n + pore) * 0.014 + stain * 0.01));

        const p = i * 4;
        cd[p] = Math.max(70, Math.min(205, Math.round(val)));
        cd[p + 1] = Math.max(72, Math.min(206, Math.round(val + 1)));
        cd[p + 2] = Math.max(74, Math.min(210, Math.round(val + 3)));
        cd[p + 3] = 255;

        const rv = Math.max(200, Math.min(252, Math.round(234 - stain * 1.1 + (rand() - 0.5) * 14)));
        rd[p] = rv; rd[p + 1] = rv; rd[p + 2] = rv; rd[p + 3] = 255;
      }
    }
    cCtx.putImageData(cImg, 0, 0);
    rCtx.putImageData(rImg, 0, 0);

    // Subtle concrete formwork panel seam & tie-rod dimples
    cCtx.strokeStyle = 'rgba(42, 46, 52, 0.22)';
    cCtx.lineWidth = Math.max(1, size / 256);
    cCtx.beginPath();
    cCtx.moveTo(0, size * 0.5); cCtx.lineTo(size, size * 0.5);
    cCtx.moveTo(size * 0.5, 0); cCtx.lineTo(size * 0.5, size);
    cCtx.stroke();

    cCtx.fillStyle = 'rgba(55, 60, 68, 0.32)';
    for (const py of [0.25, 0.75]) {
      for (const px of [0.25, 0.75]) {
        cCtx.beginPath();
        cCtx.arc(size * px, size * py, Math.max(2, size * 0.008), 0, Math.PI * 2);
        cCtx.fill();
      }
    }
  }

  // Crisp industrial safety stripes with weathered concrete overlay
  const hazSize = lowSpec ? 256 : 512;
  const hazC = makeCanvas(hazSize);
  const hazRC = makeCanvas(hazSize);
  const hCtx = hazC.getContext('2d');
  const hrCtx = hazRC.getContext('2d');
  if (hCtx && hrCtx) {
    hCtx.fillStyle = '#e5a71a';
    hCtx.fillRect(0, 0, hazSize, hazSize);
    hCtx.fillStyle = '#181a1f';
    const stripe = hazSize / 8;
    for (let i = -hazSize; i < hazSize * 2; i += stripe * 2) {
      hCtx.beginPath();
      hCtx.moveTo(i, 0);
      hCtx.lineTo(i + stripe, 0);
      hCtx.lineTo(i + stripe - hazSize, hazSize);
      hCtx.lineTo(i - hazSize, hazSize);
      hCtx.closePath();
      hCtx.fill();
    }
    // Subtle scuffing over safety stripes so they feel painted on real concrete
    hCtx.globalAlpha = 0.22;
    hCtx.drawImage(colorC, 0, 0, hazSize, hazSize);
    hCtx.globalAlpha = 1.0;

    // Slight gloss difference between painted yellow/black stripes and scuffed concrete
    hrCtx.fillStyle = '#c8c8c8';
    hrCtx.fillRect(0, 0, hazSize, hazSize);
    hrCtx.globalAlpha = 0.35;
    hrCtx.drawImage(roughC, 0, 0, hazSize, hazSize);
    hrCtx.globalAlpha = 1.0;
  }

  const normalC = heightToNormalCanvas(height, size, 1.5);
  const pack: ConcreteTexturePack = {
    map: configureTex(colorC, true, 1, 1, aniso),
    normalMap: configureTex(normalC, false, 1, 1, aniso),
    roughnessMap: configureTex(roughC, false, 1, 1, aniso),
    hazardMap: configureTex(hazC, true, 2, 1, aniso),
    hazardRoughnessMap: configureTex(hazRC, false, 2, 1, aniso),
  };
  caches[tier].concrete = pack;
  return pack;
}

// ============================================================================
// 3. WOVEN 2x2 TWILL CARBON FIBER (Splitters, Diffusers, Spoilers)
// ============================================================================

export function getCarbonFiberTextures(lowSpec: boolean): PBRTexturePack {
  const tier = lowSpec ? 'low' : 'high';
  if (caches[tier].carbon) return caches[tier].carbon!;

  const size = lowSpec ? 128 : 256;
  const aniso = lowSpec ? 4 : 8;
  const colorC = makeCanvas(size);
  const roughC = makeCanvas(size);
  const height = new Float32Array(size * size);

  const cCtx = colorC.getContext('2d');
  const rCtx = roughC.getContext('2d');
  if (cCtx && rCtx) {
    const cImg = cCtx.createImageData(size, size);
    const rImg = rCtx.createImageData(size, size);
    const cd = cImg.data;
    const rd = rImg.data;
    const cell = size / 16; // 16 weave cells per tile

    for (let y = 0; y < size; y++) {
      const cy = Math.floor(y / cell);
      const fy = (y % cell) / cell;
      for (let x = 0; x < size; x++) {
        const cx = Math.floor(x / cell);
        const fx = (x % cell) / cell;
        // 2x2 diagonal twill weave pattern
        const phase = (cx + cy) & 3;
        const isWarp = phase < 2;
        const along = isWarp ? fx : fy;
        const across = isWarp ? fy : fx;
        // Rounded fiber tow crown profile
        const crown = Math.sin(along * Math.PI) * (0.65 + 0.35 * Math.sin(across * Math.PI));
        const sheen = isWarp ? 28 + crown * 34 : 15 + crown * 22;

        const i = y * size + x;
        height[i] = 0.25 + crown * 0.7;

        const p = i * 4;
        cd[p] = Math.round(sheen);
        cd[p + 1] = Math.round(sheen + 2);
        cd[p + 2] = Math.round(sheen + 5);
        cd[p + 3] = 255;

        // Roughness alternates subtly between warp and weft tows under clearcoat
        const rv = Math.round(isWarp ? 58 + (1 - crown) * 32 : 84 + (1 - crown) * 38);
        rd[p] = rv; rd[p + 1] = rv; rd[p + 2] = rv; rd[p + 3] = 255;
      }
    }
    cCtx.putImageData(cImg, 0, 0);
    rCtx.putImageData(rImg, 0, 0);
  }

  const normalC = heightToNormalCanvas(height, size, 1.85);
  const pack: PBRTexturePack = {
    map: configureTex(colorC, true, 6, 6, aniso),
    normalMap: configureTex(normalC, false, 6, 6, aniso),
    roughnessMap: configureTex(roughC, false, 6, 6, aniso),
  };
  caches[tier].carbon = pack;
  return pack;
}

// ============================================================================
// 4. BRUSHED CNC METAL (Engine Block, Intake, Heads)
// ============================================================================

export function getBrushedMetalTextures(lowSpec: boolean): PBRTexturePack {
  const tier = lowSpec ? 'low' : 'high';
  if (caches[tier].brushedMetal) return caches[tier].brushedMetal!;

  const size = lowSpec ? 128 : 256;
  const aniso = lowSpec ? 4 : 8;
  const rand = mulberry32(9903);

  const colorC = makeCanvas(size);
  const roughC = makeCanvas(size);
  const height = new Float32Array(size * size);

  const cCtx = colorC.getContext('2d');
  const rCtx = roughC.getContext('2d');
  if (cCtx && rCtx) {
    const cImg = cCtx.createImageData(size, size);
    const rImg = rCtx.createImageData(size, size);
    const cd = cImg.data;
    const rd = rImg.data;

    // Horizontal directional CNC brush lines
    const rowBias = new Float32Array(size);
    for (let y = 0; y < size; y++) rowBias[y] = (rand() - 0.5) * 28;

    for (let y = 0; y < size; y++) {
      const rb = rowBias[y];
      for (let x = 0; x < size; x++) {
        const fine = (rand() - 0.5) * 10;
        const streak = Math.sin((x / size) * Math.PI * 6 + y * 0.4) * 5;
        const v = 136 + rb + fine + streak;
        const i = y * size + x;
        height[i] = Math.max(0, Math.min(1, 0.5 + (rb + fine) * 0.015));

        const p = i * 4;
        cd[p] = Math.max(70, Math.min(210, Math.round(v)));
        cd[p + 1] = Math.max(74, Math.min(214, Math.round(v + 3)));
        cd[p + 2] = Math.max(80, Math.min(220, Math.round(v + 7)));
        cd[p + 3] = 255;

        const rv = Math.max(60, Math.min(170, Math.round(112 - rb * 1.1 + fine)));
        rd[p] = rv; rd[p + 1] = rv; rd[p + 2] = rv; rd[p + 3] = 255;
      }
    }
    cCtx.putImageData(cImg, 0, 0);
    rCtx.putImageData(rImg, 0, 0);
  }

  const normalC = heightToNormalCanvas(height, size, 1.25);
  const pack: PBRTexturePack = {
    map: configureTex(colorC, true, 3, 3, aniso),
    normalMap: configureTex(normalC, false, 3, 3, aniso),
    roughnessMap: configureTex(roughC, false, 3, 3, aniso),
  };
  caches[tier].brushedMetal = pack;
  return pack;
}

// ============================================================================
// 5. CROSS-DRILLED & GROOVED BRAKE DISCS (Behind Rims)
// ============================================================================

export function getBrakeDiscTextures(lowSpec: boolean): PBRTexturePack {
  const tier = lowSpec ? 'low' : 'high';
  if (caches[tier].brakeDisc) return caches[tier].brakeDisc!;

  const size = lowSpec ? 256 : 512;
  const aniso = lowSpec ? 4 : 8;
  const rand = mulberry32(31415);

  const colorC = makeCanvas(size);
  const roughC = makeCanvas(size);
  const height = new Float32Array(size * size);

  const cCtx = colorC.getContext('2d');
  const rCtx = roughC.getContext('2d');
  if (cCtx && rCtx) {
    const cImg = cCtx.createImageData(size, size);
    const rImg = rCtx.createImageData(size, size);
    const cd = cImg.data;
    const rd = rImg.data;
    const half = size * 0.5;

    for (let y = 0; y < size; y++) {
      const dy = (y - half) / half;
      for (let x = 0; x < size; x++) {
        const dx = (x - half) / half;
        const r = Math.sqrt(dx * dx + dy * dy);
        const ang = Math.atan2(dy, dx);

        let v = 135;
        let rough = 105;
        let h = 0.55;

        if (r < 0.38) {
          // Dark anodized rotor hat / center hub
          v = 48 + (rand() - 0.5) * 6;
          rough = 148;
          h = 0.65;
        } else if (r <= 0.96) {
          // Concentric brushed steel rotor scoring grooves
          const rings = Math.sin(r * 190) * 14 + Math.sin(r * 65) * 8;
          const radialSheen = Math.cos(ang * 2) * 12;
          v = 146 + rings + radialSheen + (rand() - 0.5) * 8;
          rough = 82 + Math.abs(rings) * 1.4;
          h = 0.5 + rings * 0.012;

          // Cross-drilled cooling holes arranged in 12 spiral sets of 3
          const sector = ((ang + Math.PI) / (Math.PI * 2)) * 12;
          const fracS = sector - Math.floor(sector);
          for (let k = 0; k < 3; k++) {
            const hr = 0.50 + k * 0.16;
            const ha = 0.25 + k * 0.14;
            const dRad = r - hr;
            const dAng = (fracS - ha) * r * 0.52;
            if (dRad * dRad + dAng * dAng < 0.00055) {
              v = 20;
              rough = 235;
              h = 0.05;
            }
          }
        } else {
          // Outer vented edge
          v = 65;
          rough = 175;
          h = 0.3;
        }

        const i = y * size + x;
        height[i] = Math.max(0, Math.min(1, h));
        const p = i * 4;
        cd[p] = Math.max(15, Math.min(220, Math.round(v)));
        cd[p + 1] = Math.max(15, Math.min(222, Math.round(v + 2)));
        cd[p + 2] = Math.max(18, Math.min(228, Math.round(v + 5)));
        cd[p + 3] = 255;

        const rv = Math.max(45, Math.min(245, Math.round(rough)));
        rd[p] = rv; rd[p + 1] = rv; rd[p + 2] = rv; rd[p + 3] = 255;
      }
    }
    cCtx.putImageData(cImg, 0, 0);
    rCtx.putImageData(rImg, 0, 0);
  }

  const normalC = heightToNormalCanvas(height, size, 1.6);
  const pack: PBRTexturePack = {
    map: configureTex(colorC, true, 1, 1, aniso),
    normalMap: configureTex(normalC, false, 1, 1, aniso),
    roughnessMap: configureTex(roughC, false, 1, 1, aniso),
  };
  caches[tier].brakeDisc = pack;
  return pack;
}

// ============================================================================
// 6. POST-IMPACT METALLIC SCRATCHES & CRUMPLED SHEET-METAL WEAR
// ============================================================================

export function getCrashWearTextures(lowSpec: boolean): CrashWearTexturePack {
  const tier = lowSpec ? 'low' : 'high';
  if (caches[tier].crashWear) return caches[tier].crashWear!;

  const size = lowSpec ? 256 : 512;
  const aniso = lowSpec ? 4 : 8;
  const rand = mulberry32(55813);

  const roughC = makeCanvas(size);
  const height = new Float32Array(size * size);
  const rCtx = roughC.getContext('2d');

  // Base crumpled metal low-frequency dents in heightfield
  const freq = (Math.PI * 2) / size;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dent =
        Math.sin(x * freq * 3 + Math.cos(y * freq * 2)) * 0.22 +
        Math.cos(y * freq * 4 - x * freq * 2) * 0.16 +
        Math.sin((x + y) * freq * 6) * 0.08;
      height[y * size + x] = 0.5 + dent;
    }
  }

  if (rCtx) {
    // Base slightly scuffed clearcoat roughness
    rCtx.fillStyle = '#383838';
    rCtx.fillRect(0, 0, size, size);

    // Draw sharp metallic impact scrape clusters & chipped-paint streaks
    const scrapeCount = lowSpec ? 95 : 170;
    for (let s = 0; s < scrapeCount; s++) {
      const x0 = rand() * size;
      const y0 = rand() * size;
      const angle = (rand() - 0.5) * 0.65 + (s % 3 === 0 ? 0.45 : -0.2);
      const len = (0.06 + rand() * 0.26) * size;
      const x1 = x0 + Math.cos(angle) * len;
      const y1 = y0 + Math.sin(angle) * len;
      const w = Math.max(0.7, rand() * (size / 140));

      rCtx.strokeStyle = rand() > 0.35 ? 'rgba(235, 235, 235, 0.85)' : 'rgba(175, 175, 175, 0.65)';
      rCtx.lineWidth = w;
      rCtx.beginPath();
      rCtx.moveTo(x0, y0);
      rCtx.lineTo(x1, y1);
      rCtx.stroke();

      // Carve matching scratch groove into heightfield for crisp normalMap reflection break-up
      const steps = Math.max(8, Math.ceil(len * 0.75));
      const mask = size - 1;
      for (let k = 0; k <= steps; k++) {
        const t = k / steps;
        const px = Math.round(x0 + (x1 - x0) * t) & mask;
        const py = Math.round(y0 + (y1 - y0) * t) & mask;
        const idx = py * size + px;
        height[idx] = Math.max(0.05, height[idx] - 0.38);
      }
    }
  }

  const normalC = heightToNormalCanvas(height, size, 2.1);
  const pack: CrashWearTexturePack = {
    normalMap: configureTex(normalC, false, 3, 3, aniso),
    roughnessMap: configureTex(roughC, false, 3, 3, aniso),
  };
  caches[tier].crashWear = pack;
  return pack;
}
