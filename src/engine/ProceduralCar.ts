import {
  BoxGeometry, BufferAttribute, BufferGeometry, CylinderGeometry, DoubleSide, Group, LatheGeometry,
  Material, Mesh, MeshPhysicalMaterial, MeshStandardMaterial, SphereGeometry, TorusGeometry, Vector2, Vector3,
} from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { CarProfile, smoothstep } from './CarProfile';
import {
  getBrakeDiscTextures,
  getBrushedMetalTextures,
  getCarbonFiberTextures,
} from './ProceduralTextures';

/**
 * ProceduralCar
 * -------------
 * Sculpts a multi-part sports sedan from a CarProfile: hood, trunk lid, front &
 * rear bumpers with grille/intakes, fenders and rear quarters with flared
 * wheel arches, four doors, sills, greenhouse (roof, pillars, glass), lights,
 * mirrors, interior, engine block and four detailed wheels (tire, rim, disc,
 * caliper). Every panel is a separate mesh tagged with the deform group whose
 * physics nodes it skins to.
 */

export const enum DeformGroup {
  Body = 0,
  Hood = 1,
  Trunk = 2,
  BumperF = 3,
  BumperR = 4,
  DoorL = 5,
  DoorR = 6,
  SideL = 7,
  SideR = 8,
}
export const DEFORM_GROUP_COUNT = 9;
export const DEFORM_GROUP_NAMES = ['body', 'hood', 'trunk', 'bumperF', 'bumperR', 'doorL', 'doorR', 'sideL', 'sideR'];

export interface CarPart {
  name: string;
  group: DeformGroup;
  mesh: Mesh;
}

export interface CarMaterials {
  paint: MeshStandardMaterial;
  glass: MeshStandardMaterial;
  trim: MeshStandardMaterial;
  chrome: MeshStandardMaterial;
  plastic: MeshStandardMaterial;
  rubber: MeshStandardMaterial;
  rim: MeshStandardMaterial;
  disc: MeshStandardMaterial;
  caliper: MeshStandardMaterial;
  headlight: MeshStandardMaterial;
  drl: MeshStandardMaterial;
  taillight: MeshStandardMaterial;
  interior: MeshStandardMaterial;
  engine: MeshStandardMaterial;
  plate: MeshStandardMaterial;
}

export function createCarMaterials(paintColor = 0x8a9299, lowSpec = false): CarMaterials {
  const carbon = getCarbonFiberTextures(lowSpec);
  const metal = getBrushedMetalTextures(lowSpec);
  const disc = getBrakeDiscTextures(lowSpec);
  return {
    paint: new MeshPhysicalMaterial({
      color: paintColor,
      metalness: 0.72,
      roughness: 0.12,
      clearcoat: 1.0,
      clearcoatRoughness: 0.04,
      reflectivity: 1.0,
      side: DoubleSide,
      envMapIntensity: 1.85,
    }),
    glass: new MeshStandardMaterial({ color: 0x0c1218, metalness: 0.92, roughness: 0.04, transparent: true, opacity: 0.86, side: DoubleSide, envMapIntensity: 1.9 }),
    trim: new MeshPhysicalMaterial({
      color: 0xffffff,
      map: carbon.map,
      normalMap: carbon.normalMap,
      normalScale: new Vector2(0.55, 0.55),
      roughnessMap: carbon.roughnessMap,
      metalness: 0.25,
      roughness: 0.32,
      clearcoat: 0.92,
      clearcoatRoughness: 0.07,
      envMapIntensity: 1.55,
      side: DoubleSide,
    }),
    chrome: new MeshStandardMaterial({ color: 0xd8dde2, metalness: 1, roughness: 0.14 }),
    plastic: new MeshStandardMaterial({ color: 0x17191c, metalness: 0.1, roughness: 0.7, side: DoubleSide }),
    rubber: new MeshStandardMaterial({ color: 0x141414, metalness: 0, roughness: 0.92 }),
    rim: new MeshStandardMaterial({ color: 0x9aa1a8, metalness: 0.95, roughness: 0.22 }),
    disc: new MeshStandardMaterial({
      color: 0xffffff,
      map: disc.map,
      normalMap: disc.normalMap,
      normalScale: new Vector2(0.5, 0.5),
      roughnessMap: disc.roughnessMap,
      metalness: 0.92,
      roughness: 0.42,
      envMapIntensity: 1.2,
    }),
    caliper: new MeshStandardMaterial({ color: 0xc8202a, metalness: 0.35, roughness: 0.32 }),
    headlight: new MeshStandardMaterial({ color: 0xdfe8f2, emissive: 0xbfd8ff, emissiveIntensity: 0.55, metalness: 0.8, roughness: 0.1, side: DoubleSide }),
    drl: new MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 2.2, side: DoubleSide }),
    taillight: new MeshStandardMaterial({ color: 0x5a0408, emissive: 0xff1020, emissiveIntensity: 0.7, metalness: 0.4, roughness: 0.2, side: DoubleSide }),
    interior: new MeshStandardMaterial({ color: 0x1b1c1e, metalness: 0, roughness: 0.85 }),
    engine: new MeshStandardMaterial({
      color: 0xd0d6dd,
      map: metal.map,
      normalMap: metal.normalMap,
      normalScale: new Vector2(0.42, 0.42),
      roughnessMap: metal.roughnessMap,
      metalness: 0.82,
      roughness: 0.42,
      envMapIntensity: 1.35,
    }),
    plate: new MeshStandardMaterial({ color: 0xe9ecef, metalness: 0.2, roughness: 0.5 }),
  };
}

// ---------------------------------------------------------------- helpers

type SurfaceFn = (u: number, v: number, out: Vector3) => void;

function sortedUnique(values: number[]): number[] {
  const s = values.slice().sort((a, b) => a - b);
  const out: number[] = [];
  for (const v of s) if (out.length === 0 || v - out[out.length - 1] > 1e-5) out.push(v);
  return out;
}
function uniform(n: number, a = 0, b = 1): number[] {
  const out: number[] = [];
  for (let i = 0; i <= n; i++) out.push(a + ((b - a) * i) / n);
  return out;
}

/**
 * Builds parametric surface geometry; quads are split into classes (e.g. paint/glass)
 * with compacted vertex buffers so each class becomes its own mesh.
 */
function surfaceGeometries(us: number[], vs: number[], fn: SurfaceFn, classify: (u: number, v: number) => number, classCount: number): (BufferGeometry | null)[] {
  const nu = us.length, nv = vs.length;
  const pos = new Float32Array(nu * nv * 3);
  const p = new Vector3();
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
    fn(us[i], vs[j], p);
    const k = (i * nv + j) * 3;
    pos[k] = p.x; pos[k + 1] = p.y; pos[k + 2] = p.z;
  }
  const out: (BufferGeometry | null)[] = [];
  for (let c = 0; c < classCount; c++) {
    const remap = new Int32Array(nu * nv).fill(-1);
    const verts: number[] = [];
    const uvs: number[] = [];
    const idx: number[] = [];
    const vid = (i: number, j: number) => {
      const key = i * nv + j;
      if (remap[key] < 0) {
        remap[key] = verts.length / 3;
        verts.push(pos[key * 3], pos[key * 3 + 1], pos[key * 3 + 2]);
        uvs.push(us[i], vs[j]);
      }
      return remap[key];
    };
    for (let i = 0; i < nu - 1; i++) for (let j = 0; j < nv - 1; j++) {
      const cu = (us[i] + us[i + 1]) / 2, cv = (vs[j] + vs[j + 1]) / 2;
      if (classify(cu, cv) !== c) continue;
      const a = vid(i, j), b = vid(i + 1, j), d = vid(i, j + 1), e = vid(i + 1, j + 1);
      idx.push(a, b, e, a, e, d);
    }
    if (idx.length === 0) { out.push(null); continue; }
    const g = new BufferGeometry();
    g.setAttribute('position', new BufferAttribute(new Float32Array(verts), 3));
    g.setAttribute('uv', new BufferAttribute(new Float32Array(uvs), 2));
    g.setIndex(idx);
    g.computeVertexNormals();
    out.push(g);
  }
  return out;
}

function surface(us: number[], vs: number[], fn: SurfaceFn): BufferGeometry {
  return surfaceGeometries(us, vs, fn, () => 0, 1)[0]!;
}

/** Mirror a geometry across X (left -> right side) keeping triangle winding consistent. */
function mirrorX(g: BufferGeometry): BufferGeometry {
  const m = g.clone();
  const pos = m.getAttribute('position') as BufferAttribute;
  for (let i = 0; i < pos.count; i++) pos.setX(i, -pos.getX(i));
  const index = m.getIndex();
  if (index) {
    const arr = index.array as Uint16Array | Uint32Array;
    for (let i = 0; i < arr.length; i += 3) { const t = arr[i + 1]; arr[i + 1] = arr[i + 2]; arr[i + 2] = t; }
  }
  m.computeVertexNormals();
  return m;
}

function nonIndexedMerge(geoms: BufferGeometry[]): BufferGeometry {
  const prepared = geoms.map((g) => {
    let x = g.index ? g.toNonIndexed() : g;
    if (!x.getAttribute('uv')) x.setAttribute('uv', new BufferAttribute(new Float32Array(x.getAttribute('position').count * 2), 2));
    if (!x.getAttribute('normal')) x.computeVertexNormals();
    for (const name of Object.keys(x.attributes)) if (!['position', 'normal', 'uv'].includes(name)) x.deleteAttribute(name);
    return x;
  });
  return mergeGeometries(prepared, false)!;
}

// ---------------------------------------------------------------- body panels

/** Side panel with a flared wheel arch, built in polar coordinates around the wheel center. */
function archPanel(P: CarProfile, z0: number, z1: number, zc: number, yc: number, R: number, side: 1 | -1): BufferGeometry {
  const corner0 = Math.atan2(P.beltY(z1) - yc, z1 - zc);
  const corner1 = Math.atan2(P.beltY(z0) - yc, z0 - zc);
  const thetas = sortedUnique([...uniform(40, 0, Math.PI), corner0, corner1]);
  const rs = uniform(9);
  const outer = (th: number): number => {
    const c = Math.cos(th), s = Math.sin(th);
    let tMax = 10;
    if (c > 1e-6) tMax = Math.min(tMax, (z1 - zc) / c);
    if (c < -1e-6) tMax = Math.min(tMax, (z0 - zc) / c);
    // find first crossing of the shoulder line
    let lo = R, hi = tMax;
    const above = (t: number) => yc + s * t >= P.beltY(zc + c * t);
    if (!above(hi)) return hi;
    for (let k = 0; k < 28; k++) { const m = (lo + hi) / 2; if (above(m)) hi = m; else lo = m; }
    return hi;
  };
  const outerCache = new Map<number, number>();
  return surface(thetas, rs, (th, r, out) => {
    let t = outerCache.get(th);
    if (t === undefined) { t = outer(th); outerCache.set(th, t); }
    const rr = R + (t - R) * Math.pow(r, 1.15);
    const z = zc + Math.cos(th) * rr;
    const y = yc + Math.sin(th) * rr;
    const flare = 0.03 * Math.exp(-Math.pow((rr - R) / 0.09, 2)) * (0.4 + 0.6 * Math.sin(th));
    const crease = 0.006 * Math.exp(-Math.pow((y - 0.74) / 0.025, 2));
    out.set(side * (P.sideHalfW(z, y) + flare + crease), y, z);
  });
}

function sidePanel(P: CarProfile, z0: number, z1: number, y0: number, side: 1 | -1, topInset = 0): BufferGeometry {
  return surface(uniform(16), uniform(12), (u, v, out) => {
    const z = z0 + (z1 - z0) * u;
    const yTop = P.beltY(z) - topInset;
    const y = y0 + (yTop - y0) * v;
    const crease = 0.006 * Math.exp(-Math.pow((y - 0.74) / 0.025, 2));
    out.set(side * (P.sideHalfW(z, y) + crease), y, z);
  });
}

function lidSurface(P: CarProfile, zA: number, zB: number, nose: number, front: boolean): BufferGeometry {
  return surface(uniform(28), uniform(20, -1, 1), (u, v, out) => {
    let z = zA + (zB - zA) * u;
    const belt = P.beltY(z), top = P.topY(z);
    const hw = P.sideHalfW(z, belt) - 0.012;
    const crownT = Math.pow(Math.max(0, 1 - v * v), 0.55);
    let y = belt + (top - belt) * crownT;
    if (front) {
      // power domes
      y += 0.012 * Math.exp(-Math.pow((Math.abs(v) - 0.42) / 0.12, 2)) * smoothstep(0.05, 0.4, u) * (1 - smoothstep(0.75, 1, u));
    }
    // leading edge rolls downwards
    const curl = smoothstep(0.88, 1, u);
    y -= nose * curl * curl;
    z -= Math.sign(zB - zA) * 0.02 * curl * curl;
    out.set(v * hw * (1 - 0.03 * curl), y, z);
  });
}

/** Superellipse wrap-around used by bumpers and lamp overlays. */
function noseSurface(P: CarProfile, front: boolean, y0: number, y1: number, offset: number, phiRange: [number, number], tRange: [number, number], nu = 30, nv = 10): BufferGeometry {
  const zc = front ? P.halfL - 0.425 : -P.halfL + 0.375;
  const depth = front ? 0.425 : 0.375;
  const A = 0.935 * (P.width / 1.88);
  return surface(uniform(nu, phiRange[0], phiRange[1]), uniform(nv, tRange[0], tRange[1]), (phi, t, out) => {
    const y = y0 + (y1 - y0) * t;
    const sp = Math.sin(phi), cp = Math.cos(phi);
    let k = 1 - 0.035 * Math.pow((t - 0.5) * 2, 2);
    if (t > 0.88) k -= 0.12 * Math.pow((t - 0.88) / 0.12, 2) * 0.4;
    if (t < 0.08) k -= 0.03 * (1 - t / 0.08);
    const x = A * Math.sign(sp) * Math.pow(Math.abs(sp), 0.5) * k + offset * sp;
    const zz = depth * Math.sign(cp) * Math.pow(Math.abs(cp), 0.5) * k + offset * cp;
    out.set(x, y, front ? zc + zz : zc - zz);
  });
}

// ---------------------------------------------------------------- wheels

export interface WheelGeometrySet {
  tire: BufferGeometry;
  rim: BufferGeometry;
  disc: BufferGeometry;
  caliper: BufferGeometry;
}

export function createWheelGeometries(radius: number, width: number): WheelGeometrySet {
  const rimR = radius * 0.72;
  const hw = width / 2;
  // tire cross-section (radius, axial) revolved around Y, then turned to the X axle
  const prof: Vector2[] = [];
  const seg = 10;
  prof.push(new Vector2(rimR * 0.98, -hw * 0.92));
  for (let i = 0; i <= seg; i++) {
    const a = -Math.PI / 2 + (i / seg) * (Math.PI / 2);
    prof.push(new Vector2(radius - 0.04 + Math.cos(a) * 0.04, -hw + 0.04 + Math.sin(a) * 0.04 + 0.04 * (1 - Math.cos(a)) * 0));
  }
  for (let i = 0; i <= seg; i++) {
    const a = (i / seg) * (Math.PI / 2);
    prof.push(new Vector2(radius - 0.04 + Math.cos(a) * 0.04, hw - 0.04 + Math.sin(a) * 0.04));
  }
  prof.push(new Vector2(rimR * 0.98, hw * 0.92));
  const tire = new LatheGeometry(prof, 44);
  tire.rotateZ(Math.PI / 2);

  // rim barrel + dished face
  const rimProf = [
    new Vector2(rimR * 0.96, -hw * 0.9), new Vector2(rimR, -hw * 0.85), new Vector2(rimR, hw * 0.75),
    new Vector2(rimR * 1.02, hw * 0.85), new Vector2(rimR * 0.95, hw * 0.88),
  ];
  const barrel = new LatheGeometry(rimProf, 36);
  barrel.rotateZ(Math.PI / 2);
  const parts: BufferGeometry[] = [barrel];
  const hub = new CylinderGeometry(rimR * 0.24, rimR * 0.28, 0.06, 20);
  hub.rotateZ(Math.PI / 2); hub.translate(hw * 0.62, 0, 0);
  parts.push(hub);
  const cap = new CylinderGeometry(rimR * 0.12, rimR * 0.12, 0.02, 16);
  cap.rotateZ(Math.PI / 2); cap.translate(hw * 0.68, 0, 0);
  parts.push(cap);
  for (let s = 0; s < 5; s++) {
    for (const off of [-0.11, 0.11]) {
      const spoke = new BoxGeometry(0.035, rimR * 0.78, 0.034);
      spoke.translate(0, rimR * 0.58, 0);
      spoke.rotateX(((s / 5) * Math.PI * 2) + off);
      spoke.translate(hw * 0.6, 0, 0);
      parts.push(spoke);
    }
  }
  const rim = nonIndexedMerge(parts);

  const disc = new CylinderGeometry(rimR * 0.82, rimR * 0.82, 0.028, 32);
  disc.rotateZ(Math.PI / 2);
  disc.translate(-hw * 0.05, 0, 0);
  const caliper = new BoxGeometry(0.07, 0.12, 0.17);
  caliper.translate(hw * 0.12, rimR * 0.62, -rimR * 0.18);
  return { tire, rim, disc, caliper };
}

/** Wheel assembly: knuckle (steers, holds caliper) -> spinner (tire, rim, disc). */
export function buildWheelVisual(set: WheelGeometrySet, mats: CarMaterials, left: boolean): { knuckle: Group; spinner: Group } {
  const knuckle = new Group();
  const spinner = new Group();
  knuckle.add(spinner);
  const tire = new Mesh(set.tire, mats.rubber);
  const rim = new Mesh(set.rim, mats.rim);
  const disc = new Mesh(set.disc, mats.disc);
  const caliper = new Mesh(set.caliper, mats.caliper);
  tire.castShadow = true; rim.castShadow = true;
  spinner.add(tire, rim, disc);
  knuckle.add(caliper);
  if (!left) {
    // outer face points to -X on the right-hand side
    spinner.scale.x = -1;
    caliper.scale.x = -1;
  }
  return { knuckle, spinner };
}

// ---------------------------------------------------------------- full car

export function buildProceduralCar(P: CarProfile, mats: CarMaterials): CarPart[] {
  const parts: CarPart[] = [];
  const add = (name: string, group: DeformGroup, geo: BufferGeometry | null, mat: Material, shadow = true) => {
    if (!geo) return;
    const mesh = new Mesh(geo, mat);
    mesh.name = name;
    mesh.castShadow = shadow;
    mesh.frustumCulled = false; // vertices move every frame; bounds would be stale
    parts.push({ name, group, mesh });
  };
  const gap = 0.006;
  const archR = P.wheelRadius + 0.065;
  const archY = P.wheelRadius + 0.005;

  // Hood (4x5 physics grid, high-res visual) and trunk lid
  add('hood', DeformGroup.Hood, lidSurface(P, P.cowlZ + gap, P.hoodFrontZ, 0.05, true), mats.paint);
  add('trunk', DeformGroup.Trunk, lidSurface(P, P.deckZ - gap, P.trunkEndZ - 0.05, 0.2, false), mats.paint);

  // Fenders ("joues"), doors, rear quarters, sills: left side then mirrored
  const fenderZ1 = P.halfL - 0.2;
  const sideParts: [string, BufferGeometry][] = [
    ['fender', archPanel(P, P.cowlZ + gap, fenderZ1, P.frontAxleZ, archY, archR, 1)],
    ['doorFront', sidePanel(P, 0.0, P.cowlZ - gap, 0.26, 1)],
    ['doorRear', sidePanel(P, P.rearAxleZ + archR + 0.02, -gap, 0.26, 1)],
    ['quarter', archPanel(P, -P.halfL + 0.14, P.rearAxleZ + archR + 0.02 - gap, P.rearAxleZ, archY, archR, 1)],
  ];
  for (const [name, g] of sideParts) {
    const isDoor = name === 'doorFront' || name === 'doorRear';
    const grpL = isDoor ? DeformGroup.DoorL : DeformGroup.SideL;
    const grpR = isDoor ? DeformGroup.DoorR : DeformGroup.SideR;
    add(`${name}L`, grpL, g, mats.paint);
    add(`${name}R`, grpR, mirrorX(g), mats.paint);
  }
  const sill = surface(uniform(14), uniform(3), (u, v, out) => {
    const z = (P.rearAxleZ + archR) + (P.frontAxleZ - archR - (P.rearAxleZ + archR)) * u;
    const y = 0.16 + 0.1 * v;
    out.set(P.sideHalfW(z, y) - 0.012 + 0.01 * v, y, z);
  });
  add('sillL', DeformGroup.SideL, sill, mats.plastic);
  add('sillR', DeformGroup.SideR, mirrorX(sill), mats.plastic);

  // Greenhouse: roof + pillars (paint), glass, window trim, from one loft
  const zs = sortedUnique([...uniform(36, P.deckZ, P.cowlZ), P.roofFrontZ, P.roofRearZ, -0.1, -0.02, -1.12, 0.86]);
  // s = 0 at the belt line, 1 at the roof centre; t = s/2 on the left, 1 - s/2 on the right
  const sBounds = [0.06, 0.54, 0.6, 0.68];
  const ts = sortedUnique([...uniform(30), ...sBounds.map((s) => s / 2), ...sBounds.map((s) => 1 - s / 2)]);
  const glassFn = (z: number, t: number): number => {
    const s = (t > 0.5 ? 1 - t : t) / 0.5;
    if (s < 0.6) {
      if (s < 0.06) return 2; // belt trim
      if (s > 0.54) return 0; // roof rail
      if (z > 0.86 || z < -1.12) return 0;
      if (z > -0.1 && z < -0.02) return 2; // B-pillar (black)
      return 1;
    }
    const inPillar = s < 0.68;
    if (z > P.roofFrontZ) return inPillar ? 0 : 1; // windshield
    if (z < P.roofRearZ) return inPillar ? 0 : 1;  // rear window
    return 0; // roof
  };
  const loft: SurfaceFn = (z, t, out) => {
    const left = t <= 0.5;
    const s = left ? t / 0.5 : (1 - t) / 0.5;
    const belt = P.beltY(z), top = P.topY(z) - 0.02;
    const xb = P.sideHalfW(z, belt) - 0.055;
    const xr = Math.min(P.roofHalfWidth(z), xb - 0.02);
    let x: number, y: number;
    const split = 0.6;
    if (s <= split) {
      const k = s / split;
      x = xb + (xr - xb) * k + 0.03 * Math.sin(Math.PI * k) * 0.5;
      y = belt + (top - 0.03 - belt) * Math.pow(k, 0.9);
    } else {
      const k = (s - split) / (1 - split);
      const ang = k * Math.PI / 2;
      x = xr * Math.cos(ang);
      y = top - 0.03 + 0.03 * Math.sin(ang);
    }
    out.set(left ? x : -x, y, z);
  };
  const greenhouse = surfaceGeometries(zs, ts, loft, (z, t) => glassFn(z, t), 3);
  add('cabin', DeformGroup.Body, greenhouse[0], mats.paint);
  add('glass', DeformGroup.Body, greenhouse[1], mats.glass, false);
  add('windowTrim', DeformGroup.Body, greenhouse[2], mats.trim);

  // Front bumper, grille, intake, plate
  add('bumperFront', DeformGroup.BumperF, noseSurface(P, true, 0.17, 0.64, 0, [-Math.PI / 2, Math.PI / 2], [0, 1], 34, 12), mats.paint);
  add('intakeFront', DeformGroup.BumperF, noseSurface(P, true, 0.17, 0.64, 0.006, [-1.05, 1.05], [0.07, 0.3], 22, 3), mats.plastic);
  {
    const zt = P.halfL + 0.008;
    const pieces: BufferGeometry[] = [];
    const grille = new BoxGeometry(0.8, 0.2, 0.02); grille.translate(0, 0.505, zt - 0.02);
    const grilleGeo = grille;
    const bar = new BoxGeometry(0.78, 0.022, 0.03); bar.translate(0, 0.505, zt - 0.002); pieces.push(bar);
    const frame = new TorusGeometry(0.42, 0.012, 6, 36); frame.scale(1, 0.26, 1); frame.translate(0, 0.505, zt - 0.006); pieces.push(frame);
    const ring = new TorusGeometry(0.075, 0.009, 8, 28); ring.translate(0, 0.505, zt + 0.012); pieces.push(ring);
    for (let i = 0; i < 3; i++) {
      const sp = new BoxGeometry(0.008, 0.075, 0.008);
      sp.translate(0, 0.0375, 0); sp.rotateZ((i / 3) * Math.PI * 2); sp.translate(0, 0.505, zt + 0.012); pieces.push(sp);
    }
    add('grille', DeformGroup.BumperF, grilleGeo, mats.trim);
    add('grilleChrome', DeformGroup.BumperF, nonIndexedMerge(pieces), mats.chrome);
    const plate = new BoxGeometry(0.52, 0.11, 0.01); plate.translate(0, 0.3, zt - 0.004);
    add('plateFront', DeformGroup.BumperF, plate, mats.plate);
  }

  // Headlights + DRL strips (mounted on the front structure)
  for (const side of [1, -1]) {
    const range: [number, number] = side > 0 ? [0.62, 1.28] : [-1.28, -0.62];
    const grp = side > 0 ? DeformGroup.SideL : DeformGroup.SideR;
    add(side > 0 ? 'headlightL' : 'headlightR', grp, noseSurface(P, true, 0.585, 0.705, 0.01, range, [0, 1], 12, 4), mats.headlight, false);
    add(side > 0 ? 'drlL' : 'drlR', grp, noseSurface(P, true, 0.585, 0.705, 0.016, range, [0.05, 0.18], 12, 1), mats.drl, false);
  }

  // Rear bumper, diffuser, exhausts, plate
  add('bumperRear', DeformGroup.BumperR, noseSurface(P, false, 0.19, 0.74, 0, [-Math.PI / 2, Math.PI / 2], [0, 1], 34, 12), mats.paint);
  add('diffuser', DeformGroup.BumperR, noseSurface(P, false, 0.19, 0.74, 0.006, [-1.0, 1.0], [0.0, 0.2], 22, 3), mats.plastic);
  {
    const zt = -P.halfL;
    const ex: BufferGeometry[] = [];
    for (const x of [-0.55, -0.42, 0.42, 0.55]) {
      const pipe = new CylinderGeometry(0.038, 0.038, 0.12, 16, 1, true);
      pipe.rotateX(Math.PI / 2); pipe.translate(x, 0.24, zt + 0.03); ex.push(pipe);
    }
    add('exhaust', DeformGroup.BumperR, nonIndexedMerge(ex), mats.chrome);
    const plate = new BoxGeometry(0.52, 0.11, 0.01); plate.translate(0, 0.52, zt - 0.006);
    add('plateRear', DeformGroup.BumperR, plate, mats.plate);
  }

  // Taillights wrap the rear corners above the bumper
  for (const side of [1, -1]) {
    const range: [number, number] = side > 0 ? [0.72, 1.5] : [-1.5, -0.72];
    add(side > 0 ? 'taillightL' : 'taillightR', DeformGroup.Body, noseSurface(P, false, 0.76, 0.93, 0.008, range, [0, 1], 12, 4), mats.taillight, false);
  }
  // rear fascia between bumper and trunk lid
  add('rearPanel', DeformGroup.Body, noseSurface(P, false, 0.72, 0.97, -0.004, [-1.5, 1.5], [0, 1], 24, 4), mats.paint);
  // front panel behind headlights/grille (closes the nose)
  add('frontPanel', DeformGroup.BumperF, noseSurface(P, true, 0.6, 0.74, -0.012, [-1.4, 1.4], [0, 1], 24, 3), mats.trim);

  // Mirrors + door handles (bound to DoorL and DoorR so they move inward with door dents)
  for (const side of [1, -1]) {
    const grp = side > 0 ? DeformGroup.DoorL : DeformGroup.DoorR;
    const sfx = side > 0 ? 'L' : 'R';
    const m: BufferGeometry[] = [];
    const h: BufferGeometry[] = [];
    const shell = new SphereGeometry(0.1, 14, 10); shell.scale(0.9, 0.55, 0.6);
    shell.translate(side * (P.sideHalfW(0.82, 0.98) + 0.07), P.beltY(0.82) + 0.07, 0.8); m.push(shell);
    const arm = new BoxGeometry(0.1, 0.03, 0.06); arm.translate(side * (P.sideHalfW(0.85, 0.95) + 0.01), P.beltY(0.85) + 0.04, 0.82); m.push(arm);
    for (const z of [0.42, -0.55]) {
      const hd = new BoxGeometry(0.02, 0.025, 0.15);
      hd.translate(side * (P.sideHalfW(z, 0.84) + 0.006), 0.84, z); h.push(hd);
    }
    add(`mirror${sfx}`, grp, nonIndexedMerge(m), mats.paint);
    add(`handle${sfx}`, grp, nonIndexedMerge(h), mats.chrome);
  }

  // Underbody, interior, engine block
  const under = surface(uniform(12, -P.halfL + 0.25, P.halfL - 0.25), uniform(4, -1, 1), (z, v, out) => {
    out.set(v * (P.planHalfW(z) - 0.08), 0.165, z);
  });
  add('underbody', DeformGroup.Body, under, mats.plastic, false);
  {
    const int: BufferGeometry[] = [];
    const tub = new BoxGeometry(1.55, 0.42, 2.25); tub.translate(0, 0.42, -0.35); int.push(tub);
    const dash = new BoxGeometry(1.5, 0.22, 0.4); dash.translate(0, 0.78, 0.62); int.push(dash);
    for (const x of [-0.4, 0.4]) {
      const back = new BoxGeometry(0.5, 0.62, 0.14); back.rotateX(-0.18); back.translate(x, 0.88, -0.22); int.push(back);
      const rear = new BoxGeometry(0.5, 0.5, 0.14); rear.rotateX(-0.22); rear.translate(x, 0.82, -1.12); int.push(rear);
    }
    const wheel = new TorusGeometry(0.17, 0.022, 8, 24); wheel.rotateX(-0.35); wheel.translate(0.4, 0.88, 0.42); int.push(wheel);
    add('interior', DeformGroup.Body, nonIndexedMerge(int), mats.interior, false);
    const engine = buildDetailedEngineAssembly(P, 'front', mats);
    for (const m of engine.rigid) parts.push({ name: m.name, group: DeformGroup.Body, mesh: m });
    for (const dp of engine.deformable) parts.push(dp);
  }
  return parts;
}

/**
 * Generates a lightweight (< 650 tris), detailed 3D engine assembly:
 * - `rigid`: metallic V8/L6 engine block, cylinder heads, intake plenum, exhaust manifolds,
 *   and hoses anchored to the central rigid chassis core (0 skinning cost).
 * - `deformable`: subdivided (10×4) front aluminum radiator core + shroud bound to
 *   `DeformGroup.Body` so it visibly bends, folds, and crushes inward on frontal impacts.
 *
 * - `placement === 'rear'`: mid/rear-engine layout for McLaren P1 & mid-engine supercars.
 * - `placement === 'front'`: front-engine layout for BMW M4 CSL, GT-S, and standard cars.
 */
export function buildDetailedEngineAssembly(
  P: CarProfile,
  placement: 'front' | 'rear',
  mats: CarMaterials,
): { rigid: Mesh[]; deformable: CarPart[] } {
  const hs = P.height / 1.38;
  const engZ = placement === 'rear'
    ? P.rearAxleZ + 0.22
    : P.frontAxleZ - 0.04;
  const baseY = 0.28 * hs + 0.14;
  const maxBayY = Math.max(baseY + 0.34, P.topY(engZ) - 0.12);
  const engH = Math.min(0.42, Math.max(0.24, maxBayY - baseY));

  const blockParts: BufferGeometry[] = [];
  const chromeParts: BufferGeometry[] = [];
  const hoseParts: BufferGeometry[] = [];

  // 1. Main engine block & sump (cast aluminum / iron)
  const core = new BoxGeometry(0.56, engH * 0.72, 0.64);
  core.translate(0, baseY + engH * 0.36, engZ);
  blockParts.push(core);

  const sump = new BoxGeometry(0.46, engH * 0.24, 0.56);
  sump.translate(0, baseY + engH * 0.08, engZ);
  blockParts.push(sump);

  // 2. Twin angled V8/L6 cylinder heads & ribbed valve covers
  for (const side of [-1, 1]) {
    const head = new BoxGeometry(0.20, engH * 0.34, 0.58);
    head.rotateZ(side * 0.24);
    head.translate(side * 0.24, baseY + engH * 0.68, engZ);
    blockParts.push(head);

    // Exhaust manifold runners (chrome/stainless)
    for (let k = -1; k <= 1; k++) {
      const runner = new CylinderGeometry(0.022, 0.022, 0.18, 8);
      runner.rotateZ(side * 1.15);
      runner.translate(side * 0.36, baseY + engH * 0.52, engZ + k * 0.16);
      chromeParts.push(runner);
    }
  }

  // 3. Central intake plenum / carbon-metal airbox cover
  const plenum = new BoxGeometry(0.40, engH * 0.22, 0.50);
  plenum.translate(0, baseY + engH * 0.88, engZ);
  blockParts.push(plenum);

  // Twin intake charge pipes (metallic)
  for (const side of [-1, 1]) {
    const pipe = new CylinderGeometry(0.028, 0.028, 0.36, 10);
    pipe.rotateX(Math.PI / 2);
    pipe.translate(side * 0.14, baseY + engH * 0.94, engZ + (placement === 'front' ? 0.18 : -0.14));
    chromeParts.push(pipe);
  }

  // 4. Deformable subdivided front radiator core + fan shroud (compact & tucked inside nose so it never pokes out!)
  const radZ = placement === 'rear' ? P.halfL - 0.58 : P.halfL - 0.50;
  const radBot = Math.max(0.20 * hs, P.floorY + 0.04);
  const radTop = Math.min(0.52 * hs, P.topY(radZ) - 0.14);
  const radH = Math.max(0.15, radTop - radBot);
  const radY = radBot + radH * 0.5;
  const radW = Math.min(0.72, P.planHalfW(radZ) * 1.10);
  const radCoreGeo = new BoxGeometry(radW, radH, 0.045, 10, 4, 1);
  radCoreGeo.translate(0, radY, radZ);
  const radFrameGeo = new BoxGeometry(radW + 0.02, radH + 0.02, 0.025, 10, 4, 1);
  radFrameGeo.translate(0, radY, radZ - 0.035);

  // Coolant hoses connecting engine bay to radiator (or rear intercoolers for mid-engine)
  if (placement === 'front') {
    const hoseLen = Math.max(0.14, radZ - (engZ + 0.24));
    for (const side of [-1, 1]) {
      const hose = new CylinderGeometry(0.018, 0.018, hoseLen, 8);
      hose.rotateX(Math.PI / 2);
      hose.translate(side * 0.24, Math.min(radTop - 0.04, baseY + engH * 0.68), engZ + 0.24 + hoseLen * 0.5);
      hoseParts.push(hose);
    }
  } else {
    // Rear twin intercooler shrouds flanking the rear V8 bay
    for (const side of [-1, 1]) {
      const ic = new BoxGeometry(0.10, 0.24, 0.38);
      ic.translate(side * 0.54, baseY + engH * 0.50, engZ - 0.06);
      chromeParts.push(ic);
      const hose = new CylinderGeometry(0.022, 0.022, 0.26, 8);
      hose.rotateZ(Math.PI / 2);
      hose.translate(side * 0.38, baseY + engH * 0.72, engZ - 0.04);
      hoseParts.push(hose);
    }
  }

  const makeMesh = (name: string, geo: BufferGeometry, mat: MeshStandardMaterial): Mesh => {
    const m = new Mesh(geo, mat);
    m.name = name;
    m.castShadow = true;
    m.receiveShadow = true;
    m.frustumCulled = false;
    return m;
  };

  const rigid: Mesh[] = [
    makeMesh(`engineBlock_${placement}`, nonIndexedMerge(blockParts), mats.engine),
    makeMesh(`engineMetal_${placement}`, nonIndexedMerge(chromeParts), mats.chrome),
  ];
  if (hoseParts.length > 0) {
    rigid.push(makeMesh(`engineHoses_${placement}`, nonIndexedMerge(hoseParts), mats.plastic));
  }

  const deformable: CarPart[] = [
    { name: 'radiatorCore', group: DeformGroup.Body, mesh: makeMesh('radiatorCore', radCoreGeo, mats.chrome) },
    { name: 'radiatorShroud', group: DeformGroup.Body, mesh: makeMesh('radiatorShroud', radFrameGeo, mats.plastic) },
  ];

  return { rigid, deformable };
}
