import {
  MeshLambertMaterial,
  MeshLambertMaterialParameters,
  MeshStandardMaterial,
  MeshStandardMaterialParameters,
  Texture,
} from 'three';

/**
 * Shared Material Registry
 * ------------------------
 * Deduplicates and caches Three.js materials by structural parameter key so
 * identical materials across arena props and zones reuse the exact same
 * WebGL shader program and uniform bindings without compiling duplicates.
 */
const lambertCache = new Map<string, MeshLambertMaterial>();
const standardCache = new Map<string, MeshStandardMaterial>();

function texKey(t: Texture | null | undefined): string {
  return t ? String(t.id) : '-';
}

export function getCachedLambertMaterial(params: MeshLambertMaterialParameters): MeshLambertMaterial {
  const key = [
    params.color ?? '',
    params.emissive ?? '',
    params.emissiveIntensity ?? '',
    params.side ?? '',
    params.transparent ? '1' : '0',
    params.opacity ?? 1,
    params.alphaTest ?? 0,
    texKey(params.map),
  ].join('|');

  const existing = lambertCache.get(key);
  if (existing) return existing;

  const mat = new MeshLambertMaterial(params);
  mat.userData.sharedMaterial = true;
  lambertCache.set(key, mat);
  return mat;
}

export function getCachedStandardMaterial(params: MeshStandardMaterialParameters): MeshStandardMaterial {
  const key = [
    params.color ?? '',
    params.roughness ?? '',
    params.metalness ?? '',
    params.envMapIntensity ?? '',
    params.side ?? '',
    params.transparent ? '1' : '0',
    params.opacity ?? 1,
    params.alphaTest ?? 0,
    texKey(params.map),
    texKey(params.normalMap),
    texKey(params.roughnessMap),
  ].join('|');

  const existing = standardCache.get(key);
  if (existing) return existing;

  const mat = new MeshStandardMaterial(params);
  mat.userData.sharedMaterial = true;
  standardCache.set(key, mat);
  return mat;
}

export function getMaterialRegistryStats(): { lambert: number; standard: number } {
  return { lambert: lambertCache.size, standard: standardCache.size };
}
