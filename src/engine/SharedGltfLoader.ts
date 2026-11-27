import { LoadingManager } from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';

let sharedDraco: DRACOLoader | null = null;
let sharedLoader: GLTFLoader | null = null;

function getSharedDracoLoader(): DRACOLoader {
  if (!sharedDraco) {
    sharedDraco = new DRACOLoader();
    // Standard Three.js Draco WASM/JS decoder path (falls back cleanly if unused by uncompressed GLBs)
    sharedDraco.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.7/');
    sharedDraco.setDecoderConfig({ type: 'wasm' });
  }
  return sharedDraco;
}

/**
 * Returns a single shared GLTFLoader configured with DRACOLoader and MeshoptDecoder
 * so compressed .glb files load fast with minimal bandwidth and zero duplicate decoder init.
 */
export function getSharedGltfLoader(): GLTFLoader {
  if (!sharedLoader) {
    sharedLoader = new GLTFLoader();
    sharedLoader.setDRACOLoader(getSharedDracoLoader());
    sharedLoader.setMeshoptDecoder(MeshoptDecoder);
  }
  return sharedLoader;
}

/**
 * Creates a manager-scoped GLTFLoader (for ZIP blob URL resolution) that still reuses
 * the singleton DRACOLoader and MeshoptDecoder instances.
 */
export function createManagedGltfLoader(manager: LoadingManager): GLTFLoader {
  const loader = new GLTFLoader(manager);
  loader.setDRACOLoader(getSharedDracoLoader());
  loader.setMeshoptDecoder(MeshoptDecoder);
  return loader;
}
