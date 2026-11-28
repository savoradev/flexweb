import {
  ACESFilmicToneMapping, BackSide, BufferAttribute, BufferGeometry, CanvasTexture, Color, CubeCamera, DirectionalLight, FogExp2,
  HemisphereLight, LinearMipmapLinearFilter, LineBasicMaterial, LineSegments, Matrix4, Mesh, MeshBasicMaterial, MeshStandardMaterial, Object3D,
  PCFSoftShadowMap, PerspectiveCamera, PlaneGeometry, PMREMGenerator, Quaternion, Scene, ShaderMaterial, SphereGeometry, SRGBColorSpace,
  Texture, Vector3, WebGLCubeRenderTarget, WebGLRenderer, WebGLRenderTarget,
} from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { getSharedGltfLoader } from './SharedGltfLoader';
import { CrashArena, CrashTestPreset, SKY_COLOR, STUDIO_SHOWROOM_Y } from './CrashArena';
import { clearOldAssetCaches, getCachedAssetBytes, putCachedAssetBytes } from './AssetCache';
import { RainWeatherSystem, WeatherMode } from './RainWeatherSystem';
import { detectDeviceProfile } from '../utils/DeviceDetector';
import { FixedTimestepLoop } from './FixedTimestepLoop';
import { BoundedVehicleTemplateCache, heapUsedMB, isConstrainedDevice, Scratch } from './MemoryGuard';
import {
  loadCachedSkinning,
  PrecomputedBindingData,
  saveCachedSkinning,
  SKINNING_CACHE_SCHEMA_VERSION,
} from './MeshCacheDB';
import { loadSavedCarBuffer, saveUploadedCarGLB, setLastSelectedCarId } from './SavedCarsDB';
import type { SkinningWorkerPartInput, SkinningWorkerRequest, SkinningWorkerResponse } from './skinningWorker';
import { BEAM_BROKEN } from './SoftBodySolver';
import type { SimulatorSettings } from '../types/settings';
import { SoundEngine } from './SoundEngine';
import {
  createProceduralVehicleSource, fitGLBToVehicleSource, SoftBodyVehicle, VehicleTelemetry,
} from './SoftBodyVehicle';
import {
  fetchLocalModelChunked,
  GITHUB_CARS,
  loadRemoteCarWithFallbacks,
  ModelAttribution,
  parseGltfFromZipBytes,
} from './ZipGltfLoader';

/**
 * EngineCore
 * ----------
 * Owns the Three.js scene, WebGLRenderer (pixelRatio capped at 1.25 on 4 GB
 * tablets and 1.5 on desktop, 1024x1024 directional shadow frustum following
 * the car), camera controller (chase / orbit / hood / bumper), optional
 * zero-allocation node-and-beam X-ray overlay, GLB/GLTF/ZIP loader with texture
 * downscaling (<= 1024x1024), and the decoupled FixedTimestepLoop.
 */

export type CameraMode = 'chase' | 'interior' | 'orbit' | 'hood' | 'bumper';

export interface CarOption {
  id: string;
  name: string;
  url?: string;          // static asset URL in public/ (undefined = built-in, remote, or IndexedDB)
  remoteUrls?: string[]; // ordered GitHub raw .gltf / .glb / .zip fallback URLs
  attribution?: ModelAttribution;
  savedInDB?: boolean;   // true = persisted in IndexedDB ('FlexWebSavedCars')
  bytes?: number;
  paint?: number;
}

export interface EngineStats {
  fps: number;
  frameMs: number;
  physicsMs: number;
  resetMs: number;
  heapMB: number;
  pixelRatio: number;
  cameraMode: CameraMode;
  timeScale: number;
  paused: boolean;
  showLattice: boolean;
  soundEnabled: boolean;
  bodyOpacity: number;
  activeCarId: string;
  telemetry: VehicleTelemetry;
}

export type LoadProgressCallback = (step: string, pct: number) => void;

export const BUILTIN_CARS: CarOption[] = [
  { id: 'builtin-silver', name: 'GT-S 450 Coupe (Iridium Silver)', paint: 0x8e969d },
  { id: 'builtin-obsidian', name: 'GT-S 450 Coupe (Obsidian Black)', paint: 0x181b1f },
  { id: 'builtin-crimson', name: 'GT-S 450 Coupe (Designo Crimson)', paint: 0x8e1922 },
  { id: 'builtin-alpine', name: 'GT-S 450 Coupe (Alpine White)', paint: 0xdcdfe4 },
];

const SETTINGS_STORAGE_KEY = 'flexweb_settings_v1';

interface PersistedEngineSettings {
  cameraMode?: CameraMode;
  showLattice?: boolean;
  bodyOpacity?: number;
  paintByCar?: Record<string, number>;
}

function readPersistedSettings(): PersistedEngineSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as PersistedEngineSettings;
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

function writePersistedSettings(patch: Partial<PersistedEngineSettings>): void {
  try {
    const cur = readPersistedSettings();
    const next: PersistedEngineSettings = {
      ...cur,
      ...patch,
      paintByCar: {
        ...(cur.paintByCar ?? {}),
        ...(patch.paintByCar ?? {}),
      },
    };
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(next));
  } catch {
    // ignore storage quota / private browsing
  }
}

export class EngineCore {
  readonly scene = new Scene();
  readonly camera: PerspectiveCamera;
  readonly renderer: WebGLRenderer;
  readonly arena: CrashArena;
  readonly loop: FixedTimestepLoop;
  readonly constrained: boolean;

  vehicle: SoftBodyVehicle;
  cameraMode: CameraMode = 'chase';
  showLattice = false;
  bodyOpacity = 1.0;
  activeCarId = BUILTIN_CARS[0].id;
  lastResetMs = 0.25;

  private readonly container: HTMLElement;
  private readonly hemiLight: HemisphereLight;
  private readonly dirLight: DirectionalLight;
  private readonly gltfLoader = getSharedGltfLoader();
  private readonly skyDome: Mesh;
  private readonly skyUniforms = { uTime: { value: 0 }, uRain: { value: 0 }, uLightning: { value: 0 }, uNight: { value: 0 } };
  private lightningTimer = 6.5;
  private lightningFlash = 0;
  private nightMode = false;
  private nightTarget = 0;
  private nightFactor = 0;
  private headlightsOn = false;
  private readonly dryHemiSky = new Color(0xeef4ff);
  private readonly wetHemiSky = new Color(0x6e8094);
  private readonly nightHemiSky = new Color(0x1b2944);
  private readonly drySunCol = new Color(0xfff6e8);
  private readonly wetSunCol = new Color(0xa9bdd4);
  private readonly nightMoonCol = new Color(0x88a8dc);
  private readonly nightSkyCol = new Color(0x090e1a);
  private readonly tmpSkyBg = new Color();
  private readonly tmpLightCol = new Color();
  private readonly pmrem: PMREMGenerator;
  private readonly studioEnvRT: WebGLRenderTarget;
  private readonly cubeRT: WebGLCubeRenderTarget;
  private readonly cubeCamera: CubeCamera;
  private envRT: WebGLRenderTarget | null = null;
  readonly sound = new SoundEngine();
  private reflFrameCounter = 0;
  private reflectionQuality: 'off' | 'low' | 'high' = 'low';
  private menuDemoMode = false;
  private overheadMapMode = false;
  private mapPanX = 0;
  private mapPanZ = 0;
  private mapZoomY = 2250;
  private readonly activePointers = new Map<number, { x: number; y: number }>();
  private lastPinchDist = 0;
  private dynamicPRScale = 1.0;
  private dynamicPRCooldown = 0;
  private studioCanvas: HTMLCanvasElement | null = null;
  private readonly projVec = new Vector3();
  private currentSettings: SimulatorSettings | null = null;
  private readonly camPos = new Vector3(0, 2.4, -6.5);
  private readonly camLook = new Vector3(0, 0.85, 2.0);
  private readonly camVel = new Vector3();
  private orbitYaw = 0.35;
  private orbitPitch = 0.24;
  private orbitDist = 6.4;
  private userDragTimer = 0;
  private isPointerDown = false;
  private lastPtrX = 0;
  private lastPtrY = 0;

  // zero-alloc lattice overlay
  private latticeLines: LineSegments | null = null;
  private latticePos: Float32Array | null = null;
  private latticeCol: Float32Array | null = null;

  private statsTimer = 0;
  private hudTimer = 0;
  private loggedDrawCalls = false;
  private onStats?: (s: EngineStats) => void;
  onHudFrame?: (tel: VehicleTelemetry, fps: number) => void;
  private resizeObserver: ResizeObserver;
  private skinningWorker: Worker | null = null;
  private workerReqId = 1;
  private loadGeneration = 0;
  private readonly memSkinCache = new Map<string, PrecomputedBindingData[]>();
  /** Bounded LRU in-memory vehicle template cache (max 2 on mobile/4GB tablets, max 4 on desktop) for < 5ms switching. */
  private readonly vehicleTemplateCache = new BoundedVehicleTemplateCache<SoftBodyVehicle>();
  private activeCacheKey = '';
  private secondaryPreloadScheduled = false;
  private paintByCar: Record<string, number> = {};
  private readonly isMobilePhone: boolean;
  private shadowFrameCounter = 0;
  private blobShadow: Mesh | null = null;
  private mobileReflBaked = false;
  /** Zero-cost weather hook (0 = Off: skips all weather/rain updates & shader work). */
  private readonly rain: RainWeatherSystem;
  private readonly wheelSprayPos = new Float32Array(12);
  private readonly wheelSprayGrounded = new Uint8Array(4);
  private rainIntensity = 0;
  /** Dirty flag to skip redundant renderer.render() passes when paused and camera is stationary. */
  private renderDirty = true;
  private prevCamX = NaN;
  private prevCamY = NaN;
  private prevCamZ = NaN;
  private prevCamQx = NaN;
  private prevCamQy = NaN;
  private prevCamQz = NaN;
  private prevCamQw = NaN;
  private lastReflX = NaN;
  private lastReflZ = NaN;
  private lastReflRain = -1;
  private readonly tmpPaintColor = new Color();

  constructor(container: HTMLElement, onStats?: (s: EngineStats) => void) {
    this.container = container;
    this.onStats = onStats;
    this.constrained = isConstrainedDevice();
    void clearOldAssetCaches();
    this.isMobilePhone =
      /Android|iPhone|iPod/i.test(navigator.userAgent) ||
      (this.constrained && Math.min(window.innerWidth || 800, window.innerHeight || 600) < 600);

    const saved = readPersistedSettings();
    if (saved.cameraMode && ['chase', 'hood', 'orbit', 'bumper'].includes(saved.cameraMode)) {
      this.cameraMode = saved.cameraMode;
    }
    if (typeof saved.showLattice === 'boolean') this.showLattice = saved.showLattice;
    if (typeof saved.bodyOpacity === 'number' && Number.isFinite(saved.bodyOpacity)) {
      this.bodyOpacity = Math.max(0.15, Math.min(1.0, saved.bodyOpacity));
    }
    if (saved.paintByCar && typeof saved.paintByCar === 'object') {
      this.paintByCar = { ...saved.paintByCar };
    }

    const w = Math.max(1, container.clientWidth);
    const h = Math.max(1, container.clientHeight);
    this.camera = new PerspectiveCamera(54, w / h, 0.35, 3600);

    const dpr = window.devicePixelRatio || 1;
    this.renderer = new WebGLRenderer({
      antialias: !this.isMobilePhone && !this.constrained && dpr < 1.5,
      powerPreference: 'high-performance',
      stencil: false,
      depth: true,
      preserveDrawingBuffer: false,
    });
    const maxPR = this.isMobilePhone || this.constrained ? 1.0 : 1.5;
    const pr = Math.min(dpr, maxPR);
    this.renderer.setPixelRatio(pr);
    this.renderer.setSize(w, h, false);
    this.renderer.outputColorSpace = SRGBColorSpace;
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.shadowMap.enabled = !this.isMobilePhone;
    this.renderer.shadowMap.autoUpdate = false;
    this.renderer.shadowMap.needsUpdate = !this.isMobilePhone;
    this.renderer.shadowMap.type = PCFSoftShadowMap;
    this.renderer.domElement.style.display = 'block';
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';
    container.appendChild(this.renderer.domElement);

    // Scene, animated procedural cloud skybox & throttled real-time CubeCamera reflections (60 FPS)
    this.scene.background = SKY_COLOR;
    this.scene.fog = new FogExp2(SKY_COLOR.getHex(), 0.00006);

    const skyMat = new ShaderMaterial({
      side: BackSide,
      depthWrite: false,
      fog: false,
      uniforms: this.skyUniforms,
      vertexShader: `
        varying vec3 vDir;
        void main() {
          vDir = normalize(position);
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float uTime;
        uniform float uRain;
        uniform float uLightning;
        uniform float uNight;
        varying vec3 vDir;
        float hash(vec2 p) {
          return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
        }
        float noise(vec2 p) {
          vec2 i = floor(p);
          vec2 f = fract(p);
          vec2 u = f * f * (3.0 - 2.0 * f);
          return mix(
            mix(hash(i + vec2(0.0, 0.0)), hash(i + vec2(1.0, 0.0)), u.x),
            mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
            u.y
          );
        }
        float fbm(vec2 p) {
          float v = 0.0;
          float a = 0.52;
          for (int i = 0; i < 4; i++) {
            v += a * noise(p);
            p = p * 2.04 + vec2(17.3, 9.2);
            a *= 0.5;
          }
          return v;
        }
        void main() {
          vec3 d = normalize(vDir);
          float y = clamp(d.y, 0.0, 1.0);
          float r = clamp(uRain, 0.0, 1.0);
          float n = clamp(uNight, 0.0, 1.0);

          // Sunny blue sky -> dark overcast storm slate when raining -> warm dusk -> deep starry midnight
          float dusk = sin(n * 3.14159265) * (1.0 - r * 0.78);
          vec3 zenith = mix(vec3(0.24, 0.46, 0.78), vec3(0.09, 0.12, 0.17), r);
          vec3 horizon = mix(vec3(0.75, 0.84, 0.93), vec3(0.27, 0.31, 0.37), r);
          vec3 ground = mix(vec3(0.45, 0.50, 0.55), vec3(0.16, 0.19, 0.23), r);
          zenith = mix(zenith, vec3(0.015, 0.025, 0.055), n);
          horizon = mix(horizon, vec3(0.045, 0.068, 0.12), n);
          horizon += vec3(0.55, 0.24, 0.12) * dusk * 0.75;
          ground = mix(ground, vec3(0.015, 0.02, 0.032), n);
          vec3 col = d.y >= 0.0 ? mix(horizon, zenith, pow(y, 0.58)) : mix(horizon, ground, clamp(-d.y * 4.0, 0.0, 1.0));

          // Warm sun glow (day) or cool silver moon glow (night)
          vec3 sunDir = normalize(vec3(0.45, 0.62, 0.40));
          float sunDot = max(dot(d, sunDir), 0.0);
          float sunVis = (1.0 - r * 0.94) * (1.0 - n);
          col += vec3(1.0, 0.94, 0.82) * (pow(sunDot, 16.0) * 0.22 + pow(sunDot, 180.0) * 0.65) * sunVis;
          float moonVis = (1.0 - r * 0.85) * n;
          col += vec3(0.75, 0.86, 1.0) * (pow(sunDot, 36.0) * 0.16 + pow(sunDot, 340.0) * 0.85) * moonVis;

          // Subtle twinkling starfield in clear Night Mode
          if (n > 0.05 && d.y > 0.06) {
            vec2 starUv = floor(d.xz / (d.y + 0.25) * 190.0);
            float sh = hash(starUv);
            float star = step(0.994, sh) * smoothstep(0.06, 0.32, d.y) * (1.0 - r * 0.92);
            float twinkle = 0.65 + 0.35 * sin(uTime * 2.4 + sh * 80.0);
            col += vec3(0.85, 0.92, 1.0) * star * twinkle * n * 0.95;
          }

          // Volumetric clouds thicken, darken & move faster during rain + catch distant sheet lightning
          if (d.y > 0.015) {
            vec2 uv = d.xz / (d.y + 0.18) * 1.15;
            float windBoost = 1.0 + r * 1.6;
            vec2 wind = vec2(uTime * 0.012 * windBoost, uTime * 0.005 * windBoost);
            float c1 = fbm(uv + wind);
            float c2 = fbm(uv * 1.9 - wind * 0.6 + 4.2);
            float rawC = c1 * 0.72 + c2 * 0.28;
            float lo = mix(0.47, 0.20, r);
            float hi = mix(0.76, 0.64, r);
            float density = smoothstep(lo, hi, rawC);
            float horizonFade = smoothstep(0.015, 0.22, d.y);

            vec3 sunnyCloud = mix(vec3(0.84, 0.88, 0.93), vec3(1.0, 0.99, 0.96), clamp(c1 * 1.2 + sunDot * 0.25, 0.0, 1.0));
            vec3 stormCloud = mix(vec3(0.17, 0.20, 0.25), vec3(0.36, 0.40, 0.46), clamp(c2 * 1.15, 0.0, 1.0));
            vec3 cloudCol = mix(sunnyCloud, stormCloud, r);
            cloudCol = mix(cloudCol, vec3(0.055, 0.075, 0.12) + vec3(0.06, 0.08, 0.12) * c1, n);
            // Distant sheet lightning illuminating storm cloud bellies
            cloudCol += vec3(0.78, 0.88, 1.0) * uLightning * smoothstep(0.35, 0.75, c1) * 1.4;
            col = mix(col, cloudCol, density * horizonFade * mix(0.86, 0.96, r) * mix(1.0, 0.65, n));
          }
          gl_FragColor = vec4(col, 1.0);
        }
      `,
    });
    this.skyDome = new Mesh(new SphereGeometry(3000, 24, 16), skyMat);
    this.skyDome.frustumCulled = false;
    this.scene.add(this.skyDome);

    this.pmrem = new PMREMGenerator(this.renderer);
    this.pmrem.compileCubemapShader();
    this.studioEnvRT = this.pmrem.fromScene(new RoomEnvironment(), 0.04);
    this.scene.environment = this.studioEnvRT.texture;

    const cubeRes = this.constrained ? 96 : 256;
    this.cubeRT = new WebGLCubeRenderTarget(cubeRes, {
      generateMipmaps: true,
      minFilter: LinearMipmapLinearFilter,
    });
    // Far plane (3200) exceeds skyDome radius (3000) so clouds & sky are captured in real-time reflections
    this.cubeCamera = new CubeCamera(0.45, 3200, this.cubeRT);

    this.hemiLight = new HemisphereLight(0xeef4ff, 0x3b4046, 0.85);
    this.scene.add(this.hemiLight);

    this.dirLight = new DirectionalLight(0xfff6e8, 2.2);
    this.dirLight.position.set(18, 28, 16);
    this.dirLight.castShadow = true;
    this.dirLight.shadow.mapSize.set(1024, 1024);
    this.dirLight.shadow.camera.near = 2;
    this.dirLight.shadow.camera.far = 90;
    const d = 15;
    this.dirLight.shadow.camera.left = -d;
    this.dirLight.shadow.camera.right = d;
    this.dirLight.shadow.camera.top = d;
    this.dirLight.shadow.camera.bottom = -d;
    this.dirLight.shadow.bias = -0.0008;
    this.dirLight.shadow.normalBias = 0.05;
    this.scene.add(this.dirLight);
    this.scene.add(this.dirLight.target);

    // 2 km × 2 km Proving Ground
    this.arena = new CrashArena();
    this.scene.add(this.arena.buildVisuals(this.constrained));

    // Internal boot placeholder hidden until first GLB car (BMW M4 CSL / Dodge Demon / McLaren P1) loads
    const src = createProceduralVehicleSource('Loading Vehicle', 0x8e969d);
    this.vehicle = new SoftBodyVehicle(src, this.arena);
    const maxSubstepsPerFrame = this.isMobilePhone || this.constrained ? 6 : 8;
    this.vehicle.solver.substeps = maxSubstepsPerFrame;
    this.vehicle.root.visible = false;
    this.scene.add(this.vehicle.root);
    this.scene.add(this.vehicle.glass.mesh);
    this.scene.add(this.vehicle.fluids.group);
    this.rain = new RainWeatherSystem(this.isMobilePhone || this.constrained);
    this.scene.add(this.rain.group);
    this.rain.attachToVehicle(this.vehicle.root);
    this.buildLatticeOverlay();

    // camera controls on canvas
    this.bindPointerEvents(this.renderer.domElement);
    this.resizeObserver = new ResizeObserver(() => this.onResize());
    this.resizeObserver.observe(container);

    this.loop = new FixedTimestepLoop({
      step: (dt) => this.step(dt),
      render: (alpha, frameDt) => this.render(alpha, frameDt),
      maxSubstepsPerFrame,
    });
    this.loop.start();
  }

  // ================================================================ weather & time of day

  /** Switch between Day and Night Mode (smoothly transitions sky, lighting & fog; auto-enables headlights). */
  setNightMode(night: boolean): void {
    const prev = this.nightMode;
    this.nightMode = night;
    this.nightTarget = night ? 1 : 0;
    if (night) this.skyDome.visible = true;
    if (night !== prev) {
      this.setHeadlights(night);
      this.mobileReflBaked = false;
    }
    this.applyLightingMood(this.rain.intensity);
    this.renderDirty = true;
  }

  isNightMode(): boolean {
    return this.nightMode;
  }

  /** Toggle or set vehicle headlights (SpotLight beams + emissive lenses + rear running lights). */
  setHeadlights(on: boolean): void {
    this.headlightsOn = on;
    this.vehicle.headlightsOn = on;
    this.renderDirty = true;
  }

  toggleHeadlights(): boolean {
    this.setHeadlights(!this.headlightsOn);
    return this.headlightsOn;
  }

  getHeadlights(): boolean {
    return this.headlightsOn;
  }

  /** Unified zero-GC Day / Dusk / Night / Rain / Lightning lighting & fog mood controller. */
  private applyLightingMood(r: number): void {
    const n = this.nightFactor;
    const flash = this.lightningFlash;
    this.skyUniforms.uNight.value = n;
    if (n > 0.01) this.skyDome.visible = true;

    const dayDirI = 2.2 - r * 1.42;
    const nightDirI = 0.34 - r * 0.14;
    this.dirLight.intensity = dayDirI + (nightDirI - dayDirI) * n + flash * (1.65 - n * 0.2);

    this.tmpLightCol.copy(this.drySunCol).lerp(this.wetSunCol, r).lerp(this.nightMoonCol, n);
    this.dirLight.color.copy(this.tmpLightCol);

    const dayHemiI = 0.85 - r * 0.30;
    const nightHemiI = 0.28 - r * 0.08;
    this.hemiLight.intensity = dayHemiI + (nightHemiI - dayHemiI) * n + flash * (0.45 - n * 0.1);

    this.tmpLightCol.copy(this.dryHemiSky).lerp(this.wetHemiSky, r).lerp(this.nightHemiSky, n);
    this.hemiLight.color.copy(this.tmpLightCol);

    this.renderer.toneMappingExposure = (1.05 - r * 0.11) * (1 - n) + 0.92 * n;

    this.tmpSkyBg.copy(r > 0.001 ? this.rain.outSkyColor : SKY_COLOR).lerp(this.nightSkyCol, n);
    if (this.scene.background instanceof Color) {
      this.scene.background.copy(this.tmpSkyBg);
    }
    if (this.scene.fog instanceof FogExp2) {
      this.scene.fog.color.copy(this.tmpSkyBg);
      const dayFog = 0.00006 + r * 0.00046;
      const nightFog = 0.00048 + r * 0.00055;
      this.scene.fog.density = dayFog + (nightFog - dayFog) * n;
    }
  }

  /** Set weather preset ('off' | 'drizzle' | 'heavy'). */
  setWeatherMode(mode: WeatherMode): void {
    this.rain.setWeatherMode(mode);
    this.rainIntensity = this.rain.targetIntensity;
    this.mobileReflBaked = false; // re-bake wet reflections to match new storm/sunny sky
    this.renderDirty = true;
  }

  /** Set optional rain intensity (0 = Off -> zero CPU/GPU work per frame). */
  setRainIntensity(intensity: number): void {
    const next = Math.max(0, Math.min(1, intensity));
    if (next !== this.rainIntensity) {
      this.rainIntensity = next;
      this.rain.setIntensity(next);
      this.mobileReflBaked = false;
      this.renderDirty = true;
    }
  }

  getRainIntensity(): number {
    return this.rainIntensity;
  }

  /** Zero-cost weather hook: immediately returns when rain is Off, surfaces are dry, and car is not on a wet patch. */
  private updateWeather(frameDt: number): void {
    const F = this.vehicle.getFrame();
    const onWetPatch = this.arena.isWetPatchAt(F[0], F[2]);
    if (!this.rain.isActive() && !onWetPatch) {
      if (this.skyUniforms.uRain.value > 0) {
        this.skyUniforms.uRain.value = 0;
        this.skyUniforms.uLightning.value = 0;
        this.lightningFlash = 0;
        this.applyLightingMood(0);
      }
      return;
    }

    const S = this.vehicle.solver;
    const vx = S.vx[0] ?? 0;
    const vy = S.vy[0] ?? 0;
    const vz = S.vz[0] ?? 0;
    const speedAbs = Math.sqrt(vx * vx + vy * vy + vz * vz);
    this.vehicle.getWheelContactPositions(this.wheelSprayPos, this.wheelSprayGrounded);

    const wet = this.rain.update(
      frameDt,
      this.camera.position.x,
      this.camera.position.y,
      this.camera.position.z,
      vx,
      vy,
      vz,
      speedAbs,
      this.wheelSprayPos,
      this.wheelSprayGrounded,
      onWetPatch,
    );

    this.arena.setWetness(wet);
    this.vehicle.wetness = wet;

    // Reactive storm skybox, occasional sheet lightning in heavy rain & moody overcast lighting (zero GC)
    const r = this.rain.intensity;
    this.skyUniforms.uRain.value = r;
    if (r > 0.72) {
      this.lightningTimer -= frameDt;
      if (this.lightningTimer <= 0) {
        this.lightningFlash = 0.95;
        this.lightningTimer = 7.5 + ((Math.sin(this.skyUniforms.uTime.value * 12.9898) * 43758.5453) % 1 + 1) * 4.5;
      }
    }
    if (this.lightningFlash > 0.001) {
      this.lightningFlash *= Math.exp(-9.5 * frameDt);
      if (this.lightningFlash < 0.004) this.lightningFlash = 0;
    }
    this.skyUniforms.uLightning.value = this.lightningFlash;
    this.applyLightingMood(r);
  }

  // ================================================================ loop

  private step(dt: number): void {
    this.arena.updateDynamic(dt, this.vehicle.solver);
    this.vehicle.step(dt);
    this.renderDirty = true;
  }

  private render(alpha: number, frameDt: number): void {
    Scratch.pool.reset();
    // Smooth live Day -> Dusk -> Night sky transition (runs even behind Settings so user sees sky change!)
    if (Math.abs(this.nightFactor - this.nightTarget) > 0.002) {
      const stepN = Math.min(0.05, Math.max(0.012, frameDt)) * 0.85;
      if (this.nightFactor < this.nightTarget) {
        this.nightFactor = Math.min(this.nightTarget, this.nightFactor + stepN);
      } else {
        this.nightFactor = Math.max(this.nightTarget, this.nightFactor - stepN);
      }
      this.applyLightingMood(this.rain.intensity);
      if (Math.abs(this.nightFactor - this.nightTarget) <= 0.002) {
        this.nightFactor = this.nightTarget;
        this.mobileReflBaked = false;
      }
      this.renderDirty = true;
    }
    if (!this.loop.paused || this.renderDirty) {
      this.skyUniforms.uTime.value += frameDt;
      this.vehicle.render(alpha, frameDt);
    }
    this.updateCamera(frameDt);
    if (this.showLattice) this.updateLatticeOverlay();

    // Smooth cinematic orbit when Main Menu demo mode is active
    if (this.menuDemoMode) {
      this.orbitYaw += frameDt * 0.24;
      this.orbitPitch = 0.22;
      this.orbitDist = 5.6;
      this.userDragTimer = 2.0;
      this.renderDirty = true;
    }

    // Zero-cost idle guard for optional weather/rain effects (0 work when rain is Off & dry)
    const rainActive = this.rain.isActive();
    if (rainActive || !this.loop.paused) {
      this.updateWeather(frameDt);
      if (rainActive) this.renderDirty = true;
    }

    // Skip redundant WebGL renderer.render() passes when paused and camera/scene are completely idle
    const camPos = this.camera.position;
    const camQ = this.camera.quaternion;
    const camMoved =
      Math.abs(camPos.x - this.prevCamX) > 1e-4 ||
      Math.abs(camPos.y - this.prevCamY) > 1e-4 ||
      Math.abs(camPos.z - this.prevCamZ) > 1e-4 ||
      Math.abs(camQ.x - this.prevCamQx) > 1e-4 ||
      Math.abs(camQ.y - this.prevCamQy) > 1e-4 ||
      Math.abs(camQ.z - this.prevCamQz) > 1e-4 ||
      Math.abs(camQ.w - this.prevCamQw) > 1e-4;
    if (this.loop.paused && !this.renderDirty && !camMoved && !this.menuDemoMode && !rainActive) {
      return;
    }

    // Smooth dynamic resolution scaling (caps pixelRatio <= 1.0 on mobile/tablet, <= 1.5 on desktop)
    this.dynamicPRCooldown -= frameDt;
    if (this.dynamicPRCooldown <= 0 && !this.menuDemoMode && (this.currentSettings?.resolutionMode ?? 'medium') === 'medium') {
      const fms = this.loop.frameMs;
      const dpr = window.devicePixelRatio || 1;
      const maxPR = this.isMobilePhone || this.constrained ? Math.min(dpr, 1.0) : Math.min(dpr, 1.5);
      const minPR = 0.80;
      if (fms > 17.0 && this.dynamicPRScale > minPR / Math.max(0.80, maxPR)) {
        this.dynamicPRScale = Math.max(minPR / Math.max(0.80, maxPR), this.dynamicPRScale - 0.06);
        this.renderer.setPixelRatio(Math.max(minPR, Math.min(maxPR, maxPR * this.dynamicPRScale)));
        this.dynamicPRCooldown = 1.0;
      } else if (fms < 13.5 && this.dynamicPRScale < 1.0) {
        this.dynamicPRScale = Math.min(1.0, this.dynamicPRScale + 0.04);
        this.renderer.setPixelRatio(Math.max(minPR, Math.min(maxPR, maxPR * this.dynamicPRScale)));
        this.dynamicPRCooldown = 1.5;
      }
    }

    // Throttled CubeCamera -> PMREM reflections:
    // - On mobile phones in Low mode: bake once per spawn (0 ms/frame while driving!)
    // - On desktop: update at most once every 90/180 frames AND only if the car moved > 12m or rain changed
    // - Always disable shadowMap during cubeCamera.update() so it NEVER consumes shadowMap.needsUpdate while the car is hidden!
    const FRefl = this.vehicle.getFrame();
    const reflDistSq = Number.isNaN(this.lastReflX)
      ? 1e9
      : (FRefl[0] - this.lastReflX) * (FRefl[0] - this.lastReflX) + (FRefl[2] - this.lastReflZ) * (FRefl[2] - this.lastReflZ);
    const reflWeatherChanged = Math.abs(this.rain.intensity - this.lastReflRain) > 0.08;
    const shouldUpdateRefl =
      this.reflectionQuality !== 'off' &&
      (this.isMobilePhone
        ? this.reflectionQuality === 'high'
          ? (this.reflFrameCounter++ % 240) === 0 && (reflDistSq > 225 || reflWeatherChanged)
          : !this.mobileReflBaked
        : (!this.mobileReflBaked ||
            ((this.reflFrameCounter++ % (this.reflectionQuality === 'high' ? 90 : 180)) === 0 &&
              (reflDistSq > 144 || reflWeatherChanged))));

    if (shouldUpdateRefl) {
      this.mobileReflBaked = true;
      this.lastReflX = FRefl[0];
      this.lastReflZ = FRefl[2];
      this.lastReflRain = this.rain.intensity;
      this.cubeCamera.position.set(FRefl[0], FRefl[1] + 0.95, FRefl[2]);
      this.skyDome.position.copy(this.cubeCamera.position);
      const prevVis = this.vehicle.root.visible;
      const prevLat = this.latticeLines ? this.latticeLines.visible : false;
      const prevBlob = this.blobShadow ? this.blobShadow.visible : false;
      const prevShadowEnabled = this.renderer.shadowMap.enabled;
      const prevShadowNeeds = this.renderer.shadowMap.needsUpdate;

      this.vehicle.root.visible = false;
      if (this.latticeLines) this.latticeLines.visible = false;
      if (this.blobShadow) this.blobShadow.visible = false;
      this.renderer.shadowMap.enabled = false;
      this.renderer.shadowMap.needsUpdate = false;

      this.cubeCamera.update(this.renderer, this.scene);

      this.renderer.shadowMap.enabled = prevShadowEnabled;
      this.renderer.shadowMap.needsUpdate = prevShadowNeeds;
      this.vehicle.root.visible = prevVis;
      if (this.latticeLines) this.latticeLines.visible = prevLat;
      if (this.blobShadow) this.blobShadow.visible = prevBlob;

      const nextEnv = this.pmrem.fromCubemap(this.cubeRT.texture);
      this.scene.environment = nextEnv.texture;
      if (this.envRT) this.envRT.dispose();
      this.envRT = nextEnv;
    }

    // Update shadow light & blob shadow AFTER cubeCamera so the main render always gets a crisp, complete car shadow!
    this.updateShadowLight();

    this.skyDome.position.copy(this.camera.position);
    this.renderer.render(this.scene, this.camera);
    this.prevCamX = camPos.x;
    this.prevCamY = camPos.y;
    this.prevCamZ = camPos.z;
    this.prevCamQx = camQ.x;
    this.prevCamQy = camQ.y;
    this.prevCamQz = camQ.z;
    this.prevCamQw = camQ.w;
    this.renderDirty = false;

    if (!this.loggedDrawCalls) {
      this.loggedDrawCalls = true;
      console.info(
        `[FlexWeb GPU] Startup render stats: calls=${this.renderer.info.render.calls}, triangles=${this.renderer.info.render.triangles}, geometries=${this.renderer.info.memory.geometries}`,
      );
    }

    const liveTel = this.vehicle.updateTelemetry();
    const FNow = this.vehicle.getFrame();
    this.arena.updateRadarDisplay(FNow[0], FNow[2], liveTel.speedKmh);
    this.sound.update(liveTel, frameDt, this.loop.paused || this.menuDemoMode, this.loop.timeScale);

    // Throttled 10 Hz direct DOM HUD update (speed, gear, RPM, G-force, FPS — zero React reconciliation!)
    this.hudTimer += frameDt;
    if (this.onHudFrame && this.hudTimer >= 0.10) {
      this.hudTimer = 0;
      this.onHudFrame(liveTel, this.loop.fps);
    }

    // Throttled 4 Hz React stats sync (diagnostics bars & damage telemetry only)
    this.statsTimer += frameDt;
    if (this.onStats && this.statsTimer >= 0.25) {
      this.statsTimer = 0;
      this.onStats(this.collectStats());
    }
  }

  captureScreenshot(quality = 0.82): string {
    try {
      this.renderer.render(this.scene, this.camera);
      return this.renderer.domElement.toDataURL('image/jpeg', quality);
    } catch {
      return '';
    }
  }

  /**
   * Photographs the front of the active vehicle inside the pure-white studio showroom
   * built underneath the map (`y = STUDIO_SHOWROOM_Y = -90`), returning a crisp 520×300 JPEG data URL.
   */
  captureStudioPortrait(quality = 0.80): string {
    try {
      const W = 480;
      const H = 270;
      if (!this.studioCanvas && typeof document !== 'undefined') {
        this.studioCanvas = document.createElement('canvas');
        this.studioCanvas.width = W;
        this.studioCanvas.height = H;
      }

      const root = this.vehicle.root;
      const prevRootMat = new Matrix4().copy(root.matrix);
      const prevCamPos = this.camera.position.clone();
      const prevCamQuat = this.camera.quaternion.clone();
      const prevCamUp = this.camera.up.clone();
      const prevFov = this.camera.fov;
      const prevAspect = this.camera.aspect;
      const prevFog = this.scene.fog;
      const prevBg = this.scene.background;
      const prevEnv = this.scene.environment;
      const prevSkyVis = this.skyDome.visible;
      const prevFluidsVis = this.vehicle.fluids.group.visible;
      const prevLatVis = this.latticeLines ? this.latticeLines.visible : false;
      const prevLightPos = this.dirLight.position.clone();
      const prevLightTarget = this.dirLight.target.position.clone();

      // Ensure pristine undeformed showroom body & straight wheels even if car was mid-crash in background demo
      const hadImpact = this.vehicle.solver.peakImpact > 0.05;
      if (hadImpact) {
        this.vehicle.deformer.restore(this.vehicle.restLocal);
      }

      // Place vehicle flush on the underground pure-white showroom floor at (0, STUDIO_SHOWROOM_Y, 0)
      root.matrix.makeTranslation(0, STUDIO_SHOWROOM_Y + 0.01, 0);
      root.matrixWorldNeedsUpdate = true;
      root.updateMatrixWorld(true);

      // Pure white studio environment & key lighting
      this.scene.fog = null;
      this.scene.background = new Color(0xf8fafc);
      this.scene.environment = this.studioEnvRT.texture;
      this.skyDome.visible = false;
      this.vehicle.fluids.group.visible = false;
      if (this.latticeLines) this.latticeLines.visible = false;

      this.dirLight.position.set(10, STUDIO_SHOWROOM_Y + 18, 16);
      this.dirLight.target.position.set(0, STUDIO_SHOWROOM_Y, 0);
      this.dirLight.target.updateMatrixWorld();

      // Front-focused studio portrait framing the grille, headlights, hood & front wheel
      this.camera.fov = 38;
      this.camera.aspect = W / H;
      this.camera.up.set(0, 1, 0);
      this.camera.position.set(1.55, STUDIO_SHOWROOM_Y + 1.04, 4.35);
      this.camera.lookAt(0, STUDIO_SHOWROOM_Y + 0.54, 0.45);
      this.camera.updateProjectionMatrix();

      this.renderer.render(this.scene, this.camera);

      let dataUrl = '';
      const ctx = this.studioCanvas?.getContext('2d');
      if (this.studioCanvas && ctx) {
        ctx.fillStyle = '#f8fafc';
        ctx.fillRect(0, 0, W, H);
        ctx.drawImage(this.renderer.domElement, 0, 0, W, H);
        dataUrl = this.studioCanvas.toDataURL('image/jpeg', quality);
      } else {
        dataUrl = this.renderer.domElement.toDataURL('image/jpeg', quality);
      }

      // Restore exact live arena state & re-render immediately so zero frames ever flash
      root.matrix.copy(prevRootMat);
      root.matrixWorldNeedsUpdate = true;
      if (hadImpact) {
        this.vehicle.render(1, 0);
      }
      this.scene.fog = prevFog;
      this.scene.background = prevBg;
      this.scene.environment = prevEnv;
      this.skyDome.visible = prevSkyVis;
      this.vehicle.fluids.group.visible = prevFluidsVis;
      if (this.latticeLines) this.latticeLines.visible = prevLatVis;
      this.dirLight.position.copy(prevLightPos);
      this.dirLight.target.position.copy(prevLightTarget);
      this.dirLight.target.updateMatrixWorld();

      this.camera.fov = prevFov;
      this.camera.aspect = prevAspect;
      this.camera.up.copy(prevCamUp);
      this.camera.position.copy(prevCamPos);
      this.camera.quaternion.copy(prevCamQuat);
      this.camera.updateProjectionMatrix();
      this.renderer.render(this.scene, this.camera);

      return dataUrl;
    } catch {
      return '';
    }
  }

  setMenuDemoMode(active: boolean): void {
    this.menuDemoMode = active;
    if (!active) {
      this.userDragTimer = 0;
    }
  }

  setOverheadMapMode(active: boolean): void {
    this.overheadMapMode = active;
    const fog = this.scene.fog as FogExp2 | null;
    if (active) {
      this.recenterMap();
      this.camera.far = 6500;
      this.camera.updateProjectionMatrix();
      if (fog) fog.density = 0;
    } else {
      this.camera.far = 6500;
      this.camera.updateProjectionMatrix();
      if (fog) fog.density = 0.00006;
    }
  }

  /** Zooms the interactive overhead map camera in (factor < 1) or out (factor > 1). */
  zoomMapBy(factor: number): void {
    this.mapZoomY = Math.max(140, Math.min(3400, this.mapZoomY * factor));
  }

  /** Re-centers the interactive overhead map camera on the full 2 km × 2 km Proving Ground (or vehicle if requested). */
  recenterMap(toVehicle = false): void {
    if (toVehicle) {
      const F = this.vehicle.getFrame();
      this.mapPanX = Math.max(-1000, Math.min(1000, F[0]));
      this.mapPanZ = Math.max(-1000, Math.min(1000, F[2]));
      this.mapZoomY = 480;
    } else {
      this.mapPanX = 0;
      this.mapPanZ = 0;
      this.mapZoomY = 2250;
    }
  }

  /** Projects a 3D world point onto 2D viewport pixel coordinates for overhead Map markers. */
  projectWorldToScreen(wx: number, wy: number, wz: number): { x: number; y: number; visible: boolean } {
    this.projVec.set(wx, wy, wz).project(this.camera);
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    const visible = this.projVec.z >= -1 && this.projVec.z <= 1;
    return {
      x: (this.projVec.x * 0.5 + 0.5) * w,
      y: (-this.projVec.y * 0.5 + 0.5) * h,
      visible,
    };
  }

  applySettings(s: SimulatorSettings): void {
    this.currentSettings = s;
    // 1. General driving assists & steering sensitivity
    this.vehicle.steeringSensitivity = s.steeringSensitivity;
    this.vehicle.absEnabled = s.absEnabled;
    this.vehicle.tcsEnabled = s.tcsEnabled;
    this.vehicle.transmissionMode = s.transmissionMode;

    // 2. Sounds volumes & mute
    this.sound.setVolumes(s.masterVolume, s.engineVolume, s.effectsVolume, s.muted);

    // 3. Graphics: Shadows (Low Settings & Mobile Phones disable real-time shadow maps completely and use a single lightweight contact shadow plane)
    const isLowPerf = s.resolutionMode === 'low' || this.isMobilePhone;
    if (s.shadowQuality === 'off' || s.resolutionMode === 'low' || (this.isMobilePhone && s.shadowQuality === 'low')) {
      this.renderer.shadowMap.enabled = false;
      this.dirLight.castShadow = false;
      this.ensureBlobShadow(s.shadowQuality !== 'off');
    } else {
      this.ensureBlobShadow(false);
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.autoUpdate = false;
      this.dirLight.castShadow = true;
      const sz = this.constrained ? 512 : 1024;
      if (this.dirLight.shadow.mapSize.x !== sz) {
        this.dirLight.shadow.mapSize.set(sz, sz);
        this.dirLight.shadow.map?.dispose();
        this.dirLight.shadow.map = null;
      }
      this.renderer.shadowMap.needsUpdate = true;
    }

    // 4. Strict fixed-timestep substeps (8 desktop, 6 mobile/tablet) for identical stability at 30/60/120 FPS
    this.vehicle.solver.substeps = this.loop.maxSubstepsPerFrame;
    this.vehicle.deformer.lowQualityNormals = isLowPerf;

    // 5. Graphics: Real-time Reflections
    this.reflectionQuality = s.reflectionQuality;
    if (s.reflectionQuality === 'off') {
      this.scene.environment = null;
    } else {
      this.scene.environment = (this.envRT ?? this.studioEnvRT).texture;
    }

    // 6. Graphics: Pixel Quality (strict adaptive caps: <= 1.0 phone/tablet/constrained, <= 1.5 desktop)
    this.dynamicPRScale = 1.0;
    const dpr = window.devicePixelRatio || 1;
    const maxAllowedPR = this.isMobilePhone || this.constrained ? 1.0 : 1.5;
    const targetPR =
      s.resolutionMode === 'low'
        ? Math.min(dpr, 0.85)
        : Math.min(dpr, maxAllowedPR);
    this.renderer.setPixelRatio(targetPR);
    this.renderDirty = true;
    this.onResize();

    // 6. Graphics: Full-Map + Skybox Far Plane (3600m > 3000m skyDome radius for crisp depth precision with zero Z-fighting)
    this.camera.far = 3600;
    this.camera.updateProjectionMatrix();
    this.skyDome.visible = s.skyboxClouds;
    this.setWeatherMode(s.weatherMode ?? 'off');
    this.setNightMode((s.timeOfDay ?? 'day') === 'night');
  }

  unlockAudio(): void {
    this.sound.unlock();
  }

  setSoundEnabled(on: boolean): void {
    this.sound.unlock();
    this.sound.setEnabled(on);
    if (this.onStats) this.onStats(this.collectStats());
  }

  collectStats(): EngineStats {
    return {
      fps: this.loop.fps,
      frameMs: this.loop.frameMs,
      physicsMs: this.loop.physicsMs,
      resetMs: this.lastResetMs,
      heapMB: heapUsedMB(),
      pixelRatio: this.renderer.getPixelRatio(),
      cameraMode: this.cameraMode,
      timeScale: this.loop.timeScale,
      paused: this.loop.paused,
      showLattice: this.showLattice,
      soundEnabled: this.sound.isEnabled(),
      bodyOpacity: this.bodyOpacity,
      activeCarId: this.activeCarId,
      telemetry: this.vehicle.updateTelemetry(),
    };
  }

  // ================================================================ camera

  private updateCamera(dt: number): void {
    if (this.overheadMapMode) {
      this.camera.up.set(0, 1, 0);
      const mapPos = Scratch.v0.set(this.mapPanX, this.mapZoomY, this.mapPanZ - this.mapZoomY * 0.08);
      const mapLook = Scratch.v1.set(this.mapPanX, 0, this.mapPanZ);
      const k = 1 - Math.exp(-5.4 * dt);
      this.camPos.lerp(mapPos, k);
      this.camLook.lerp(mapLook, k);
      this.camera.position.copy(this.camPos);
      this.camera.lookAt(this.camLook);
      return;
    }

    const F = this.vehicle.getFrame();
    const ox = F[0], oy = F[1], oz = F[2];
    const Yx = F[6], Yy = F[7], Yz = F[8];
    const Zx = F[9], Zy = F[10], Zz = F[11];

    if (this.userDragTimer > 0) this.userDragTimer = Math.max(0, this.userDragTimer - dt);

    if (this.cameraMode === 'interior') {
      const P = this.vehicle.profile;
      const Xx = F[3], Xy = F[4], Xz = F[5];
      // Left-hand driver seat eye position inside cabin looking forward over dash & steering wheel
      const ix = 0.36;
      const iz = P.cowlZ - 0.56;
      const iy = Math.min(P.topY(iz) - 0.16, P.beltY(iz) + 0.19);
      this.camera.position.set(
        ox + Xx * ix + Yx * iy + Zx * iz,
        oy + Xy * ix + Yy * iy + Zy * iz,
        oz + Xz * ix + Yz * iy + Zz * iz,
      );
      this.camera.up.set(Yx, Yy, Yz);
      this.camera.lookAt(
        ox + Xx * (ix * 0.65) + Yx * (iy - 0.08) + Zx * (iz + 8.0),
        oy + Xy * (ix * 0.65) + Yy * (iy - 0.08) + Zy * (iz + 8.0),
        oz + Xz * (ix * 0.65) + Yz * (iy - 0.08) + Zz * (iz + 8.0),
      );
      return;
    }

    if (this.cameraMode === 'hood') {
      const cz = this.vehicle.profile.cowlZ - 0.12;
      const cy = this.vehicle.profile.topY(cz) + 0.16;
      this.camera.position.set(
        ox + Yx * cy + Zx * cz,
        oy + Yy * cy + Zy * cz,
        oz + Yz * cy + Zz * cz,
      );
      this.camera.up.set(Yx, Yy, Yz);
      this.camera.lookAt(
        ox + Yx * (cy - 0.18) + Zx * (cz + 6),
        oy + Yy * (cy - 0.18) + Zy * (cz + 6),
        oz + Yz * (cy - 0.18) + Zz * (cz + 6),
      );
      return;
    }

    if (this.cameraMode === 'bumper') {
      const bz = this.vehicle.profile.halfL + 0.06;
      const by = 0.48;
      this.camera.position.set(
        ox + Yx * by + Zx * bz,
        oy + Yy * by + Zy * bz,
        oz + Yz * by + Zz * bz,
      );
      this.camera.up.set(Yx, Yy, Yz);
      this.camera.lookAt(
        ox + Yx * by + Zx * (bz + 10),
        oy + Yy * by + Zy * (bz + 10),
        oz + Yz * by + Zz * (bz + 10),
      );
      return;
    }

    this.camera.up.set(0, 1, 0);
    const targetLook = Scratch.v0.set(ox + Zx * 0.6, oy + 0.85, oz + Zz * 0.6);

    if (this.cameraMode === 'orbit' || this.userDragTimer > 0) {
      const cp = Math.cos(this.orbitPitch), sp = Math.sin(this.orbitPitch);
      const cy = Math.cos(this.orbitYaw), sy = Math.sin(this.orbitYaw);
      const desired = Scratch.v1.set(
        ox + this.orbitDist * cp * sy,
        Math.max(0.35, oy + 0.75 + this.orbitDist * sp),
        oz + this.orbitDist * cp * cy,
      );
      const k = 1 - Math.exp(-14 * dt);
      this.camPos.lerp(desired, k);
      this.camLook.lerp(targetLook, k);
      this.camera.position.copy(this.camPos);
      this.camera.lookAt(this.camLook);
      return;
    }

    // Dynamic speed-sensitive FOV (62° at rest -> 78° at 200+ km/h) so 60 km/h genuinely feels like 60 km/h!
    const spdKmh = this.vehicle.telemetry.speedKmh;
    const speedRatio = Math.min(1, spdKmh / 200);
    const targetFov = 62 + speedRatio * 16;
    if (Math.abs(this.camera.fov - targetFov) > 0.06) {
      this.camera.fov += (targetFov - this.camera.fov) * (1 - Math.exp(-6.5 * dt));
      this.camera.updateProjectionMatrix();
    }

    // Closer, lower spring chase camera for authentic road speed sensation
    const chaseDist = 4.85 + speedRatio * 0.55;
    const chaseHeight = 1.58 + speedRatio * 0.14;
    const desired = Scratch.v1.set(
      ox - Zx * chaseDist,
      Math.max(0.55, oy + chaseHeight),
      oz - Zz * chaseDist,
    );
    const stiffness = 13;
    const damping = 7.2;
    const ax = (desired.x - this.camPos.x) * stiffness - this.camVel.x * damping;
    const ay = (desired.y - this.camPos.y) * stiffness - this.camVel.y * damping;
    const az = (desired.z - this.camPos.z) * stiffness - this.camVel.z * damping;
    this.camVel.x += ax * dt;
    this.camVel.y += ay * dt;
    this.camVel.z += az * dt;
    this.camPos.x += this.camVel.x * dt;
    this.camPos.y = Math.max(0.35, this.camPos.y + this.camVel.y * dt);
    this.camPos.z += this.camVel.z * dt;

    const lk = 1 - Math.exp(-15 * dt);
    this.camLook.lerp(targetLook, lk);
    this.camera.position.copy(this.camPos);
    this.camera.lookAt(this.camLook);
    this.orbitYaw = Math.atan2(this.camPos.x - ox, this.camPos.z - oz);
  }

  private ensureBlobShadow(enabled: boolean): void {
    if (!enabled) {
      if (this.blobShadow) this.blobShadow.visible = false;
      return;
    }
    if (!this.blobShadow) {
      const w = 128;
      const h = 256;
      const cv = document.createElement('canvas');
      cv.width = w;
      cv.height = h;
      const ctx = cv.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, w, h);
        // Soft car-shaped contact shadow (rounded chassis footprint with feathered penumbra)
        const drawLayer = (padX: number, padY: number, r: number, alpha: number) => {
          const x = padX, y = padY, rw = w - padX * 2, rh = h - padY * 2;
          ctx.beginPath();
          ctx.moveTo(x + r, y);
          ctx.lineTo(x + rw - r, y);
          ctx.quadraticCurveTo(x + rw, y, x + rw, y + r);
          ctx.lineTo(x + rw, y + rh - r);
          ctx.quadraticCurveTo(x + rw, y + rh, x + rw - r, y + rh);
          ctx.lineTo(x + r, y + rh);
          ctx.quadraticCurveTo(x, y + rh, x, y + rh - r);
          ctx.lineTo(x, y + r);
          ctx.quadraticCurveTo(x, y, x + r, y);
          ctx.closePath();
          ctx.fillStyle = `rgba(6, 9, 14, ${alpha})`;
          ctx.fill();
        };
        ctx.filter = 'blur(14px)';
        drawLayer(18, 22, 26, 0.34);
        ctx.filter = 'blur(8px)';
        drawLayer(24, 30, 22, 0.42);
        ctx.filter = 'blur(4px)';
        drawLayer(30, 38, 18, 0.48);
        ctx.filter = 'none';
      }
      const tex = new CanvasTexture(cv);
      const geo = new PlaneGeometry(2.25, 4.95);
      geo.rotateX(-Math.PI * 0.5);
      const mat = new MeshBasicMaterial({
        map: tex,
        transparent: true,
        depthWrite: false,
        polygonOffset: true,
        polygonOffsetFactor: -1,
        polygonOffsetUnits: -2,
      });
      const mesh = new Mesh(geo, mat);
      mesh.renderOrder = 1;
      mesh.frustumCulled = false;
      this.scene.add(mesh);
      this.blobShadow = mesh;
    }
    this.blobShadow.visible = true;
  }

  private readonly blobGround = { h: 0, nx: 0, ny: 1, nz: 0, mu: 1 };
  private readonly blobNormal = new Vector3(0, 1, 0);
  private readonly blobUp = new Vector3(0, 1, 0);
  private readonly blobQuat = new Quaternion();
  private readonly blobYawQuat = new Quaternion();
  private lastShadowX = Number.NaN;
  private lastShadowY = Number.NaN;
  private lastShadowZ = Number.NaN;
  private lastShadowYaw = Number.NaN;

  private updateShadowLight(): void {
    const F = this.vehicle.getFrame();
    const ox = F[0], oy = F[1], oz = F[2];
    const yaw = Math.atan2(F[9], F[11]);

    if (this.blobShadow && this.blobShadow.visible) {
      if (oy < -50 || this.overheadMapMode) {
        this.blobShadow.scale.set(0, 0, 0);
      } else {
        this.arena.groundAt(ox, oz, this.blobGround);
        const groundY = this.blobGround.h;
        const clearance = Math.max(0, oy - groundY - 0.38);
        const fade = Math.max(0.12, 0.88 - clearance * 0.22);
        const scale = 1 + clearance * 0.12;
        this.blobShadow.position.set(
          ox + this.blobGround.nx * 0.025,
          groundY + this.blobGround.ny * 0.025,
          oz + this.blobGround.nz * 0.025,
        );
        this.blobNormal.set(this.blobGround.nx, this.blobGround.ny, this.blobGround.nz).normalize();
        this.blobQuat.setFromUnitVectors(this.blobUp, this.blobNormal);
        this.blobYawQuat.setFromAxisAngle(this.blobUp, yaw);
        this.blobShadow.quaternion.copy(this.blobQuat).multiply(this.blobYawQuat);
        this.blobShadow.scale.set(scale, 1, scale);
        (this.blobShadow.material as MeshBasicMaterial).opacity = fade;
      }
    }

    if (!this.renderer.shadowMap.enabled) return;

    // Track vehicle X, Y, Z so shadow never clips on ramps/jumps, and only re-render shadowMap when car moves or deforms
    const moved =
      Number.isNaN(this.lastShadowX) ||
      Math.abs(ox - this.lastShadowX) > 0.004 ||
      Math.abs(oy - this.lastShadowY) > 0.004 ||
      Math.abs(oz - this.lastShadowZ) > 0.004 ||
      Math.abs(yaw - this.lastShadowYaw) > 0.003 ||
      this.vehicle.solver.lastImpact > 0.2;

    if (moved || (this.shadowFrameCounter++ % 12) === 0) {
      this.dirLight.position.set(ox + 18, oy + 28, oz + 16);
      this.dirLight.target.position.set(ox, oy, oz);
      this.dirLight.target.updateMatrixWorld();
      this.renderer.shadowMap.needsUpdate = true;
      this.lastShadowX = ox;
      this.lastShadowY = oy;
      this.lastShadowZ = oz;
      this.lastShadowYaw = yaw;
    }
  }

  // ================================================================ lattice X-ray

  private buildLatticeOverlay(): void {
    if (this.latticeLines) {
      this.latticeLines.removeFromParent();
      this.latticeLines.geometry.dispose();
      (this.latticeLines.material as LineBasicMaterial).dispose();
    }
    const m = this.vehicle.solver.beamCount;
    this.latticePos = new Float32Array(m * 6);
    this.latticeCol = new Float32Array(m * 6);
    const geo = new BufferGeometry();
    const pa = new BufferAttribute(this.latticePos, 3);
    const ca = new BufferAttribute(this.latticeCol, 3);
    pa.setUsage(35048); ca.setUsage(35048);
    geo.setAttribute('position', pa);
    geo.setAttribute('color', ca);
    const mat = new LineBasicMaterial({ vertexColors: true, depthTest: false, transparent: true, opacity: 0.9 });
    this.latticeLines = new LineSegments(geo, mat);
    this.latticeLines.frustumCulled = false;
    this.latticeLines.renderOrder = 10;
    this.latticeLines.visible = this.showLattice;
    this.scene.add(this.latticeLines);
  }

  private updateLatticeOverlay(): void {
    if (!this.latticeLines || !this.latticePos || !this.latticeCol) return;
    const S = this.vehicle.solver;
    const P = this.latticePos, C = this.latticeCol;
    const { px, py, pz, beamA, beamB, rest, rest0, flags } = S;
    for (let j = 0; j < S.beamCount; j++) {
      const o = j * 6;
      if (flags[j] & BEAM_BROKEN) {
        P[o] = P[o + 1] = P[o + 2] = P[o + 3] = P[o + 4] = P[o + 5] = 0;
        continue;
      }
      const a = beamA[j], b = beamB[j];
      P[o] = px[a]; P[o + 1] = py[a]; P[o + 2] = pz[a];
      P[o + 3] = px[b]; P[o + 4] = py[b]; P[o + 5] = pz[b];
      const plastic = Math.min(1, Math.abs(rest[j] - rest0[j]) / (rest0[j] * 0.14));
      const r = Math.min(1, 0.18 + plastic * 1.4);
      const g = Math.max(0.15, 0.88 - plastic * 0.85);
      const bl = Math.max(0.1, 0.35 - plastic * 0.35);
      C[o] = C[o + 3] = r;
      C[o + 1] = C[o + 4] = g;
      C[o + 2] = C[o + 5] = bl;
    }
    (this.latticeLines.geometry.getAttribute('position') as BufferAttribute).needsUpdate = true;
    (this.latticeLines.geometry.getAttribute('color') as BufferAttribute).needsUpdate = true;
  }

  setCameraMode(mode: CameraMode): void {
    this.cameraMode = mode;
    writePersistedSettings({ cameraMode: mode });
  }

  setShowLattice(v: boolean): void {
    this.showLattice = v;
    if (this.latticeLines) this.latticeLines.visible = v;
    writePersistedSettings({ showLattice: v });
  }

  setBodyOpacity(opacity: number): void {
    this.bodyOpacity = Math.max(0.15, Math.min(1.0, opacity));
    this.vehicle.setBodyOpacity(this.bodyOpacity);
    writePersistedSettings({ bodyOpacity: this.bodyOpacity });
  }

  // ================================================================ actions

  /** Fast < 2 ms reset using pre-cached baseline typed arrays. */
  reset(x = 0, z = 0, yaw = 0, speedKmh = 0, lateral = false): void {
    const t0 = performance.now();
    this.arena.resetDynamic();
    this.rain.reset();
    this.vehicle.spawn(x, z, yaw, speedKmh / 3.6, lateral);
    this.vehicle.headlightsOn = this.headlightsOn;
    if (speedKmh > 0 && !lateral) this.vehicle.autopilotSpeed = speedKmh / 3.6;
    const F = this.vehicle.getFrame();
    this.camPos.set(F[0] - F[9] * 6.2, F[1] + 2.1, F[2] - F[11] * 6.2);
    this.camLook.set(F[0], F[1] + 0.85, F[2]);
    this.camVel.set(0, 0, 0);
    this.lastShadowX = Number.NaN;
    this.mobileReflBaked = false;
    this.renderDirty = true;
    this.lastResetMs = performance.now() - t0;
  }

  /** Repairs the vehicle completely in-place at its current world position & heading (no teleport to spawn). */
  repairInPlace(): void {
    const t0 = performance.now();
    const F = this.vehicle.getFrame();
    const x = F[0];
    const z = F[2];
    const yaw = Math.atan2(F[9], F[11]);
    this.vehicle.spawn(x, z, yaw, 0, false);
    this.vehicle.headlightsOn = this.headlightsOn;
    this.lastShadowX = Number.NaN;
    this.renderDirty = true;
    this.lastResetMs = performance.now() - t0;
  }

  launchPreset(preset: CrashTestPreset): void {
    this.reset(preset.x, preset.z, preset.yaw, preset.speedKmh, Boolean(preset.lateral));
  }

  setPaintColor(hex: number, targetCarId?: string): void {
    const carId = targetCarId || this.activeCarId;
    if (carId === this.activeCarId) {
      const c = this.tmpPaintColor.set(hex);
      const vSrc = (this.vehicle as unknown as {
        src: { materials: { paint: { color: Color; needsUpdate?: boolean } }; paintMaterials?: { color: Color; needsUpdate?: boolean }[] };
      }).src;
      vSrc.materials.paint.color.copy(c);
      vSrc.materials.paint.needsUpdate = true;
      if (vSrc.paintMaterials) {
        for (const pm of vSrc.paintMaterials) {
          pm.color.copy(c);
          pm.needsUpdate = true;
        }
      }
      this.vehicle.root.traverse((obj) => {
        const mesh = obj as Mesh;
        if (!mesh.isMesh) return;
        const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        for (const m of mats) {
          const sm = m as { name?: string; color?: Color; needsUpdate?: boolean } | null;
          if (sm?.color && sm.name && /^(Car_?paint|Car_?Paint_?\.?\d*|CARS|carpaint_second|ARm4_main)$/i.test(sm.name)) {
            sm.color.copy(c);
            sm.needsUpdate = true;
          }
        }
      });
    }
    this.paintByCar[carId] = hex;
    writePersistedSettings({ paintByCar: { [carId]: hex } });
  }

  /**
   * Switches to `option`, parking the previous vehicle in a bounded LRU cache
   * (max 2 on 4 GB RAM tablets/phones, max 4 on desktop) for < 5 ms switching.
   *
   * Pipeline (all stages profiled with console.time):
   *  1. GLTFLoader parse + parallel FlexWebMeshCache IndexedDB lookup
   *  2. Mesh merge + 0.002 m spatial vertex welding cap (>45k verts) + shared material templates
   *  3. Async createImageBitmap texture downscaling (<= 1024px)
   *  4. Cached TypedArray hydration (<200ms) OR Web Worker O(V) skinning + adjacency (zero-copy Transferables)
   */
  async selectCar(option: CarOption, onProgress?: LoadProgressCallback): Promise<void> {
    const gen = ++this.loadGeneration;
    const isRemote = Boolean(option.remoteUrls && option.remoteUrls.length > 0);
    // Fast path: switching paint colour between built-in GT-S variants reuses the active vehicle
    if (!option.url && !option.savedInDB && !isRemote && this.activeCarId.startsWith('builtin-') && option.id.startsWith('builtin-')) {
      this.activeCarId = option.id;
      setLastSelectedCarId(option.id);
      if (option.paint !== undefined) this.setPaintColor(option.paint);
      this.reset(0, 0, 0, 0);
      return;
    }

    if (!option.url && !option.savedInDB && !isRemote) {
      onProgress?.('Building Sculpted Coupe…', 55);
      await yieldFrame();
      if (gen !== this.loadGeneration) return;
      const src = createProceduralVehicleSource(option.name, option.paint ?? 0x8e969d);
      this.activeCarId = option.id;
      setLastSelectedCarId(option.id);
      this.replaceActiveVehicle(src);
      onProgress?.('Ready', 100);
      return;
    }

    const totalTag = `[FlexWeb Load] Total (${option.name})`;
    console.time(totalTag);
    const cacheKey = `v${SKINNING_CACHE_SCHEMA_VERSION}:${option.id}:${option.bytes ?? option.url ?? option.remoteUrls?.[0] ?? option.name}`;

    // Instant < 5 ms switching if this vehicle is already active or parked in the bounded LRU template cache
    if (this.activeCacheKey === cacheKey && this.activeCarId === option.id) {
      if (option.paint !== undefined) this.setPaintColor(option.paint);
      this.reset(0, 0, 0, 0);
      onProgress?.('Vehicle Spawned', 100);
      console.timeEnd(totalTag);
      return;
    }
    const cachedVehicle = this.vehicleTemplateCache.take(cacheKey);
    if (cachedVehicle) {
      if (this.activeCacheKey && this.activeCacheKey !== cacheKey) {
        this.scene.remove(this.vehicle.root);
        this.scene.remove(this.vehicle.glass.mesh);
        this.scene.remove(this.vehicle.fluids.group);
        this.vehicleTemplateCache.set(this.activeCacheKey, this.vehicle);
      } else {
        this.vehicle.dispose();
      }
      this.vehicle = cachedVehicle;
      this.activeCacheKey = cacheKey;
      this.vehicle.solver.substeps = this.loop.maxSubstepsPerFrame;
      this.vehicle.root.visible = true;
      if (this.currentSettings) {
        this.vehicle.steeringSensitivity = this.currentSettings.steeringSensitivity;
        this.vehicle.absEnabled = this.currentSettings.absEnabled;
        this.vehicle.tcsEnabled = this.currentSettings.tcsEnabled;
        this.vehicle.transmissionMode = this.currentSettings.transmissionMode;
      }
      this.sound.setProfileFromCarName(this.vehicle.name);
      this.activeCarId = option.id;
      setLastSelectedCarId(option.id);
      const savedHex = this.paintByCar[option.id] ?? option.paint;
      if (savedHex !== undefined) this.setPaintColor(savedHex, option.id);
      if (this.bodyOpacity < 0.99) this.vehicle.setBodyOpacity(this.bodyOpacity);
      this.scene.add(this.vehicle.root);
      this.scene.add(this.vehicle.glass.mesh);
      this.scene.add(this.vehicle.fluids.group);
      this.rain.attachToVehicle(this.vehicle.root);
      this.buildLatticeOverlay();
      this.renderDirty = true;
      this.reset(0, 0, 0, 0);
      this.vehicle.render(1, 0);
      this.updateCamera(0.016);
      this.updateShadowLight();
      if (this.showLattice) this.updateLatticeOverlay();
      this.renderer.render(this.scene, this.camera);
      onProgress?.('Vehicle Spawned', 100);
      console.timeEnd(totalTag);
      return;
    }

    const memHit = this.memSkinCache.get(cacheKey);
    const cachePromise = memHit ? Promise.resolve(null) : loadCachedSkinning(cacheKey);

    try {
      let sceneRoot: Object3D;
      console.time('[FlexWeb Load] GLTFLoader parse');
      if (isRemote && option.remoteUrls) {
        const res = await loadRemoteCarWithFallbacks(option.id, option.remoteUrls, onProgress);
        sceneRoot = res.scene;
      } else if (option.savedInDB) {
        onProgress?.('Reading saved vehicle from storage…', 15);
        const rawBuf = await loadSavedCarBuffer(option.id);
        if (!rawBuf) {
          throw new Error(`Saved car "${option.name}" not found in IndexedDB`);
        }
        const bufCopy = rawBuf.slice(0);
        const header = new Uint8Array(bufCopy, 0, Math.min(4, bufCopy.byteLength));
        if (header.length >= 2 && header[0] === 0x50 && header[1] === 0x4b) {
          sceneRoot = await parseGltfFromZipBytes(new Uint8Array(bufCopy), onProgress);
        } else {
          onProgress?.('Parsing 3D Mesh…', 35);
          const gltf = await this.gltfLoader.parseAsync(bufCopy, '/');
          sceneRoot = gltf.scene;
        }
      } else {
        const cleanUrl = option.url!.replace(/^\/?public\//i, '/');
        const resolvedUrl = cleanUrl.startsWith('/') || /^https?:\/\//i.test(cleanUrl) ? cleanUrl : `/${cleanUrl}`;
        console.info(`[FlexWeb Load] Loading GLB URL: ${resolvedUrl}`);
        onProgress?.('Loading 3D Mesh…', 14);
        const bytes = resolvedUrl.startsWith('/')
          ? await fetchLocalModelChunked(resolvedUrl, 14, 42, onProgress)
          : await (async () => {
              const cached = await getCachedAssetBytes(resolvedUrl);
              if (cached && cached.byteLength >= 12 && cached[0] !== 0x3c) {
                onProgress?.('Loaded 3D Mesh from local cache…', 42);
                return cached;
              }
              const res = await fetch(resolvedUrl, { cache: 'default' });
              if (!res.ok) throw new Error(`HTTP ${res.status} while fetching "${resolvedUrl}"`);
              const fetched = new Uint8Array(await res.arrayBuffer());
              if (fetched.byteLength >= 12 && fetched[0] !== 0x3c) {
                void putCachedAssetBytes(resolvedUrl, fetched);
              }
              return fetched;
            })();
        if (bytes.byteLength < 12) {
          throw new Error(`Empty or truncated GLB response (${bytes.byteLength} bytes) from "${resolvedUrl}"`);
        }
        if (bytes[0] === 0x3c /* '<' HTML fallback */) {
          throw new Error(`Received HTML (likely index.html SPA fallback) instead of model binary from "${resolvedUrl}"`);
        }
        if (bytes[0] === 0x50 && bytes[1] === 0x4b) {
          sceneRoot = await parseGltfFromZipBytes(bytes, onProgress);
        } else {
          const magic = new DataView(bytes.buffer, bytes.byteOffset, 4).getUint32(0, true);
          if (/\.glb(\?|$)/i.test(resolvedUrl) && magic !== 0x46546c67 /* 'glTF' */) {
            throw new Error(`Invalid GLB magic header (0x${magic.toString(16)}) from "${resolvedUrl}"`);
          }
          onProgress?.('Parsing 3D Mesh…', 46);
          await yieldFrame();
          const basePath = resolvedUrl.substring(0, resolvedUrl.lastIndexOf('/') + 1) || '/';
          const ab: ArrayBuffer =
            bytes.buffer instanceof ArrayBuffer && bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
              ? bytes.buffer
              : bytes.slice().buffer;
          const gltf = await this.gltfLoader.parseAsync(ab, basePath);
          sceneRoot = gltf.scene;
        }
      }
      console.timeEnd('[FlexWeb Load] GLTFLoader parse');
      if (gen !== this.loadGeneration) return;

      await this.installGLBScene(sceneRoot, option.id, option.name, cacheKey, memHit ?? null, cachePromise, option.paint, gen, onProgress);
    } catch (err) {
      console.error(`[FlexWeb Load] Failed to load vehicle "${option.name}"; keeping active vehicle:`, err);
      if (!this.vehicle.root.parent) {
        this.scene.add(this.vehicle.root);
      }
      throw err;
    } finally {
      console.timeEnd(totalTag);
    }
  }

  /**
   * Saves an uploaded `.glb` / `.gltf` / `.zip` file into IndexedDB (`FlexWebSavedCars`),
   * parses and fits its soft-body lattice via safe skinning + cache, and makes it the single active car in RAM.
   */
  async loadCustomGLBFile(file: File, onProgress?: LoadProgressCallback): Promise<CarOption> {
    const gen = ++this.loadGeneration;
    const totalTag = `[FlexWeb Load] Total (${file.name})`;
    console.time(totalTag);
    try {
      onProgress?.('Saving vehicle to local storage…', 15);
      const { meta, buffer } = await saveUploadedCarGLB(file);
      const cacheKey = `v${SKINNING_CACHE_SCHEMA_VERSION}:${meta.id}:${meta.bytes}`;
      const memHit = this.memSkinCache.get(cacheKey);
      const cachePromise = memHit ? Promise.resolve(null) : loadCachedSkinning(cacheKey);

      console.time('[FlexWeb Load] GLTFLoader parse');
      const copy = buffer.slice(0);
      const hdr = new Uint8Array(copy, 0, Math.min(4, copy.byteLength));
      let sceneRoot: Object3D;
      if (hdr.length >= 2 && hdr[0] === 0x50 && hdr[1] === 0x4b) {
        sceneRoot = await parseGltfFromZipBytes(new Uint8Array(copy), onProgress);
      } else {
        onProgress?.('Parsing 3D Mesh…', 38);
        const gltf = await this.gltfLoader.parseAsync(copy, '/');
        sceneRoot = gltf.scene;
      }
      console.timeEnd('[FlexWeb Load] GLTFLoader parse');

      const opt: CarOption = {
        id: meta.id,
        name: `${meta.name} (Saved)`,
        savedInDB: true,
        bytes: meta.bytes,
      };
      await this.installGLBScene(sceneRoot, opt.id, meta.name, cacheKey, memHit ?? null, cachePromise, undefined, gen, onProgress);
      return opt;
    } catch (err) {
      console.error(`[FlexWeb Load] Failed to load uploaded file "${file.name}"; keeping active vehicle:`, err);
      throw err;
    } finally {
      console.timeEnd(totalTag);
    }
  }

  /**
   * Parses and spawns a `.glb` / `.gltf` / `.zip` directly from an in-memory local `ArrayBuffer`
   * (used immediately after saving a newly uploaded car so the browser never re-downloads it).
   */
  async loadGLBFromLocalBuffer(
    option: CarOption,
    buffer: ArrayBuffer,
    onProgress?: LoadProgressCallback,
  ): Promise<void> {
    const gen = ++this.loadGeneration;
    const totalTag = `[FlexWeb Load] Total (${option.name})`;
    console.time(totalTag);
    const cacheKey = `v${SKINNING_CACHE_SCHEMA_VERSION}:${option.id}:${option.bytes ?? buffer.byteLength}`;
    const memHit = this.memSkinCache.get(cacheKey);
    const cachePromise = memHit ? Promise.resolve(null) : loadCachedSkinning(cacheKey);
    try {
      console.time('[FlexWeb Load] GLTFLoader parse');
      const copy = buffer.slice(0);
      const hdr = new Uint8Array(copy, 0, Math.min(4, copy.byteLength));
      let sceneRoot: Object3D;
      if (hdr.length >= 2 && hdr[0] === 0x50 && hdr[1] === 0x4b) {
        sceneRoot = await parseGltfFromZipBytes(new Uint8Array(copy), onProgress);
      } else {
        onProgress?.('Parsing 3D Mesh…', 38);
        const gltf = await this.gltfLoader.parseAsync(copy, '/');
        sceneRoot = gltf.scene;
      }
      console.timeEnd('[FlexWeb Load] GLTFLoader parse');
      if (gen !== this.loadGeneration) return;
      await this.installGLBScene(sceneRoot, option.id, option.name, cacheKey, memHit ?? null, cachePromise, option.paint, gen, onProgress);
    } catch (err) {
      console.error(`[FlexWeb Load] Failed to parse local model "${option.name}"; keeping active vehicle:`, err);
      if (!this.vehicle.root.parent) this.scene.add(this.vehicle.root);
    } finally {
      console.timeEnd(totalTag);
    }
  }

  private async installGLBScene(
    sceneRoot: Object3D,
    carId: string,
    carName: string,
    cacheKey: string,
    memBindings: PrecomputedBindingData[] | null,
    cachePromise: Promise<Awaited<ReturnType<typeof loadCachedSkinning>>>,
    paint: number | undefined,
    gen: number,
    onProgress?: LoadProgressCallback,
  ): Promise<void> {
    const devType = detectDeviceProfile().deviceType;
    const isMobileOrTablet = this.constrained || devType === 'phone' || devType === 'tablet';
    const maxTexDim = isMobileOrTablet ? 512 : 1024;

    // Decode & resize oversized textures asynchronously in parallel with IndexedDB cache lookup
    onProgress?.('Decoding & optimizing textures…', 52);
    await yieldFrame();
    console.time('[FlexWeb Load] GPU texture decode/resize');
    const [cached] = await Promise.all([
      cachePromise,
      downscaleLargeTexturesAsync(sceneRoot, maxTexDim, isMobileOrTablet),
    ]);
    console.timeEnd('[FlexWeb Load] GPU texture decode/resize');
    if (gen !== this.loadGeneration) return;

    onProgress?.('Fitting soft-body chassis & wheels…', 68);
    await yieldFrame();
    console.time('[FlexWeb Load] mesh merge');
    const savedPaint = this.paintByCar[carId];
    const src = fitGLBToVehicleSource(sceneRoot, carName, savedPaint ?? paint, savedPaint);
    console.timeEnd('[FlexWeb Load] mesh merge');
    if (gen !== this.loadGeneration) {
      src.dispose();
      return;
    }

    const candidateBindings = memBindings ?? cached?.bindings ?? null;
    const cacheMatches = Boolean(
      candidateBindings &&
      candidateBindings.length === src.parts.length &&
      candidateBindings.every((b, idx) =>
        b.group === src.parts[idx].group &&
        b.vertCount === src.parts[idx].mesh.geometry.getAttribute('position').count,
      ),
    );

    // Build & bind the NEW vehicle non-blockingly BEFORE disposing the currently active vehicle (atomic swap)
    let nextVehicle: SoftBodyVehicle;
    if (cacheMatches && candidateBindings) {
      onProgress?.('Restoring cached soft-body skinning…', 88);
      await yieldFrame();
      console.time('[FlexWeb Load] skinning weights (cache hit)');
      nextVehicle = new SoftBodyVehicle(src, this.arena, 'defer');
      await yieldFrame();
      await nextVehicle.applyPrecomputedBindingsAsync(candidateBindings);
      console.timeEnd('[FlexWeb Load] skinning weights (cache hit)');
    } else {
      onProgress?.('Computing soft-body skinning weights…', 80);
      await yieldFrame();
      console.time('[FlexWeb Load] skinning weights + adjacency build');
      nextVehicle = new SoftBodyVehicle(src, this.arena, 'defer');
      await yieldFrame();
      try {
        const bindings = await this.computeSkinningOffThread(nextVehicle);
        if (gen !== this.loadGeneration) {
          nextVehicle.dispose();
          return;
        }
        await nextVehicle.applyPrecomputedBindingsAsync(bindings);
      } catch (workerErr) {
        console.warn('[FlexWeb Load] Worker skinning unavailable, falling back to chunked main-thread O(V) grid:', workerErr);
        let sliceStart = performance.now();
        for (const part of nextVehicle.getVisualParts()) {
          nextVehicle.deformer.bind(part.mesh, part.group, nextVehicle.restLocal);
          if (performance.now() - sliceStart > 10) {
            await yieldFrame();
            sliceStart = performance.now();
          }
        }
        nextVehicle.deformer.warmup();
      }
      console.timeEnd('[FlexWeb Load] skinning weights + adjacency build');

      onProgress?.('Saving skinning cache & settings…', 92);
      await yieldFrame();
      const exported = nextVehicle.deformer.exportBindingsForCache();
      this.memSkinCache.set(cacheKey, exported);
      try {
        await saveCachedSkinning({
          key: cacheKey,
          schemaVersion: SKINNING_CACHE_SCHEMA_VERSION,
          nodeCount: nextVehicle.solver.nodeCount,
          totalVerts: nextVehicle.deformer.totalVehicleVerts,
          savedAt: Date.now(),
          bindings: exported,
        });
      } catch {
        // non-fatal if IndexedDB quota is full
      }
    }

    if (gen !== this.loadGeneration) {
      nextVehicle.dispose();
      return;
    }

    onProgress?.('Spawning vehicle in arena…', 96);
    await yieldFrame();

    // Atomic swap: park old vehicle in bounded LRU cache (auto-disposes oldest if over RAM limit) so switching back takes < 5 ms!
    if (this.activeCacheKey && this.activeCacheKey !== cacheKey) {
      this.scene.remove(this.vehicle.root);
      this.scene.remove(this.vehicle.glass.mesh);
      this.scene.remove(this.vehicle.fluids.group);
      this.vehicleTemplateCache.set(this.activeCacheKey, this.vehicle);
    } else {
      this.vehicle.dispose();
    }
    this.vehicle = nextVehicle;
    this.activeCacheKey = cacheKey;
    this.vehicle.solver.substeps = this.loop.maxSubstepsPerFrame;
    this.vehicle.root.visible = true;
    this.vehicle.headlightsOn = this.headlightsOn;
    if (this.currentSettings) {
      this.vehicle.steeringSensitivity = this.currentSettings.steeringSensitivity;
      this.vehicle.absEnabled = this.currentSettings.absEnabled;
      this.vehicle.tcsEnabled = this.currentSettings.tcsEnabled;
      this.vehicle.transmissionMode = this.currentSettings.transmissionMode;
    }
    this.sound.setProfileFromCarName(this.vehicle.name);
    this.activeCarId = carId;
    setLastSelectedCarId(carId);
    writePersistedSettings({
      cameraMode: this.cameraMode,
      showLattice: this.showLattice,
      bodyOpacity: this.bodyOpacity,
    });
    if (this.bodyOpacity < 0.99) this.vehicle.setBodyOpacity(this.bodyOpacity);
    this.scene.add(this.vehicle.root);
    this.scene.add(this.vehicle.glass.mesh);
    this.scene.add(this.vehicle.fluids.group);
    this.rain.attachToVehicle(this.vehicle.root);
    this.buildLatticeOverlay();
    this.renderDirty = true;
    this.reset(0, 0, 0, 0);

    // Warm up vehicle frame & render without freezing mobile GPUs with heavy synchronous compile stalls
    this.vehicle.render(1, 0);
    this.updateCamera(0.016);
    this.updateShadowLight();
    if (this.showLattice) this.updateLatticeOverlay();
    if (!this.isMobilePhone && !this.constrained) {
      await yieldFrame();
      this.renderer.compile(this.scene, this.camera);
    }
    await yieldFrame();
    this.renderer.render(this.scene, this.camera);
    await yieldFrame();
    if (gen !== this.loadGeneration) return;

    onProgress?.('Vehicle Spawned', 100);
    this.scheduleSecondaryPreload();
  }

  /**
   * Staggered low-priority background preloading for secondary vehicles into AssetCache
   * using requestIdleCallback (never blocks active gameplay or saves data on metered connections).
   */
  private scheduleSecondaryPreload(): void {
    if (this.secondaryPreloadScheduled) return;
    this.secondaryPreloadScheduled = true;
    const nav = navigator as Navigator & { connection?: { saveData?: boolean } };
    if (nav.connection?.saveData) return;

    const urls: string[] = [];
    for (const c of GITHUB_CARS) {
      if (c.id === this.activeCarId) continue;
      const localUrl = c.remoteUrls.find((u) => u.startsWith('/'));
      if (localUrl) urls.push(localUrl);
    }
    if (urls.length === 0) return;

    const scheduleIdle = (fn: () => void, delayMs: number) => {
      const w = window as unknown as { requestIdleCallback?: (cb: () => void, opts?: { timeout: number }) => number };
      setTimeout(() => {
        if (w.requestIdleCallback) w.requestIdleCallback(fn, { timeout: 4000 });
        else fn();
      }, delayMs);
    };

    urls.forEach((url: string, idx: number) => {
      scheduleIdle(() => {
        void (async () => {
          try {
            const existing = await getCachedAssetBytes(url);
            if (existing && existing.byteLength >= 12) return;
            const res = await fetch(url, { priority: 'low' } as RequestInit);
            if (!res.ok) return;
            const bytes = new Uint8Array(await res.arrayBuffer());
            if (bytes.byteLength >= 12 && bytes[0] !== 0x3c) {
              await putCachedAssetBytes(url, bytes);
            }
          } catch {
            // non-fatal background cache warm-up
          }
        })();
      }, 3500 + idx * 4500);
    });
  }

  private getSkinningWorker(): Worker | null {
    if (typeof Worker === 'undefined') return null;
    if (!this.skinningWorker) {
      this.skinningWorker = new Worker(new URL('./skinningWorker.ts', import.meta.url), { type: 'module' });
    }
    return this.skinningWorker;
  }

  /** Offloads top-4 skinning weight & Laplacian adjacency table calculation to skinningWorker.ts using cloned Transferables. */
  private computeSkinningOffThread(vehicle: SoftBodyVehicle): Promise<PrecomputedBindingData[]> {
    const worker = this.getSkinningWorker();
    const parts = vehicle.getVisualParts();
    if (!worker) {
      for (const part of parts) vehicle.deformer.bind(part.mesh, part.group, vehicle.restLocal);
      return Promise.resolve(vehicle.deformer.exportBindingsForCache());
    }

    const reqId = this.workerReqId++;
    const workerParts: SkinningWorkerPartInput[] = [];
    const transferables: Transferable[] = [];
    for (let i = 0; i < parts.length; i++) {
      const geo = parts[i].mesh.geometry as BufferGeometry;
      if (!geo.getAttribute('normal')) geo.computeVertexNormals();
      // Always clone position/normal arrays before transferring so original Three.js geometry buffers are NEVER detached
      const posCopy = new Float32Array((geo.getAttribute('position') as BufferAttribute).array as Float32Array);
      const nrmCopy = new Float32Array((geo.getAttribute('normal') as BufferAttribute).array as Float32Array);
      const idxAttr = geo.getIndex();
      let idxCopy: Uint16Array | Uint32Array | null = null;
      if (idxAttr) {
        idxCopy = idxAttr.array instanceof Uint16Array
          ? new Uint16Array(idxAttr.array)
          : new Uint32Array(idxAttr.array);
        transferables.push(idxCopy.buffer);
      }
      transferables.push(posCopy.buffer, nrmCopy.buffer);
      workerParts.push({ group: parts[i].group, positions: posCopy, normals: nrmCopy, indices: idxCopy });
    }

    const restLocalCopy = new Float32Array(vehicle.restLocal);
    const restFramesCopy = new Float32Array(vehicle.deformer.getRestFrames());
    const groupNodesCopy = vehicle.deformer.getGroupNodes().map((g) => new Uint16Array(g));
    transferables.push(restLocalCopy.buffer, restFramesCopy.buffer);
    for (const g of groupNodesCopy) transferables.push(g.buffer);

    return new Promise((resolve, reject) => {
      const onMessage = (ev: MessageEvent<SkinningWorkerResponse>) => {
        if (ev.data.id !== reqId) return;
        worker.removeEventListener('message', onMessage);
        worker.removeEventListener('error', onError);
        resolve(ev.data.bindings);
      };
      const onError = (err: ErrorEvent) => {
        worker.removeEventListener('message', onMessage);
        worker.removeEventListener('error', onError);
        reject(err.error ?? new Error(err.message));
      };
      worker.addEventListener('message', onMessage);
      worker.addEventListener('error', onError);
      const req: SkinningWorkerRequest = {
        id: reqId,
        restLocal: restLocalCopy,
        restFrames: restFramesCopy,
        groupNodes: groupNodesCopy,
        parts: workerParts,
      };
      worker.postMessage(req, transferables);
    });
  }

  private replaceActiveVehicle(src: ReturnType<typeof createProceduralVehicleSource>): void {
    // Dispose previous vehicle completely so only 1 car stays in WebGL/RAM
    this.vehicle.dispose();
    this.vehicle = new SoftBodyVehicle(src, this.arena);
    const isLowPerf = this.isMobilePhone || this.currentSettings?.resolutionMode === 'low';
    this.vehicle.solver.substeps = this.loop.maxSubstepsPerFrame;
    this.vehicle.deformer.lowQualityNormals = isLowPerf;
    this.vehicle.root.visible = true;
    if (this.currentSettings) {
      this.vehicle.steeringSensitivity = this.currentSettings.steeringSensitivity;
      this.vehicle.absEnabled = this.currentSettings.absEnabled;
      this.vehicle.tcsEnabled = this.currentSettings.tcsEnabled;
    }
    this.sound.setProfileFromCarName(this.vehicle.name);
    if (this.bodyOpacity < 0.99) this.vehicle.setBodyOpacity(this.bodyOpacity);
    this.scene.add(this.vehicle.root);
    this.scene.add(this.vehicle.glass.mesh);
    this.scene.add(this.vehicle.fluids.group);
    this.rain.attachToVehicle(this.vehicle.root);
    this.buildLatticeOverlay();
    this.renderDirty = true;
    this.reset(0, 0, 0, 0);
  }

  // ================================================================ pointer / resize

  private bindPointerEvents(el: HTMLElement): void {
    el.addEventListener('pointerdown', (e) => {
      this.sound.unlock();
      this.activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (this.activePointers.size === 2) {
        const pts = Array.from(this.activePointers.values());
        this.lastPinchDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
      }
      this.isPointerDown = true;
      this.lastPtrX = e.clientX;
      this.lastPtrY = e.clientY;
      try { el.setPointerCapture(e.pointerId); } catch { /* ignore */ }
    });
    el.addEventListener('pointermove', (e) => {
      if (!this.activePointers.has(e.pointerId)) return;
      this.activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

      // 2-finger pinch-to-zoom on touchscreens
      if (this.activePointers.size === 2) {
        const pts = Array.from(this.activePointers.values());
        const distNow = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
        if (this.lastPinchDist > 10 && distNow > 10) {
          const ratio = this.lastPinchDist / distNow;
          if (this.overheadMapMode) {
            this.mapZoomY = Math.max(140, Math.min(3400, this.mapZoomY * ratio));
          } else {
            this.orbitDist = Math.max(3.2, Math.min(16, this.orbitDist * ratio));
            this.userDragTimer = 3.0;
          }
        }
        this.lastPinchDist = distNow;
        return;
      }

      if (!this.isPointerDown) return;
      const dx = e.clientX - this.lastPtrX;
      const dy = e.clientY - this.lastPtrY;
      this.lastPtrX = e.clientX;
      this.lastPtrY = e.clientY;
      if (this.overheadMapMode) {
        const panScale = this.mapZoomY * 0.0014;
        this.mapPanX = Math.max(-1000, Math.min(1000, this.mapPanX + dx * panScale));
        this.mapPanZ = Math.max(-1000, Math.min(1000, this.mapPanZ + dy * panScale));
        return;
      }
      this.orbitYaw -= dx * 0.007;
      this.orbitPitch = Math.max(0.05, Math.min(1.35, this.orbitPitch + dy * 0.006));
      this.userDragTimer = 3.0;
    });
    const up = (e: PointerEvent) => {
      this.activePointers.delete(e.pointerId);
      if (this.activePointers.size < 2) this.lastPinchDist = 0;
      if (this.activePointers.size === 0) this.isPointerDown = false;
    };
    el.addEventListener('pointerup', up);
    el.addEventListener('pointercancel', up);
    el.addEventListener('wheel', (e) => {
      e.preventDefault();
      if (this.overheadMapMode) {
        this.mapZoomY = Math.max(140, Math.min(3400, this.mapZoomY + e.deltaY * 1.1));
        return;
      }
      this.orbitDist = Math.max(3.2, Math.min(16, this.orbitDist + e.deltaY * 0.004));
      this.userDragTimer = 3.0;
    }, { passive: false });
  }

  private onResize(): void {
    const w = Math.max(1, this.container.clientWidth);
    const h = Math.max(1, this.container.clientHeight);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
    this.renderDirty = true;
  }

  dispose(): void {
    this.loop.stop();
    this.sound.dispose();
    this.resizeObserver.disconnect();
    if (this.skinningWorker) {
      this.skinningWorker.terminate();
      this.skinningWorker = null;
    }
    this.memSkinCache.clear();
    this.vehicleTemplateCache.clear();
    this.rain.dispose();
    this.vehicle.dispose();
    this.arena.dispose();
    if (this.latticeLines) {
      this.latticeLines.geometry.dispose();
      (this.latticeLines.material as LineBasicMaterial).dispose();
    }
    this.skyDome.geometry.dispose();
    (this.skyDome.material as ShaderMaterial).dispose();
    this.cubeRT.dispose();
    this.studioEnvRT.dispose();
    this.envRT?.dispose();
    this.pmrem.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}

function yieldFrame(): Promise<void> {
  return new Promise((resolve) => {
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(() => resolve());
    else setTimeout(resolve, 0);
  });
}

/**
 * Decodes & resizes oversized GLB textures (> `maxDim`, e.g. 1024px) asynchronously
 * off the main thread using `createImageBitmap(..., { resizeWidth, resizeHeight })`,
 * without closing source bitmaps that may be shared across GLB textures.
 */
async function downscaleLargeTexturesAsync(root: Object3D, maxDim: number, isMobileOrTablet: boolean): Promise<void> {
  if (typeof createImageBitmap !== 'function') return;
  const seen = new Set<Texture>();
  const texKeys: (keyof MeshStandardMaterial)[] = ['map', 'normalMap', 'roughnessMap', 'metalnessMap', 'emissiveMap', 'aoMap'];
  const tasks: Promise<void>[] = [];

  root.traverse((obj) => {
    const m = obj as Mesh;
    if (!m.isMesh || !m.material) return;
    const mats = Array.isArray(m.material) ? m.material : [m.material];
    for (const mat of mats) {
      const sm = mat as MeshStandardMaterial;
      for (const k of texKeys) {
        const tex = sm[k] as Texture | null | undefined;
        if (!tex || !tex.isTexture || seen.has(tex)) continue;
        seen.add(tex);
        tex.generateMipmaps = true;
        tex.anisotropy = isMobileOrTablet ? 1 : Math.min(4, tex.anisotropy || 1);
        const img = tex.image as (ImageBitmapSource & { width?: number; height?: number }) | undefined;
        if (!img || !img.width || !img.height) continue;
        const w = img.width, h = img.height;
        if (w <= maxDim && h <= maxDim) continue;
        const scale = maxDim / Math.max(w, h);
        const nw = Math.max(1, Math.round(w * scale));
        const nh = Math.max(1, Math.round(h * scale));
        tasks.push(
          createImageBitmap(img, { resizeWidth: nw, resizeHeight: nh, resizeQuality: 'medium' })
            .then((bitmap) => {
              tex.image = bitmap;
              tex.needsUpdate = true;
            })
            .catch(() => { /* keep original image if browser rejects resize */ }),
        );
      }
    }
  });

  if (tasks.length > 0) await Promise.all(tasks);
}

