import {
  AdditiveBlending,
  BoxGeometry,
  BufferAttribute,
  BufferGeometry,
  CanvasTexture,
  Color,
  Euler,
  Group,
  InstancedMesh,
  Material,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  MeshPhysicalMaterial,
  MeshStandardMaterial,
  NormalBlending,
  Object3D,
  Points,
  Quaternion,
  ShaderMaterial,
  Vector3,
} from 'three';

export type WeatherMode = 'off' | 'drizzle' | 'heavy';

export interface RainShaderUniforms {
  uRainTime: { value: number };
  uRainDropPhase: { value: number };
  uRainAirflowPhase: { value: number };
  uRainIntensity: { value: number };
  uRainWetness: { value: number };
  uRainWind: { value: Vector3 };
  uRainSpeed: { value: number };
}

interface TrackedCarMaterial {
  mat: MeshStandardMaterial;
  baseRoughness: number;
  baseEnvMapIntensity: number;
  baseClearcoat: number;
  baseClearcoatRoughness: number;
  isGlass: boolean;
}

/**
 * RainWeatherSystem
 * -----------------
 * HD PBR Rain & Dripping Raindrops System with strictly Zero-GC per-frame updates
 * and a zero-cost idle guard when `rainIntensity === 0` and `wetness === 0`.
 *
 * Features:
 *  1. `onBeforeCompile` procedural raindrop & streaking-trail shader injected into
 *     vehicle paint & glass materials (beading on horizontal panels, dripping &
 *     wind-deflected trails on sloped/vertical windows and doors, PBR normal
 *     perturbation + glossy wet roughness response).
 *  2. Dynamic wet PBR roughness/clearcoat/envMap modulation on car paint & glass
 *     without recreating materials.
 *  3. Pooled `InstancedMesh` airborne rain streaks wrapped around the camera/car.
 *  4. Pooled soft water-mist `Points` tire spray emitted behind grounded wheels
 *     when driving at speed in rain or wet conditions.
 */
export class RainWeatherSystem {
  readonly group = new Group();
  readonly uniforms: RainShaderUniforms;

  /** Target rain intensity: 0 = Off, 0.45 = Light Drizzle, 1.0 = Heavy Rain. */
  targetIntensity = 0;
  /** Smoothly interpolated current rain intensity (0..1). */
  intensity = 0;
  /** Smoothly interpolated surface wetness (0..1) — builds up in rain and dries when off. */
  wetness = 0;

  private readonly maxDrops: number;
  private readonly maxSpray: number;

  // Pre-allocated airborne raindrop streak pool (SoA Float32Arrays)
  private readonly dropMesh: InstancedMesh;
  private readonly dropMat: MeshBasicMaterial;
  private readonly rx: Float32Array;
  private readonly ry: Float32Array;
  private readonly rz: Float32Array;
  private readonly rSpeed: Float32Array;
  private readonly rLen: Float32Array;
  private readonly rPhase: Float32Array;

  // Pre-allocated tire water spray pool (SoA Float32Arrays)
  private readonly sprayPoints: Points;
  private readonly sprayGeo: BufferGeometry;
  private readonly sprayMat: ShaderMaterial;
  private readonly sprayTex: CanvasTexture;
  private readonly spPos: Float32Array;
  private readonly spAlpha: Float32Array;
  private readonly spSize: Float32Array;
  private readonly spVx: Float32Array;
  private readonly spVy: Float32Array;
  private readonly spVz: Float32Array;
  private readonly spLife: Float32Array;
  private readonly spMaxLife: Float32Array;
  private readonly spActive: Uint8Array;
  private sprayCursor = 0;
  private activeSprayCount = 0;
  private sprayEmitTimer = 0;

  // Tracked vehicle PBR materials for zero-allocation dynamic wet response
  private readonly trackedMaterials: TrackedCarMaterial[] = [];

  // Pre-allocated scratch math objects (strictly zero allocations inside update)
  private readonly tmpPos = new Vector3();
  private readonly tmpScale = new Vector3();
  private readonly tmpQuat = new Quaternion();
  private readonly tmpEuler = new Euler();
  private readonly tmpMat = new Matrix4();
  private readonly tmpUp = new Vector3(0, 1, 0);
  private readonly tmpDir = new Vector3(0, -1, 0);
  private readonly rainTimeAcc = { t: 0, dropPhase: 0, airflowPhase: 0 };

  // Pre-allocated sky/fog mood colors
  readonly drySkyColor = new Color(0x8ec5f0);
  readonly wetSkyColor = new Color(0x526478);
  readonly outSkyColor = new Color(0x8ec5f0);

  constructor(isConstrained: boolean) {
    this.group.name = 'RainWeatherSystem';
    this.maxDrops = isConstrained ? 320 : 680;
    this.maxSpray = isConstrained ? 56 : 110;

    this.uniforms = {
      uRainTime: { value: 0 },
      uRainDropPhase: { value: 0 },
      uRainAirflowPhase: { value: 0 },
      uRainIntensity: { value: 0 },
      uRainWetness: { value: 0 },
      uRainWind: { value: new Vector3(0, -1, 0) },
      uRainSpeed: { value: 0 },
    };

    // 1. Airborne rain streaks (1 InstancedMesh draw call)
    const streakGeo = new BoxGeometry(0.008, 0.62, 0.008);
    this.dropMat = new MeshBasicMaterial({
      color: 0xe2efff,
      transparent: true,
      opacity: 0.34,
      blending: AdditiveBlending,
      depthWrite: false,
    });
    this.dropMesh = new InstancedMesh(streakGeo, this.dropMat, this.maxDrops);
    this.dropMesh.frustumCulled = false; // always centered around camera when visible
    this.dropMesh.instanceMatrix.setUsage(35048); // DynamicDrawUsage
    this.dropMesh.visible = false;

    this.rx = new Float32Array(this.maxDrops);
    this.ry = new Float32Array(this.maxDrops);
    this.rz = new Float32Array(this.maxDrops);
    this.rSpeed = new Float32Array(this.maxDrops);
    this.rLen = new Float32Array(this.maxDrops);
    this.rPhase = new Float32Array(this.maxDrops);

    for (let i = 0; i < this.maxDrops; i++) {
      // Deterministic golden-ratio spread around camera box (34m x 18m x 34m)
      this.rx[i] = (((i * 0.6180339887) % 1) - 0.5) * 34;
      this.ry[i] = (((i * 0.3819660113) % 1)) * 18;
      this.rz[i] = (((i * 0.7548776662) % 1) - 0.5) * 34;
      this.rSpeed[i] = 15.5 + ((i * 19) % 11) * 0.65;
      this.rLen[i] = 0.72 + ((i * 13) % 9) * 0.08;
      this.rPhase[i] = (i * 0.271828) % 6.283;
    }

    // 2. Tire water mist spray (1 Points draw call, zero GC)
    this.spPos = new Float32Array(this.maxSpray * 3);
    this.spAlpha = new Float32Array(this.maxSpray);
    this.spSize = new Float32Array(this.maxSpray);
    this.spVx = new Float32Array(this.maxSpray);
    this.spVy = new Float32Array(this.maxSpray);
    this.spVz = new Float32Array(this.maxSpray);
    this.spLife = new Float32Array(this.maxSpray);
    this.spMaxLife = new Float32Array(this.maxSpray);
    this.spActive = new Uint8Array(this.maxSpray);

    this.sprayGeo = new BufferGeometry();
    const posAttr = new BufferAttribute(this.spPos, 3);
    posAttr.setUsage(35048);
    const alphaAttr = new BufferAttribute(this.spAlpha, 1);
    alphaAttr.setUsage(35048);
    const sizeAttr = new BufferAttribute(this.spSize, 1);
    sizeAttr.setUsage(35048);
    this.sprayGeo.setAttribute('position', posAttr);
    this.sprayGeo.setAttribute('aAlpha', alphaAttr);
    this.sprayGeo.setAttribute('aSize', sizeAttr);

    this.sprayTex = createSoftMistTexture();
    this.sprayMat = new ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: NormalBlending,
      uniforms: {
        uMap: { value: this.sprayTex },
        uColor: { value: new Color(0xdbe8f4) },
      },
      vertexShader: `
        attribute float aAlpha;
        attribute float aSize;
        varying float vAlpha;
        void main() {
          vAlpha = aAlpha;
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = aSize * (260.0 / max(1.0, -mvPosition.z));
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        uniform sampler2D uMap;
        uniform vec3 uColor;
        varying float vAlpha;
        void main() {
          if (vAlpha <= 0.005) discard;
          float mask = texture2D(uMap, gl_PointCoord).a;
          float a = mask * vAlpha;
          if (a <= 0.005) discard;
          gl_FragColor = vec4(uColor, a);
        }
      `,
    });
    this.sprayPoints = new Points(this.sprayGeo, this.sprayMat);
    this.sprayPoints.frustumCulled = false;
    this.sprayPoints.visible = false;

    this.group.add(this.dropMesh, this.sprayPoints);
  }

  /** Set weather preset ('off' | 'drizzle' | 'heavy'). */
  setWeatherMode(mode: WeatherMode): void {
    this.targetIntensity = mode === 'heavy' ? 1.0 : mode === 'drizzle' ? 0.45 : 0.0;
  }

  /** Set continuous rain intensity (0..1). */
  setIntensity(value: number): void {
    this.targetIntensity = Math.max(0, Math.min(1, value));
  }

  /** Returns true if any weather effect or wet drying transition is active. */
  isActive(): boolean {
    return this.targetIntensity > 0.001 || this.intensity > 0.001 || this.wetness > 0.001 || this.activeSprayCount > 0;
  }

  /**
   * Injects the animated raindrop & streaking-trail shader hook (`onBeforeCompile`)
   * into the active vehicle's paint & glass materials and caches baseline PBR values
   * so dynamic wetness updates never recreate materials.
   */
  attachToVehicle(vehicleRoot: Object3D): void {
    this.trackedMaterials.length = 0;
    const seen = new Set<Material>();

    vehicleRoot.traverse((obj) => {
      const mesh = obj as Mesh;
      if (!mesh.isMesh) return;
      const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      for (let i = 0; i < mats.length; i++) {
        const m = mats[i];
        if (!m || seen.has(m)) continue;
        seen.add(m);
        if (!(m as MeshStandardMaterial).isMeshStandardMaterial) continue;
        const sm = m as MeshStandardMaterial;
        const nameLower = `${sm.name || ''} ${mesh.name || ''}`.toLowerCase();

        // Skip tires, rubber, interior cabin fabrics, and brake discs
        if (
          nameLower.includes('tire') ||
          nameLower.includes('tyre') ||
          nameLower.includes('rubber') ||
          nameLower.includes('interior') ||
          nameLower.includes('seat') ||
          nameLower.includes('carpet') ||
          nameLower.includes('disc') ||
          nameLower.includes('caliper')
        ) {
          continue;
        }

        const isGlass =
          sm.transparent ||
          nameLower.includes('glass') ||
          nameLower.includes('window') ||
          nameLower.includes('windshield') ||
          nameLower.includes('headlight') ||
          nameLower.includes('taillight');

        const pm = sm as MeshPhysicalMaterial;
        this.trackedMaterials.push({
          mat: sm,
          baseRoughness: sm.roughness,
          baseEnvMapIntensity: sm.envMapIntensity ?? 1.0,
          baseClearcoat: pm.isMeshPhysicalMaterial ? (pm.clearcoat ?? 0) : 0,
          baseClearcoatRoughness: pm.isMeshPhysicalMaterial ? (pm.clearcoatRoughness ?? 0.1) : 0.1,
          isGlass,
        });

        this.injectRainShader(sm);
      }
    });

    // Apply current wetness immediately if vehicle was swapped during active rain
    if (this.wetness > 0.001) {
      this.applyVehicleWetness();
    }
  }

  private injectRainShader(mat: MeshStandardMaterial): void {
    if (mat.userData.__flexwebRainInjected) return;
    mat.userData.__flexwebRainInjected = true;

    const prevOnBeforeCompile = mat.onBeforeCompile;
    const uniforms = this.uniforms;

    mat.customProgramCacheKey = () => 'flexweb_hd_pbr_rain_v3';
    mat.onBeforeCompile = (shader, renderer) => {
      if (prevOnBeforeCompile) prevOnBeforeCompile(shader, renderer);

      shader.uniforms.uRainTime = uniforms.uRainTime;
      shader.uniforms.uRainDropPhase = uniforms.uRainDropPhase;
      shader.uniforms.uRainAirflowPhase = uniforms.uRainAirflowPhase;
      shader.uniforms.uRainIntensity = uniforms.uRainIntensity;
      shader.uniforms.uRainWetness = uniforms.uRainWetness;
      shader.uniforms.uRainWind = uniforms.uRainWind;
      shader.uniforms.uRainSpeed = uniforms.uRainSpeed;

      // Pass car-local position and normal to fragment shader for seamless triplanar raindrops
      shader.vertexShader = shader.vertexShader
        .replace(
          '#include <common>',
          `#include <common>
          varying vec3 vRainLocalPos;
          varying vec3 vRainLocalNormal;`,
        )
        .replace(
          '#include <beginnormal_vertex>',
          `#include <beginnormal_vertex>
          vRainLocalNormal = normalize(objectNormal);`,
        )
        .replace(
          '#include <begin_vertex>',
          `#include <begin_vertex>
          vRainLocalPos = transformed;`,
        );

      // Smooth speed-integrated procedural dripping raindrops
      shader.fragmentShader = shader.fragmentShader
        .replace(
          '#include <common>',
          `#include <common>
          uniform float uRainTime;
          uniform float uRainDropPhase;
          uniform float uRainAirflowPhase;
          uniform float uRainIntensity;
          uniform float uRainWetness;
          uniform vec3 uRainWind;
          uniform float uRainSpeed;
          varying vec3 vRainLocalPos;
          varying vec3 vRainLocalNormal;

          float rainHash21(vec2 p) {
            p = fract(p * vec2(123.34, 456.21));
            p += dot(p, p + 45.32);
            return fract(p.x * p.y);
          }

          // Returns vec4(gradX, gradY, wetTrailMask, dropRimGlint) using smoothly integrated phase
          vec4 rainDropLayer(vec2 uv, float phase, float dropSize) {
            vec2 aspect = vec2(2.1, 1.0);
            vec2 st = uv * aspect;
            vec2 id = floor(st);
            float n = rainHash21(id);
            float tLocal = phase * (0.68 + n * 0.65) + n * 6.2831;
            float ySlide = fract(tLocal * 0.26);
            vec2 gv = fract(st) - 0.5;

            // Organic surface-tension wiggle as drops stream down paint & glass
            float wiggle = sin(tLocal * 1.35 + gv.y * 5.2 + n * 12.0) * 0.16;
            float dropX = (n - 0.5) * 0.52 + wiggle;
            float dropY = 0.44 - ySlide * 0.88;

            vec2 d = (gv - vec2(dropX, dropY)) / aspect;
            float dist = length(d);
            float dropMask = smoothstep(dropSize, dropSize * 0.18, dist);
            float dropRim = smoothstep(dropSize * 1.05, dropSize * 0.42, dist) * smoothstep(dropSize * 0.15, dropSize * 0.65, dist);

            // Tapered glistening wet trail above the sliding droplet
            float above = smoothstep(dropY - 0.02, dropY + 0.08, gv.y);
            float trailFade = smoothstep(0.52, dropY, gv.y);
            float trailWidth = mix(dropSize * 0.22, dropSize * 0.62, clamp((gv.y - dropY) * 1.7, 0.0, 1.0));
            float trailMask = above * trailFade * smoothstep(trailWidth, 0.005, abs(gv.x - dropX));

            // Dense beaded micro-droplets across the cell
            vec2 mGv = (fract(st * 2.0 + vec2(n * 3.1, n * 5.7)) - 0.5) / aspect;
            float mDist = length(mGv);
            float micro = smoothstep(dropSize * 0.52, dropSize * 0.12, mDist);
            float microRim = smoothstep(dropSize * 0.54, dropSize * 0.25, mDist) * 0.55;

            float mask = clamp(dropMask + trailMask * 0.72 + micro * 0.45, 0.0, 1.0);
            // Smooth, physically bounded convex water-bead slope (never tilts normal into black grazing angles)
            vec2 grad = d * (dropMask * 4.5) + mGv * (micro * 2.2);
            float rim = clamp(dropRim + microRim + trailMask * 0.18, 0.0, 1.0);
            return vec4(grad, mask, rim);
          }`,
        )
        .replace(
          '#include <roughnessmap_fragment>',
          `#include <roughnessmap_fragment>
          float rainTrailWet = 0.0;
          float rainDropGlint = 0.0;
          vec2 rainNormalGrad = vec2(0.0);
          if (uRainIntensity > 0.001 || uRainWetness > 0.001) {
            vec3 rn = normalize(vRainLocalNormal);
            float upness = abs(rn.y);
            float rAmt = clamp(uRainIntensity * 0.90 + uRainWetness * 0.50, 0.0, 1.0);

            if (upness > 0.60) {
              // Horizontal panels (hood, roof, trunk): crisp visible raindrops swept smoothly aft by speed
              vec2 hUv = vRainLocalPos.xz * 5.2;
              hUv.y += uRainAirflowPhase;
              vec4 hero = rainDropLayer(hUv, uRainDropPhase * 0.55, 0.064);
              vec4 med = rainDropLayer(hUv * 1.75 + vec2(3.7, 1.9), uRainDropPhase * 0.50, 0.050);
              vec4 fine = rainDropLayer(hUv * 2.85 + vec2(8.1, 4.3), uRainDropPhase * 0.35, 0.038);
              rainNormalGrad = clamp((hero.xy + med.xy * 0.65 + fine.xy * 0.35) * rAmt, vec2(-0.34), vec2(0.34));
              rainTrailWet = clamp((hero.z + med.z * 0.78 + fine.z * 0.48) * rAmt, 0.0, 1.0);
              rainDropGlint = clamp((hero.w + med.w * 0.70 + fine.w * 0.40) * rAmt, 0.0, 1.0);
            } else {
              // Sloped/vertical surfaces (windshield, windows, doors, fenders): continuous speed-driven streaming
              vec2 vUv = (abs(rn.x) > abs(rn.z))
                ? vec2(vRainLocalPos.z, vRainLocalPos.y)
                : vec2(vRainLocalPos.x, vRainLocalPos.y);
              vUv *= 5.0;
              // Side windows sweep aft at speed; windshield streams faster with speed
              if (abs(rn.x) > abs(rn.z)) {
                vUv.x += uRainAirflowPhase * 0.85;
              }
              vec4 hero = rainDropLayer(vUv, uRainDropPhase * 1.0, 0.062);
              vec4 med = rainDropLayer(vUv * 1.70 + vec2(7.3, 2.1), uRainDropPhase * 1.12, 0.048);
              vec4 fine = rainDropLayer(vUv * 2.75 + vec2(2.4, 9.2), uRainDropPhase * 0.78, 0.036);
              rainNormalGrad = clamp((hero.xy + med.xy * 0.65 + fine.xy * 0.35) * rAmt, vec2(-0.34), vec2(0.34));
              rainTrailWet = clamp((hero.z + med.z * 0.78 + fine.z * 0.48) * rAmt, 0.0, 1.0);
              rainDropGlint = clamp((hero.w + med.w * 0.70 + fine.w * 0.40) * rAmt, 0.0, 1.0);
            }
            // Crystal-clear glossy wet sheen inside droplets & streak trails (never darkens paint)
            roughnessFactor *= mix(1.0, 0.07, clamp(rainTrailWet + uRainWetness * 0.48, 0.0, 1.0));
            diffuseColor.rgb += vec3(0.26, 0.32, 0.38) * rainDropGlint * 0.38;
          }`,
        )
        .replace(
          '#include <normal_fragment_maps>',
          `#include <normal_fragment_maps>
          if (uRainIntensity > 0.001 || uRainWetness > 0.001) {
            vec3 pertN = normalize(normal + vec3(rainNormalGrad.x, rainNormalGrad.y, 0.0) * 0.36);
            normal = pertN.z > 0.05 ? pertN : normal;
          }`,
        )
        .replace(
          '#include <clearcoat_normal_fragment_maps>',
          `#include <clearcoat_normal_fragment_maps>
          #ifdef USE_CLEARCOAT
          if (uRainIntensity > 0.001 || uRainWetness > 0.001) {
            vec3 pertCC = normalize(clearcoatNormal + vec3(rainNormalGrad.x, rainNormalGrad.y, 0.0) * 0.46);
            clearcoatNormal = pertCC.z > 0.05 ? pertCC : clearcoatNormal;
          }
          #endif`,
        );
    };
    mat.needsUpdate = true;
  }

  private applyVehicleWetness(): void {
    const w = this.wetness;
    for (let i = 0; i < this.trackedMaterials.length; i++) {
      const item = this.trackedMaterials[i];
      const sm = item.mat;
      // Lower surface roughness & boost wet specular/clearcoat sheen smoothly (zero material recreation!)
      const minRough = item.isGlass ? 0.03 : 0.07;
      sm.roughness = item.baseRoughness + (minRough - item.baseRoughness) * (w * 0.72);
      sm.envMapIntensity = item.baseEnvMapIntensity * (1.0 + w * 0.48);
      const pm = sm as MeshPhysicalMaterial;
      if (pm.isMeshPhysicalMaterial) {
        pm.clearcoat = Math.min(1.0, item.baseClearcoat + w * 0.45);
        pm.clearcoatRoughness = Math.max(0.02, item.baseClearcoatRoughness * (1.0 - w * 0.65));
      }
    }
  }

  /**
   * Zero-GC per-frame update.
   * Immediately returns with zero work when rain is Off and surfaces are dry.
   */
  update(
    frameDt: number,
    camX: number,
    camY: number,
    camZ: number,
    carVx: number,
    carVy: number,
    carVz: number,
    carSpeedAbs: number,
    wheelWorldPos: Float32Array,
    wheelGrounded: Uint8Array,
    onWetPatch: boolean,
  ): number {
    // 1. Zero-cost idle guard when weather is Off and everything is dry & settled
    if (!this.isActive() && !onWetPatch) {
      if (this.dropMesh.visible) this.dropMesh.visible = false;
      if (this.sprayPoints.visible) this.sprayPoints.visible = false;
      return 0;
    }

    // 2. Smoothly ramp rain intensity & surface wetness
    const dt = Math.min(frameDt, 0.05);
    const prevWet = this.wetness;
    this.intensity += (this.targetIntensity - this.intensity) * Math.min(1, dt * 2.4);
    if (Math.abs(this.targetIntensity - this.intensity) < 0.001) this.intensity = this.targetIntensity;

    const wetTarget = this.targetIntensity > 0 ? Math.min(1, this.targetIntensity * 1.15) : 0;
    const wetRate = wetTarget > this.wetness ? 0.45 : 0.22; // wets faster than it dries
    this.wetness += (wetTarget - this.wetness) * Math.min(1, dt * wetRate);
    if (Math.abs(wetTarget - this.wetness) < 0.001) this.wetness = wetTarget;

    // 3. Smoothly integrate raindrop phase by TRUE vehicle speed (no acceleration phase jumps!)
    this.rainTimeAcc.t += dt;
    const speedRate = 1.0 + Math.min(4.2, carSpeedAbs * 0.14);
    this.rainTimeAcc.dropPhase += dt * speedRate;
    this.rainTimeAcc.airflowPhase += dt * Math.min(7.5, carSpeedAbs * 0.22);
    this.uniforms.uRainTime.value = this.rainTimeAcc.t;
    this.uniforms.uRainDropPhase.value = this.rainTimeAcc.dropPhase;
    this.uniforms.uRainAirflowPhase.value = this.rainTimeAcc.airflowPhase;
    this.uniforms.uRainIntensity.value = this.intensity;
    this.uniforms.uRainWetness.value = this.wetness;
    this.uniforms.uRainSpeed.value = carSpeedAbs;
    this.uniforms.uRainWind.value.set(-carVx * 0.12, -1.0, -carVz * 0.12);

    if (Math.abs(this.wetness - prevWet) > 0.0008) {
      this.applyVehicleWetness();
    }

    // 4. Pooled airborne rain streaks around camera
    if (this.intensity > 0.01) {
      this.dropMesh.visible = true;
      const activeCount = Math.max(1, Math.floor(this.maxDrops * this.intensity));
      this.dropMesh.count = activeCount;
      this.dropMat.opacity = 0.16 + this.intensity * 0.28;

      // Wind + slight car motion slant for rain streaks
      const windX = -carVx * 0.22 + 1.4;
      const windZ = -carVz * 0.22 + 0.8;
      this.tmpDir.set(windX, -17.5, windZ).normalize();
      this.tmpQuat.setFromUnitVectors(this.tmpUp, this.tmpDir);

      const boxXZ = 34;
      const halfXZ = 17;
      const boxY = 18;
      const minY = Math.max(0.2, camY - 3.5);

      for (let i = 0; i < activeCount; i++) {
        let x = this.rx[i] + windX * dt;
        let y = this.ry[i] - this.rSpeed[i] * dt;
        let z = this.rz[i] + windZ * dt;

        // Wrap seamlessly around active camera box (no teleport pops)
        const dx = x - camX;
        const dz = z - camZ;
        if (dx > halfXZ) x -= boxXZ * Math.ceil((dx - halfXZ) / boxXZ);
        else if (dx < -halfXZ) x += boxXZ * Math.ceil((-halfXZ - dx) / boxXZ);
        if (dz > halfXZ) z -= boxXZ * Math.ceil((dz - halfXZ) / boxXZ);
        else if (dz < -halfXZ) z += boxXZ * Math.ceil((-halfXZ - dz) / boxXZ);
        if (y < minY || y > minY + boxY) {
          y = minY + boxY - ((minY - y) % boxY + boxY) % boxY;
        }

        this.rx[i] = x;
        this.ry[i] = y;
        this.rz[i] = z;

        this.tmpPos.set(x, y, z);
        this.tmpScale.set(1, this.rLen[i] * (0.85 + this.intensity * 0.45), 1);
        this.tmpMat.compose(this.tmpPos, this.tmpQuat, this.tmpScale);
        this.dropMesh.setMatrixAt(i, this.tmpMat);
      }
      this.dropMesh.instanceMatrix.needsUpdate = true;
    } else {
      this.dropMesh.visible = false;
    }

    // 5. Tire water spray mist behind grounded wheels on wet tarmac
    const effectiveWet = onWetPatch ? Math.max(this.wetness, 0.85) : this.wetness;
    if (effectiveWet > 0.12 && carSpeedAbs > 5.5) {
      this.sprayEmitTimer -= dt;
      if (this.sprayEmitTimer <= 0) {
        this.sprayEmitTimer = Math.max(0.018, 0.055 - Math.min(0.035, carSpeedAbs * 0.0009));
        for (let w = 0; w < 4; w++) {
          if (!wheelGrounded[w]) continue;
          const idx = this.sprayCursor;
          this.sprayCursor = (this.sprayCursor + 1) % this.maxSpray;
          const w3 = w * 3;
          const jitter = ((idx * 17) % 11 - 5) * 0.032;
          this.spPos[idx * 3] = wheelWorldPos[w3] + jitter;
          this.spPos[idx * 3 + 1] = wheelWorldPos[w3 + 1] + 0.12;
          this.spPos[idx * 3 + 2] = wheelWorldPos[w3 + 2] + jitter;
          this.spVx[idx] = carVx * 0.35 + jitter * 8;
          this.spVy[idx] = 0.75 + Math.min(2.6, carSpeedAbs * 0.085);
          this.spVz[idx] = carVz * 0.35 - jitter * 8;
          this.spLife[idx] = 0;
          this.spMaxLife[idx] = 0.32 + Math.min(0.32, carSpeedAbs * 0.008);
          this.spSize[idx] = 0.42 + effectiveWet * 0.35;
          this.spAlpha[idx] = Math.min(0.32, effectiveWet * Math.min(1, (carSpeedAbs - 4) / 22) * 0.34);
          if (!this.spActive[idx]) this.activeSprayCount++;
          this.spActive[idx] = 1;
        }
        this.sprayPoints.visible = true;
      }
    }

    if (this.activeSprayCount > 0) {
      let alive = 0;
      for (let i = 0; i < this.maxSpray; i++) {
        if (!this.spActive[i]) continue;
        this.spLife[i] += dt;
        const t = this.spLife[i] / this.spMaxLife[i];
        if (t >= 1) {
          this.spActive[i] = 0;
          this.spAlpha[i] = 0;
          this.spSize[i] = 0;
          continue;
        }
        alive++;
        this.spVy[i] -= 3.2 * dt;
        this.spVx[i] *= 1 - 1.8 * dt;
        this.spVz[i] *= 1 - 1.8 * dt;
        this.spPos[i * 3] += this.spVx[i] * dt;
        this.spPos[i * 3 + 1] = Math.max(0.05, this.spPos[i * 3 + 1] + this.spVy[i] * dt);
        this.spPos[i * 3 + 2] += this.spVz[i] * dt;
        this.spSize[i] += dt * 1.45;
        this.spAlpha[i] *= 1 - dt * 2.4;
      }
      this.activeSprayCount = alive;
      (this.sprayGeo.getAttribute('position') as BufferAttribute).needsUpdate = true;
      (this.sprayGeo.getAttribute('aAlpha') as BufferAttribute).needsUpdate = true;
      (this.sprayGeo.getAttribute('aSize') as BufferAttribute).needsUpdate = true;
      if (alive === 0) this.sprayPoints.visible = false;
    }

    // Update atmospheric sky tint color (zero allocations)
    this.outSkyColor.copy(this.drySkyColor).lerp(this.wetSkyColor, this.intensity * 0.72);
    // Silence unused-variable lint while keeping carVy available for vertical pitch airflow
    void carVy;
    void this.tmpEuler;
    return this.wetness;
  }

  reset(): void {
    this.spActive.fill(0);
    this.spAlpha.fill(0);
    this.spSize.fill(0);
    this.activeSprayCount = 0;
    this.sprayPoints.visible = false;
    (this.sprayGeo.getAttribute('aAlpha') as BufferAttribute).needsUpdate = true;
  }

  dispose(): void {
    this.group.removeFromParent();
    this.dropMesh.geometry.dispose();
    this.dropMat.dispose();
    this.sprayGeo.dispose();
    this.sprayMat.dispose();
    this.sprayTex.dispose();
    this.trackedMaterials.length = 0;
  }
}

function createSoftMistTexture(): CanvasTexture {
  const sz = 64;
  const c = document.createElement('canvas');
  c.width = sz;
  c.height = sz;
  const ctx = c.getContext('2d');
  if (ctx) {
    const g = ctx.createRadialGradient(sz / 2, sz / 2, 2, sz / 2, sz / 2, sz / 2);
    g.addColorStop(0, 'rgba(255,255,255,0.9)');
    g.addColorStop(0.45, 'rgba(235,245,255,0.45)');
    g.addColorStop(1, 'rgba(220,235,250,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, sz, sz);
  }
  const tex = new CanvasTexture(c);
  tex.generateMipmaps = false;
  return tex;
}
