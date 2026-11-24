export type ShadowQuality = 'off' | 'low' | 'high';
export type ReflectionQuality = 'off' | 'low' | 'high';
export type ResolutionMode = 'auto' | 'performance' | 'quality';
export type SpeedUnit = 'kmh' | 'mph';

export interface SimulatorSettings {
  // General
  steeringSensitivity: number; // 0.5 .. 1.5
  absEnabled: boolean;
  tcsEnabled: boolean;
  speedUnit: SpeedUnit;
  showHud: boolean;
  showDiagnostics: boolean;
  // Sounds
  masterVolume: number;   // 0 .. 1
  engineVolume: number;   // 0 .. 1
  effectsVolume: number;  // 0 .. 1 (tires & crashes)
  muted: boolean;
  // Graphics
  shadowQuality: ShadowQuality;
  reflectionQuality: ReflectionQuality;
  resolutionMode: ResolutionMode;
  drawDistance: number;   // 180 | 300 | 450
  skyboxClouds: boolean;
}

export const DEFAULT_SETTINGS: SimulatorSettings = {
  steeringSensitivity: 1.0,
  absEnabled: true,
  tcsEnabled: true,
  speedUnit: 'kmh',
  showHud: true,
  showDiagnostics: true,
  masterVolume: 0.75,
  engineVolume: 0.85,
  effectsVolume: 0.85,
  muted: false,
  shadowQuality: 'low',
  reflectionQuality: 'low',
  resolutionMode: 'auto',
  drawDistance: 300,
  skyboxClouds: true,
};

const STORAGE_KEY = 'flexweb_sim_settings_v1';

export function loadSimulatorSettings(): SimulatorSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_SETTINGS };
    const parsed = JSON.parse(raw) as Partial<SimulatorSettings>;
    return { ...DEFAULT_SETTINGS, ...parsed };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSimulatorSettings(settings: SimulatorSettings): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // ignore private mode / quota
  }
}
