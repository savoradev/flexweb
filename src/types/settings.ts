export type ShadowQuality = 'off' | 'low' | 'high';
export type ReflectionQuality = 'off' | 'low' | 'high';
export type ResolutionMode = 'low' | 'medium' | 'high' | 'ultra';
export type WeatherMode = 'off' | 'drizzle' | 'heavy';
export type TimeOfDay = 'day' | 'night';
export type SpeedUnit = 'kmh' | 'mph';
export type TransmissionModeSetting = 'auto' | 'semi' | 'manual';
export type TouchSteeringMode = 'arrows' | 'wheel';

export interface SimulatorSettings {
  // General
  steeringSensitivity: number; // 0.5 .. 1.5
  absEnabled: boolean;
  tcsEnabled: boolean;
  transmissionMode: TransmissionModeSetting;
  touchSteering: TouchSteeringMode;
  analogPedals: boolean;
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
  drawDistance: number;   // 2200 | 3500 | 5000
  skyboxClouds: boolean;
  weatherMode: WeatherMode;
  timeOfDay: TimeOfDay;
}

export const DEFAULT_SETTINGS: SimulatorSettings = {
  steeringSensitivity: 1.0,
  absEnabled: true,
  tcsEnabled: true,
  transmissionMode: 'auto',
  touchSteering: 'arrows',
  analogPedals: true,
  speedUnit: 'kmh',
  showHud: true,
  showDiagnostics: true,
  masterVolume: 0.75,
  engineVolume: 0.85,
  effectsVolume: 0.85,
  muted: false,
  shadowQuality: 'low',
  reflectionQuality: 'low',
  resolutionMode: 'medium',
  drawDistance: 3500,
  skyboxClouds: true,
  weatherMode: 'off',
  timeOfDay: 'day',
};

const STORAGE_KEY = 'flexweb_sim_settings_v1';

import { detectDeviceProfile } from '../utils/DeviceDetector';

export function loadSimulatorSettings(): SimulatorSettings {
  const { deviceType } = detectDeviceProfile();
  const deviceDefault: SimulatorSettings =
    deviceType === 'phone'
      ? { ...DEFAULT_SETTINGS, resolutionMode: 'low', shadowQuality: 'low', reflectionQuality: 'low' }
      : { ...DEFAULT_SETTINGS };
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return deviceDefault;
    const parsed = JSON.parse(raw) as Partial<SimulatorSettings> & { resolutionMode?: string };
    const merged = { ...deviceDefault, ...parsed } as SimulatorSettings;
    if (merged.drawDistance < 2200) merged.drawDistance = 3500;
    const rm = String(parsed.resolutionMode ?? '');
    if (rm === 'performance') merged.resolutionMode = 'low';
    else if (rm === 'auto') merged.resolutionMode = 'medium';
    else if (rm === 'quality') merged.resolutionMode = 'high';
    return merged;
  } catch {
    return deviceDefault;
  }
}

export function saveSimulatorSettings(settings: SimulatorSettings): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // ignore private mode / quota
  }
}
