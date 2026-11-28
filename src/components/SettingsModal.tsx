import React, { useState } from 'react';
import type {
  ReflectionQuality,
  ResolutionMode,
  ShadowQuality,
  SimulatorSettings,
} from '../types/settings';

type SettingsTab = 'general' | 'controls' | 'sounds' | 'graphics' | 'weather';

interface SettingsModalProps {
  settings: SimulatorSettings;
  deviceLabel?: string;
  onChange: (next: SimulatorSettings) => void;
  onResetDefaults: () => void;
  onClose: () => void;
}

export function SettingsModal({
  settings,
  deviceLabel,
  onChange,
  onResetDefaults,
  onClose,
}: SettingsModalProps): React.ReactElement {
  const [tab, setTab] = useState<SettingsTab>('general');

  const patch = <K extends keyof SimulatorSettings>(key: K, val: SimulatorSettings[K]) => {
    onChange({ ...settings, [key]: val });
  };

  return (
    <div className="bng-overlay-backdrop" role="dialog" aria-label="FlexWeb Settings">
      <div className="bng-modal-shell bng-settings-shell">
        <div className="bng-modal-header">
          <div>
            <span className="bng-eyebrow">CONFIGURATION</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
              <h2 className="bng-modal-title">Settings</h2>
              {deviceLabel && <span className="device-badge">{deviceLabel}</span>}
            </div>
          </div>
          <div className="bng-header-actions">
            <button type="button" className="btn btn-menu" onClick={onResetDefaults}>
              Reset Defaults
            </button>
            <button type="button" className="btn btn-menu" onClick={onClose}>
              ← Close
            </button>
          </div>
        </div>

        <div className="bng-settings-body">
          {/* Left Category Sidebar */}
          <nav className="bng-settings-tabs" aria-label="Settings Categories">
            <button
              type="button"
              className={`bng-tab-btn ${tab === 'general' ? 'active' : ''}`}
              onClick={() => setTab('general')}
            >
              General
            </button>
            <button
              type="button"
              className={`bng-tab-btn ${tab === 'controls' ? 'active' : ''}`}
              onClick={() => setTab('controls')}
            >
              Controls
            </button>
            <button
              type="button"
              className={`bng-tab-btn ${tab === 'sounds' ? 'active' : ''}`}
              onClick={() => setTab('sounds')}
            >
              Sounds
            </button>
            <button
              type="button"
              className={`bng-tab-btn ${tab === 'graphics' ? 'active' : ''}`}
              onClick={() => setTab('graphics')}
            >
              Graphics
            </button>
            <button
              type="button"
              className={`bng-tab-btn ${tab === 'weather' ? 'active' : ''}`}
              onClick={() => setTab('weather')}
            >
              Météo
            </button>
          </nav>

          {/* Right Settings Content */}
          <div className="bng-settings-content">
            {tab === 'general' && (
              <div className="bng-settings-group">
                <div className="bng-setting-row">
                  <div>
                    <strong>Steering Sensitivity</strong>
                    <span>Steering response speed for keyboard and touch controls</span>
                  </div>
                  <div className="bng-slider-wrap">
                    <input
                      type="range"
                      min={0.5}
                      max={1.5}
                      step={0.05}
                      value={settings.steeringSensitivity}
                      onChange={(e) => patch('steeringSensitivity', Number(e.target.value))}
                    />
                    <span className="bng-val-chip">{Math.round(settings.steeringSensitivity * 100)}%</span>
                  </div>
                </div>

                <div className="bng-setting-row">
                  <div>
                    <strong>ABS (Anti-Lock Braking)</strong>
                    <span>14 Hz brake modulation with emergency brake light flash</span>
                  </div>
                  <button
                    type="button"
                    className={`btn btn-menu ${settings.absEnabled ? 'active' : ''}`}
                    onClick={() => patch('absEnabled', !settings.absEnabled)}
                  >
                    {settings.absEnabled ? 'ON' : 'OFF'}
                  </button>
                </div>

                <div className="bng-setting-row">
                  <div>
                    <strong>TCS (Traction Control)</strong>
                    <span>Limits driven wheelspin during hard acceleration and cornering</span>
                  </div>
                  <button
                    type="button"
                    className={`btn btn-menu ${settings.tcsEnabled ? 'active' : ''}`}
                    onClick={() => patch('tcsEnabled', !settings.tcsEnabled)}
                  >
                    {settings.tcsEnabled ? 'ON' : 'OFF'}
                  </button>
                </div>

                <div className="bng-setting-row">
                  <div>
                    <strong>Transmission</strong>
                    <span>Auto, Semi-Auto (E/Q paddle shift), or Full Manual (R / N / 1–7)</span>
                  </div>
                  <div className="bng-seg-group">
                    {(['auto', 'semi', 'manual'] as const).map((mode) => (
                      <button
                        key={mode}
                        type="button"
                        className={`btn ${settings.transmissionMode === mode ? 'active' : ''}`}
                        onClick={() => patch('transmissionMode', mode)}
                      >
                        {mode === 'auto' ? 'Auto' : mode === 'semi' ? 'Semi' : 'Manual'}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="bng-setting-row">
                  <div>
                    <strong>Speed Units</strong>
                    <span>In-game speedometer and telemetry display unit</span>
                  </div>
                  <div className="bng-seg-group">
                    <button
                      type="button"
                      className={`btn ${settings.speedUnit === 'kmh' ? 'active' : ''}`}
                      onClick={() => patch('speedUnit', 'kmh')}
                    >
                      km/h
                    </button>
                    <button
                      type="button"
                      className={`btn ${settings.speedUnit === 'mph' ? 'active' : ''}`}
                      onClick={() => patch('speedUnit', 'mph')}
                    >
                      mph
                    </button>
                  </div>
                </div>

                <div className="bng-setting-row">
                  <div>
                    <strong>Show HUD</strong>
                    <span>Display tachometer and quick action bar (H key)</span>
                  </div>
                  <button
                    type="button"
                    className={`btn btn-menu ${settings.showHud ? 'active' : ''}`}
                    onClick={() => patch('showHud', !settings.showHud)}
                  >
                    {settings.showHud ? 'ON' : 'OFF'}
                  </button>
                </div>

                <div className="bng-setting-row">
                  <div>
                    <strong>Vehicle Diagnostics Overlay</strong>
                    <span>Real-time damage schematic panel on the right</span>
                  </div>
                  <button
                    type="button"
                    className={`btn btn-menu ${settings.showDiagnostics ? 'active' : ''}`}
                    onClick={() => patch('showDiagnostics', !settings.showDiagnostics)}
                  >
                    {settings.showDiagnostics ? 'ON' : 'OFF'}
                  </button>
                </div>

                <div className="bng-setting-row">
                  <div>
                    <strong>Keyboard Controls</strong>
                    <span>Drive WASD / Arrows · Handbrake Space · Map M · Pause Esc</span>
                  </div>
                  <span className="bng-val-chip">Reset: R</span>
                </div>
              </div>
            )}

            {tab === 'controls' && (
              <div className="bng-settings-section">
                <h3>Driving &amp; Touch Controls</h3>

                <div className="bng-setting-row">
                  <div>
                    <strong>Touch Steering Style</strong>
                    <span>Choose between interactive rotary Steering Wheel or Left/Right Arrows</span>
                  </div>
                  <div className="bng-seg-group">
                    <button
                      type="button"
                      className={`btn ${settings.touchSteering === 'arrows' ? 'active' : ''}`}
                      onClick={() => patch('touchSteering', 'arrows')}
                    >
                      Arrows (◀ ▶)
                    </button>
                    <button
                      type="button"
                      className={`btn ${settings.touchSteering === 'wheel' ? 'active' : ''}`}
                      onClick={() => patch('touchSteering', 'wheel')}
                    >
                      Steering Wheel
                    </button>
                  </div>
                </div>

                <div className="bng-setting-row">
                  <div>
                    <strong>Analog Gas &amp; Brake Pedals</strong>
                    <span>Slide vertically on pedals for smooth 0%–100% throttle &amp; braking</span>
                  </div>
                  <button
                    type="button"
                    className={`btn btn-menu ${settings.analogPedals ? 'active' : ''}`}
                    onClick={() => patch('analogPedals', !settings.analogPedals)}
                  >
                    {settings.analogPedals ? 'ANALOG ON' : 'DIGITAL'}
                  </button>
                </div>

                <div className="bng-setting-row">
                  <div>
                    <strong>Transmission Mode</strong>
                    <span>Auto, Semi-Auto (E/Q or buttons above pedals), or Full Manual (R / N / 1–7)</span>
                  </div>
                  <div className="bng-seg-group">
                    {(['auto', 'semi', 'manual'] as const).map((mode) => (
                      <button
                        key={mode}
                        type="button"
                        className={`btn ${settings.transmissionMode === mode ? 'active' : ''}`}
                        onClick={() => patch('transmissionMode', mode)}
                      >
                        {mode === 'auto' ? 'Auto' : mode === 'semi' ? 'Semi' : 'Manual'}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="bng-setting-row">
                  <div>
                    <strong>Steering Sensitivity</strong>
                    <span>Adjusts front wheel steering speed and responsiveness</span>
                  </div>
                  <div className="bng-range-wrap">
                    <input
                      type="range"
                      min={0.5}
                      max={1.5}
                      step={0.05}
                      value={settings.steeringSensitivity}
                      onChange={(e) => patch('steeringSensitivity', Number(e.target.value))}
                    />
                    <span className="bng-val-chip">{Math.round(settings.steeringSensitivity * 100)}%</span>
                  </div>
                </div>
              </div>
            )}

            {tab === 'sounds' && (
              <div className="bng-settings-group">
                <div className="bng-setting-row">
                  <div>
                    <strong>Mute Audio</strong>
                    <span>Instantly silence the entire Web Audio sound engine</span>
                  </div>
                  <button
                    type="button"
                    className={`btn btn-menu ${settings.muted ? 'active' : ''}`}
                    onClick={() => patch('muted', !settings.muted)}
                  >
                    {settings.muted ? 'MUTED' : 'ACTIVE'}
                  </button>
                </div>

                <div className="bng-setting-row">
                  <div>
                    <strong>Master Volume</strong>
                    <span>Overall simulator audio level</span>
                  </div>
                  <div className="bng-slider-wrap">
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.05}
                      value={settings.masterVolume}
                      onChange={(e) => patch('masterVolume', Number(e.target.value))}
                    />
                    <span className="bng-val-chip">{Math.round(settings.masterVolume * 100)}%</span>
                  </div>
                </div>

                <div className="bng-setting-row">
                  <div>
                    <strong>Engine, Exhaust &amp; Supercharger</strong>
                    <span>BMW I6 S58 · McLaren Flat-Plane V8 · Dodge HEMI Supercharger</span>
                  </div>
                  <div className="bng-slider-wrap">
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.05}
                      value={settings.engineVolume}
                      onChange={(e) => patch('engineVolume', Number(e.target.value))}
                    />
                    <span className="bng-val-chip">{Math.round(settings.engineVolume * 100)}%</span>
                  </div>
                </div>

                <div className="bng-setting-row">
                  <div>
                    <strong>Tires &amp; Crash Impacts</strong>
                    <span>Tire screech, ABS braking, and metallic structural crashes</span>
                  </div>
                  <div className="bng-slider-wrap">
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.05}
                      value={settings.effectsVolume}
                      onChange={(e) => patch('effectsVolume', Number(e.target.value))}
                    />
                    <span className="bng-val-chip">{Math.round(settings.effectsVolume * 100)}%</span>
                  </div>
                </div>
              </div>
            )}

            {tab === 'graphics' && (
              <div className="bng-settings-group">
                <div className="bng-setting-row">
                  <div>
                    <strong>Shadow Quality</strong>
                    <span>Dynamic sun shadows on vehicles and obstacles</span>
                  </div>
                  <div className="bng-seg-group">
                    {(['off', 'low', 'high'] as ShadowQuality[]).map((q) => (
                      <button
                        key={q}
                        type="button"
                        className={`btn ${settings.shadowQuality === q ? 'active' : ''}`}
                        onClick={() => patch('shadowQuality', q)}
                      >
                        {q === 'off' ? 'Off' : q === 'low' ? 'Low' : 'High'}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="bng-setting-row">
                  <div>
                    <strong>Real-Time Reflections (CubeCamera)</strong>
                    <span>Dynamic sky and arena reflections on car paint &amp; glass</span>
                  </div>
                  <div className="bng-seg-group">
                    {(['off', 'low', 'high'] as ReflectionQuality[]).map((q) => (
                      <button
                        key={q}
                        type="button"
                        className={`btn ${settings.reflectionQuality === q ? 'active' : ''}`}
                        onClick={() => patch('reflectionQuality', q)}
                      >
                        {q === 'off' ? 'Off' : q === 'low' ? 'Low' : 'High'}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="bng-setting-row">
                  <div>
                    <strong>Pixel Quality</strong>
                    <span>Low (0.85×), Medium (balanced), High (1.75×), or Ultra (max native pixels)</span>
                  </div>
                  <div className="bng-seg-group">
                    {([
                      { id: 'low', label: 'Low' },
                      { id: 'medium', label: 'Medium' },
                      { id: 'high', label: 'High' },
                      { id: 'ultra', label: 'Ultra' },
                    ] as { id: ResolutionMode; label: string }[]).map((r) => (
                      <button
                        key={r.id}
                        type="button"
                        className={`btn ${settings.resolutionMode === r.id ? 'active' : ''}`}
                        onClick={() => patch('resolutionMode', r.id)}
                      >
                        {r.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {tab === 'weather' && (
              <div className="bng-settings-group">
                <div className="bng-setting-row">
                  <div>
                    <strong>Jour / Nuit</strong>
                    <span>Transition douce du ciel entre le jour, le crépuscule et la nuit étoilée</span>
                  </div>
                  <div className="bng-seg-group">
                    {([
                      { id: 'day', label: 'Jour' },
                      { id: 'night', label: 'Nuit' },
                    ] as const).map((t) => (
                      <button
                        key={t.id}
                        type="button"
                        className={`btn ${(settings.timeOfDay ?? 'day') === t.id ? 'active' : ''}`}
                        onClick={() => patch('timeOfDay', t.id)}
                      >
                        {t.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="bng-setting-row">
                  <div>
                    <strong>Pluie HD</strong>
                    <span>Gouttes sur carrosserie et vitres, asphalte mouillé et éclaboussures</span>
                  </div>
                  <div className="bng-seg-group">
                    {([
                      { id: 'off', label: 'Off' },
                      { id: 'drizzle', label: 'Bruine' },
                      { id: 'heavy', label: 'Forte pluie' },
                    ] as const).map((w) => (
                      <button
                        key={w.id}
                        type="button"
                        className={`btn ${(settings.weatherMode ?? 'off') === w.id ? 'active' : ''}`}
                        onClick={() => patch('weatherMode', w.id)}
                      >
                        {w.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="bng-setting-row">
                  <div>
                    <strong>Nuages animés</strong>
                    <span>Nuages volumétriques légers dans le ciel</span>
                  </div>
                  <button
                    type="button"
                    className={`btn btn-menu ${settings.skyboxClouds ? 'active' : ''}`}
                    onClick={() => patch('skyboxClouds', !settings.skyboxClouds)}
                  >
                    {settings.skyboxClouds ? 'ON' : 'OFF'}
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
