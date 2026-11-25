import React, { useState } from 'react';
import type {
  ReflectionQuality,
  ResolutionMode,
  ShadowQuality,
  SimulatorSettings,
} from '../types/settings';

type SettingsTab = 'general' | 'sounds' | 'graphics';

interface SettingsModalProps {
  settings: SimulatorSettings;
  onChange: (next: SimulatorSettings) => void;
  onResetDefaults: () => void;
  onClose: () => void;
}

export function SettingsModal({
  settings,
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
            <h2 className="bng-modal-title">Settings</h2>
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
                    <strong>Resolution / Pixel Ratio</strong>
                    <span>Auto caps 1.25× on tablets &amp; 1.5× on desktop for locked 60 FPS</span>
                  </div>
                  <div className="bng-seg-group">
                    {([
                      { id: 'auto', label: 'Auto 60 FPS' },
                      { id: 'performance', label: 'Performance' },
                      { id: 'quality', label: 'Quality' },
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

                <div className="bng-setting-row">
                  <div>
                    <strong>Render Distance</strong>
                    <span>Full-map camera view range across the 2 km × 2 km Proving Ground</span>
                  </div>
                  <div className="bng-seg-group">
                    {[2200, 3500, 5000].map((d) => (
                      <button
                        key={d}
                        type="button"
                        className={`btn ${settings.drawDistance === d ? 'active' : ''}`}
                        onClick={() => patch('drawDistance', d)}
                      >
                        {d}m
                      </button>
                    ))}
                  </div>
                </div>

                <div className="bng-setting-row">
                  <div>
                    <strong>Animated Skybox Clouds</strong>
                    <span>Lightweight procedural volumetric cloud shader</span>
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
