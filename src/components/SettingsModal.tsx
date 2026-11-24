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
    <div className="bng-overlay-backdrop" role="dialog" aria-label="Paramètres FlexWeb">
      <div className="bng-modal-shell bng-settings-shell">
        <div className="bng-modal-header">
          <div>
            <span className="bng-eyebrow">CONFIGURATION</span>
            <h2 className="bng-modal-title">Settings</h2>
          </div>
          <div className="bng-header-actions">
            <button type="button" className="btn btn-menu" onClick={onResetDefaults}>
              Réinitialiser
            </button>
            <button type="button" className="btn btn-menu" onClick={onClose}>
              ← Fermer
            </button>
          </div>
        </div>

        <div className="bng-settings-body">
          {/* Left Category Sidebar */}
          <nav className="bng-settings-tabs" aria-label="Catégories de réglages">
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
                    <strong>Sensibilité de direction</strong>
                    <span>Vitesse de braquage au clavier et tactile</span>
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
                    <strong>ABS (Antiblocage de freins)</strong>
                    <span>Modulation 14 Hz et clignotement d&apos;urgence des feux stop</span>
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
                    <strong>TCS (Antipatinage)</strong>
                    <span>Limite le patinage des roues motrices en virage</span>
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
                    <strong>Unités de vitesse</strong>
                    <span>Affichage du tachymètre en jeu</span>
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
                    <strong>Afficher le HUD</strong>
                    <span>Affiche le compteur et les contrôles rapides (touche H)</span>
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
                    <strong>Diagnostic Véhicule style BeamNG</strong>
                    <span>Silhouette temps réel des dégâts à droite</span>
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
                    <strong>Raccourcis clavier</strong>
                    <span>Conduite WASD / Flèches · Frein à main Espace · Pause Échap</span>
                  </div>
                  <span className="bng-val-chip">Reset : R</span>
                </div>
              </div>
            )}

            {tab === 'sounds' && (
              <div className="bng-settings-group">
                <div className="bng-setting-row">
                  <div>
                    <strong>Couper le son (Mute)</strong>
                    <span>Désactive instantanément tout le moteur Web Audio</span>
                  </div>
                  <button
                    type="button"
                    className={`btn btn-menu ${settings.muted ? 'active' : ''}`}
                    onClick={() => patch('muted', !settings.muted)}
                  >
                    {settings.muted ? 'MUTED' : 'ACTIF'}
                  </button>
                </div>

                <div className="bng-setting-row">
                  <div>
                    <strong>Volume principal</strong>
                    <span>Niveau global du simulateur</span>
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
                    <strong>Moteur, Échappement &amp; Suralimentation</strong>
                    <span>BMW L6 S58 · McLaren V8 Flat-Plane · Dodge HEMI Supercharger</span>
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
                    <strong>Pneus &amp; Impacts de crash</strong>
                    <span>Crissements de dérive, freinages ABS et chocs métalliques</span>
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
                    <strong>Qualité des ombres</strong>
                    <span>Ombres dynamiques du soleil et des obstacles</span>
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
                    <strong>Reflets temps réel (CubeCamera)</strong>
                    <span>Reflets dynamiques du ciel et de l&apos;arène sur la carrosserie</span>
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
                    <strong>Résolution / Pixel Ratio</strong>
                    <span>Auto est optimisé 60 FPS pour tablettes 4 Go de RAM</span>
                  </div>
                  <div className="bng-seg-group">
                    {([
                      { id: 'auto', label: 'Auto 4GB' },
                      { id: 'performance', label: 'Performance' },
                      { id: 'quality', label: 'Qualité' },
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
                    <strong>Distance d&apos;affichage</strong>
                    <span>Portée de la caméra sur la GridMap 400 m</span>
                  </div>
                  <div className="bng-seg-group">
                    {[180, 300, 450].map((d) => (
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
                    <strong>Nuages Skybox animés</strong>
                    <span>Shader procédural léger de nuages dans le ciel</span>
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
