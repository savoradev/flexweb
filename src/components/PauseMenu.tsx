import React from 'react';

interface PauseMenuProps {
  showLattice: boolean;
  bodyOpacity: number;
  touchEnabled: boolean;
  onContinue: () => void;
  onOpenMap: () => void;
  onOpenCars: () => void;
  onOpenSettings: () => void;
  onMainMenu: () => void;
  onToggleLattice: () => void;
  onChangeOpacity: (v: number) => void;
  onToggleTouch: () => void;
}

export function PauseMenu({
  showLattice,
  bodyOpacity,
  touchEnabled,
  onContinue,
  onOpenMap,
  onOpenCars,
  onOpenSettings,
  onMainMenu,
  onToggleLattice,
  onChangeOpacity,
  onToggleTouch,
}: PauseMenuProps): React.ReactElement {
  return (
    <div className="bng-overlay-backdrop" role="dialog" aria-label="Menu Pause">
      <div className="bng-pause-card">
        <div className="bng-pause-header">
          <span className="bng-eyebrow">SIMULATION EN PAUSE</span>
          <h2>FLEXWEB</h2>
        </div>

        <div className="bng-pause-actions">
          <button id="pause-continue-btn" type="button" className="bng-cta-btn" onClick={onContinue}>
            Continue
          </button>
          <button
            id="pause-map-btn"
            type="button"
            className="bng-pause-btn"
            onClick={onOpenMap}
          >
            Map
          </button>
          <button id="pause-cars-btn" type="button" className="bng-pause-btn" onClick={onOpenCars}>
            Cars
          </button>
          <button id="pause-settings-btn" type="button" className="bng-pause-btn" onClick={onOpenSettings}>
            Settings
          </button>

          <div className="bng-pause-quick-tools">
            <button
              type="button"
              className={`btn btn-menu ${showLattice ? 'active' : ''}`}
              onClick={onToggleLattice}
            >
              X-Ray : {showLattice ? 'ON' : 'OFF'}
            </button>
            <button
              type="button"
              className={`btn btn-menu ${touchEnabled ? 'active' : ''}`}
              onClick={onToggleTouch}
            >
              Pédales : {touchEnabled ? 'ON' : 'OFF'}
            </button>
            <button
              type="button"
              className={`btn btn-menu ${bodyOpacity < 0.5 ? 'active' : ''}`}
              onClick={() => onChangeOpacity(bodyOpacity < 0.5 ? 1.0 : 0.35)}
            >
              {bodyOpacity < 0.5 ? 'Carrosserie 100%' : 'Ghost 35%'}
            </button>
          </div>

          <button id="pause-mainmenu-btn" type="button" className="bng-pause-btn danger" onClick={onMainMenu}>
            Main Menu
          </button>
        </div>
      </div>
    </div>
  );
}
