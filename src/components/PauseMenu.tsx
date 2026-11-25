import React from 'react';

interface PauseMenuProps {
  touchEnabled: boolean;
  onContinue: () => void;
  onOpenMap: () => void;
  onOpenCars: () => void;
  onOpenSettings: () => void;
  onMainMenu: () => void;
  onToggleTouch: () => void;
}

export function PauseMenu({
  touchEnabled,
  onContinue,
  onOpenMap,
  onOpenCars,
  onOpenSettings,
  onMainMenu,
  onToggleTouch,
}: PauseMenuProps): React.ReactElement {
  return (
    <div className="bng-overlay-backdrop" role="dialog" aria-label="Pause Menu">
      <div className="bng-pause-card">
        <div className="bng-pause-header">
          <span className="bng-eyebrow">SIMULATION PAUSED</span>
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
              className={`btn btn-menu ${touchEnabled ? 'active' : ''}`}
              onClick={onToggleTouch}
            >
              Touch Controls: {touchEnabled ? 'ON' : 'OFF'}
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
