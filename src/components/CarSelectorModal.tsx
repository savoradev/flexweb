import React, { useState } from 'react';
import type { CarOption } from '../engine/EngineCore';

const PAINT_SWATCHES = [
  { name: 'Papaya Orange', hex: 0xe06a10, finish: 'Metallic' },
  { name: 'Iridium Silver', hex: 0x8e969d, finish: 'Metallic' },
  { name: 'Obsidian Black', hex: 0x181b1f, finish: 'Gloss' },
  { name: 'Frozen Brooklyn Grey', hex: 0x6b7280, finish: 'Matte' },
  { name: 'Designo Crimson', hex: 0x991b24, finish: 'Metallic' },
  { name: 'Hellcat Plum', hex: 0x4c1d95, finish: 'Metallic' },
  { name: 'Solarbeam Gold', hex: 0xd49b27, finish: 'Metallic' },
  { name: 'Emerald Green', hex: 0x174f38, finish: 'Metallic' },
  { name: 'Alpine White', hex: 0xe3e6eb, finish: 'Gloss' },
];

interface CarSpecInfo {
  engine: string;
  power: string;
  layout: string;
  redline: string;
  accent: string;
  badge: string;
}

function getCarSpecs(opt: CarOption): CarSpecInfo {
  const n = opt.name.toLowerCase();
  if (n.includes('bmw') || n.includes('m4')) {
    return {
      engine: '3.0L Twin-Turbo I6 (S58)',
      power: '543 hp · 650 Nm',
      layout: 'Front-Engine · RWD',
      redline: '7,200 RPM',
      accent: '#3b82f6',
      badge: 'TWIN-TURBO I6',
    };
  }
  if (n.includes('mclaren') || n.includes('p1')) {
    return {
      engine: '3.8L Twin-Turbo V8 + IPAS Hybrid',
      power: '903 hp · 900 Nm',
      layout: 'Mid-Engine · Active Aero Wing',
      redline: '8,500 RPM',
      accent: '#f97316',
      badge: 'HYBRID V8',
    };
  }
  if (n.includes('dodge') || n.includes('demon') || n.includes('challenger')) {
    return {
      engine: '6.2L Supercharged HEMI V8',
      power: '840 hp · 1,044 Nm',
      layout: 'Front-Engine · Supercharged RWD',
      redline: '6,500 RPM',
      accent: '#ef4444',
      badge: 'HEMI V8 SC',
    };
  }
  return {
    engine: '4.0L Twin-Turbo V8',
    power: '510 hp · 650 Nm',
    layout: 'Front-Engine · RWD',
    redline: '7,200 RPM',
    accent: '#94a3b8',
    badge: 'V8 COUPE',
  };
}

interface CarSelectorModalProps {
  cars: CarOption[];
  activeCarId: string;
  carThumbs: Record<string, string>;
  onSelectCar: (opt: CarOption) => Promise<void>;
  onPickPaint: (car: CarOption, hex: number) => void;
  onDeleteSavedCar: (opt: CarOption) => void;
  onConfirmPlay: (car: CarOption) => void;
  onClose: () => void;
  confirmLabel?: string;
}

export function CarSelectorModal({
  cars,
  activeCarId,
  carThumbs,
  onSelectCar,
  onPickPaint,
  onDeleteSavedCar,
  onConfirmPlay,
  onClose,
  confirmLabel = 'Select & Drive This Vehicle',
}: CarSelectorModalProps): React.ReactElement {
  const [focusedId, setFocusedId] = useState(activeCarId);
  const [pickedHexByCar, setPickedHexByCar] = useState<Record<string, number>>({});
  const selected = cars.find((c) => c.id === focusedId) ?? cars.find((c) => c.id === activeCarId) ?? cars[0];
  const specs = selected ? getCarSpecs(selected) : null;
  const isLoaded = selected?.id === activeCarId;

  const handleCardClick = (opt: CarOption) => {
    setFocusedId(opt.id);
    if (!carThumbs[opt.id] && opt.id !== activeCarId) {
      void onSelectCar(opt);
    }
  };

  return (
    <div className="bng-overlay-backdrop" role="dialog" aria-label="Vehicle Selector">
      <div className="bng-modal-shell bng-cars-shell">
        <div className="bng-modal-header">
          <div>
            <span className="bng-eyebrow">FLEXWEB SHOWROOM</span>
            <h2 className="bng-modal-title">Vehicle Selector</h2>
          </div>
          <div className="bng-header-actions">
            <button type="button" className="btn btn-menu" onClick={onClose}>
              ← Close
            </button>
          </div>
        </div>

        <div className="bng-cars-split">
          {/* Left: Vehicle Cards Grid */}
          <div className="bng-car-grid">
            {cars.map((car) => {
              const sp = getCarSpecs(car);
              const thumb = carThumbs[car.id];
              const isSelected = car.id === selected?.id;
              const cleanTitle = car.name.replace(/\s*\((public\/|Saved)\)$/i, '');
              return (
                <div
                  key={car.id}
                  role="button"
                  tabIndex={0}
                  className={`bng-car-card ${isSelected ? 'selected' : ''}`}
                  onClick={() => handleCardClick(car)}
                  onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') handleCardClick(car); }}
                >
                  <div className="bng-car-card-thumb" style={{ borderBottomColor: sp.accent }}>
                    {thumb ? (
                      <img src={thumb} alt={cleanTitle} />
                    ) : (
                      <div className="bng-car-silhouette">
                        <span className="bng-car-badge" style={{ borderColor: sp.accent, color: sp.accent }}>
                          {sp.badge}
                        </span>
                        <span className="bng-car-code">{cleanTitle}</span>
                      </div>
                    )}
                    {car.id === activeCarId && <span className="bng-active-pill">ACTIVE</span>}
                    {car.savedInDB && (
                      <button
                        type="button"
                        className="bng-car-del"
                        title={`Delete ${cleanTitle}`}
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteSavedCar(car);
                        }}
                      >
                        ×
                      </button>
                    )}
                  </div>
                  <div className="bng-car-card-info">
                    <strong>{cleanTitle}</strong>
                    <span>{sp.engine}</span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Right: Detailed Vehicle Sheet + Live Paint Picker + Embedded Attribution Credits */}
          {selected && specs && (
            <aside className="bng-car-detail">
              <div className="bng-detail-preview">
                {carThumbs[selected.id] ? (
                  <img src={carThumbs[selected.id]} alt={selected.name} />
                ) : (
                  <div className="bng-detail-placeholder">
                    <span className="bng-car-badge" style={{ borderColor: specs.accent, color: specs.accent }}>
                      {specs.badge}
                    </span>
                    <strong>{selected.name.replace(/\s*\((public\/|Saved)\)$/i, '')}</strong>
                    <span>{isLoaded ? 'White Studio Showroom' : 'Studio Showroom Preview'}</span>
                  </div>
                )}
              </div>

              {/* Mandatory Creative Commons Attribution neatly placed under the vehicle preview */}
              {selected.attribution && (
                <div className="bng-car-credit" role="contentinfo">
                  <span className="attr-badge">3D CREDIT</span>
                  <span>
                    &ldquo;{selected.attribution.title}&rdquo; (
                    <a href={selected.attribution.modelUrl} target="_blank" rel="noopener noreferrer">
                      {selected.attribution.modelUrl}
                    </a>
                    ) by {selected.attribution.author} is licensed under{' '}
                    <a href={selected.attribution.licenseUrl} target="_blank" rel="noopener noreferrer">
                      {selected.attribution.licenseName}
                    </a>{' '}
                    (
                    <a href={selected.attribution.licenseUrl} target="_blank" rel="noopener noreferrer">
                      {selected.attribution.licenseUrl}
                    </a>
                    ).
                  </span>
                </div>
              )}

              <div className="bng-detail-specs">
                <div><span>Engine</span><strong>{specs.engine}</strong></div>
                <div><span>Output</span><strong>{specs.power}</strong></div>
                <div><span>Layout</span><strong>{specs.layout}</strong></div>
                <div><span>Redline</span><strong>{specs.redline}</strong></div>
              </div>

              <div className="bng-detail-paint">
                <span className="popover-label">Body Paint (Real-Time)</span>
                <div className="bng-swatch-grid">
                  {PAINT_SWATCHES.map((sw) => {
                    const isPicked = pickedHexByCar[selected.id] === sw.hex;
                    return (
                      <button
                        key={sw.name}
                        type="button"
                        className={`bng-swatch-btn ${isPicked ? 'active' : ''}`}
                        style={isPicked ? { borderColor: '#f59e0b', boxShadow: '0 0 0 1px rgba(245,158,11,0.5)' } : undefined}
                        title={`${sw.name} (${sw.finish})`}
                        onClick={() => {
                          setPickedHexByCar((prev) => ({ ...prev, [selected.id]: sw.hex }));
                          onPickPaint(selected, sw.hex);
                        }}
                      >
                        <span
                          className="bng-swatch-dot"
                          style={{ background: `#${sw.hex.toString(16).padStart(6, '0')}` }}
                        />
                        <span className="bng-swatch-name">{sw.name}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              <button
                id="confirm-select-car-btn"
                type="button"
                className="bng-cta-btn"
                onClick={async () => {
                  await onSelectCar(selected);
                  onConfirmPlay(selected);
                }}
              >
                {confirmLabel} →
              </button>
            </aside>
          )}
        </div>
      </div>
    </div>
  );
}
