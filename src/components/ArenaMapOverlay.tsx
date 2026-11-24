import React, { useEffect, useState } from 'react';
import type { ArenaMapZone } from '../engine/CrashArena';
import type { EngineCore } from '../engine/EngineCore';

interface ProjectedPin {
  num: number;
  x: number;
  y: number;
  visible: boolean;
}

interface ArenaMapOverlayProps {
  engine: EngineCore | null;
  speedUnit: 'kmh' | 'mph';
  onTeleport: (zone: ArenaMapZone) => void;
  onLaunchCrash: (zone: ArenaMapZone, speedKmh: number) => void;
  onClose: () => void;
}

const SPEED_PRESETS_KMH = [40, 65, 100, 140, 190];

export function ArenaMapOverlay({
  engine,
  speedUnit,
  onTeleport,
  onLaunchCrash,
  onClose,
}: ArenaMapOverlayProps): React.ReactElement {
  const zones = engine?.arena.mapZones ?? [];
  const [selectedNum, setSelectedNum] = useState<number>(zones[0]?.num ?? 1);
  const selectedZone = zones.find((z) => z.num === selectedNum) ?? zones[0] ?? null;
  const [speedKmh, setSpeedKmh] = useState<number>(selectedZone?.defaultSpeedKmh ?? 65);
  const [pins, setPins] = useState<ProjectedPin[]>([]);
  const [carPin, setCarPin] = useState<{ x: number; y: number; visible: boolean }>({ x: 0, y: 0, visible: false });

  useEffect(() => {
    if (selectedZone) {
      setSpeedKmh(selectedZone.defaultSpeedKmh);
    }
  }, [selectedZone?.num]);

  useEffect(() => {
    if (!engine) return;
    let rafId = 0;
    const tick = () => {
      const zs = engine.arena.mapZones;
      const nextPins: ProjectedPin[] = new Array(zs.length);
      for (let i = 0; i < zs.length; i++) {
        const z = zs[i];
        const p = engine.projectWorldToScreen(z.markerX, 1.8, z.markerZ);
        nextPins[i] = { num: z.num, x: p.x, y: p.y, visible: p.visible };
      }
      setPins(nextPins);
      const F = engine.vehicle.getFrame();
      setCarPin(engine.projectWorldToScreen(F[0], F[1] + 1.2, F[2]));
      rafId = window.requestAnimationFrame(tick);
    };
    rafId = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(rafId);
  }, [engine]);

  const fmtSpeed = (kmh: number) =>
    speedUnit === 'mph' ? `${Math.round(kmh * 0.621371)} mph` : `${Math.round(kmh)} km/h`;

  return (
    <div className="bng-mapview-layer" role="dialog" aria-label="Carte interactive des tests">
      {/* Top Bar */}
      <div className="bng-mapview-topbar">
        <div className="bng-mapview-title">
          <span className="bng-eyebrow">VUE AÉRIENNE · CRASH TEST ARENA</span>
          <strong>Cliquez sur un numéro pour vous téléporter ou lancer un crash</strong>
        </div>
        <button id="close-mapview-btn" type="button" className="btn btn-menu" onClick={onClose}>
          ← Retour (Échap)
        </button>
      </div>

      {/* Live 3D-Projected Numbered Pins on every Arena Test */}
      <div className="bng-mapview-pins">
        {carPin.visible && (
          <div
            className="bng-map-car-dot"
            style={{ left: `${carPin.x}px`, top: `${carPin.y}px` }}
            title="Position actuelle du véhicule"
          >
            VOUS
          </div>
        )}
        {pins.map((p) => {
          if (!p.visible) return null;
          const z = zones.find((item) => item.num === p.num);
          if (!z) return null;
          const isSel = z.num === selectedNum;
          return (
            <button
              key={z.num}
              id={`map-pin-${z.num}`}
              type="button"
              className={`bng-map-pin ${z.category === 'suspension' ? 'susp' : 'crash'} ${isSel ? 'selected' : ''}`}
              style={{ left: `${p.x}px`, top: `${p.y}px` }}
              title={`${z.num}. ${z.label}`}
              onClick={() => setSelectedNum(z.num)}
            >
              <span>{z.num}</span>
            </button>
          );
        })}
      </div>

      {/* Bottom Interactive Teleport / Crash Speed Control Dock */}
      {selectedZone && (
        <div className="bng-mapview-dock">
          <div className="bng-mapview-nums">
            {zones.map((z) => (
              <button
                key={z.num}
                type="button"
                className={`bng-num-chip ${z.num === selectedNum ? 'active' : ''}`}
                onClick={() => setSelectedNum(z.num)}
                title={z.label}
              >
                {z.num}
              </button>
            ))}
          </div>

          <div className="bng-mapview-card">
            <div className="bng-mapview-info">
              <span className="bng-zone-badge">TEST #{selectedZone.num}</span>
              <h3>{selectedZone.label}</h3>
            </div>

            <div className="bng-mapview-speed">
              <div className="bng-speed-head">
                <span>Vitesse d&apos;impact / test :</span>
                <strong>{fmtSpeed(speedKmh)}</strong>
              </div>
              <input
                type="range"
                min={20}
                max={220}
                step={5}
                value={speedKmh}
                onChange={(e) => setSpeedKmh(Number(e.target.value))}
              />
              <div className="bng-speed-pills">
                {SPEED_PRESETS_KMH.map((s) => (
                  <button
                    key={s}
                    type="button"
                    className={`btn ${speedKmh === s ? 'active' : ''}`}
                    onClick={() => setSpeedKmh(s)}
                  >
                    {fmtSpeed(s)}
                  </button>
                ))}
              </div>
            </div>

            <div className="bng-mapview-actions">
              <button
                id="map-teleport-btn"
                type="button"
                className="bng-pause-btn"
                onClick={() => onTeleport(selectedZone)}
              >
                Se téléporter ici (0 {speedUnit === 'mph' ? 'mph' : 'km/h'})
              </button>
              <button
                id="map-launch-crash-btn"
                type="button"
                className="bng-cta-btn"
                onClick={() => onLaunchCrash(selectedZone, speedKmh)}
              >
                Crash / Lancer à {fmtSpeed(speedKmh)} →
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
