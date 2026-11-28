import React, { useEffect, useRef, useState } from 'react';
import type { VehicleTelemetry } from '../engine/SoftBodyVehicle';

interface VehicleDiagnosticsProps {
  telemetry: VehicleTelemetry | undefined;
  collapsed: boolean;
  onToggle: () => void;
}

interface DamageAlert {
  id: number;
  text: string;
  severity: 'warn' | 'crit';
  expiresAt: number;
}

/** Maps a 0..1 damage value to clean motorsport green -> amber -> red hex color. */
function dmgColor(dmg: number): string {
  if (dmg < 0.14) return '#22c55e';
  if (dmg < 0.45) return '#f59e0b';
  return '#ef4444';
}

function partColor(state: string | undefined, fallbackDmg: number): string {
  if (state === 'detached') return '#ef4444';
  if (state === 'unlatched') return '#f59e0b';
  return dmgColor(fallbackDmg);
}

const TIRE_NAMES = ['Front Left Tire', 'Front Right Tire', 'Rear Left Tire', 'Rear Right Tire'] as const;

export const VehicleDiagnostics: React.FC<VehicleDiagnosticsProps> = ({ telemetry, collapsed, onToggle }) => {
    const [alerts, setAlerts] = useState<DamageAlert[]>([]);
    const prevFlagsRef = useRef<Record<string, boolean>>({});
    const alertIdRef = useRef(1);

    // Run on every 4 Hz stats update (telemetry is mutated in-place for zero-GC, so never gate on object identity!)
    useEffect(() => {
      if (!telemetry) return;
      const d = telemetry.damage;
      const m = telemetry.mech;
      const parts = telemetry.parts ?? [];
      const findState = (label: string) => parts.find((p) => p.label === label)?.state ?? 'attached';

      const currentFlags: Record<string, { active: boolean; text: string; severity: 'warn' | 'crit' }> = {
        hood_dmg: { active: d.hood > 0.16 && findState('Hood') === 'attached', text: 'Hood Damaged', severity: 'warn' },
        hood_unlatched: { active: findState('Hood') === 'unlatched', text: 'Hood Unlatched', severity: 'warn' },
        hood_detached: { active: findState('Hood') === 'detached', text: 'Hood Broken', severity: 'crit' },
        trunk_dmg: { active: d.rear > 0.22 && findState('Trunk') === 'attached', text: 'Trunk Damaged', severity: 'warn' },
        trunk_detached: { active: findState('Trunk') === 'detached', text: 'Trunk Broken', severity: 'crit' },
        bumperF_dmg: { active: d.front > 0.16 && findState('Front bumper') === 'attached', text: 'Front Bumper Damaged', severity: 'warn' },
        bumperF_detached: { active: findState('Front bumper') === 'detached', text: 'Front Bumper Broken', severity: 'crit' },
        bumperR_detached: { active: findState('Rear bumper') === 'detached', text: 'Rear Bumper Broken', severity: 'crit' },
        fender_dmg: { active: d.fenders > 0.22, text: 'Side Panel Damaged', severity: 'warn' },
        doorL_broken: { active: findState('Left door') !== 'attached', text: 'Front Left Door Broken', severity: 'crit' },
        doorR_broken: { active: findState('Right door') !== 'attached', text: 'Front Right Door Broken', severity: 'crit' },
        radiator: { active: Boolean(m.radiatorDamaged), text: 'Radiator Damaged', severity: 'warn' },
        coolant: { active: Boolean(m.coolantLeak), text: 'Coolant Leak', severity: 'warn' },
        oil: { active: Boolean(m.oilLeak), text: 'Oil Leak', severity: 'warn' },
        fuel: { active: Boolean(m.fuelLeak), text: 'Fuel Leak', severity: 'warn' },
        engine_dmg: { active: d.engine > 0.25 && !m.engineBroken, text: 'Engine Damaged', severity: 'warn' },
        overheat: { active: Boolean(m.overheating), text: 'Engine Overheating!', severity: 'crit' },
        engine_broken: { active: Boolean(m.engineBroken), text: 'Engine Broken!', severity: 'crit' },
        out_of_fuel: { active: m.fuelLevel <= 0.001, text: 'Out of Fuel!', severity: 'crit' },
        steer_dmg: { active: m.steeringDamage > 0.18 && !m.steeringLocked, text: 'Steering Bent', severity: 'warn' },
        steer_lock: { active: Boolean(m.steeringLocked), text: 'Steering Locked', severity: 'crit' },
        hlL_broken: { active: Boolean(m.headlightBrokenL), text: 'Left Headlight Broken', severity: 'warn' },
        hlR_broken: { active: Boolean(m.headlightBrokenR), text: 'Right Headlight Broken', severity: 'warn' },
      };

      for (let i = 0; i < 4; i++) {
        currentFlags[`flat_${i}`] = {
          active: Boolean(m.flatTires?.[i]),
          text: `${TIRE_NAMES[i]} Flat`,
          severity: 'warn',
        };
        currentFlags[`axle_${i}`] = {
          active: (m.axleDamage?.[i] ?? 0) > 0.28,
          text: `${TIRE_NAMES[i].replace('Tire', 'Axle')} Damaged`,
          severity: 'warn',
        };
      }

      const prev = prevFlagsRef.current;
      const now = Date.now();
      const added: DamageAlert[] = [];
      let anyActive = false;

      for (const key of Object.keys(currentFlags)) {
        const cur = currentFlags[key];
        if (cur.active) anyActive = true;
        if (cur.active && !prev[key]) {
          added.push({
            id: alertIdRef.current++,
            text: cur.text,
            severity: cur.severity,
            expiresAt: now + 4600,
          });
        }
        prev[key] = cur.active;
      }

      if (!anyActive) {
        // Vehicle was repaired/reset — clear stale alerts immediately
        setAlerts((old) => (old.length > 0 ? [] : old));
        return;
      }

      if (added.length > 0) {
        setAlerts((old) => [...old.filter((a) => a.expiresAt > now), ...added].slice(-3));
      }
    });

    // Periodic expiry cleanup for active alerts
    useEffect(() => {
      if (alerts.length === 0) return;
      const timer = window.setInterval(() => {
        const now = Date.now();
        setAlerts((old) => old.filter((a) => a.expiresAt > now));
      }, 450);
      return () => window.clearInterval(timer);
    }, [alerts.length]);

    const d = telemetry?.damage;
    const m = telemetry?.mech;
    const parts = telemetry?.parts ?? [];
    const findState = (label: string) => parts.find((p) => p.label === label)?.state;

    const hoodState = findState('Hood');
    const trunkState = findState('Trunk');
    const bumperFState = findState('Front bumper');
    const bumperRState = findState('Rear bumper');
    const doorLState = findState('Left door');
    const doorRState = findState('Right door');

    const frontDmg = Math.min(1, d?.front ?? 0);
    const hoodDmg = Math.min(1, d?.hood ?? 0);
    const cabinDmg = Math.min(1, d?.cabin ?? 0);
    const rearDmg = Math.min(1, d?.rear ?? 0);
    const fenderDmg = Math.min(1, d?.fenders ?? 0);
    const engineDmg = Math.min(1, d?.engine ?? 0);
    const bumperDmg = Math.min(1, d?.bumpers ?? 0);
    const midRear = Boolean(telemetry?.midRearEngine);

    const axleD = m?.axleDamage ?? [0, 0, 0, 0];
    const flats = m?.flatTires ?? [false, false, false, false];
    const wheelCols = [0, 1, 2, 3].map((i) => (flats[i] ? '#f59e0b' : dmgColor(axleD[i] ?? 0)));

    const bumperFCol = partColor(bumperFState, bumperDmg * 0.9);
    const bumperRCol = partColor(bumperRState, rearDmg * 0.9);
    const hoodCol = partColor(hoodState, Math.max(hoodDmg, frontDmg * 0.65));
    const cabinCol = dmgColor(cabinDmg);
    const trunkCol = partColor(trunkState, rearDmg);
    const doorLCol = partColor(doorLState, fenderDmg);
    const doorRCol = partColor(doorRState, fenderDmg);
    const engineCol = m?.engineBroken || m?.engineStalled ? '#ef4444' : m?.overheating ? '#f97316' : dmgColor(engineDmg);
    const radCol = m?.radiatorDamaged || m?.coolantLeak ? '#ef4444' : dmgColor(frontDmg * 0.85);
    const fuelCol = m?.fuelLeak || (m?.fuelLevel ?? 1) < 0.15 ? '#f59e0b' : dmgColor(rearDmg * 0.8);
    const steerCol = m?.steeringLocked ? '#ef4444' : dmgColor(m?.steeringDamage ?? 0);

    if (collapsed) {
      const hasAlert = Boolean(
        m?.engineBroken || m?.overheating || m?.radiatorDamaged || m?.coolantLeak || m?.oilLeak || m?.fuelLeak || flats.some(Boolean),
      );
      return (
        <div className="diag-bottom-wrap">
          {alerts.length > 0 && (
            <div className="diag-alert-stack" role="status" aria-live="polite">
              {alerts.map((a) => (
                <div key={a.id} className={`diag-alert-pill ${a.severity}`}>
                  {a.text}
                </div>
              ))}
            </div>
          )}
          <button
            id="expand-diagnostics-drawer"
            type="button"
            className="pill-drawer diag-mini-pill"
            style={hasAlert ? { borderColor: 'rgba(239,68,68,0.7)' } : undefined}
            onClick={onToggle}
          >
            <span>DIAG</span>
            {hasAlert && <span style={{ color: '#f87171' }}>!</span>}
          </button>
        </div>
      );
    }

    return (
      <div className="diag-bottom-wrap">
        {/* Live Breakage & Mechanical Alerts ABOVE the Horizontal Schematic */}
        {alerts.length > 0 && (
          <div className="diag-alert-stack" role="status" aria-live="polite">
            {alerts.map((a) => (
              <div key={a.id} className={`diag-alert-pill ${a.severity}`}>
                {a.text}
              </div>
            ))}
          </div>
        )}

        {/* Horizontal Side-Oriented Compact X-Ray Schematic (No percentage list below) */}
        <section className="diag-horiz-card" aria-label="Vehicle Diagnostics">
          <button
            id="collapse-diagnostics-drawer"
            type="button"
            className="diag-horiz-close"
            title="Hide vehicle diagnostics"
            onClick={onToggle}
          >
            ×
          </button>
          <svg
            viewBox="0 0 210 92"
            className="diag-svg-horiz"
            role="img"
            aria-label="Vehicle structural and mechanical health diagram"
          >
            {/* Rear Bumper (left) */}
            <path d="M 10 21 Q 3 46 10 71 L 21 69 L 21 23 Z" fill={bumperRCol} fillOpacity="0.86" stroke="#0b0f17" strokeWidth="1.2" />
            {/* Trunk + Fuel Tank */}
            <path d="M 23 23 L 58 25 L 58 67 L 23 69 Z" fill={trunkCol} fillOpacity="0.46" stroke={trunkCol} strokeWidth="1.2" />
            <rect x="26" y="34" width="11" height="24" rx="3" fill={fuelCol} fillOpacity="0.88" stroke="#0b0f17" strokeWidth="1" />
            {/* Rear Axle */}
            <line x1="42" y1="15" x2="42" y2="77" stroke={dmgColor(m?.rearAxleDamage ?? 0)} strokeWidth="2.6" strokeLinecap="round" />

            {/* Left (top) & Right (bottom) Doors / Flanks */}
            <path d="M 32 15 L 166 15 L 168 23 L 30 23 Z" fill={doorLCol} fillOpacity="0.65" stroke="#0b0f17" strokeWidth="1.1" />
            <path d="M 30 69 L 168 69 L 166 77 L 32 77 Z" fill={doorRCol} fillOpacity="0.65" stroke="#0b0f17" strokeWidth="1.1" />

            {/* Survival Cell / Cabin */}
            <rect x="61" y="25" width="58" height="42" rx="7" fill={cabinCol} fillOpacity="0.48" stroke={cabinCol} strokeWidth="1.5" />

            {/* Front Hood (right) */}
            <path d="M 122 25 L 178 23 L 178 69 L 122 67 Z" fill={hoodCol} fillOpacity="0.46" stroke={hoodCol} strokeWidth="1.2" />

            {/* Steering Rack (front axle) */}
            <line x1="155" y1="15" x2="155" y2="77" stroke={steerCol} strokeWidth="2.8" strokeLinecap="round" />

            {/* Engine Block Core (Front or Mid-Rear) */}
            <rect
              x={midRear ? 40 : 136}
              y="33"
              width="28"
              height="26"
              rx="4"
              fill={engineCol}
              fillOpacity="0.92"
              stroke="#0b0f17"
              strokeWidth="1.4"
            />
            <text x={midRear ? 54 : 150} y="49" textAnchor="middle" fill="#0b0f17" fontSize="8" fontWeight="900">
              {m?.engineBroken || m?.engineStalled ? 'OFF' : 'ENG'}
            </text>

            {/* Radiator bar */}
            <rect x="180" y="29" width="5" height="34" rx="2" fill={radCol} fillOpacity="0.95" />

            {/* Front Bumper (right) */}
            <path d="M 187 22 L 198 20 Q 206 46 198 72 L 187 70 Z" fill={bumperFCol} fillOpacity="0.86" stroke="#0b0f17" strokeWidth="1.2" />

            {/* 4 Wheels: RL (top-left), RR (bottom-left), FL (top-right), FR (bottom-right) */}
            <rect x="29" y="4" width="26" height="10" rx="3" fill="#0b0f17" stroke={wheelCols[2]} strokeWidth="2.2" />
            <rect x="29" y="78" width="26" height="10" rx="3" fill="#0b0f17" stroke={wheelCols[3]} strokeWidth="2.2" />
            <rect x="142" y="4" width="26" height="10" rx="3" fill="#0b0f17" stroke={wheelCols[0]} strokeWidth="2.2" />
            <rect x="142" y="78" width="26" height="10" rx="3" fill="#0b0f17" stroke={wheelCols[1]} strokeWidth="2.2" />
          </svg>
        </section>
      </div>
    );
  };
