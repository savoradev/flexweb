import React from 'react';
import type { VehicleTelemetry } from '../engine/SoftBodyVehicle';

interface VehicleDiagnosticsProps {
  telemetry: VehicleTelemetry | undefined;
  collapsed: boolean;
  onToggle: () => void;
}

/** Maps a 0..1 damage value to BeamNG-style green -> amber -> red hex color. */
function dmgColor(dmg: number): string {
  if (dmg < 0.12) return '#22c55e';
  if (dmg < 0.45) return '#f59e0b';
  return '#ef4444';
}

/** Maps a part attachment state + damage to BeamNG-style color. */
function partColor(state: string | undefined, fallbackDmg: number): string {
  if (state === 'detached') return '#ef4444';
  if (state === 'unlatched') return '#f59e0b';
  return dmgColor(fallbackDmg);
}

function HealthBar({ pct, color }: { pct: number; color: string }): React.ReactElement {
  const clamped = Math.max(0, Math.min(100, pct));
  return (
    <div
      style={{
        height: 4,
        width: '100%',
        borderRadius: 99,
        background: 'rgba(148,163,184,0.16)',
        overflow: 'hidden',
        marginTop: 3,
      }}
    >
      <div
        style={{
          height: '100%',
          width: `${clamped}%`,
          background: color,
          borderRadius: 99,
          transition: 'width 120ms ease-out, background 180ms ease-out',
        }}
      />
    </div>
  );
}

export const VehicleDiagnostics: React.FC<VehicleDiagnosticsProps> = React.memo(
  ({ telemetry, collapsed, onToggle }) => {
    if (collapsed) {
      const structDmg = Math.min(
        100,
        Math.round(((telemetry?.damage.front ?? 0) * 0.45 + (telemetry?.damage.cabin ?? 0) * 0.35 + (telemetry?.damage.rear ?? 0) * 0.2) * 100),
      );
      const hasAlert = Boolean(
        telemetry?.mech?.engineStalled ||
        telemetry?.mech?.steeringLocked ||
        telemetry?.mech?.oilLeak ||
        telemetry?.mech?.coolantLeak ||
        telemetry?.mech?.fuelLeak,
      );
      return (
        <button
          id="expand-diagnostics-drawer"
          type="button"
          className="pill-drawer"
          style={hasAlert ? { borderColor: 'rgba(239,68,68,0.65)' } : undefined}
          onClick={onToggle}
        >
          <span className="pill-icon">+</span>
          <span>Diagnostics · {100 - structDmg}%{hasAlert ? ' ⚠' : ''}</span>
        </button>
      );
    }

    const d = telemetry?.damage;
    const m = telemetry?.mech;
    const parts = telemetry?.parts ?? [];
    const findState = (label: string) => parts.find((p) => p.label === label)?.state;

    const hoodState = findState('Hood');
    const trunkState = findState('Trunk');
    const bumperFState = findState('Front bumper');
    const bumperRState = findState('Rear bumper');

    const frontDmg = Math.min(1, d?.front ?? 0);
    const hoodDmg = Math.min(1, d?.hood ?? 0);
    const cabinDmg = Math.min(1, d?.cabin ?? 0);
    const rearDmg = Math.min(1, d?.rear ?? 0);
    const fenderDmg = Math.min(1, d?.fenders ?? 0);
    const engineDmg = Math.min(1, d?.engine ?? 0);
    const bumperDmg = Math.min(1, d?.bumpers ?? 0);
    const midRear = Boolean(telemetry?.midRearEngine);

    const axleD = m?.axleDamage ?? [0, 0, 0, 0];
    const wheelCols = [
      dmgColor(axleD[0]),
      dmgColor(axleD[1]),
      dmgColor(axleD[2]),
      dmgColor(axleD[3]),
    ];

    const bumperFCol = partColor(bumperFState, bumperDmg * 0.9);
    const bumperRCol = partColor(bumperRState, rearDmg * 0.9);
    const hoodCol = partColor(hoodState, Math.max(hoodDmg, frontDmg * 0.65));
    const cabinCol = dmgColor(cabinDmg);
    const trunkCol = partColor(trunkState, rearDmg);
    const sideCol = dmgColor(fenderDmg);
    const engineCol = m?.engineStalled ? '#ef4444' : dmgColor(engineDmg);
    const radCol = m?.coolantLeak ? '#16e874' : dmgColor(frontDmg * 0.85);
    const fuelCol = m?.fuelLeak ? '#38bdf8' : dmgColor(rearDmg * 0.8);
    const steerCol = m?.steeringLocked ? '#ef4444' : dmgColor(m?.steeringDamage ?? 0);

    const chassisPct = Math.max(
      0,
      Math.min(100, Math.round((1 - (frontDmg * 0.35 + cabinDmg * 0.4 + rearDmg * 0.25)) * 100)),
    );
    const enginePct = m?.engineStalled ? 0 : Math.max(0, Math.round((1 - engineDmg) * 100));
    const steerPct = m?.steeringLocked ? 0 : Math.max(0, Math.round((1 - (m?.steeringDamage ?? 0)) * 100));

    const chassisCls = chassisPct > 82 ? 'diag-ok' : chassisPct > 52 ? 'diag-warn' : 'diag-red';
    const engineCls = m?.engineStalled || enginePct < 45 ? 'diag-red' : enginePct < 85 ? 'diag-warn' : 'diag-ok';
    const steerCls = m?.steeringLocked ? 'diag-red' : (m?.steeringDamage ?? 0) > 0.18 ? 'diag-warn' : 'diag-ok';

    const anyLeak = Boolean(m?.coolantLeak || m?.oilLeak || m?.fuelLeak);

    return (
      <section className="panel-card diag-card" aria-label="Vehicle Diagnostics">
        <div className="panel-header">
          <h2 className="panel-title">Diagnostics</h2>
          <button
            id="collapse-diagnostics-drawer"
            type="button"
            className="drawer-toggle-btn"
            title="Collapse vehicle diagnostics"
            onClick={onToggle}
          >
            −
          </button>
        </div>

        {/* Pro BeamNG-style Top-Down Mechanical X-Ray Schematic */}
        <div className="diag-schematic-wrap" style={{ padding: '6px 0 4px' }}>
          <svg
            viewBox="0 0 120 186"
            className="diag-svg-top"
            role="img"
            aria-label="Vehicle structural and mechanical health diagram"
          >
            {/* Front Bumper */}
            <path d="M 32 10 Q 60 2 88 10 L 86 22 L 34 22 Z" fill={bumperFCol} fillOpacity="0.82" stroke="#0f172a" strokeWidth="1.4" />
            {/* Radiator bar */}
            <rect x="42" y="23" width="36" height="5" rx="2" fill={radCol} fillOpacity="0.95" />
            {/* Front Hood */}
            <path d="M 34 29 L 86 29 L 84 68 L 36 68 Z" fill={hoodCol} fillOpacity="0.48" stroke={hoodCol} strokeWidth="1.2" />
            {/* Steering Rack connecting FL & FR hubs */}
            <line x1="25" y1="44" x2="95" y2="44" stroke={steerCol} strokeWidth="3" strokeLinecap="round" />
            {/* Left & Right Flanks */}
            <path d="M 24 30 L 34 29 L 34 152 L 24 150 Z" fill={sideCol} fillOpacity="0.62" stroke="#0f172a" strokeWidth="1.2" />
            <path d="M 86 29 L 96 30 L 96 150 L 86 152 Z" fill={sideCol} fillOpacity="0.62" stroke="#0f172a" strokeWidth="1.2" />
            {/* Survival Cell / Cabin */}
            <rect x="35" y="70" width="50" height="52" rx="7" fill={cabinCol} fillOpacity="0.52" stroke={cabinCol} strokeWidth="1.6" />
            {/* Engine Block Core (Front or Mid-Rear) */}
            <rect
              x="46"
              y={midRear ? 124 : 36}
              width="28"
              height="24"
              rx="4"
              fill={engineCol}
              fillOpacity="0.92"
              stroke="#0f172a"
              strokeWidth="1.5"
            />
            <text x="60" y={midRear ? 139 : 51} textAnchor="middle" fill="#0b0f17" fontSize="8" fontWeight="900">
              {m?.engineStalled ? 'OFF' : 'ENG'}
            </text>
            {/* Rear Axle Shaft */}
            <line x1="25" y1="142" x2="95" y2="142" stroke={dmgColor(m?.rearAxleDamage ?? 0)} strokeWidth="2.8" strokeLinecap="round" />
            {/* Trunk + Fuel Tank */}
            <path d="M 36 124 L 84 124 L 86 158 L 34 158 Z" fill={trunkCol} fillOpacity="0.48" stroke={trunkCol} strokeWidth="1.2" />
            <rect x="46" y="147" width="28" height="9" rx="3" fill={fuelCol} fillOpacity="0.88" stroke="#0f172a" strokeWidth="1" />
            {/* Rear Bumper */}
            <path d="M 32 160 L 88 160 L 86 173 Q 60 180 34 173 Z" fill={bumperRCol} fillOpacity="0.82" stroke="#0f172a" strokeWidth="1.4" />
            {/* 4 Individual Wheels / Axles (FL, FR, RL, RR) */}
            <rect x="13" y="31" width="11" height="26" rx="3" fill="#0b0f17" stroke={wheelCols[0]} strokeWidth="2.4" />
            <rect x="96" y="31" width="11" height="26" rx="3" fill="#0b0f17" stroke={wheelCols[1]} strokeWidth="2.4" />
            <rect x="13" y="129" width="11" height="26" rx="3" fill="#0b0f17" stroke={wheelCols[2]} strokeWidth="2.4" />
            <rect x="96" y="129" width="11" height="26" rx="3" fill="#0b0f17" stroke={wheelCols[3]} strokeWidth="2.4" />
          </svg>
        </div>

        {/* Pro Mechanical Health Bars */}
        <div className="diag-rows" style={{ gap: 7 }}>
          <div>
            <div className="diag-row">
              <span>Chassis</span>
              <strong className={chassisCls}>{chassisPct}%</strong>
            </div>
            <HealthBar pct={chassisPct} color={dmgColor(1 - chassisPct / 100)} />
          </div>

          <div>
            <div className="diag-row">
              <span>Engine ({midRear ? 'Rear' : 'Front'})</span>
              <strong className={engineCls}>
                {m?.engineStalled ? 'STALLED' : `${enginePct}%`}
              </strong>
            </div>
            <HealthBar pct={enginePct} color={engineCol} />
          </div>

          <div>
            <div className="diag-row">
              <span>Steering</span>
              <strong className={steerCls}>
                {m?.steeringLocked
                  ? 'LOCKED'
                  : (m?.steeringDamage ?? 0) > 0.16
                    ? `Bent (${m?.steeringBiasDeg ?? 0}°)`
                    : 'OK'}
              </strong>
            </div>
            <HealthBar pct={steerPct} color={steerCol} />
          </div>

          {/* 4-Wheel Individual Axle Health Grid */}
          <div>
            <div className="diag-row" style={{ marginBottom: 4 }}>
              <span>Axles &amp; Hubs</span>
              <span style={{ fontSize: 10, opacity: 0.72 }}>FL · FR · RL · RR</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 4 }}>
              {(['FL', 'FR', 'RL', 'RR'] as const).map((lbl, idx) => {
                const pct = Math.max(0, Math.round((1 - (axleD[idx] ?? 0)) * 100));
                const col = wheelCols[idx];
                return (
                  <div
                    key={lbl}
                    style={{
                      background: 'rgba(15,23,42,0.58)',
                      border: `1px solid ${col}66`,
                      borderRadius: 6,
                      padding: '3px 4px',
                      textAlign: 'center',
                      fontSize: 10,
                    }}
                  >
                    <div style={{ opacity: 0.7, fontSize: 9 }}>{lbl}</div>
                    <strong style={{ color: col, fontSize: 10.5 }}>{pct}%</strong>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Live Color-Coded Fluid Leak Badges */}
          {anyLeak && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginTop: 2 }}>
              {m?.coolantLeak && (
                <span
                  style={{
                    fontSize: 10,
                    fontWeight: 700,
                    padding: '2px 7px',
                    borderRadius: 99,
                    background: 'rgba(22,232,116,0.16)',
                    border: '1px solid rgba(22,232,116,0.55)',
                    color: '#4ade80',
                  }}
                >
                  ● COOLANT {Math.round((m.coolantSeverity ?? 0) * 100)}%
                </span>
              )}
              {m?.oilLeak && (
                <span
                  style={{
                    fontSize: 10,
                    fontWeight: 700,
                    padding: '2px 7px',
                    borderRadius: 99,
                    background: 'rgba(245,158,11,0.16)',
                    border: '1px solid rgba(245,158,11,0.55)',
                    color: '#fbbf24',
                  }}
                >
                  ● OIL {Math.round((m.oilSeverity ?? 0) * 100)}%
                </span>
              )}
              {m?.fuelLeak && (
                <span
                  style={{
                    fontSize: 10,
                    fontWeight: 700,
                    padding: '2px 7px',
                    borderRadius: 99,
                    background: 'rgba(56,189,248,0.16)',
                    border: '1px solid rgba(56,189,248,0.55)',
                    color: '#38bdf8',
                  }}
                >
                  ● FUEL {Math.round((m.fuelSeverity ?? 0) * 100)}%
                </span>
              )}
            </div>
          )}
        </div>
      </section>
    );
  },
);
