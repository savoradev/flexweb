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
  if (dmg < 0.42) return '#f59e0b';
  return '#ef4444';
}

/** Maps a part attachment state + damage to BeamNG-style color. */
function partColor(state: string | undefined, fallbackDmg: number): string {
  if (state === 'detached') return '#ef4444';
  if (state === 'unlatched') return '#f59e0b';
  return dmgColor(fallbackDmg);
}

function bumperStateLabel(state: string | undefined): { text: string; cls: string } {
  if (state === 'detached') return { text: 'Detached', cls: 'diag-red' };
  if (state === 'unlatched') return { text: 'Hanging', cls: 'diag-warn' };
  return { text: 'Attached', cls: 'diag-ok' };
}

function engineStateLabel(dmg: number): { text: string; cls: string } {
  if (dmg >= 0.45) return { text: `Critical (${Math.round(dmg * 100)}%)`, cls: 'diag-red' };
  if (dmg >= 0.12) return { text: `Damaged (${Math.round(dmg * 100)}%)`, cls: 'diag-warn' };
  return { text: 'OK', cls: 'diag-ok' };
}

export const VehicleDiagnostics: React.FC<VehicleDiagnosticsProps> = React.memo(
  ({ telemetry, collapsed, onToggle }) => {
    if (collapsed) {
      const structDmg = Math.min(
        100,
        Math.round(((telemetry?.damage.front ?? 0) * 0.45 + (telemetry?.damage.cabin ?? 0) * 0.35 + (telemetry?.damage.rear ?? 0) * 0.2) * 100),
      );
      return (
        <button
          id="expand-diagnostics-drawer"
          type="button"
          className="pill-drawer"
          onClick={onToggle}
        >
          <span className="pill-icon">+</span>
          <span>Diagnostics · {100 - structDmg}%</span>
        </button>
      );
    }

    const d = telemetry?.damage;
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

    const susp = telemetry?.suspCompression ?? [0, 0, 0, 0];
    const wheelDmgFront = Math.min(1, frontDmg * 0.65 + fenderDmg * 0.35);
    const wheelDmgRear = Math.min(1, rearDmg * 0.65 + fenderDmg * 0.35);
    const wheelCols = [
      dmgColor(wheelDmgFront),
      dmgColor(wheelDmgFront),
      dmgColor(wheelDmgRear),
      dmgColor(wheelDmgRear),
    ];

    const bumperFCol = partColor(bumperFState, bumperDmg * 0.9);
    const bumperRCol = partColor(bumperRState, rearDmg * 0.9);
    const hoodCol = partColor(hoodState, Math.max(hoodDmg, frontDmg * 0.65));
    const cabinCol = dmgColor(cabinDmg);
    const trunkCol = partColor(trunkState, rearDmg);
    const sideCol = dmgColor(fenderDmg);
    const engineCol = dmgColor(engineDmg);

    const chassisIntegrity = Math.max(
      0,
      Math.min(100, Math.round((1 - (frontDmg * 0.35 + cabinDmg * 0.4 + rearDmg * 0.25)) * 100)),
    );
    const chassisCls = chassisIntegrity > 82 ? 'diag-ok' : chassisIntegrity > 52 ? 'diag-warn' : 'diag-red';
    const bfInfo = bumperStateLabel(bumperFState);
    const brInfo = bumperStateLabel(bumperRState);
    const engInfo = engineStateLabel(engineDmg);
    const avgSuspPct = Math.round(((susp[0] + susp[1] + susp[2] + susp[3]) / 4) * 100);

    return (
      <section className="panel-card diag-card" aria-label="Vehicle Diagnostics">
        <div className="panel-header">
          <h2 className="panel-title">Vehicle Diagnostics</h2>
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

        {/* BeamNG-style Top-Down + Side-Profile Schematic */}
        <div className="diag-schematic-wrap">
          <svg
            viewBox="0 0 120 184"
            className="diag-svg"
            role="img"
            aria-label="Vehicle damage schematic"
          >
            {/* Front Bumper */}
            <path
              d="M 28 14 Q 60 4 92 14 L 89 24 Q 60 17 31 24 Z"
              fill={bumperFCol}
              fillOpacity={bumperFState === 'detached' ? 0.35 : 0.78}
              stroke={bumperFCol}
              strokeWidth="1.6"
              strokeDasharray={bumperFState === 'detached' ? '3 2' : undefined}
            />

            {/* Hood / Front Crumple Zone */}
            <path
              d="M 31 27 L 89 27 L 86 64 L 34 64 Z"
              fill={hoodCol}
              fillOpacity="0.62"
              stroke={hoodCol}
              strokeWidth="1.4"
            />

            {/* Left & Right Doors / Flanks */}
            <rect x="21" y="62" width="11" height="62" rx="3" fill={sideCol} fillOpacity="0.7" stroke={sideCol} strokeWidth="1.2" />
            <rect x="88" y="62" width="11" height="62" rx="3" fill={sideCol} fillOpacity="0.7" stroke={sideCol} strokeWidth="1.2" />

            {/* Cabin / Safety Cage */}
            <path
              d="M 34 67 L 86 67 L 84 124 L 36 124 Z"
              fill={cabinCol}
              fillOpacity="0.58"
              stroke={cabinCol}
              strokeWidth="1.5"
            />
            {/* Windshield & Rear Glass hint lines */}
            <line x1="36" y1="77" x2="84" y2="77" stroke="rgba(255,255,255,0.28)" strokeWidth="1" />
            <line x1="38" y1="114" x2="82" y2="114" stroke="rgba(255,255,255,0.24)" strokeWidth="1" />

            {/* Trunk / Rear Engine Bay */}
            <path
              d="M 34 127 L 86 127 L 89 158 L 31 158 Z"
              fill={trunkCol}
              fillOpacity="0.62"
              stroke={trunkCol}
              strokeWidth="1.4"
            />

            {/* Rear Bumper */}
            <path
              d="M 30 161 Q 60 167 90 161 L 92 172 Q 60 180 28 172 Z"
              fill={bumperRCol}
              fillOpacity={bumperRState === 'detached' ? 0.35 : 0.78}
              stroke={bumperRCol}
              strokeWidth="1.6"
              strokeDasharray={bumperRState === 'detached' ? '3 2' : undefined}
            />

            {/* Engine Block Icon (Front for BMW/Dodge, Mid-Rear for McLaren P1) */}
            <g transform={midRear ? 'translate(46, 132)' : 'translate(46, 34)'}>
              <rect
                x="0"
                y="0"
                width="28"
                height="21"
                rx="3"
                fill="rgba(10, 14, 22, 0.85)"
                stroke={engineCol}
                strokeWidth="1.8"
              />
              <rect x="4" y="4" width="20" height="13" rx="1.5" fill={engineCol} fillOpacity="0.72" />
              <text x="14" y="13.5" textAnchor="middle" fontSize="7" fontWeight="700" fill="#0b0f17">
                ENG
              </text>
            </g>

            {/* 4 Wheels & Suspension Travel Bars (FL, FR, RL, RR) */}
            <rect x="10" y="32" width="11" height="26" rx="3" fill={wheelCols[0]} fillOpacity="0.88" stroke="#0b0f17" strokeWidth="1" />
            <rect x="99" y="32" width="11" height="26" rx="3" fill={wheelCols[1]} fillOpacity="0.88" stroke="#0b0f17" strokeWidth="1" />
            <rect x="10" y="128" width="11" height="26" rx="3" fill={wheelCols[2]} fillOpacity="0.88" stroke="#0b0f17" strokeWidth="1" />
            <rect x="99" y="128" width="11" height="26" rx="3" fill={wheelCols[3]} fillOpacity="0.88" stroke="#0b0f17" strokeWidth="1" />
          </svg>

          {/* Compact Side-Profile Silhouette */}
          <svg
            viewBox="0 0 180 46"
            className="diag-profile-svg"
            role="img"
            aria-label="Vehicle side profile"
          >
            <rect x="150" y="21" width="18" height="14" rx="3" fill={bumperFCol} fillOpacity="0.78" />
            <polygon points="108,18 150,21 150,35 108,35" fill={hoodCol} fillOpacity="0.65" />
            <polygon points="58,18 74,7 112,7 124,18 124,35 58,35" fill={cabinCol} fillOpacity="0.62" />
            <polygon points="24,19 58,18 58,35 24,35" fill={trunkCol} fillOpacity="0.65" />
            <rect x="10" y="21" width="14" height="14" rx="3" fill={bumperRCol} fillOpacity="0.78" />
            <circle cx="136" cy="35" r="9" fill="#0b0f17" stroke={wheelCols[0]} strokeWidth="3" />
            <circle cx="42" cy="35" r="9" fill="#0b0f17" stroke={wheelCols[2]} strokeWidth="3" />
          </svg>
        </div>

        {/* Real-time Key Subsystem Status Rows */}
        <div className="diag-rows">
          <div className="diag-row">
            <span>Chassis</span>
            <strong className={chassisCls}>{chassisIntegrity}%</strong>
          </div>
          <div className="diag-row">
            <span>Engine ({midRear ? 'Rear' : 'Front'})</span>
            <strong className={telemetry?.mech?.engineStalled ? 'diag-red' : engInfo.cls}>
              {telemetry?.mech?.engineStalled
                ? 'STALLED (0%)'
                : engInfo.text}
            </strong>
          </div>
          <div className="diag-row">
            <span>Steering</span>
            <strong
              className={
                telemetry?.mech?.steeringLocked
                  ? 'diag-red'
                  : (telemetry?.mech?.steeringDamage ?? 0) > 0.18
                    ? 'diag-warn'
                    : 'diag-ok'
              }
            >
              {telemetry?.mech?.steeringLocked
                ? 'LOCKED'
                : (telemetry?.mech?.steeringDamage ?? 0) > 0.18
                  ? `Bent (${telemetry?.mech?.steeringBiasDeg ?? 0}°)`
                  : 'OK'}
            </strong>
          </div>
          <div className="diag-row">
            <span>Axles (F / R)</span>
            <strong
              className={
                Math.max(telemetry?.mech?.frontAxleDamage ?? 0, telemetry?.mech?.rearAxleDamage ?? 0) > 0.65
                  ? 'diag-red'
                  : Math.max(telemetry?.mech?.frontAxleDamage ?? 0, telemetry?.mech?.rearAxleDamage ?? 0) > 0.18
                    ? 'diag-warn'
                    : 'diag-ok'
              }
            >
              {Math.round((1 - (telemetry?.mech?.frontAxleDamage ?? 0)) * 100)}% / {Math.round((1 - (telemetry?.mech?.rearAxleDamage ?? 0)) * 100)}%
            </strong>
          </div>
          {(telemetry?.mech?.oilLeak || telemetry?.mech?.coolantLeak) && (
            <div className="diag-row">
              <span>Fluid Leaks</span>
              <strong className="diag-red">
                {telemetry.mech.oilLeak && telemetry.mech.coolantLeak
                  ? 'OIL + COOLANT'
                  : telemetry.mech.coolantLeak
                    ? 'COOLANT LEAK'
                    : 'OIL LEAK'}
              </strong>
            </div>
          )}
          <div className="diag-row">
            <span>Front / Rear Bumper</span>
            <strong className={bfInfo.cls === 'diag-red' || brInfo.cls === 'diag-red' ? 'diag-red' : bfInfo.cls === 'diag-warn' || brInfo.cls === 'diag-warn' ? 'diag-warn' : 'diag-ok'}>
              {bfInfo.text} · {brInfo.text}
            </strong>
          </div>
          <div className="diag-row">
            <span>Suspension</span>
            <strong className={dmgColor(Math.max(wheelDmgFront, wheelDmgRear)) === '#22c55e' ? 'diag-ok' : 'diag-warn'}>
              4/4 · {avgSuspPct}%
            </strong>
          </div>
        </div>
      </section>
    );
  },
);
