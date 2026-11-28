import { forwardRef, useImperativeHandle, useRef } from 'react';
import type { VehicleTelemetry } from '../engine/SoftBodyVehicle';

export interface MotorsportClusterHandle {
  update(tel: VehicleTelemetry, fps: number, mph: boolean, dt?: number): void;
}

interface MotorsportClusterProps {
  speedUnitLabel: string;
}

const RPM_START_DEG = -135;
const RPM_SWEEP_DEG = 215; // 0 -> 7.5k RPM arc on left pod
const SPD_START_DEG = -135;
const SPD_SWEEP_DEG = 270; // 0 -> 300 km/h (or 0 -> 200 mph) arc on right pod

/**
 * Dual-Pod Asymmetric Automotive Instrument Cluster
 * - Left pod: Analog Tachometer (0–7/8 x1000 RPM, redline band, orange needle, digital RPM box)
 * - Right pod: Large Analog Speedometer (0–300 km/h with red minor ticks, red needle, digital speed box)
 * - Bottom telemetry strip: Gear, Temp, Throttle, Brake, ABS, Fuel, G-Force, Lights, FPS
 * - 100% direct DOM/SVG ref updates inside `update()` at full frame rate -> zero React re-renders!
 */
export const MotorsportCluster = forwardRef<MotorsportClusterHandle, MotorsportClusterProps>(
  function MotorsportCluster({ speedUnitLabel }, ref) {
    const smoothRpm = useRef(850);
    const smoothSpd = useRef(0);
    const slowTextTimer = useRef(0);
    const lastText = useRef({
      rpm: '',
      spd: '',
      unit: '',
      gear: '',
      temp: '',
      tempHot: false,
      thr: '',
      brk: '',
      abs: '',
      absOn: false,
      fuel: '',
      fuelLow: false,
      g: '',
      hl: '',
      hlOn: false,
      fps: '',
    });

    const rpmNeedleRef = useRef<SVGGElement | null>(null);
    const spdNeedleRef = useRef<SVGGElement | null>(null);
    const rpmDigitsRef = useRef<HTMLSpanElement | null>(null);
    const spdDigitsRef = useRef<HTMLSpanElement | null>(null);
    const spdUnitRef = useRef<HTMLSpanElement | null>(null);

    const gearRef = useRef<HTMLSpanElement | null>(null);
    const tempRef = useRef<HTMLSpanElement | null>(null);
    const thrRef = useRef<HTMLSpanElement | null>(null);
    const brkRef = useRef<HTMLSpanElement | null>(null);
    const absRef = useRef<HTMLSpanElement | null>(null);
    const fuelRef = useRef<HTMLSpanElement | null>(null);
    const gRef = useRef<HTMLSpanElement | null>(null);
    const hlRef = useRef<HTMLSpanElement | null>(null);
    const fpsRef = useRef<HTMLSpanElement | null>(null);

    useImperativeHandle(ref, () => ({
      update(tel: VehicleTelemetry, fps: number, mph: boolean, dt = 1 / 60) {
        const clampedDt = Math.min(0.05, Math.max(0.001, dt));
        const rawSpeed = mph ? tel.speedKmh * 0.621371 : tel.speedKmh;
        const maxScale = mph ? 200 : 300;

        // Frame-rate-independent snappy damping (~35ms response time: instant feel, zero jitter)
        const rpmAlpha = 1 - Math.exp(-32 * clampedDt);
        const spdAlpha = 1 - Math.exp(-28 * clampedDt);
        smoothRpm.current += (tel.rpm - smoothRpm.current) * rpmAlpha;
        smoothSpd.current += (rawSpeed - smoothSpd.current) * spdAlpha;

        const rpmFrac = Math.min(1, Math.max(0, smoothRpm.current / 7500));
        const spdFrac = Math.min(1, Math.max(0, smoothSpd.current / maxScale));

        if (rpmNeedleRef.current) {
          const deg = RPM_START_DEG + rpmFrac * RPM_SWEEP_DEG;
          rpmNeedleRef.current.style.transform = `rotate(${deg.toFixed(2)}deg)`;
        }
        if (spdNeedleRef.current) {
          const deg = SPD_START_DEG + spdFrac * SPD_SWEEP_DEG;
          spdNeedleRef.current.style.transform = `rotate(${deg.toFixed(2)}deg)`;
        }

        const c = lastText.current;

        const rpmStr = String(Math.round(tel.rpm));
        if (rpmDigitsRef.current && c.rpm !== rpmStr) {
          c.rpm = rpmStr;
          rpmDigitsRef.current.textContent = rpmStr;
        }
        const spdStr = rawSpeed.toFixed(1);
        if (spdDigitsRef.current && c.spd !== spdStr) {
          c.spd = spdStr;
          spdDigitsRef.current.textContent = spdStr;
        }
        const unitStr = mph ? 'MPH' : 'KM/H';
        if (spdUnitRef.current && c.unit !== unitStr) {
          c.unit = unitStr;
          spdUnitRef.current.textContent = unitStr;
        }
        if (gearRef.current && c.gear !== tel.gear) {
          c.gear = tel.gear;
          gearRef.current.textContent = tel.gear;
        }

        const thrStr = `${Math.round(tel.throttle * 100)}%`;
        if (thrRef.current && c.thr !== thrStr) {
          c.thr = thrStr;
          thrRef.current.textContent = thrStr;
        }
        const brkStr = `${Math.round(tel.brake * 100)}%`;
        if (brkRef.current && c.brk !== brkStr) {
          c.brk = brkStr;
          brkRef.current.textContent = brkStr;
        }
        if (absRef.current && c.absOn !== tel.absActive) {
          c.absOn = tel.absActive;
          absRef.current.textContent = tel.absActive ? 'ON' : 'OFF';
          absRef.current.classList.toggle('alert-red', tel.absActive);
        }

        // Throttle slower-changing telemetry cells (TEMP, FUEL, G-FORCE, LIGHTS, FPS) to ~12 Hz for readability
        slowTextTimer.current += clampedDt;
        if (slowTextTimer.current >= 0.08) {
          slowTextTimer.current = 0;

          const tempStr = `${tel.mech.engineTemp}°C`;
          const tempHot = tel.mech.engineTemp >= 110;
          if (tempRef.current && (c.temp !== tempStr || c.tempHot !== tempHot)) {
            c.temp = tempStr;
            c.tempHot = tempHot;
            tempRef.current.textContent = tempStr;
            tempRef.current.classList.toggle('alert-red', tempHot);
          }

          const fPct = Math.round(Math.max(0, Math.min(1, tel.mech.fuelLevel)) * 100);
          const fuelStr = `${fPct}%`;
          const fuelLow = fPct <= 18;
          if (fuelRef.current && (c.fuel !== fuelStr || c.fuelLow !== fuelLow)) {
            c.fuel = fuelStr;
            c.fuelLow = fuelLow;
            fuelRef.current.textContent = fuelStr;
            fuelRef.current.classList.toggle('alert-amber', fuelLow);
          }

          const gStr = `${tel.gForce.toFixed(2)}G`;
          if (gRef.current && c.g !== gStr) {
            c.g = gStr;
            gRef.current.textContent = gStr;
          }

          if (hlRef.current && c.hlOn !== tel.headlightsOn) {
            c.hlOn = tel.headlightsOn;
            hlRef.current.textContent = tel.headlightsOn ? 'ON' : 'OFF';
            hlRef.current.classList.toggle('alert-blue', tel.headlightsOn);
          }

          const fpsStr = `${Math.round(fps)}`;
          if (fpsRef.current && c.fps !== fpsStr) {
            c.fps = fpsStr;
            fpsRef.current.textContent = fpsStr;
          }
        }
      },
    }), []);

    // Left Pod (Tachometer) center=(78, 122), radius=58
    const rpmCx = 78;
    const rpmCy = 122;

    // Right Pod (Speedometer) center=(214, 108), radius=86
    const spdCx = 214;
    const spdCy = 108;

    return (
      <footer className="dual-cluster-hud" aria-label="Automotive Instrument Cluster">
        <div className="dual-cluster-dials">
          <svg viewBox="0 0 318 206" className="dual-cluster-svg" aria-hidden="true">
            <defs>
              <linearGradient id="bezelMetallic" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#d4d7dc" />
                <stop offset="45%" stopColor="#7a7f87" />
                <stop offset="100%" stopColor="#35383e" />
              </linearGradient>
              <linearGradient id="housingDark" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#262930" />
                <stop offset="100%" stopColor="#121418" />
              </linearGradient>
              <radialGradient id="dialCharcoal" cx="50%" cy="42%" r="60%">
                <stop offset="0%" stopColor="#24262b" />
                <stop offset="100%" stopColor="#141518" />
              </radialGradient>
            </defs>

            {/* Unified dark matte outer housing silhouette */}
            <circle cx={rpmCx} cy={rpmCy} r="69" fill="url(#housingDark)" />
            <circle cx={spdCx} cy={spdCy} r="99" fill="url(#housingDark)" />
            <rect x="78" y="135" width="136" height="56" rx="8" fill="url(#housingDark)" />

            {/* Left Pod: Brushed metallic bezel + charcoal face */}
            <circle cx={rpmCx} cy={rpmCy} r="61" fill="url(#dialCharcoal)" stroke="url(#bezelMetallic)" strokeWidth="4.5" />
            <circle cx={rpmCx} cy={rpmCy} r="56.5" fill="none" stroke="rgba(0,0,0,0.65)" strokeWidth="1.2" />

            {/* Left Pod Redline Arc (6.0 -> 7.5k RPM) */}
            {(() => {
              const a0 = ((RPM_START_DEG + (6.0 / 7.5) * RPM_SWEEP_DEG - 90) * Math.PI) / 180;
              const a1 = ((RPM_START_DEG + RPM_SWEEP_DEG - 90) * Math.PI) / 180;
              const r = 51.5;
              const x0 = rpmCx + Math.cos(a0) * r;
              const y0 = rpmCy + Math.sin(a0) * r;
              const x1 = rpmCx + Math.cos(a1) * r;
              const y1 = rpmCy + Math.sin(a1) * r;
              return (
                <path
                  d={`M ${x0.toFixed(1)} ${y0.toFixed(1)} A ${r} ${r} 0 0 1 ${x1.toFixed(1)} ${y1.toFixed(1)}`}
                  fill="none"
                  stroke="#dc2626"
                  strokeWidth="5.5"
                />
              );
            })()}

            {/* Left Pod Ticks & Numerals (0..7) */}
            {Array.from({ length: 31 }, (_, i) => {
              const val = i * 0.25;
              if (val > 7.5) return null;
              const isMajor = i % 4 === 0;
              const deg = RPM_START_DEG + (val / 7.5) * RPM_SWEEP_DEG;
              const rad = ((deg - 90) * Math.PI) / 180;
              const rIn = isMajor ? 45.5 : 49;
              const rOut = 54;
              const isRed = val >= 6.0;
              return (
                <g key={i}>
                  <line
                    x1={(rpmCx + Math.cos(rad) * rIn).toFixed(1)}
                    y1={(rpmCy + Math.sin(rad) * rIn).toFixed(1)}
                    x2={(rpmCx + Math.cos(rad) * rOut).toFixed(1)}
                    y2={(rpmCy + Math.sin(rad) * rOut).toFixed(1)}
                    stroke={isRed ? '#ef4444' : isMajor ? '#f1f5f9' : 'rgba(226,232,240,0.45)'}
                    strokeWidth={isMajor ? '1.8' : '0.9'}
                  />
                  {isMajor && val <= 7 && (
                    <text
                      x={(rpmCx + Math.cos(rad) * 37.5).toFixed(1)}
                      y={(rpmCy + Math.sin(rad) * 37.5 + 3.2).toFixed(1)}
                      textAnchor="middle"
                      fill={val >= 7 ? '#ef4444' : '#f1f5f9'}
                      fontSize="9.5"
                      fontWeight="700"
                      fontFamily="Inter, sans-serif"
                    >
                      {val}
                    </text>
                  )}
                </g>
              );
            })}
            <text x={rpmCx} y={rpmCy - 18} textAnchor="middle" fill="rgba(226,232,240,0.72)" fontSize="6.5" fontWeight="600" letterSpacing="0.04em">
              RPM x 1000
            </text>

            {/* Left Pod Orange Needle */}
            <g
              ref={rpmNeedleRef}
              className="cluster-needle-rpm"
              style={{ transformOrigin: `${rpmCx}px ${rpmCy}px`, transform: `rotate(${RPM_START_DEG}deg)` }}
            >
              <polygon
                points={`${rpmCx - 1.8},${rpmCy + 10} ${rpmCx + 1.8},${rpmCy + 10} ${rpmCx + 0.9},${rpmCy - 49} ${rpmCx - 0.9},${rpmCy - 49}`}
                fill="#f97316"
              />
              <circle cx={rpmCx} cy={rpmCy} r="5.5" fill="#181a1f" stroke="#52525b" strokeWidth="1.2" />
            </g>

            {/* Right Pod: Larger Speedometer overlapping cleanly */}
            <circle cx={spdCx} cy={spdCy} r="91" fill="url(#dialCharcoal)" stroke="url(#bezelMetallic)" strokeWidth="5.5" />
            <circle cx={spdCx} cy={spdCy} r="85.5" fill="none" stroke="rgba(0,0,0,0.7)" strokeWidth="1.4" />

            {/* Right Pod Ticks: Major white every 20, fine red minor ticks in between (0..300) */}
            {Array.from({ length: 76 }, (_, i) => {
              const val = i * 4; // 0..300 in steps of 4
              const isMajor = val % 20 === 0;
              const isMid = !isMajor && val % 10 === 0;
              const deg = SPD_START_DEG + (val / 300) * SPD_SWEEP_DEG;
              const rad = ((deg - 90) * Math.PI) / 180;
              const rIn = isMajor ? 72 : isMid ? 75.5 : 77.5;
              const rOut = 83;
              const isEndRed = val === 300;
              return (
                <g key={i}>
                  <line
                    x1={(spdCx + Math.cos(rad) * rIn).toFixed(1)}
                    y1={(spdCy + Math.sin(rad) * rIn).toFixed(1)}
                    x2={(spdCx + Math.cos(rad) * rOut).toFixed(1)}
                    y2={(spdCy + Math.sin(rad) * rOut).toFixed(1)}
                    stroke={isMajor ? (isEndRed ? '#ef4444' : '#f8fafc') : 'rgba(220, 38, 38, 0.72)'}
                    strokeWidth={isMajor ? '2.1' : '0.95'}
                  />
                  {isMajor && (
                    <text
                      x={(spdCx + Math.cos(rad) * 61).toFixed(1)}
                      y={(spdCy + Math.sin(rad) * 61 + 3.6).toFixed(1)}
                      textAnchor="middle"
                      fill="#f1f5f9"
                      fontSize="10.2"
                      fontWeight="700"
                      fontFamily="Inter, sans-serif"
                    >
                      {val}
                    </text>
                  )}
                </g>
              );
            })}
            <text x={spdCx} y={spdCy - 30} textAnchor="middle" fill="rgba(226,232,240,0.78)" fontSize="9.5" fontWeight="600">
              {speedUnitLabel}
            </text>

            {/* Right Pod Red Speed Needle with subtle center glow */}
            <circle cx={spdCx} cy={spdCy} r="12" fill="rgba(239, 68, 68, 0.14)" />
            <g
              ref={spdNeedleRef}
              className="cluster-needle-spd"
              style={{ transformOrigin: `${spdCx}px ${spdCy}px`, transform: `rotate(${SPD_START_DEG}deg)` }}
            >
              <polygon
                points={`${spdCx - 2.6},${spdCy + 16} ${spdCx + 2.6},${spdCy + 16} ${spdCx + 1.0},${spdCy - 78} ${spdCx - 1.0},${spdCy - 78}`}
                fill="#ef4444"
              />
              <circle cx={spdCx} cy={spdCy} r="8.5" fill="#181a1f" stroke="#3f3f46" strokeWidth="1.4" />
            </g>
          </svg>

          {/* Digital RPM Box (recessed at bottom of left pod) */}
          <div className="cluster-rpm-lcd">
            <span ref={rpmDigitsRef} className="cluster-rpm-val">850</span>
            <span className="cluster-lcd-unit">RPM</span>
          </div>

          {/* Digital Speed Box (recessed at bottom of right pod) */}
          <div className="cluster-spd-lcd">
            <span id="speed-value" ref={spdDigitsRef} className="cluster-spd-val">0.0</span>
            <span ref={spdUnitRef} className="cluster-lcd-unit">{speedUnitLabel.toUpperCase()}</span>
          </div>
        </div>

        {/* Compact Dark Recessed Telemetry Strip Underneath */}
        <div className="cluster-tele-grid">
          <div className="tele-cell">
            <span>GEAR:</span>
            <strong id="gear-badge" ref={gearRef}>N</strong>
          </div>
          <div className="tele-cell">
            <span>TEMP:</span>
            <strong ref={tempRef}>88°C</strong>
          </div>
          <div className="tele-cell">
            <span>THROTTLE:</span>
            <strong ref={thrRef}>0%</strong>
          </div>
          <div className="tele-cell">
            <span>BRAKE:</span>
            <strong ref={brkRef}>0%</strong>
          </div>
          <div className="tele-cell">
            <span>ABS:</span>
            <strong ref={absRef}>OFF</strong>
          </div>
          <div className="tele-cell">
            <span>FUEL:</span>
            <strong ref={fuelRef}>100%</strong>
          </div>
          <div className="tele-cell">
            <span>G-FORCE:</span>
            <strong ref={gRef}>0.00G</strong>
          </div>
          <div className="tele-cell">
            <span>LIGHTS:</span>
            <strong ref={hlRef}>OFF</strong>
          </div>
          <div className="tele-cell">
            <span>FPS:</span>
            <strong ref={fpsRef}>60</strong>
          </div>
        </div>
      </footer>
    );
  },
);
