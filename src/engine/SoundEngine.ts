import type { VehicleTelemetry } from './SoftBodyVehicle';

export type EngineSoundProfileId = 'bmw_s58' | 'mclaren_p1' | 'dodge_demon' | 'gts_v8';

interface EngineAcousticProfile {
  id: EngineSoundProfileId;
  redlineRpm: number;
  /** Harmonic multipliers relative to crank revolutions per second (RPM / 60). */
  orders: [number, number, number, number];
  /** Relative gain mix of the 4 harmonic oscillators. */
  mix: [number, number, number, number];
  /** Oscillator waveforms for the 4 combustion voices. */
  waves: [OscillatorType, OscillatorType, OscillatorType, OscillatorType];
  /** Exhaust lowpass base & top cutoff (Hz). */
  lpMin: number;
  lpMax: number;
  /** Resonant exhaust character peak (Hz, Q, dB). */
  peakFreq: number;
  peakQ: number;
  peakGainDb: number;
  /** Turbo or supercharger whine frequency range (Hz) and gain under full load. */
  forcedMinHz: number;
  forcedMaxHz: number;
  forcedGain: number;
  /** True for belt-driven supercharger (instant RPM-locked whine, e.g. Dodge Demon). */
  isSupercharger: boolean;
  /** Electric / hybrid whine gain (McLaren P1 IPAS). */
  hybridGain: number;
  /** Overrun pops & bangs intensity vs crisp BOV pschitt. */
  popsIntensity: number;
  bovIntensity: number;
  /** Drive saturation amount into waveshaper. */
  driveGain: number;
}

const PROFILES: Record<EngineSoundProfileId, EngineAcousticProfile> = {
  // BMW M4 CSL — S58 3.0L Inline-6 Bi-Turbo:
  // Deep, bassy, throaty, raspy low/mid-range growl (strong sub + 1.5x + 3.0x I6 firing orders),
  // capped lowpass so it NEVER sounds high-pitched or synthetic, plus sharp dry pops & bangs on lift.
  bmw_s58: {
    id: 'bmw_s58',
    redlineRpm: 7200,
    orders: [0.75, 1.5, 3.0, 4.5],
    mix: [0.42, 0.56, 0.36, 0.14],
    waves: ['triangle', 'sawtooth', 'triangle', 'sawtooth'],
    lpMin: 175,
    lpMax: 1550,
    peakFreq: 145,
    peakQ: 2.0,
    peakGainDb: 8.5,
    forcedMinHz: 1700,
    forcedMaxHz: 4600,
    forcedGain: 0.024,
    isSupercharger: false,
    hybridGain: 0.0,
    popsIntensity: 1.0,
    bovIntensity: 0.22,
    driveGain: 1.75,
  },
  // McLaren P1 — M838TQ 3.8L Flat-Plane-Crank Twin-Turbo V8 + IPAS Hybrid:
  // Sharp, nervous mechanical flat-plane V8 up to 8500 RPM, prominent twin-turbo spool,
  // crisp BOV "pschitt" on gear shifts, and subtle IPAS hybrid electric whine.
  mclaren_p1: {
    id: 'mclaren_p1',
    redlineRpm: 8500,
    orders: [1.5, 3.0, 6.0, 9.0],
    mix: [0.24, 0.52, 0.36, 0.18],
    waves: ['sawtooth', 'sawtooth', 'triangle', 'sawtooth'],
    lpMin: 320,
    lpMax: 3400,
    peakFreq: 420,
    peakQ: 2.4,
    peakGainDb: 7.0,
    forcedMinHz: 2200,
    forcedMaxHz: 6800,
    forcedGain: 0.072,
    isSupercharger: false,
    hybridGain: 0.032,
    popsIntensity: 0.45,
    bovIntensity: 1.0,
    driveGain: 1.8,
  },
  // Dodge Challenger SRT Demon — 6.2L Supercharged HEMI V8:
  // Heavy, thunderous low-end American cross-plane V8 rumble + unmistakable
  // belt-driven high-pitched supercharger whine rising linearly with RPM & throttle!
  dodge_demon: {
    id: 'dodge_demon',
    redlineRpm: 6500,
    orders: [0.5, 1.0, 2.0, 4.0],
    mix: [0.48, 0.62, 0.38, 0.16],
    waves: ['triangle', 'sawtooth', 'sawtooth', 'triangle'],
    lpMin: 150,
    lpMax: 1380,
    peakFreq: 98,
    peakQ: 1.9,
    peakGainDb: 9.5,
    forcedMinHz: 1350,
    forcedMaxHz: 5900,
    forcedGain: 0.095,
    isSupercharger: true,
    hybridGain: 0.0,
    popsIntensity: 0.65,
    bovIntensity: 0.15,
    driveGain: 1.95,
  },
  // Built-in GT-S — Cross-Plane V8
  gts_v8: {
    id: 'gts_v8',
    redlineRpm: 7200,
    orders: [0.75, 1.5, 3.0, 5.0],
    mix: [0.36, 0.52, 0.30, 0.14],
    waves: ['triangle', 'sawtooth', 'sawtooth', 'triangle'],
    lpMin: 210,
    lpMax: 1950,
    peakFreq: 155,
    peakQ: 2.0,
    peakGainDb: 6.0,
    forcedMinHz: 1650,
    forcedMaxHz: 4400,
    forcedGain: 0.02,
    isSupercharger: false,
    hybridGain: 0.0,
    popsIntensity: 0.55,
    bovIntensity: 0.2,
    driveGain: 1.45,
  },
};

/**
 * SoundEngine
 * -----------
 * 100% procedural Web Audio API synthesizer with ZERO per-frame allocations:
 * all oscillators, looping noise buffer, waveshaper, and biquad filters are
 * created once on first user interaction and modulated via AudioParam scheduling
 * for locked 60 FPS even on 4 GB RAM tablets.
 */
export class SoundEngine {
  private ctx: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private engineBusGain: GainNode | null = null;
  private effectsBusGain: GainNode | null = null;

  // Combustion bank (4 persistent oscillators)
  private oscs: OscillatorNode[] = [];
  private oscGains: GainNode[] = [];
  private engineDriveGain: GainNode | null = null;
  private exhaustLowpass: BiquadFilterNode | null = null;
  private exhaustPeak: BiquadFilterNode | null = null;
  private engineOutGain: GainNode | null = null;

  // Forced induction (twin-turbo spool OR belt-driven supercharger whine) & hybrid whine
  private forcedOsc: OscillatorNode | null = null;
  private forcedGain: GainNode | null = null;
  private hybridOsc: OscillatorNode | null = null;
  private hybridGain: GainNode | null = null;

  // Persistent looping noise source feeding tire screech, BOV, pops & crash crunch
  private noiseSource: AudioBufferSourceNode | null = null;
  private tireFilter: BiquadFilterNode | null = null;
  private tireGain: GainNode | null = null;
  private bovFilter: BiquadFilterNode | null = null;
  private bovGain: GainNode | null = null;
  private popFilter: BiquadFilterNode | null = null;
  private popGain: GainNode | null = null;

  // Crash metallic crunch + low impact thud
  private crashNoiseFilter: BiquadFilterNode | null = null;
  private crashNoiseGain: GainNode | null = null;
  private crashSubOsc: OscillatorNode | null = null;
  private crashSubGain: GainNode | null = null;

  private profile: EngineAcousticProfile = PROFILES.bmw_s58;
  private enabled = true;
  private unlocked = false;
  private masterVol = 0.75;
  private engineVol = 0.85;
  private effectsVol = 0.85;

  // Continuous state (zero allocations)
  private turboBoost = 0;
  private prevThrottle = 0;
  private prevShiftCount = 0;
  private prevBrokenBeams = 0;
  private popsRemaining = 0;
  private popCooldown = 0;
  private crashCooldown = 0;
  private revLimiterCooldown = 0;

  isEnabled(): boolean {
    return this.enabled;
  }

  setEnabled(on: boolean): void {
    this.enabled = on;
    this.applyVolumes();
    if (on && this.ctx && this.ctx.state === 'suspended') {
      void this.ctx.resume();
    }
  }

  setVolumes(master: number, engine: number, effects: number, muted: boolean): void {
    this.masterVol = Math.max(0, Math.min(1, master));
    this.engineVol = Math.max(0, Math.min(1, engine));
    this.effectsVol = Math.max(0, Math.min(1, effects));
    this.enabled = !muted;
    this.applyVolumes();
  }

  private applyVolumes(): void {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    const m = this.enabled ? this.masterVol * 0.55 : 0.0;
    this.masterGain?.gain.setTargetAtTime(m, t, 0.03);
    this.engineBusGain?.gain.setTargetAtTime(this.engineVol, t, 0.03);
    this.effectsBusGain?.gain.setTargetAtTime(this.effectsVol, t, 0.03);
  }

  setProfileFromCarName(name: string): void {
    let next: EngineAcousticProfile = PROFILES.gts_v8;
    if (/mclaren|p1|ferrari|lamborghini/i.test(name)) {
      next = PROFILES.mclaren_p1;
    } else if (/dodge|challenger|demon|hemi|hellcat/i.test(name)) {
      next = PROFILES.dodge_demon;
    } else if (/bmw|m4|csl|s58/i.test(name)) {
      next = PROFILES.bmw_s58;
    }
    this.profile = next;
    this.applyProfileParams();
  }

  /** Must be called from a user gesture (click / keydown / pointerdown). Safe to call repeatedly. */
  unlock(): void {
    if (this.unlocked) {
      if (this.ctx && this.ctx.state === 'suspended' && this.enabled) {
        void this.ctx.resume();
      }
      return;
    }
    const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    if (!AudioCtx) return;

    const ctx = new AudioCtx();
    this.ctx = ctx;
    this.unlocked = true;

    const master = ctx.createGain();
    master.gain.value = this.enabled ? this.masterVol * 0.55 : 0.0;
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -10;
    comp.knee.value = 12;
    comp.ratio.value = 6;
    comp.attack.value = 0.004;
    comp.release.value = 0.12;
    master.connect(comp);
    comp.connect(ctx.destination);
    this.masterGain = master;

    const engineBus = ctx.createGain();
    engineBus.gain.value = this.engineVol;
    engineBus.connect(master);
    this.engineBusGain = engineBus;

    const effectsBus = ctx.createGain();
    effectsBus.gain.value = this.effectsVol;
    effectsBus.connect(master);
    this.effectsBusGain = effectsBus;

    // 1. Engine combustion bank -> drive -> waveshaper -> resonant exhaust filters
    const drive = ctx.createGain();
    drive.gain.value = this.profile.driveGain;
    const shaper = ctx.createWaveShaper();
    shaper.curve = buildExhaustCurve();
    shaper.oversample = '2x';

    const peak = ctx.createBiquadFilter();
    peak.type = 'peaking';
    peak.frequency.value = this.profile.peakFreq;
    peak.Q.value = this.profile.peakQ;
    peak.gain.value = this.profile.peakGainDb;

    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.value = this.profile.lpMin;
    lp.Q.value = 1.25;

    const engOut = ctx.createGain();
    engOut.gain.value = 0.30;

    drive.connect(shaper);
    shaper.connect(peak);
    peak.connect(lp);
    lp.connect(engOut);
    engOut.connect(engineBus);

    this.engineDriveGain = drive;
    this.exhaustPeak = peak;
    this.exhaustLowpass = lp;
    this.engineOutGain = engOut;

    for (let i = 0; i < 4; i++) {
      const osc = ctx.createOscillator();
      osc.type = this.profile.waves[i];
      osc.frequency.value = 50 * (i + 1);
      const g = ctx.createGain();
      g.gain.value = this.profile.mix[i];
      osc.connect(g);
      g.connect(drive);
      osc.start();
      this.oscs.push(osc);
      this.oscGains.push(g);
    }

    // 2. Forced Induction (Twin-Turbo Spool OR Dodge Demon Supercharger Whine)
    const forcedOsc = ctx.createOscillator();
    forcedOsc.type = 'sine';
    forcedOsc.frequency.value = this.profile.forcedMinHz;
    const forcedBp = ctx.createBiquadFilter();
    forcedBp.type = 'bandpass';
    forcedBp.frequency.value = 3000;
    forcedBp.Q.value = 1.6;
    const forcedGain = ctx.createGain();
    forcedGain.gain.value = 0.0;
    forcedOsc.connect(forcedBp);
    forcedBp.connect(forcedGain);
    forcedGain.connect(engineBus);
    forcedOsc.start();
    this.forcedOsc = forcedOsc;
    this.forcedGain = forcedGain;

    // 3. Hybrid IPAS Electric Whine (McLaren P1)
    const hybridOsc = ctx.createOscillator();
    hybridOsc.type = 'sine';
    hybridOsc.frequency.value = 420;
    const hybridGain = ctx.createGain();
    hybridGain.gain.value = 0.0;
    hybridOsc.connect(hybridGain);
    hybridGain.connect(engineBus);
    hybridOsc.start();
    this.hybridOsc = hybridOsc;
    this.hybridGain = hybridGain;

    // 4. Single persistent looping white noise buffer (2 seconds) for tires, BOV, pops & crash
    const sampleRate = ctx.sampleRate;
    const bufLen = sampleRate * 2;
    const noiseBuf = ctx.createBuffer(1, bufLen, sampleRate);
    const data = noiseBuf.getChannelData(0);
    let seed = 987654321;
    for (let i = 0; i < bufLen; i++) {
      seed = (seed * 1664525 + 1013904223) | 0;
      data[i] = seed / 2147483648;
    }
    const noise = ctx.createBufferSource();
    noise.buffer = noiseBuf;
    noise.loop = true;

    // 4a. Tire screech branch -> effectsBus
    const tireFilter = ctx.createBiquadFilter();
    tireFilter.type = 'bandpass';
    tireFilter.frequency.value = 1380;
    tireFilter.Q.value = 8.5;
    const tireGain = ctx.createGain();
    tireGain.gain.value = 0.0;
    noise.connect(tireFilter);
    tireFilter.connect(tireGain);
    tireGain.connect(effectsBus);
    this.tireFilter = tireFilter;
    this.tireGain = tireGain;

    // 4b. Blow-Off Valve ("pschitt") branch -> engineBus
    const bovFilter = ctx.createBiquadFilter();
    bovFilter.type = 'bandpass';
    bovFilter.frequency.value = 2600;
    bovFilter.Q.value = 2.8;
    const bovGain = ctx.createGain();
    bovGain.gain.value = 0.0;
    noise.connect(bovFilter);
    bovFilter.connect(bovGain);
    bovGain.connect(engineBus);
    this.bovFilter = bovFilter;
    this.bovGain = bovGain;

    // 4c. Exhaust Pops & Bangs branch -> engineBus
    const popFilter = ctx.createBiquadFilter();
    popFilter.type = 'bandpass';
    popFilter.frequency.value = 210;
    popFilter.Q.value = 1.5;
    const popGain = ctx.createGain();
    popGain.gain.value = 0.0;
    noise.connect(popFilter);
    popFilter.connect(popGain);
    popGain.connect(engineBus);
    this.popFilter = popFilter;
    this.popGain = popGain;

    // 4d. Crash metallic crunch + sub thud -> effectsBus
    const crashFilter = ctx.createBiquadFilter();
    crashFilter.type = 'bandpass';
    crashFilter.frequency.value = 820;
    crashFilter.Q.value = 1.1;
    const crashGain = ctx.createGain();
    crashGain.gain.value = 0.0;
    noise.connect(crashFilter);
    crashFilter.connect(crashGain);
    crashGain.connect(effectsBus);
    this.crashNoiseFilter = crashFilter;
    this.crashNoiseGain = crashGain;

    const crashSub = ctx.createOscillator();
    crashSub.type = 'triangle';
    crashSub.frequency.value = 68;
    const crashSubGain = ctx.createGain();
    crashSubGain.gain.value = 0.0;
    crashSub.connect(crashSubGain);
    crashSubGain.connect(effectsBus);
    crashSub.start();
    this.crashSubOsc = crashSub;
    this.crashSubGain = crashSubGain;

    noise.start();
    this.noiseSource = noise;

    this.applyProfileParams();
    this.applyVolumes();
    if (ctx.state === 'suspended' && this.enabled) {
      void ctx.resume();
    }
  }

  private applyProfileParams(): void {
    if (!this.ctx || !this.unlocked) return;
    const t = this.ctx.currentTime;
    const P = this.profile;
    for (let i = 0; i < this.oscs.length; i++) {
      this.oscs[i].type = P.waves[i];
      this.oscGains[i].gain.setTargetAtTime(P.mix[i], t, 0.05);
    }
    if (this.forcedOsc) {
      this.forcedOsc.type = P.isSupercharger ? 'triangle' : 'sine';
    }
    this.exhaustPeak?.frequency.setTargetAtTime(P.peakFreq, t, 0.05);
    this.exhaustPeak?.Q.setTargetAtTime(P.peakQ, t, 0.05);
    this.exhaustPeak?.gain.setTargetAtTime(P.peakGainDb, t, 0.05);
    this.engineDriveGain?.gain.setTargetAtTime(P.driveGain, t, 0.05);
  }

  /** Called once per animation frame. Zero allocations. */
  update(tel: VehicleTelemetry, dt: number, paused: boolean, timeScale: number): void {
    if (!this.ctx || !this.unlocked || !this.enabled) return;
    const t = this.ctx.currentTime;

    if (paused) {
      this.engineOutGain?.gain.setTargetAtTime(0.0, t, 0.04);
      this.forcedGain?.gain.setTargetAtTime(0.0, t, 0.04);
      this.hybridGain?.gain.setTargetAtTime(0.0, t, 0.04);
      this.tireGain?.gain.setTargetAtTime(0.0, t, 0.04);
      return;
    }

    const P = this.profile;
    const rpm = Math.max(750, Math.min(P.redlineRpm + 200, tel.rpm));
    const rpmNorm = Math.max(0, Math.min(1, (rpm - 800) / (P.redlineRpm - 800)));
    const throttle = tel.throttle;
    const pitchScale = Math.max(0.35, Math.min(1.0, timeScale));
    const crankHz = (rpm / 60) * pitchScale;

    // 1. Update 4 combustion harmonic oscillators
    for (let i = 0; i < 4; i++) {
      const f = Math.max(22, crankHz * P.orders[i]);
      this.oscs[i].frequency.setTargetAtTime(f, t, 0.018);
    }
    const topHarmonic = P.mix[3] * (0.55 + rpmNorm * 0.95 + throttle * 0.35);
    this.oscGains[3].gain.setTargetAtTime(topHarmonic, t, 0.025);

    // Exhaust lowpass filter opens smoothly with RPM & throttle load
    const lpCutoff = P.lpMin + (P.lpMax - P.lpMin) * Math.pow(rpmNorm, 0.9) * (0.52 + 0.48 * throttle);
    this.exhaustLowpass?.frequency.setTargetAtTime(lpCutoff, t, 0.025);
    this.engineDriveGain?.gain.setTargetAtTime(P.driveGain * (0.76 + 0.52 * throttle + 0.22 * rpmNorm), t, 0.03);
    this.engineOutGain?.gain.setTargetAtTime(0.24 + 0.16 * throttle + 0.10 * rpmNorm, t, 0.03);

    // 2. Forced Induction: belt-driven Supercharger whine (Dodge Demon) OR Twin-Turbo spool (BMW / McLaren)
    if (this.forcedOsc && this.forcedGain) {
      if (P.isSupercharger) {
        // Instant RPM-locked high-pitched supercharger scream on throttle!
        const scHz = (P.forcedMinHz + (P.forcedMaxHz - P.forcedMinHz) * rpmNorm) * pitchScale;
        const scAmp = P.forcedGain * (0.12 + 0.88 * throttle) * (0.28 + 0.72 * rpmNorm);
        this.forcedOsc.frequency.setTargetAtTime(scHz, t, 0.015);
        this.forcedGain.gain.setTargetAtTime(scAmp, t, 0.02);
      } else {
        const targetBoost = throttle * Math.max(0, Math.min(1, (rpm - 1800) / (P.redlineRpm - 2200)));
        const spoolRate = targetBoost > this.turboBoost ? 3.5 : 5.5;
        this.turboBoost += (targetBoost - this.turboBoost) * Math.min(1, dt * spoolRate);
        const turboHz = (P.forcedMinHz + (P.forcedMaxHz - P.forcedMinHz) * this.turboBoost) * pitchScale;
        this.forcedOsc.frequency.setTargetAtTime(turboHz, t, 0.025);
        this.forcedGain.gain.setTargetAtTime(P.forcedGain * this.turboBoost * (0.4 + 0.6 * throttle), t, 0.03);
      }
    }

    // 3. McLaren P1 IPAS electric/hybrid whine
    if (this.hybridOsc && this.hybridGain) {
      if (P.hybridGain > 0) {
        const speedNorm = Math.min(1, tel.speedKmh / 220);
        const hybHz = (380 + speedNorm * 2100 + rpmNorm * 650) * pitchScale;
        const hybAmp = P.hybridGain * (0.25 + 0.75 * throttle) * (0.35 + 0.65 * Math.min(1, tel.speedKmh / 18 + throttle));
        this.hybridOsc.frequency.setTargetAtTime(hybHz, t, 0.03);
        this.hybridGain.gain.setTargetAtTime(hybAmp, t, 0.04);
      } else {
        this.hybridGain.gain.setTargetAtTime(0.0, t, 0.05);
      }
    }

    // 4. Gear shift -> crisp Blow-Off Valve ("pschitt") & upshift exhaust crack
    if (tel.shiftCount !== this.prevShiftCount) {
      this.prevShiftCount = tel.shiftCount;
      if (this.turboBoost > 0.25 && P.bovIntensity > 0) {
        this.triggerBov(t, P.bovIntensity * Math.min(1, 0.45 + this.turboBoost * 0.7));
      }
      if (rpm > 4200) {
        this.triggerPop(t, 0.38 + 0.35 * P.popsIntensity, 240);
      }
    }

    // 5. Sharp throttle lift > 3800 RPM -> BMW M4 CSL dry overrun Pops & Bangs + BOV
    const throttleDrop = this.prevThrottle - throttle;
    if (throttleDrop > 0.45 && rpm > 3800) {
      if (P.popsIntensity > 0.2) {
        this.popsRemaining = P.id === 'bmw_s58' ? 4 : 2;
        this.popCooldown = 0.045;
      }
      if (this.turboBoost > 0.3 && P.bovIntensity > 0) {
        this.triggerBov(t, P.bovIntensity * this.turboBoost * 0.85);
      }
    }
    this.prevThrottle = throttle;

    if (this.popsRemaining > 0) {
      this.popCooldown -= dt;
      if (this.popCooldown <= 0) {
        this.triggerPop(t, (0.42 + this.popsRemaining * 0.09) * P.popsIntensity, 165 + this.popsRemaining * 30);
        this.popsRemaining--;
        this.popCooldown = 0.075 + (this.popsRemaining % 2) * 0.045;
      }
    }

    // Rev-limiter crack near redline
    if (this.revLimiterCooldown > 0) this.revLimiterCooldown -= dt;
    if (rpm > P.redlineRpm - 180 && throttle > 0.7 && this.revLimiterCooldown <= 0) {
      this.triggerPop(t, 0.45, 290);
      this.revLimiterCooldown = 0.14;
    }

    // 6. Tire screech modulated by lateral slip & hard braking
    if (this.tireGain && this.tireFilter) {
      const slipMag = Math.max(0, tel.maxTireSlip - 0.14);
      const speedGate = Math.min(1, tel.speedKmh / 14);
      const screech = Math.min(1, slipMag * 2.4 + (tel.absActive ? 0.28 : 0)) * speedGate;
      this.tireFilter.frequency.setTargetAtTime(1180 + Math.min(620, tel.speedKmh * 4.5), t, 0.04);
      this.tireGain.gain.setTargetAtTime(screech * 0.16, t, 0.03);
    }

    // 7. Metallic crash impact proportional to G-force & newly broken beams
    if (this.crashCooldown > 0) this.crashCooldown -= dt;
    const newBroken = Math.max(0, tel.brokenBeams - this.prevBrokenBeams);
    this.prevBrokenBeams = tel.brokenBeams;
    if ((tel.gForce > 4.2 || newBroken >= 2) && this.crashCooldown <= 0) {
      const severity = Math.min(1, Math.max((tel.gForce - 3.5) / 22, newBroken / 10));
      this.triggerCrash(t, severity);
      this.crashCooldown = 0.11;
    }
  }

  private triggerBov(t: number, intensity: number): void {
    if (!this.bovGain || !this.bovFilter) return;
    this.bovFilter.frequency.cancelScheduledValues(t);
    this.bovFilter.frequency.setValueAtTime(3400, t);
    this.bovFilter.frequency.exponentialRampToValueAtTime(1450, t + 0.19);
    this.bovGain.gain.cancelScheduledValues(t);
    this.bovGain.gain.setValueAtTime(0.0001, t);
    this.bovGain.gain.linearRampToValueAtTime(0.24 * intensity, t + 0.015);
    this.bovGain.gain.exponentialRampToValueAtTime(0.0001, t + 0.22);
  }

  private triggerPop(t: number, intensity: number, freq: number): void {
    if (!this.popGain || !this.popFilter) return;
    this.popFilter.frequency.setValueAtTime(freq, t);
    this.popGain.gain.cancelScheduledValues(t);
    this.popGain.gain.setValueAtTime(0.0001, t);
    this.popGain.gain.linearRampToValueAtTime(0.48 * intensity, t + 0.005);
    this.popGain.gain.exponentialRampToValueAtTime(0.0001, t + 0.085);
  }

  private triggerCrash(t: number, severity: number): void {
    if (!this.crashNoiseGain || !this.crashNoiseFilter || !this.crashSubOsc || !this.crashSubGain) return;
    const dur = 0.16 + severity * 0.28;
    this.crashNoiseFilter.frequency.cancelScheduledValues(t);
    this.crashNoiseFilter.frequency.setValueAtTime(1150 + severity * 650, t);
    this.crashNoiseFilter.frequency.exponentialRampToValueAtTime(280, t + dur);
    this.crashNoiseGain.gain.cancelScheduledValues(t);
    this.crashNoiseGain.gain.setValueAtTime(0.0001, t);
    this.crashNoiseGain.gain.linearRampToValueAtTime(0.55 * severity, t + 0.006);
    this.crashNoiseGain.gain.exponentialRampToValueAtTime(0.0001, t + dur);

    this.crashSubOsc.frequency.cancelScheduledValues(t);
    this.crashSubOsc.frequency.setValueAtTime(105 + severity * 35, t);
    this.crashSubOsc.frequency.exponentialRampToValueAtTime(34, t + dur * 0.9);
    this.crashSubGain.gain.cancelScheduledValues(t);
    this.crashSubGain.gain.setValueAtTime(0.0001, t);
    this.crashSubGain.gain.linearRampToValueAtTime(0.65 * severity, t + 0.005);
    this.crashSubGain.gain.exponentialRampToValueAtTime(0.0001, t + dur * 0.95);
  }

  dispose(): void {
    try {
      for (const o of this.oscs) o.stop();
      this.forcedOsc?.stop();
      this.hybridOsc?.stop();
      this.noiseSource?.stop();
      this.crashSubOsc?.stop();
      void this.ctx?.close();
    } catch { /* ignore */ }
    this.oscs = [];
    this.oscGains = [];
    this.ctx = null;
    this.unlocked = false;
  }
}

function buildExhaustCurve(): Float32Array<ArrayBuffer> {
  const n = 512;
  const curve = new Float32Array(new ArrayBuffer(n * 4));
  for (let i = 0; i < n; i++) {
    const x = (i * 2) / n - 1;
    // Deep, warm asymmetric exhaust saturation
    curve[i] = Math.tanh(x * 2.6 + 0.22 * x * x);
  }
  return curve;
}
