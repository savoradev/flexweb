import React, { useEffect, useRef, useState } from 'react';
import { CameraMode, CarOption, EngineCore, EngineStats } from './engine/EngineCore';
import { useDeviceProfile } from './utils/DeviceDetector';
import {
  discoverPublicCars,
  fetchPublicCarsFromServer,
  persistCarToPublicFolder,
} from './engine/PublicCarScanner';
import {
  deleteSavedCarGLB,
  getLastSelectedCarId,
  listSavedCarsMetadata,
  saveCarBufferToDB,
  setLastSelectedCarId,
} from './engine/SavedCarsDB';
import { GITHUB_CARS } from './engine/ZipGltfLoader';
import { VehicleDiagnostics } from './components/VehicleDiagnostics';
import { MainMenu } from './components/MainMenu';
import { CarSelectorModal } from './components/CarSelectorModal';
import { SettingsModal } from './components/SettingsModal';
import { PauseMenu } from './components/PauseMenu';
import { ArenaMapOverlay } from './components/ArenaMapOverlay';
import type { ArenaMapZone } from './engine/CrashArena';
import {
  DEFAULT_SETTINGS,
  loadSimulatorSettings,
  saveSimulatorSettings,
  SimulatorSettings,
} from './types/settings';

const GITHUB_CAR_OPTIONS: CarOption[] = GITHUB_CARS.map((g) => ({
  id: g.id,
  name: g.name,
  remoteUrls: g.remoteUrls,
  attribution: g.attribution,
}));

const CAMERA_MODES: { id: CameraMode; label: string }[] = [
  { id: 'chase', label: 'Chase' },
  { id: 'interior', label: 'Interior' },
  { id: 'hood', label: 'Hood' },
  { id: 'orbit', label: 'Orbit' },
  { id: 'bumper', label: 'Bumper' },
];

type ScreenState = 'menu' | 'maps' | 'cars' | 'settings' | 'playing' | 'paused' | 'mapview';

interface LoadingState {
  title: string;
  step: string;
  pct: number;
}

const STUDIO_PORTRAIT_CACHE_KEY = 'flexweb_studio_portraits_v2';

function loadCachedStudioPortraits(): Record<string, string> {
  try {
    const raw = localStorage.getItem(STUDIO_PORTRAIT_CACHE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Record<string, string>;
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

export function App(): React.ReactElement {
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const engineRef = useRef<EngineCore | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const loadSeqRef = useRef(0);

  const [screen, setScreen] = useState<ScreenState>('menu');
  const [returnScreen, setReturnScreen] = useState<'menu' | 'paused'>('menu');
  const screenRef = useRef<ScreenState>('menu');
  screenRef.current = screen;

  const [settings, setSettings] = useState<SimulatorSettings>(() => loadSimulatorSettings());
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  const [menuSlides, setMenuSlides] = useState<string[]>([]);
  const [carThumbs, setCarThumbs] = useState<Record<string, string>>(() => loadCachedStudioPortraits());
  const carThumbsRef = useRef(carThumbs);
  carThumbsRef.current = carThumbs;

  const initialCars: CarOption[] = [
    ...GITHUB_CAR_OPTIONS,
    ...discoverPublicCars()
      .filter((c) => !/(bmw_m4_f82|dodge_challenger_srt_demon)\.glb$/i.test(c.url))
      .map((c) => ({ id: c.id, name: c.name, url: c.url, bytes: c.bytes })),
  ];
  const [carOptions, setCarOptions] = useState<CarOption[]>(initialCars);
  const [chosenCarId, setChosenCarId] = useState<string>(() => {
    const saved = getLastSelectedCarId();
    return saved && !saved.startsWith('builtin-') ? saved : GITHUB_CAR_OPTIONS[0].id;
  });
  const [stats, setStats] = useState<EngineStats | null>(null);
  // Start with NO loading screen on Main Menu — only show after user selects/confirms a car
  const [loadingState, setLoadingState] = useState<LoadingState | null>(null);
  const [draggingFile, setDraggingFile] = useState(false);

  const device = useDeviceProfile();
  const isTouchDevice = device.isTouch;
  const isPhone = device.deviceType === 'phone';
  const isPhoneLandscape = isPhone && !device.isPortrait;
  const pausedBeforeLandscapeRef = useRef<boolean | null>(null);
  const [diagCollapsed, setDiagCollapsed] = useState(() => device.deviceType === 'phone');
  const [touchEnabled, setTouchEnabled] = useState(
    () => device.isTouch || device.deviceType === 'phone' || device.deviceType === 'tablet',
  );
  const [touchPressed, setTouchPressed] = useState({ left: false, right: false, throttle: false, brake: false, handbrake: false });

  useEffect(() => {
    setTouchEnabled(device.isTouch || device.deviceType === 'phone' || device.deviceType === 'tablet');
    if (device.deviceType === 'phone') {
      setDiagCollapsed(true);
    }
  }, [device.deviceType, device.isTouch]);

  // Attempt native Screen Orientation API portrait lock on mobile phones
  useEffect(() => {
    if (!isPhone) return;
    const lockPortrait = () => {
      try {
        const ori = window.screen.orientation as ScreenOrientation & { lock?: (o: string) => Promise<void> };
        if (ori && typeof ori.lock === 'function') {
          void ori.lock('portrait-primary').catch(() => { /* ignored on browsers requiring fullscreen */ });
        }
      } catch {
        // unsupported
      }
    };
    lockPortrait();
    window.addEventListener('pointerdown', lockPortrait, { once: true, passive: true });
    return () => window.removeEventListener('pointerdown', lockPortrait);
  }, [isPhone]);

  // Auto-pause simulation when a phone is rotated to landscape, resume when back in portrait
  useEffect(() => {
    const eng = engineRef.current;
    if (!eng) return;
    if (isPhoneLandscape) {
      if (pausedBeforeLandscapeRef.current === null) {
        pausedBeforeLandscapeRef.current = eng.loop.paused;
      }
      eng.loop.paused = true;
    } else if (pausedBeforeLandscapeRef.current !== null) {
      eng.loop.paused = pausedBeforeLandscapeRef.current;
      pausedBeforeLandscapeRef.current = null;
    }
  }, [isPhoneLandscape]);

  const [isFullscreen, setIsFullscreen] = useState(
    () => typeof document !== 'undefined' && Boolean(document.fullscreenElement),
  );

  const enterFullscreen = () => {
    try {
      const el = document.documentElement as HTMLElement & {
        webkitRequestFullscreen?: () => Promise<void> | void;
      };
      if (!document.fullscreenElement) {
        if (typeof el.requestFullscreen === 'function') {
          void el.requestFullscreen({ navigationUI: 'hide' }).catch(() => {
            void el.requestFullscreen().catch(() => {});
          });
        } else if (typeof el.webkitRequestFullscreen === 'function') {
          el.webkitRequestFullscreen();
        }
      }
      const orient = window.screen?.orientation as unknown as { lock?: (mode: string) => Promise<void> } | undefined;
      if (orient && typeof orient.lock === 'function') {
        void orient.lock('landscape').catch(() => {});
      }
    } catch {
      // Ignore browser restrictions
    }
  };

  const toggleFullscreen = () => {
    try {
      if (document.fullscreenElement) {
        void document.exitFullscreen?.().catch(() => {});
      } else {
        enterFullscreen();
      }
    } catch {
      // Ignore browser restrictions
    }
  };

  useEffect(() => {
    const onFsChange = () => setIsFullscreen(Boolean(document.fullscreenElement));
    document.addEventListener('fullscreenchange', onFsChange);
    document.addEventListener('webkitfullscreenchange', onFsChange);
    return () => {
      document.removeEventListener('fullscreenchange', onFsChange);
      document.removeEventListener('webkitfullscreenchange', onFsChange);
    };
  }, []);

  // Automatically request true Chrome Android fullscreen + landscape lock on first user interaction
  useEffect(() => {
    if (!isTouchDevice) return;
    let triggered = false;
    const onFirstTap = () => {
      if (triggered) return;
      triggered = true;
      enterFullscreen();
      window.removeEventListener('pointerdown', onFirstTap);
      window.removeEventListener('touchstart', onFirstTap);
    };
    window.addEventListener('pointerdown', onFirstTap, { passive: true });
    window.addEventListener('touchstart', onFirstTap, { passive: true });
    return () => {
      window.removeEventListener('pointerdown', onFirstTap);
      window.removeEventListener('touchstart', onFirstTap);
    };
  }, [isTouchDevice]);

  const keyState = useRef({ up: false, down: false, left: false, right: false, space: false });
  const touchState = useRef({ left: false, right: false, throttle: false, brake: false, handbrake: false });

  const updateSettings = (next: SimulatorSettings) => {
    setSettings(next);
    saveSimulatorSettings(next);
    engineRef.current?.applySettings(next);
  };

  const analogTouchRef = useRef({ steer: 0, throttle: 0, brake: 0 });
  const wheelPtrRef = useRef<{ id: number; startX: number } | null>(null);
  const [analogUI, setAnalogUI] = useState({ steer: 0, throttle: 0, brake: 0 });

  const syncVehicleInputs = () => {
    const eng = engineRef.current;
    if (!eng) return;
    if (screenRef.current !== 'playing') {
      eng.vehicle.input.throttle = 0;
      eng.vehicle.input.brake = 0;
      eng.vehicle.input.steer = 0;
      eng.vehicle.input.handbrake = 0;
      return;
    }
    const k = keyState.current;
    const t = touchState.current;
    const a = analogTouchRef.current;
    const inp = eng.vehicle.input;
    inp.throttle = Math.max(k.up ? 1 : 0, t.throttle ? 1 : 0, a.throttle);
    inp.brake = Math.max(k.down ? 1 : 0, t.brake ? 1 : 0, a.brake);
    inp.handbrake = k.space || t.handbrake ? 1 : 0;
    const keySteer = (k.right || t.right ? 1 : 0) - (k.left || t.left ? 1 : 0);
    inp.steer = Math.abs(a.steer) > 0.01 ? a.steer : keySteer;
  };

  const saveStudioThumb = (carId: string, shot: string) => {
    if (!shot) return;
    setCarThumbs((prev) => {
      const next = { ...prev, [carId]: shot };
      try {
        localStorage.setItem(STUDIO_PORTRAIT_CACHE_KEY, JSON.stringify(next));
      } catch {
        // Ignore storage quota errors
      }
      return next;
    });
  };

  const captureCarThumb = (carId: string) => {
    const eng = engineRef.current;
    if (!eng) return;
    window.setTimeout(() => {
      if (!engineRef.current || engineRef.current.activeCarId !== carId) return;
      const shot = engineRef.current.captureStudioPortrait(0.88);
      if (shot) saveStudioThumb(carId, shot);
    }, 80);
  };

  const handleSelectCar = async (opt: CarOption) => {
    const eng = engineRef.current;
    if (!eng) return;
    setChosenCarId(opt.id);
    setLastSelectedCarId(opt.id);
    if (eng.activeCarId === opt.id) {
      captureCarThumb(opt.id);
      return;
    }
    const seq = ++loadSeqRef.current;
    const cleanName = opt.name.replace(/\s*\((public\/|Saved)\)$/i, '');
    setLoadingState({ title: cleanName, step: 'Loading', pct: 10 });
    try {
      await eng.selectCar(opt, (step, pct) => {
        if (seq === loadSeqRef.current) {
          setLoadingState({ title: cleanName, step, pct });
        }
      });
      if (seq === loadSeqRef.current) {
        eng.applySettings(settingsRef.current);
        setStats(eng.collectStats());
        const shot = eng.captureStudioPortrait(0.88);
        if (shot) saveStudioThumb(opt.id, shot);
      }
    } catch {
      // Handled gracefully
    } finally {
      if (seq === loadSeqRef.current) {
        setLoadingState(null);
      }
    }
  };

  // Sync Engine demo mode, overhead map zoom & pause state whenever screen changes
  useEffect(() => {
    const eng = engineRef.current;
    if (!eng) return;
    const inMenu = screen === 'menu' || screen === 'maps' || (screen === 'cars' && returnScreen === 'menu') || (screen === 'settings' && returnScreen === 'menu');
    const isPausedOverlay = screen === 'paused' || (screen === 'cars' && returnScreen === 'paused') || (screen === 'settings' && returnScreen === 'paused');
    eng.setMenuDemoMode(inMenu);
    eng.setOverheadMapMode(screen === 'mapview');
    eng.loop.paused = isPausedOverlay;
    if (screen === 'playing' || screen === 'mapview') {
      eng.loop.paused = false;
    }
    if (screen !== 'playing') {
      keyState.current = { up: false, down: false, left: false, right: false, space: false };
      syncVehicleInputs();
    }
  }, [screen, returnScreen]);

  // Automatic background crash showcase cycling BMW M4 CSL, Dodge Demon & McLaren P1 across the updated 2 km Proving Ground
  useEffect(() => {
    if (screen !== 'menu' && screen !== 'maps') return;
    let cancelled = false;
    let busy = false;
    let cycleIdx = 0;
    const demoCars = [
      GITHUB_CAR_OPTIONS[0], // BMW M4 CSL
      GITHUB_CAR_OPTIONS[2] ?? GITHUB_CAR_OPTIONS[0], // Dodge Challenger SRT Demon
      GITHUB_CAR_OPTIONS[1] ?? GITHUB_CAR_OPTIONS[0], // McLaren P1
    ].filter(Boolean);
    const demoZoneIds = ['wall', 'double-press', 'offset', 'pendulums', 'pole', 'rollers', 'mega-canyon', 'small-overlap'];

    const runCrashCycle = async () => {
      const eng = engineRef.current;
      if (!eng || cancelled || busy || loadingState) return;
      busy = true;
      const startLoadSeq = loadSeqRef.current;
      try {
        const car = demoCars[cycleIdx % demoCars.length];
        const zid = demoZoneIds[cycleIdx % demoZoneIds.length];
        cycleIdx++;

        if (car && eng.activeCarId !== car.id) {
          await eng.selectCar(car);
          if (cancelled || startLoadSeq !== loadSeqRef.current) return;
          if (screenRef.current !== 'menu' && screenRef.current !== 'maps') return;
          eng.applySettings(settingsRef.current);
          setStats(eng.collectStats());
        }

        // Capture pure-white underground studio front portrait if not yet cached
        if (car && !carThumbsRef.current[car.id]) {
          const portrait = eng.captureStudioPortrait(0.80);
          if (portrait) saveStudioThumb(car.id, portrait);
        }

        const zone = eng.arena.mapZones.find((z) => z.id === zid);
        if (zone) {
          eng.reset(zone.spawnX, zone.spawnZ, zone.yaw, zone.defaultSpeedKmh, Boolean(zone.lateral));
        } else {
          const preset = eng.arena.presets[0];
          if (preset) eng.launchPreset(preset);
        }

        // Capture arena thumbnail for the PLAY map card 2.0s after launch (post-impact with fluid leaks & steam)
        window.setTimeout(() => {
          if (cancelled || !engineRef.current) return;
          if (screenRef.current !== 'menu' && screenRef.current !== 'maps') return;
          const shot = engineRef.current.captureScreenshot(0.78);
          if (shot) {
            setMenuSlides((prev) => {
              const next = [...prev, shot];
              return next.length > 4 ? next.slice(next.length - 4) : next;
            });
          }
        }, 2000);
      } catch {
        // Keep background showcase resilient
      } finally {
        busy = false;
      }
    };

    const startTimer = window.setTimeout(() => { void runCrashCycle(); }, 450);
    const interval = window.setInterval(() => { void runCrashCycle(); }, 6800);
    return () => {
      cancelled = true;
      window.clearTimeout(startTimer);
      window.clearInterval(interval);
    };
  }, [screen, loadingState]);

  const [headlightsOn, setHeadlightsOn] = useState(false);
  const hudSpeedRef = useRef<HTMLSpanElement | null>(null);
  const hudGearRef = useRef<HTMLDivElement | null>(null);
  const hudRpmArcRef = useRef<SVGCircleElement | null>(null);
  const hudNeedleRef = useRef<SVGGElement | null>(null);
  const hudFuelFillRef = useRef<HTMLDivElement | null>(null);
  const hudTempFillRef = useRef<HTMLDivElement | null>(null);
  const hudTempTextRef = useRef<HTMLSpanElement | null>(null);
  const hudAbsRef = useRef<HTMLSpanElement | null>(null);
  const hudLightsRef = useRef<HTMLSpanElement | null>(null);
  const hudFpsRef = useRef<HTMLSpanElement | null>(null);
  const wheelSvgRef = useRef<SVGSVGElement | null>(null);
  const brakeFillRef = useRef<HTMLDivElement | null>(null);
  const brakePctRef = useRef<HTMLSpanElement | null>(null);
  const gasFillRef = useRef<HTMLDivElement | null>(null);
  const gasPctRef = useRef<HTMLSpanElement | null>(null);

  // Boot 3D Engine & discover cars
  useEffect(() => {
    const el = viewportRef.current;
    if (!el) return;
    const bootSeq = ++loadSeqRef.current;
    const engine = new EngineCore(el, (s) => setStats({ ...s }));
    engine.onHudFrame = (tel, fps) => {
      const mph = settingsRef.current.speedUnit === 'mph';
      const spd = Math.round(mph ? tel.speedKmh * 0.621371 : tel.speedKmh);
      if (hudSpeedRef.current) hudSpeedRef.current.textContent = String(spd);
      if (hudGearRef.current) hudGearRef.current.textContent = tel.gear;
      const rpmFrac = Math.min(1, Math.max(0, tel.rpm / 8200));
      // Circular 240-deg arc (r=54, circumference=339.29, active arc=226.2)
      if (hudRpmArcRef.current) {
        const arcLen = 226.2;
        hudRpmArcRef.current.style.strokeDashoffset = (arcLen * (1 - rpmFrac)).toFixed(1);
        hudRpmArcRef.current.style.stroke = rpmFrac > 0.84 ? '#ef4444' : rpmFrac > 0.66 ? '#f59e0b' : '#f8fafc';
      }
      if (hudNeedleRef.current) {
        const deg = -120 + rpmFrac * 240;
        hudNeedleRef.current.style.transform = `rotate(${deg.toFixed(1)}deg)`;
      }
      if (hudFuelFillRef.current) {
        const fPct = Math.round(Math.max(0, Math.min(1, tel.mech.fuelLevel)) * 100);
        hudFuelFillRef.current.style.width = `${fPct}%`;
        hudFuelFillRef.current.classList.toggle('low', fPct <= 18);
      }
      if (hudTempFillRef.current) {
        const tempNorm = Math.max(0, Math.min(1, (tel.mech.engineTemp - 65) / 65));
        hudTempFillRef.current.style.width = `${Math.round(tempNorm * 100)}%`;
        hudTempFillRef.current.classList.toggle('hot', tel.mech.engineTemp >= 110);
      }
      if (hudTempTextRef.current) {
        hudTempTextRef.current.textContent = `${tel.mech.engineTemp}°C`;
        hudTempTextRef.current.classList.toggle('hot', tel.mech.engineTemp >= 110);
      }
      if (hudAbsRef.current) hudAbsRef.current.classList.toggle('active', tel.absActive);
      if (hudLightsRef.current) hudLightsRef.current.classList.toggle('active', tel.headlightsOn);
      if (hudFpsRef.current) hudFpsRef.current.textContent = `${Math.round(fps)}`;
    };
    engineRef.current = engine;
    (window as unknown as { __flexweb?: EngineCore }).__flexweb = engine;
    engine.applySettings(settings);
    engine.setMenuDemoMode(true);
    setStats(engine.collectStats());

    let disposed = false;
    void (async () => {
      const [discoveredRaw, savedMeta] = await Promise.all([
        fetchPublicCarsFromServer(),
        listSavedCarsMetadata(),
      ]);
      if (disposed || bootSeq !== loadSeqRef.current) return;

      const discovered = discoveredRaw.filter(
        (c) => !/(bmw_m4_f82|dodge_challenger_srt_demon)\.glb$/i.test(c.url),
      );
      const publicIds = new Set(discovered.map((c) => c.id));
      const publicNames = new Set(discovered.map((c) => c.name.toLowerCase()));
      const savedOptions: CarOption[] = savedMeta
        .filter((m) => !publicIds.has(m.id) && !publicNames.has(m.name.toLowerCase()))
        .map((m) => ({
          id: m.id,
          name: `${m.name} (Saved)`,
          savedInDB: true,
          bytes: m.bytes,
        }));

      const allOptions: CarOption[] = [
        ...GITHUB_CAR_OPTIONS,
        ...discovered.map((c) => ({ id: c.id, name: c.name, url: c.url, bytes: c.bytes })),
        ...savedOptions,
      ];
      setCarOptions(allOptions);

      const lastId = getLastSelectedCarId();
      if (lastId && !lastId.startsWith('builtin-')) {
        const restored = allOptions.find(
          (c) =>
            c.id === lastId ||
            c.id === `models/${lastId}` ||
            c.url === lastId ||
            c.url === `/models/${lastId}`,
        );
        if (restored) setChosenCarId(restored.id);
      }
    })();

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      engine.unlockAudio();

      if (e.code === 'Escape') {
        if (screenRef.current === 'playing') {
          setScreen('paused');
        } else if (screenRef.current === 'paused' || screenRef.current === 'mapview') {
          setScreen('playing');
        } else if (screenRef.current === 'cars' || screenRef.current === 'settings' || screenRef.current === 'maps') {
          setScreen(returnScreen);
        }
        return;
      }

      if (screenRef.current !== 'playing') return;
      const k = keyState.current;
      switch (e.code) {
        case 'KeyW':
        case 'ArrowUp':
          k.up = true; syncVehicleInputs(); break;
        case 'KeyS':
        case 'ArrowDown':
          k.down = true; syncVehicleInputs(); break;
        case 'KeyA':
        case 'ArrowLeft':
          k.left = true; syncVehicleInputs(); break;
        case 'KeyD':
        case 'ArrowRight':
          k.right = true; syncVehicleInputs(); break;
        case 'Space':
          e.preventDefault();
          k.space = true; syncVehicleInputs(); break;
        case 'KeyR':
          engine.reset(); break;
        case 'KeyL':
          setHeadlightsOn(engine.toggleHeadlights()); break;
        case 'KeyN':
          setSettings((prev) => {
            const nextTod = (prev.timeOfDay ?? 'day') === 'night' ? 'day' : 'night';
            const next = { ...prev, timeOfDay: nextTod } as SimulatorSettings;
            saveSimulatorSettings(next);
            engine.applySettings(next);
            setHeadlightsOn(engine.getHeadlights());
            return next;
          });
          break;
        case 'KeyH':
          setSettings((prev) => {
            const next = { ...prev, showHud: !prev.showHud };
            saveSimulatorSettings(next);
            return next;
          });
          break;
        case 'KeyB':
          engine.setShowLattice(!engine.showLattice); break;
        case 'KeyC': {
          const idx = CAMERA_MODES.findIndex((m) => m.id === engine.cameraMode);
          engine.setCameraMode(CAMERA_MODES[(idx + 1) % CAMERA_MODES.length].id);
          break;
        }
        case 'KeyT': {
          const cur = engine.loop.timeScale;
          engine.loop.timeScale = cur === 1 ? 0.25 : cur === 0.25 ? 0.1 : 1;
          break;
        }
        case 'KeyE':
          engine.vehicle.shiftUp(); break;
        case 'KeyQ':
          engine.vehicle.shiftDown(); break;
        case 'KeyP':
          setScreen('paused'); break;
        case 'KeyM':
          setScreen('mapview'); break;
        case 'Digit1':
        case 'Digit2':
        case 'Digit3':
        case 'Digit4':
        case 'Digit5':
        case 'Digit6':
        case 'Digit7':
        case 'Digit8':
        case 'Digit9': {
          const p = engine.arena.presets[Number(e.code.replace('Digit', '')) - 1];
          if (p) engine.launchPreset(p);
          break;
        }
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      const k = keyState.current;
      switch (e.code) {
        case 'KeyW':
        case 'ArrowUp':
          k.up = false; syncVehicleInputs(); break;
        case 'KeyS':
        case 'ArrowDown':
          k.down = false; syncVehicleInputs(); break;
        case 'KeyA':
        case 'ArrowLeft':
          k.left = false; syncVehicleInputs(); break;
        case 'KeyD':
        case 'ArrowRight':
          k.right = false; syncVehicleInputs(); break;
        case 'Space':
          k.space = false; syncVehicleInputs(); break;
      }
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      disposed = true;
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      engine.dispose();
      engineRef.current = null;
    };
  }, []);

  const setTouch = (field: keyof typeof touchState.current, val: boolean) => {
    if (val) engineRef.current?.unlockAudio();
    touchState.current[field] = val;
    setTouchPressed({ ...touchState.current });
    syncVehicleInputs();
  };

  const handleFileUpload = async (file: File) => {
    const eng = engineRef.current;
    if (!eng) return;
    const seq = ++loadSeqRef.current;
    setLoadingState({ title: file.name, step: 'Importing 3D model…', pct: 12 });
    try {
      const rawBuffer = await file.arrayBuffer();
      const isZip = /\.zip$/i.test(file.name);
      const [idbSaved, persisted] = await Promise.all([
        saveCarBufferToDB(file.name, rawBuffer).catch(() => null),
        isZip ? Promise.resolve(null) : persistCarToPublicFolder(file, rawBuffer),
      ]);
      if (persisted) {
        const opt: CarOption = { id: persisted.id, name: persisted.name, url: persisted.url, bytes: persisted.bytes };
        setLastSelectedCarId(opt.id);
        setCarOptions((prev) => [...prev.filter((c) => c.id !== opt.id && (!idbSaved || c.id !== idbSaved.meta.id)), opt]);
        await eng.loadGLBFromLocalBuffer(opt, rawBuffer.slice(0), (step, pct) => {
          if (seq === loadSeqRef.current) setLoadingState({ title: persisted.name, step, pct });
        });
        captureCarThumb(opt.id);
      } else if (idbSaved) {
        const opt: CarOption = { id: idbSaved.meta.id, name: `${idbSaved.meta.name} (Saved)`, savedInDB: true, bytes: idbSaved.meta.bytes };
        setLastSelectedCarId(opt.id);
        setCarOptions((prev) => [...prev.filter((c) => c.id !== opt.id), opt]);
        await eng.loadGLBFromLocalBuffer(opt, idbSaved.buffer, (step, pct) => {
          if (seq === loadSeqRef.current) setLoadingState({ title: idbSaved.meta.name, step, pct });
        });
        captureCarThumb(opt.id);
      } else {
        const opt = await eng.loadCustomGLBFile(file, (step, pct) => {
          if (seq === loadSeqRef.current) setLoadingState({ title: file.name, step, pct });
        });
        setCarOptions((prev) => [...prev.filter((c) => c.id !== opt.id), opt]);
        captureCarThumb(opt.id);
      }
      if (seq === loadSeqRef.current) {
        eng.applySettings(settings);
        setStats(eng.collectStats());
      }
    } finally {
      if (seq === loadSeqRef.current) {
        setTimeout(() => { if (seq === loadSeqRef.current) setLoadingState(null); }, 180);
      }
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleDeleteSavedCar = async (opt: CarOption) => {
    await deleteSavedCarGLB(opt.id);
    const remaining = carOptions.filter((c) => c.id !== opt.id);
    setCarOptions(remaining);
    const eng = engineRef.current;
    if (eng && eng.activeCarId === opt.id) {
      await handleSelectCar(GITHUB_CAR_OPTIONS[0]);
    }
  };

  const launchArenaPlay = async (explicitCar?: CarOption) => {
    const eng = engineRef.current;
    const targetCar =
      explicitCar ??
      carOptions.find((c) => c.id === chosenCarId) ??
      GITHUB_CAR_OPTIONS[0];
    if (eng && targetCar && eng.activeCarId !== targetCar.id) {
      await handleSelectCar(targetCar);
    }
    if (eng) {
      eng.unlockAudio();
      eng.setMenuDemoMode(false);
      eng.setOverheadMapMode(false);
      eng.loop.paused = false;
      eng.loop.timeScale = 1.0;
      eng.reset(0, 0, 0, 0);
    }
    setScreen('playing');
  };

  const handleTeleportToZone = (zone: ArenaMapZone) => {
    const eng = engineRef.current;
    if (eng) {
      eng.unlockAudio();
      eng.setOverheadMapMode(false);
      eng.loop.paused = false;
      eng.loop.timeScale = 1.0;
      eng.reset(zone.teleportX, zone.teleportZ, zone.yaw, 0, false);
    }
    setScreen('playing');
  };

  const handleLaunchCrashZone = (zone: ArenaMapZone, speedKmh: number) => {
    const eng = engineRef.current;
    if (eng) {
      eng.unlockAudio();
      eng.setOverheadMapMode(false);
      eng.loop.paused = false;
      eng.loop.timeScale = 1.0;
      eng.reset(zone.spawnX, zone.spawnZ, zone.yaw, speedKmh, Boolean(zone.lateral));
    }
    setScreen('playing');
  };

  const t = stats?.telemetry;
  const chosenCar = carOptions.find((c) => c.id === chosenCarId) ?? carOptions[0];
  const maxRpm = t?.midRearEngine ? 8500 : 7200;
  const rpmPct = Math.min(100, Math.max(4, (((t?.rpm ?? 850) - 700) / (maxRpm - 700)) * 100));
  const speedDisplay = settings.speedUnit === 'mph'
    ? Math.round((t?.speedKmh ?? 0) * 0.621371)
    : Math.round(t?.speedKmh ?? 0);
  const speedUnitLabel = settings.speedUnit === 'mph' ? 'mph' : 'km/h';

  return (
    <div
      className="app-shell"
      onDragOver={(e) => { e.preventDefault(); setDraggingFile(true); }}
      onDragLeave={() => setDraggingFile(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDraggingFile(false);
        const f = e.dataTransfer.files?.[0];
        if (f && /\.(glb|gltf|zip)$/i.test(f.name)) void handleFileUpload(f);
      }}
    >
      <div id="viewport-canvas" ref={viewportRef} className="viewport-canvas" />

      <input
        ref={fileInputRef}
        type="file"
        accept=".glb,.gltf,.zip"
        style={{ display: 'none' }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void handleFileUpload(f);
        }}
      />

      {draggingFile && (
        <div className="drop-banner">Drop a .GLB / .GLTF / .ZIP 3D vehicle file to import</div>
      )}

      {loadingState && (
        <div className="loading-overlay" role="status" aria-live="polite">
          <div className="loading-card loading-minimal">
            <span className="loading-word">Loading</span>
            <span className="loading-dots" aria-hidden="true">
              <span />
              <span />
              <span />
            </span>
          </div>
        </div>
      )}

      {/* ---------------- 1. Full-Screen BeamNG Main Menu & Map Selector ---------------- */}
      {(screen === 'menu' || screen === 'maps') && (
        <MainMenu
          mode={screen === 'maps' ? 'maps' : 'home'}
          slides={menuSlides}
          activeCarName={chosenCar?.name.replace(/\s*\((public\/|Saved)\)$/i, '') ?? 'BMW M4 F82'}
          fps={stats?.fps ?? 60}
          onOpenMaps={() => { engineRef.current?.unlockAudio(); setScreen('maps'); }}
          onOpenCars={() => { engineRef.current?.unlockAudio(); setReturnScreen('menu'); setScreen('cars'); }}
          onOpenSettings={() => { engineRef.current?.unlockAudio(); setReturnScreen('menu'); setScreen('settings'); }}
          onBackHome={() => setScreen('menu')}
          onLaunchArena={() => { void launchArenaPlay(); }}
        />
      )}

      {/* ---------------- 2. Pro BeamNG Car Selector Modal ---------------- */}
      {screen === 'cars' && (
        <CarSelectorModal
          cars={carOptions}
          activeCarId={chosenCarId}
          carThumbs={carThumbs}
          onSelectCar={handleSelectCar}
          onPickPaint={(car, hex) => {
            void (async () => {
              const eng = engineRef.current;
              if (!eng) return;
              eng.setPaintColor(hex, car.id);
              if (eng.activeCarId !== car.id) {
                await handleSelectCar(car);
                if (!engineRef.current || engineRef.current.activeCarId !== car.id) return;
                engineRef.current.setPaintColor(hex, car.id);
              }
              const shot = engineRef.current?.captureStudioPortrait(0.80);
              if (shot) saveStudioThumb(car.id, shot);
            })();
          }}
          onDeleteSavedCar={(opt) => void handleDeleteSavedCar(opt)}
          confirmLabel={returnScreen === 'paused' ? 'Resume with This Vehicle' : 'Start on Crash Test Arena'}
          onConfirmPlay={(car) => { void launchArenaPlay(car); }}
          onClose={() => setScreen(returnScreen)}
        />
      )}

      {/* ---------------- 3. Pro Categorized Settings Modal ---------------- */}
      {screen === 'settings' && (
        <SettingsModal
          settings={settings}
          deviceLabel={device.label}
          onChange={updateSettings}
          onResetDefaults={() => updateSettings({ ...DEFAULT_SETTINGS })}
          onClose={() => setScreen(returnScreen)}
        />
      )}

      {/* ---------------- 4. Clean Pause Menu Overlay (Esc) ---------------- */}
      {screen === 'paused' && (
        <PauseMenu
          touchEnabled={touchEnabled}
          onContinue={() => setScreen('playing')}
          onOpenMap={() => setScreen('mapview')}
          onOpenCars={() => { setReturnScreen('paused'); setScreen('cars'); }}
          onOpenSettings={() => { setReturnScreen('paused'); setScreen('settings'); }}
          onToggleTouch={() => setTouchEnabled((v) => !v)}
          onMainMenu={() => setScreen('menu')}
        />
      )}

      {/* ---------------- 4b. Zoomed-Out Interactive Overhead Arena Map ---------------- */}
      {screen === 'mapview' && (
        <ArenaMapOverlay
          engine={engineRef.current}
          speedUnit={settings.speedUnit}
          onTeleport={handleTeleportToZone}
          onLaunchCrash={handleLaunchCrashZone}
          onClose={() => setScreen('playing')}
        />
      )}

      {/* ---------------- 5. Ultra-Clean In-Game Minimalist HUD ---------------- */}
      {screen === 'playing' && (
        <>
          {/* Top-left: single clean PAUSE (Esc) button + ultra-compact quick strip */}
          <div className="bng-ingame-topleft">
            <button
              id="pause-menu-btn"
              type="button"
              className="bng-pause-pill"
              title="Pause (Esc)"
              onClick={() => setScreen('paused')}
            >
              <span>❚❚</span>
              {!isPhone && <span>PAUSE</span>}
              {!isPhone && !device.isTouch && <kbd>Esc</kbd>}
            </button>

            {settings.showHud && (
              <div className="bng-quick-strip">
                <button
                  id="quick-map-btn"
                  type="button"
                  className="btn"
                  title="Arena Map"
                  onClick={() => setScreen('mapview')}
                >
                  {isPhone ? '🗺' : 'Map'}
                </button>
                <button
                  type="button"
                  className="btn"
                  title="Cycle Camera"
                  onClick={() => {
                    const eng = engineRef.current;
                    if (!eng) return;
                    const idx = CAMERA_MODES.findIndex((m) => m.id === eng.cameraMode);
                    eng.setCameraMode(CAMERA_MODES[(idx + 1) % CAMERA_MODES.length].id);
                  }}
                >
                  {isPhone ? `🎥 ${stats?.cameraMode ?? 'chase'}` : `Cam: ${stats?.cameraMode ?? 'chase'}`}
                </button>
                <button
                  id="toggle-nodes-btn"
                  type="button"
                  className={`btn ${stats?.showLattice ? 'active' : ''}`}
                  title="Show JBeam Nodes & Beams with maximum Ghost bodywork"
                  onClick={() => {
                    const eng = engineRef.current;
                    if (!eng) return;
                    const nextOn = !eng.showLattice;
                    eng.setShowLattice(nextOn);
                    eng.setBodyOpacity(nextOn ? 0.18 : 1.0);
                  }}
                >
                  Nodes: {stats?.showLattice ? 'ON' : 'OFF'}
                </button>
                <button
                  type="button"
                  className={`btn ${stats?.timeScale === 0.25 ? 'active' : ''}`}
                  title="Toggle Slow Motion"
                  onClick={() => {
                    const eng = engineRef.current;
                    if (!eng) return;
                    eng.loop.timeScale = eng.loop.timeScale === 1 ? 0.25 : 1;
                  }}
                >
                  {stats?.timeScale === 0.25 ? '0.25x' : '1x'}
                </button>
                <button
                  id="toggle-fullscreen-btn"
                  type="button"
                  className={`btn ${isFullscreen ? 'active' : ''}`}
                  title={isFullscreen ? 'Exit Fullscreen' : 'Enter True Fullscreen'}
                  onClick={toggleFullscreen}
                >
                  {isPhone ? '⛶' : isFullscreen ? '⛶ Exit' : '⛶ Full'}
                </button>
                <button
                  id="toggle-headlights-btn"
                  type="button"
                  className={`btn ${headlightsOn ? 'active' : ''}`}
                  title="Phares (L)"
                  onClick={() => {
                    const eng = engineRef.current;
                    if (!eng) return;
                    setHeadlightsOn(eng.toggleHeadlights());
                  }}
                >
                  {isPhone ? '💡' : headlightsOn ? 'Phares: ON' : 'Phares'}
                </button>
                <button
                  id="toggle-night-btn"
                  type="button"
                  className={`btn ${(settings.timeOfDay ?? 'day') === 'night' ? 'active' : ''}`}
                  title="Mode Nuit / Jour (N)"
                  onClick={() => {
                    const nextTod = (settings.timeOfDay ?? 'day') === 'night' ? 'day' : 'night';
                    updateSettings({ ...settings, timeOfDay: nextTod });
                    if (engineRef.current) setHeadlightsOn(engineRef.current.getHeadlights());
                  }}
                >
                  {(settings.timeOfDay ?? 'day') === 'night' ? (isPhone ? '🌙' : 'Nuit') : (isPhone ? '☀' : 'Jour')}
                </button>
                <button
                  id="repair-vehicle-btn"
                  type="button"
                  className="btn btn-repair"
                  title="Réparer la voiture sur place"
                  onClick={() => engineRef.current?.repairInPlace()}
                >
                  {isPhone ? '🔧' : 'Réparer'}
                </button>
                <button
                  id="reset-vehicle-btn"
                  type="button"
                  className="btn btn-reset"
                  title="Reset Vehicle (R)"
                  onClick={() => engineRef.current?.reset()}
                >
                  {isPhone ? '↺' : 'Reset (R)'}
                </button>
                <span className="device-badge hud-device-badge" title={`Detected: ${device.label} (${device.viewportWidth}×${device.viewportHeight})`}>
                  {device.label}
                </span>
              </div>
            )}
          </div>

          {/* Bottom-Center: Side-Oriented Horizontal Vehicle Diagnostics + Live Breakage Alerts ABOVE */}
          {settings.showHud && settings.showDiagnostics && (
            <aside className="diag-bottom-center">
              <VehicleDiagnostics
                telemetry={t}
                collapsed={diagCollapsed}
                onToggle={() => setDiagCollapsed((v) => !v)}
              />
            </aside>
          )}

          {/* Upgraded Touch Controls (Steering Wheel OR Arrows + Shift Buttons ABOVE Pedals + Analog Gas/Brake Pedals) */}
          {settings.showHud && touchEnabled && (
            <div className="touch-layer">
              {/* Left: Steering Wheel OR Left/Right Arrows */}
              <div className="touch-pad-group">
                {settings.touchSteering === 'wheel' ? (
                  <div
                    id="touch-steering-wheel"
                    className="touch-wheel-wrap"
                    role="slider"
                    aria-label="Touch Steering Wheel"
                    aria-valuemin={-100}
                    aria-valuemax={100}
                    aria-valuenow={Math.round(analogUI.steer * 100)}
                    onPointerDown={(e) => {
                      e.currentTarget.setPointerCapture(e.pointerId);
                      engineRef.current?.unlockAudio();
                      wheelPtrRef.current = { id: e.pointerId, startX: e.clientX };
                    }}
                    onPointerMove={(e) => {
                      if (!wheelPtrRef.current || wheelPtrRef.current.id !== e.pointerId) return;
                      const dx = e.clientX - wheelPtrRef.current.startX;
                      const val = Math.max(-1, Math.min(1, dx / 72));
                      analogTouchRef.current.steer = val;
                      if (wheelSvgRef.current) wheelSvgRef.current.style.transform = `rotate(${val * 135}deg)`;
                      syncVehicleInputs();
                    }}
                    onPointerUp={(e) => {
                      if (!wheelPtrRef.current || wheelPtrRef.current.id !== e.pointerId) return;
                      wheelPtrRef.current = null;
                      analogTouchRef.current.steer = 0;
                      if (wheelSvgRef.current) wheelSvgRef.current.style.transform = 'rotate(0deg)';
                      syncVehicleInputs();
                    }}
                    onPointerCancel={() => {
                      wheelPtrRef.current = null;
                      analogTouchRef.current.steer = 0;
                      if (wheelSvgRef.current) wheelSvgRef.current.style.transform = 'rotate(0deg)';
                      syncVehicleInputs();
                    }}
                  >
                    <svg
                      ref={wheelSvgRef}
                      viewBox="0 0 120 120"
                      className="touch-wheel-svg"
                      style={{ transform: `rotate(${analogUI.steer * 135}deg)` }}
                    >
                      <circle cx="60" cy="60" r="50" fill="rgba(15,23,42,0.52)" stroke="rgba(226,232,240,0.85)" strokeWidth="11" />
                      <rect x="56" y="7" width="8" height="14" rx="2" fill="#f59e0b" />
                      <line x1="15" y1="60" x2="105" y2="60" stroke="rgba(203,213,225,0.85)" strokeWidth="9" strokeLinecap="round" />
                      <line x1="60" y1="60" x2="60" y2="105" stroke="rgba(203,213,225,0.85)" strokeWidth="9" strokeLinecap="round" />
                      <circle cx="60" cy="60" r="15" fill="#1e293b" stroke="#f59e0b" strokeWidth="3" />
                    </svg>
                  </div>
                ) : (
                  <>
                    <button
                      id="touch-steer-left"
                      type="button"
                      className={`touch-btn ${touchPressed.left ? 'pressed' : ''}`}
                      onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); setTouch('left', true); }}
                      onPointerUp={() => setTouch('left', false)}
                      onPointerCancel={() => setTouch('left', false)}
                    >
                      <span>◀</span>
                      <span>LEFT</span>
                    </button>
                    <button
                      id="touch-steer-right"
                      type="button"
                      className={`touch-btn ${touchPressed.right ? 'pressed' : ''}`}
                      onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); setTouch('right', true); }}
                      onPointerUp={() => setTouch('right', false)}
                      onPointerCancel={() => setTouch('right', false)}
                    >
                      <span>▶</span>
                      <span>RIGHT</span>
                    </button>
                  </>
                )}
              </div>

              {/* Right: Shift Buttons directly ABOVE Pedals + Handbrake + Analog Brake & Gas Pedals */}
              <div className="touch-right-stack">
                {settings.transmissionMode !== 'auto' && (
                  <div className="touch-shift-row">
                    <button
                      id="touch-shift-down"
                      type="button"
                      className="touch-shift-btn"
                      onPointerDown={(e) => {
                        e.stopPropagation();
                        engineRef.current?.unlockAudio();
                        engineRef.current?.vehicle.shiftDown();
                      }}
                    >
                      − SHIFT
                    </button>
                    <button
                      id="touch-shift-up"
                      type="button"
                      className="touch-shift-btn up"
                      onPointerDown={(e) => {
                        e.stopPropagation();
                        engineRef.current?.unlockAudio();
                        engineRef.current?.vehicle.shiftUp();
                      }}
                    >
                      SHIFT +
                    </button>
                  </div>
                )}

                <div className="touch-pad-group touch-pedals-row">
                  <button
                    id="touch-handbrake"
                    type="button"
                    className={`touch-btn touch-hb-btn ${touchPressed.handbrake ? 'pressed' : ''}`}
                    onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); setTouch('handbrake', true); }}
                    onPointerUp={() => setTouch('handbrake', false)}
                    onPointerCancel={() => setTouch('handbrake', false)}
                  >
                    <span>HB</span>
                    <span>DRIFT</span>
                  </button>

                  <div
                    id="touch-brake"
                    role="slider"
                    aria-label="Brake Pedal"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={Math.round((analogUI.brake || (touchPressed.brake ? 1 : 0)) * 100)}
                    className={`touch-pedal pedal-stop ${analogUI.brake > 0.05 || touchPressed.brake ? 'pressed' : ''}`}
                    onPointerDown={(e) => {
                      e.currentTarget.setPointerCapture(e.pointerId);
                      engineRef.current?.unlockAudio();
                      if (settings.analogPedals) {
                        const rect = e.currentTarget.getBoundingClientRect();
                        const val = Math.max(0.15, Math.min(1, (rect.bottom - e.clientY) / rect.height));
                        analogTouchRef.current.brake = val;
                        setAnalogUI((prev) => ({ ...prev, brake: val }));
                        syncVehicleInputs();
                      } else {
                        setTouch('brake', true);
                      }
                    }}
                    onPointerMove={(e) => {
                      if (!settings.analogPedals || e.buttons === 0) return;
                      const rect = e.currentTarget.getBoundingClientRect();
                      const val = Math.max(0.05, Math.min(1, (rect.bottom - e.clientY) / rect.height));
                      analogTouchRef.current.brake = val;
                      const pct = Math.round(val * 100);
                      if (brakeFillRef.current) brakeFillRef.current.style.height = `${pct}%`;
                      if (brakePctRef.current) brakePctRef.current.textContent = `${pct}%`;
                      syncVehicleInputs();
                    }}
                    onPointerUp={() => {
                      analogTouchRef.current.brake = 0;
                      if (brakeFillRef.current) brakeFillRef.current.style.height = '0%';
                      if (brakePctRef.current) brakePctRef.current.textContent = settings.analogPedals ? '0%' : 'REV';
                      setAnalogUI((prev) => ({ ...prev, brake: 0 }));
                      setTouch('brake', false);
                    }}
                    onPointerCancel={() => {
                      analogTouchRef.current.brake = 0;
                      if (brakeFillRef.current) brakeFillRef.current.style.height = '0%';
                      if (brakePctRef.current) brakePctRef.current.textContent = settings.analogPedals ? '0%' : 'REV';
                      setAnalogUI((prev) => ({ ...prev, brake: 0 }));
                      setTouch('brake', false);
                    }}
                  >
                    <div
                      ref={brakeFillRef}
                      className="touch-pedal-fill brake"
                      style={{ height: `${Math.round((settings.analogPedals ? analogUI.brake : touchPressed.brake ? 1 : 0) * 100)}%` }}
                    />
                    <span className="touch-pedal-label">BRAKE</span>
                    <span ref={brakePctRef} className="touch-pedal-pct">
                      {settings.analogPedals ? `${Math.round(analogUI.brake * 100)}%` : 'REV'}
                    </span>
                  </div>

                  <div
                    id="touch-throttle"
                    role="slider"
                    aria-label="Gas Pedal"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={Math.round((analogUI.throttle || (touchPressed.throttle ? 1 : 0)) * 100)}
                    className={`touch-pedal pedal-go ${analogUI.throttle > 0.05 || touchPressed.throttle ? 'pressed' : ''}`}
                    onPointerDown={(e) => {
                      e.currentTarget.setPointerCapture(e.pointerId);
                      engineRef.current?.unlockAudio();
                      if (settings.analogPedals) {
                        const rect = e.currentTarget.getBoundingClientRect();
                        const val = Math.max(0.15, Math.min(1, (rect.bottom - e.clientY) / rect.height));
                        analogTouchRef.current.throttle = val;
                        setAnalogUI((prev) => ({ ...prev, throttle: val }));
                        syncVehicleInputs();
                      } else {
                        setTouch('throttle', true);
                      }
                    }}
                    onPointerMove={(e) => {
                      if (!settings.analogPedals || e.buttons === 0) return;
                      const rect = e.currentTarget.getBoundingClientRect();
                      const val = Math.max(0.05, Math.min(1, (rect.bottom - e.clientY) / rect.height));
                      analogTouchRef.current.throttle = val;
                      const pct = Math.round(val * 100);
                      if (gasFillRef.current) gasFillRef.current.style.height = `${pct}%`;
                      if (gasPctRef.current) gasPctRef.current.textContent = `${pct}%`;
                      syncVehicleInputs();
                    }}
                    onPointerUp={() => {
                      analogTouchRef.current.throttle = 0;
                      if (gasFillRef.current) gasFillRef.current.style.height = '0%';
                      if (gasPctRef.current) gasPctRef.current.textContent = settings.analogPedals ? '0%' : 'GO';
                      setAnalogUI((prev) => ({ ...prev, throttle: 0 }));
                      setTouch('throttle', false);
                    }}
                    onPointerCancel={() => {
                      analogTouchRef.current.throttle = 0;
                      if (gasFillRef.current) gasFillRef.current.style.height = '0%';
                      if (gasPctRef.current) gasPctRef.current.textContent = settings.analogPedals ? '0%' : 'GO';
                      setAnalogUI((prev) => ({ ...prev, throttle: 0 }));
                      setTouch('throttle', false);
                    }}
                  >
                    <div
                      ref={gasFillRef}
                      className="touch-pedal-fill gas"
                      style={{ height: `${Math.round((settings.analogPedals ? analogUI.throttle : touchPressed.throttle ? 1 : 0) * 100)}%` }}
                    />
                    <span className="touch-pedal-label">GAS</span>
                    <span ref={gasPctRef} className="touch-pedal-pct">
                      {settings.analogPedals ? `${Math.round(analogUI.throttle * 100)}%` : 'GO'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Bottom-Right: Round Motorsport Speedometer with Fuel & Engine Temp Gauges */}
          {settings.showHud && (
            <footer className="hud-speedo-round">
              <div className="speedo-dial-wrap">
                <svg viewBox="0 0 132 132" className="speedo-svg" aria-hidden="true">
                  {/* Track arc (240° sweep starting at 150°) */}
                  <circle
                    cx="66"
                    cy="66"
                    r="54"
                    fill="rgba(9, 12, 17, 0.88)"
                    stroke="rgba(255, 255, 255, 0.10)"
                    strokeWidth="7"
                    strokeDasharray="226.2 340"
                    strokeLinecap="round"
                    transform="rotate(150 66 66)"
                  />
                  {/* Redline accent zone */}
                  <circle
                    cx="66"
                    cy="66"
                    r="54"
                    fill="none"
                    stroke="rgba(239, 68, 68, 0.36)"
                    strokeWidth="7"
                    strokeDasharray="36 340"
                    strokeDashoffset="-190"
                    strokeLinecap="round"
                    transform="rotate(150 66 66)"
                  />
                  {/* Live RPM sweep arc */}
                  <circle
                    ref={hudRpmArcRef}
                    cx="66"
                    cy="66"
                    r="54"
                    fill="none"
                    stroke="#f8fafc"
                    strokeWidth="7"
                    strokeDasharray="226.2 340"
                    strokeDashoffset={(226.2 * (1 - rpmPct / 100)).toFixed(1)}
                    strokeLinecap="round"
                    transform="rotate(150 66 66)"
                  />
                  {/* Analog tachometer needle */}
                  <g
                    ref={hudNeedleRef}
                    className="speedo-needle-group"
                    style={{ transform: `rotate(${(-120 + (rpmPct / 100) * 240).toFixed(1)}deg)` }}
                  >
                    <line x1="66" y1="66" x2="66" y2="18" stroke="#f97316" strokeWidth="2.4" strokeLinecap="round" />
                    <circle cx="66" cy="66" r="4.5" fill="#f97316" />
                  </g>
                </svg>

                <div className="speedo-center-readout">
                  <span id="speed-value" ref={hudSpeedRef} className="speedo-num">{speedDisplay}</span>
                  <span className="speedo-unit">{speedUnitLabel}</span>
                  <div className="speedo-bottom-row">
                    <div id="gear-badge" ref={hudGearRef} className="speedo-gear">{t?.gear ?? 'N'}</div>
                    <span ref={hudAbsRef} className={`speedo-lamp ${t?.absActive ? 'active' : ''}`}>ABS</span>
                    <span ref={hudLightsRef} className={`speedo-lamp lights ${t?.headlightsOn ? 'active' : ''}`}>HL</span>
                  </div>
                </div>
              </div>

              {/* Integrated Fuel & Engine Temp Gauges */}
              <div className="speedo-subgauges">
                <div className="subgauge-row">
                  <span className="subgauge-lbl">FUEL</span>
                  <div className="subgauge-track">
                    <div
                      ref={hudFuelFillRef}
                      className={`subgauge-fill fuel ${(t?.mech?.fuelLevel ?? 1) <= 0.18 ? 'low' : ''}`}
                      style={{ width: `${Math.round((t?.mech?.fuelLevel ?? 1) * 100)}%` }}
                    />
                  </div>
                </div>
                <div className="subgauge-row">
                  <span className="subgauge-lbl">TEMP</span>
                  <div className="subgauge-track">
                    <div
                      ref={hudTempFillRef}
                      className={`subgauge-fill temp ${(t?.mech?.engineTemp ?? 88) >= 110 ? 'hot' : ''}`}
                      style={{ width: `${Math.round(Math.max(0, Math.min(1, ((t?.mech?.engineTemp ?? 88) - 65) / 65)) * 100)}%` }}
                    />
                  </div>
                  <span
                    ref={hudTempTextRef}
                    className={`subgauge-val ${(t?.mech?.engineTemp ?? 88) >= 110 ? 'hot' : ''}`}
                  >
                    {t?.mech?.engineTemp ?? 88}°C
                  </span>
                </div>
                <div className="speedo-fps-row">
                  <span>FPS</span>
                  <span ref={hudFpsRef}>{Math.round(stats?.fps ?? 60)}</span>
                </div>
              </div>
            </footer>
          )}
        </>
      )}

      {isPhoneLandscape && (
        <div className="mobile-portrait-guard" role="dialog" aria-modal="true" aria-label="Portrait mode required">
          <svg className="mobile-portrait-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <rect x="6" y="2.5" width="12" height="19" rx="2.5" />
            <line x1="10" y1="18.5" x2="14" y2="18.5" />
          </svg>
          <h2 className="mobile-portrait-title">Mode Portrait uniquement</h2>
          <p className="mobile-portrait-sub">
            Tournez votre téléphone à la verticale pour continuer à jouer.
          </p>
        </div>
      )}
    </div>
  );
}
