import { useEffect, useState } from 'react';

export type DeviceType = 'phone' | 'tablet' | 'laptop' | 'desktop';

export interface SafeAreaInsets {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export interface DeviceProfile {
  deviceType: DeviceType;
  isTouch: boolean;
  isPortrait: boolean;
  viewportWidth: number;
  viewportHeight: number;
  safeAreaInsets: SafeAreaInsets;
  label: string;
}

function readSafeAreaInsets(): SafeAreaInsets {
  if (typeof window === 'undefined' || typeof document === 'undefined' || !document.documentElement) {
    return { top: 0, right: 0, bottom: 0, left: 0 };
  }
  const style = getComputedStyle(document.documentElement);
  const parsePx = (prop: string): number => {
    const val = parseFloat(style.getPropertyValue(prop));
    return Number.isFinite(val) ? val : 0;
  };
  return {
    top: parsePx('--sat'),
    right: parsePx('--sar'),
    bottom: parsePx('--sab'),
    left: parsePx('--sal'),
  };
}

export function detectDeviceProfile(): DeviceProfile {
  if (typeof window === 'undefined' || typeof navigator === 'undefined') {
    return {
      deviceType: 'desktop',
      isTouch: false,
      isPortrait: false,
      viewportWidth: 1920,
      viewportHeight: 1080,
      safeAreaInsets: { top: 0, right: 0, bottom: 0, left: 0 },
      label: 'Desktop',
    };
  }

  const vw = Math.max(1, Math.round(window.visualViewport?.width ?? window.innerWidth));
  const vh = Math.max(1, Math.round(window.visualViewport?.height ?? window.innerHeight));
  const shortSide = Math.min(vw, vh);
  const isPortrait = vh > vw;

  const coarsePointer = typeof window.matchMedia === 'function' && window.matchMedia('(pointer: coarse)').matches;
  const mobileUA = /Android|iPhone|iPod|Mobile/i.test(navigator.userAgent);
  const tabletUA = /iPad|Tablet|SM-X|SM-T/i.test(navigator.userAgent);
  const isTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0 || coarsePointer || mobileUA || tabletUA;

  let deviceType: DeviceType;
  if ((isTouch || mobileUA) && !tabletUA && (vw < 768 || shortSide < 500)) {
    deviceType = 'phone';
  } else if (isTouch && (tabletUA || (vw >= 768 && vw <= 1280))) {
    deviceType = 'tablet';
  } else if (!isTouch && vw <= 1440) {
    deviceType = 'laptop';
  } else if (isTouch && vw > 1280) {
    deviceType = vw > 1440 ? 'desktop' : 'laptop';
  } else {
    deviceType = 'desktop';
  }

  const baseName =
    deviceType === 'phone'
      ? 'Phone'
      : deviceType === 'tablet'
        ? 'Tablet'
        : deviceType === 'laptop'
          ? 'Laptop'
          : 'Desktop';
  const label = isTouch && (deviceType === 'phone' || deviceType === 'tablet') ? `${baseName} · Touch` : baseName;

  return {
    deviceType,
    isTouch,
    isPortrait,
    viewportWidth: vw,
    viewportHeight: vh,
    safeAreaInsets: readSafeAreaInsets(),
    label,
  };
}

export function applyDeviceAttributes(profile: DeviceProfile): void {
  if (typeof document === 'undefined' || !document.documentElement) return;
  const root = document.documentElement;
  root.dataset.device = profile.deviceType;
  root.dataset.touch = String(profile.isTouch);
  root.dataset.orientation = profile.isPortrait ? 'portrait' : 'landscape';
}

// Immediately tag <html> on module load so CSS applies before first paint
if (typeof document !== 'undefined') {
  applyDeviceAttributes(detectDeviceProfile());
}

export function useDeviceProfile(): DeviceProfile {
  const [profile, setProfile] = useState<DeviceProfile>(() => {
    const p = detectDeviceProfile();
    applyDeviceAttributes(p);
    return p;
  });

  useEffect(() => {
    let rafId = 0;
    const sync = () => {
      if (rafId) cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(() => {
        rafId = 0;
        const next = detectDeviceProfile();
        applyDeviceAttributes(next);
        setProfile((prev) =>
          prev.deviceType === next.deviceType &&
          prev.isTouch === next.isTouch &&
          prev.isPortrait === next.isPortrait &&
          prev.viewportWidth === next.viewportWidth &&
          prev.viewportHeight === next.viewportHeight
            ? prev
            : next,
        );
      });
    };

    sync();
    window.addEventListener('resize', sync, { passive: true });
    window.addEventListener('orientationchange', sync, { passive: true });
    window.visualViewport?.addEventListener('resize', sync, { passive: true });
    return () => {
      if (rafId) cancelAnimationFrame(rafId);
      window.removeEventListener('resize', sync);
      window.removeEventListener('orientationchange', sync);
      window.visualViewport?.removeEventListener('resize', sync);
    };
  }, []);

  return profile;
}
