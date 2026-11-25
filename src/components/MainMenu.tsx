import React from 'react';

interface MainMenuProps {
  mode: 'home' | 'maps';
  slides: string[];
  activeCarName: string;
  fps: number;
  onOpenMaps: () => void;
  onOpenCars: () => void;
  onOpenSettings: () => void;
  onBackHome: () => void;
  onLaunchArena: () => void;
}

export function MainMenu({
  mode,
  slides,
  activeCarName,
  fps,
  onOpenMaps,
  onOpenCars,
  onOpenSettings,
  onBackHome,
  onLaunchArena,
}: MainMenuProps): React.ReactElement {
  const mapThumb = slides.length > 0 ? slides[slides.length - 1] : null;

  return (
    <div className="bng-menu-backdrop" role="dialog" aria-label="FlexWeb Main Menu">
      {/* Subtle left-side readability vignette — keeps live 3D crash showcase 100% visible behind the menu */}
      <div className="bng-slideshow-layer" aria-hidden="true">
        <div className="bng-menu-vignette" />
      </div>

      {mode === 'home' ? (
        <div className="bng-home-layout">
          <div className="bng-hero-col">
            <div className="bng-logo-lockup">
              <span className="bng-logo-badge">SOFT-BODY PHYSICS SIMULATOR</span>
              <h1 className="bng-logo-title">
                <span className="brand-mark lg" />
                FLEXWEB
              </h1>
              <p className="bng-logo-sub">
                Real-Time Soft-Body Crash Engine · Active Vehicle: <strong>{activeCarName}</strong>
              </p>
            </div>

            <nav className="bng-main-actions" aria-label="Main Menu">
              <button id="menu-play-btn" type="button" className="bng-nav-btn primary" onClick={onOpenMaps}>
                <span className="bng-nav-title">PLAY</span>
                <span className="bng-nav-desc">Select a map and launch the simulation</span>
              </button>
              <button id="menu-cars-btn" type="button" className="bng-nav-btn" onClick={onOpenCars}>
                <span className="bng-nav-title">CARS</span>
                <span className="bng-nav-desc">Studio showroom, live body paints &amp; vehicle specs</span>
              </button>
              <button id="menu-settings-btn-main" type="button" className="bng-nav-btn" onClick={onOpenSettings}>
                <span className="bng-nav-title">SETTINGS</span>
                <span className="bng-nav-desc">Controls, Web Audio engine sound &amp; 60 FPS graphics</span>
              </button>
            </nav>

            <div className="bng-menu-footer">
              <span>{Math.round(fps)} FPS</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="bng-modal-shell">
          <div className="bng-modal-header">
            <div>
              <span className="bng-eyebrow">MAP SELECTION</span>
              <h2 className="bng-modal-title">Choose an Environment</h2>
            </div>
            <button type="button" className="btn btn-menu" onClick={onBackHome}>
              ← Back
            </button>
          </div>

          <div className="bng-map-grid">
            {/* Map 1: Crash Test Arena (Playable immediately) */}
            <article className="bng-map-card playable">
              <div className="bng-map-thumb">
                {mapThumb ? (
                  <img src={mapThumb} alt="Crash Test Arena" />
                ) : (
                  <div className="bng-map-placeholder arena">PROVING GROUND 2 km × 2 km</div>
                )}
                <span className="bng-map-tag ready">PLAYABLE</span>
              </div>
              <div className="bng-map-body">
                <h3>Crash Test Arena</h3>
                <p>
                  2 km × 2 km Proving Ground: 500 m NCAP radar runway, double hydraulic press, pendulum hammers,
                  crusher rollers, suspension &amp; 25° banked circuit, mega-ramp canyon, and industrial roundabout.
                </p>
                <div className="bng-map-meta">
                  <span>Vehicle: <strong>{activeCarName}</strong></span>
                </div>
                <button
                  id="launch-arena-btn"
                  type="button"
                  className="bng-cta-btn"
                  onClick={onLaunchArena}
                >
                  Start on Crash Test Arena →
                </button>
              </div>
            </article>

            {/* Map 2: City (Locked — Coming Soon) */}
            <article className="bng-map-card locked" aria-disabled="true">
              <div className="bng-map-thumb">
                <div className="bng-map-placeholder city">
                  <div className="bng-city-grid" />
                  <span className="bng-coming-soon-banner">COMING SOON</span>
                </div>
                <span className="bng-map-tag soon">LOCKED</span>
              </div>
              <div className="bng-map-body">
                <h3>City</h3>
                <p>
                  Open urban district with wide avenues, intersections, highway overpasses, traffic, and stunt ramps
                  currently in development for upcoming updates.
                </p>
                <div className="bng-map-meta">
                  <span>Status: <strong>In Development</strong></span>
                </div>
                <button type="button" className="bng-cta-btn locked" disabled>
                  🔒 COMING SOON
                </button>
              </div>
            </article>
          </div>
        </div>
      )}
    </div>
  );
}
