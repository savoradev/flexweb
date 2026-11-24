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
      {/* Subtle left-side readability vignette — keeps live 3D crash videos 100% visible behind the menu */}
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
                Moteur de déformation temps réel · Véhicule actif : <strong>{activeCarName}</strong>
              </p>
            </div>

            <nav className="bng-main-actions" aria-label="Menu Principal">
              <button id="menu-play-btn" type="button" className="bng-nav-btn primary" onClick={onOpenMaps}>
                <span className="bng-nav-title">PLAY</span>
                <span className="bng-nav-desc">Choisir une carte et lancer la simulation</span>
              </button>
              <button id="menu-cars-btn" type="button" className="bng-nav-btn" onClick={onOpenCars}>
                <span className="bng-nav-title">CARS</span>
                <span className="bng-nav-desc">Showroom blanc, peintures carrosserie &amp; fiches techniques</span>
              </button>
              <button id="menu-settings-btn-main" type="button" className="bng-nav-btn" onClick={onOpenSettings}>
                <span className="bng-nav-title">SETTINGS</span>
                <span className="bng-nav-desc">Conduite, Audio moteur Web Audio &amp; Graphismes 60 FPS</span>
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
              <span className="bng-eyebrow">SÉLECTION DE CARTE</span>
              <h2 className="bng-modal-title">Choisir un environnement</h2>
            </div>
            <button type="button" className="btn btn-menu" onClick={onBackHome}>
              ← Retour
            </button>
          </div>

          <div className="bng-map-grid">
            {/* Map 1: Crash Test Arena (Playable immediately) */}
            <article className="bng-map-card playable">
              <div className="bng-map-thumb">
                {mapThumb ? (
                  <img src={mapThumb} alt="Crash Test Arena" />
                ) : (
                  <div className="bng-map-placeholder arena">GRIDMAP 400m × 400m</div>
                )}
                <span className="bng-map-tag ready">DISPONIBLE</span>
              </div>
              <div className="bng-map-body">
                <h3>Crash Test Arena</h3>
                <p>
                  Arène GridMap 400×400 m : barrières NCAP, presse hydraulique cabine, double tremplin,
                  pistes de suspension Flexbody, poteaux, hangars solides et 9 scénarios.
                </p>
                <div className="bng-map-meta">
                  <span>Véhicule : <strong>{activeCarName}</strong></span>
                </div>
                <button
                  id="launch-arena-btn"
                  type="button"
                  className="bng-cta-btn"
                  onClick={onLaunchArena}
                >
                  Démarrer sur Crash Test Arena →
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
                <span className="bng-map-tag soon">VERROUILLÉ</span>
              </div>
              <div className="bng-map-body">
                <h3>City</h3>
                <p>
                  Zone urbaine ouverte avec avenues, carrefours, autoroute, circulation et rampes de saut
                  en cours de préparation pour les prochaines mises à jour.
                </p>
                <div className="bng-map-meta">
                  <span>Statut : <strong>En développement</strong></span>
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
