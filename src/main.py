"""Point d'entrée principal — pipeline unifié audio-visuel adaptatif.

Architecture refonte :
  1. Audio spectral (AudioDetector) → segments actifs
  2. Qualité de scène mesurée une fois (luminance, glare_frac)
  3. YOLO avec prétraitement adaptatif (AdaptivePreprocessor) sur tous les segments
  4. Confiance au son sur les segments sans détection visuelle si conditions difficiles
  5. Fallback audio global en dernier recours

  Plus de modes nuit/crépuscule/jour ni de NightDetector.
  Un seul détecteur visuel s'adapte à toutes les conditions.
"""
import logging
import logging.handlers
import os
import signal
import sys
import threading
import time
from pathlib import Path

from .audio_detector import AudioDetector, bootstrap_calibration
from .config import load_config
from .dashboard import run_dashboard
from .database import Database
from .detector import CrossingEvent, VehicleDetector, sunrise_sunset_local
from .frame_sampler import FrameSampler, purge_old_frames
from .ingestion import FileWatcher
from .preprocessor import AdaptivePreprocessor, measure_scene_quality
from .progress import tracker as progress_tracker

_DEBUG_FRAMES_DIR = Path("/app/data/debug_frames")
_DEBUG_FRAMES_MAX_AGE_H = 48


def _check_video(path: Path) -> "str | None":
    """Sanity check — retourne un message d'erreur ou None si OK."""
    import cv2 as _cv2
    cap = _cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return "Impossible d'ouvrir le fichier vidéo (fichier corrompu ou format non supporté)"
    ret, _ = cap.read()
    cap.release()
    if not ret:
        return "Le fichier vidéo est vide ou ne contient aucune image lisible"
    return None


def setup_logging(level: str, log_dir: Path, max_mb: int, backup_count: int):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "comptage.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_mb * 1024 * 1024, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)

_shutdown = threading.Event()
_processing_start: "float | None" = None
_PROCESSING_TIMEOUT_SEC = 7200  # watchdog : 2h max par fichier


def _handle_signal(signum, frame):
    logger.info("Signal %s reçu — arrêt en cours…", signum)
    _shutdown.set()


def _watchdog_loop():
    """Daemon thread : tue le process si un fichier prend plus de 2h."""
    while not _shutdown.is_set():
        _shutdown.wait(timeout=30)
        start = _processing_start
        if start is not None:
            elapsed = time.monotonic() - start
            if elapsed > _PROCESSING_TIMEOUT_SEC:
                logger.error("Timeout : bloqué depuis %d s — arrêt forcé.", int(elapsed))
                import os as _os
                _os.kill(_os.getpid(), signal.SIGTERM)
                return


def _auto_purge(config, db: Database):
    days = config.data_retention_days
    if days > 0:
        count = db.delete_old_crossings(days)
        if count:
            logger.info("Rétention : %d franchissement(s) > %d jours supprimé(s).", count, days)

    old_paths = db.delete_old_debug_frames(_DEBUG_FRAMES_MAX_AGE_H)
    if old_paths:
        import os
        for p in old_paths:
            try:
                os.remove(p)
            except OSError:
                pass
    purge_old_frames(_DEBUG_FRAMES_DIR, _DEBUG_FRAMES_MAX_AGE_H)


def _synthesize_audio_crossings(
    segments: list,
    existing: list[dict],
    video_start_dt,
    source_name: str,
) -> list[dict]:
    """Crée 1 véhicule par segment audio actif sans détection visuelle.

    Utilisé quand la visibilité est mauvaise (nuit, contre-jour) et que l'audio
    a détecté un passage mais YOLO n'a rien vu. Le son moteur reste fiable même
    quand l'image est inexploitable.
    """
    from datetime import datetime, timedelta

    existing_secs: "list[float] | None" = None
    if video_start_dt is not None:
        existing_secs = []
        for c in existing:
            try:
                t = datetime.fromisoformat(c["timestamp"])
                existing_secs.append((t - video_start_dt).total_seconds())
            except Exception:
                pass

    out: list[dict] = []
    for seg in segments:
        if existing_secs is not None:
            covered = any(seg.start_sec <= s <= seg.end_sec for s in existing_secs)
        else:
            covered = len(existing) > 0
        if covered:
            continue
        mid = (seg.start_sec + seg.end_sec) / 2.0
        ts = (video_start_dt + timedelta(seconds=mid)) if video_start_dt else datetime.utcnow()
        out.append({
            "timestamp":    ts.isoformat(),
            "vehicle_type": "car",
            "direction":    None,
            "confidence":   0.2,
            "source_file":  source_name,
        })
    return out


def _track_detection_fingerprint(config, db: Database):
    current_fp = config.detection_fingerprint()
    stored_fp  = db.get_state("detection_fingerprint")
    db.set_state("detection_fingerprint", current_fp)

    if stored_fp is None:
        logger.info("Empreinte de config enregistrée : %s", current_fp)
    elif stored_fp != current_fp:
        logger.info(
            "Paramètres de détection modifiés (%s → %s) — appliqués aux nouveaux fichiers.",
            stored_fp, current_fp,
        )


def processing_loop(
    watcher: FileWatcher,
    audio: AudioDetector,
    detector: VehicleDetector,
    preprocessor: AdaptivePreprocessor,
    db: Database,
):
    logger.info("Boucle de traitement démarrée (pipeline adaptatif unifié).")
    config = watcher.config
    _prev_imgsz = config.imgsz
    _prev_model_name = config.model_name
    _prev_log_level = config.log_level

    while not _shutdown.is_set():
        changed = config.reload()
        if changed:
            new_log = config.log_level
            if new_log != _prev_log_level:
                logging.getLogger().setLevel(getattr(logging, new_log.upper(), logging.INFO))
                logger.info("Niveau de log mis à jour : %s", new_log)
                _prev_log_level = new_log

            new_imgsz = config.imgsz
            new_model = config.model_name
            if new_imgsz != _prev_imgsz or new_model != _prev_model_name:
                logger.info(
                    "Modèle/imgsz modifié (%s imgsz=%d → %s imgsz=%d) — rechargement au prochain fichier.",
                    _prev_model_name, _prev_imgsz, new_model, new_imgsz,
                )
                detector._model = None
                _prev_imgsz = new_imgsz
                _prev_model_name = new_model

            _track_detection_fingerprint(config, db)

        try:
            new_files = watcher.scan_new_files()
        except Exception as e:
            logger.error("Erreur lors du scan : %s", e)
            _shutdown.wait(timeout=30)
            continue

        config._backlog_size = len(new_files)
        if len(new_files) > config.backlog_throttle_threshold > 0:
            logger.info(
                "Backlog important (%d fichiers) — FPS réduit : %.1f",
                len(new_files), config.effective_detector_fps,
            )

        status = db.get_processing_status()
        queue_done  = status.get("done", 0) + status.get("errors", 0)
        queue_total = queue_done + len(new_files)

        if new_files:
            for i, video_path in enumerate(new_files):
                if _shutdown.is_set():
                    break
                _process_file(
                    video_path, watcher, audio, detector, preprocessor, db,
                    queue_done + i, queue_total,
                )

        if not new_files:
            config._backlog_size = 0
            progress_tracker.set_idle()

        _shutdown.wait(timeout=watcher.config.scan_interval)

    logger.info("Boucle de traitement terminée.")


def _process_file(
    video_path: Path,
    watcher: FileWatcher,
    audio: AudioDetector,
    detector: VehicleDetector,
    preprocessor: AdaptivePreprocessor,
    db: Database,
    queue_done: int,
    queue_total: int,
):
    global _processing_start
    filename = video_path.name
    logger.info("=== Traitement : %s ===", filename)
    _processing_start = time.monotonic()

    err = _check_video(video_path)
    if err:
        logger.warning("Fichier ignoré — %s : %s", err, filename)
        db.mark_file_error(filename, err)
        progress_tracker.finish_file()
        return

    video_start_dt = watcher.extract_datetime(filename)
    progress_tracker.start_file(filename, queue_done, queue_total)
    start_time = time.monotonic()
    config = watcher.config

    try:
        # ── Vérifier checkpoint ──────────────────────────────────────────
        checkpoint = db.get_checkpoint(filename)

        if checkpoint and checkpoint.get("segments"):
            from .motion_filter import Segment as _Seg
            segments = [_Seg(s["start_sec"], s["end_sec"]) for s in checkpoint["segments"]]
            start_seg_idx  = checkpoint.get("cursor", 0)
            saved_crossings = checkpoint.get("crossings", [])
            logger.info(
                "%s — reprise depuis segment %d/%d (%d franchissement(s) déjà trouvés).",
                filename, start_seg_idx, len(segments), len(saved_crossings),
            )
            progress_tracker.set_phase("audio", frames_total=0)
        else:
            # ── Phase 1 : détection audio spectrale ─────────────────────
            progress_tracker.set_phase("audio", frames_total=0)
            try:
                segments = audio.analyze_video(video_path)
            except Exception as e:
                logger.warning("Audio échoué sur %s : %s — fichier ignoré.", filename, e)
                db.mark_file_error(filename, f"Filtre audio échoué : {e}")
                db.clear_checkpoint(filename)
                progress_tracker.finish_file()
                return

            if not segments:
                logger.info("%s — aucun segment actif.", filename)
                db.mark_file_done(
                    filename, vehicle_count=0,
                    duration_seconds=time.monotonic() - start_time,
                    detection_mode="audio_only",
                )
                db.clear_checkpoint(filename)
                progress_tracker.finish_file()
                return

            db.save_checkpoint(filename, segments, 0, [])
            start_seg_idx  = 0
            saved_crossings: list = []

        # ── Phase 2 : qualité de scène ───────────────────────────────────
        luminance, glare_frac = measure_scene_quality(video_path, config)
        preprocessor.log_scene(filename, luminance, glare_frac)
        scene_mode = preprocessor.scene_label(luminance, glare_frac)

        # Garde astronomique : cohérence heure solaire / luminance mesurée
        if video_start_dt is not None:
            sunrise_h, sunset_h = sunrise_sunset_local(
                config.latitude, config.longitude, config.timezone_offset, video_start_dt,
            )
            h_local = video_start_dt.hour + video_start_dt.minute / 60.0
            in_daylight = (sunrise_h + 0.5) <= h_local <= (sunset_h - 0.5)
            in_night    = h_local >= sunset_h + 0.75 or h_local <= sunrise_h - 0.75

            if scene_mode == "night" and in_daylight:
                # Image sous-exposée (ombre dense, tunnel) mais c'est le jour
                logger.info(
                    "%s — garde astro : %.1fh en plein jour (lever %.1fh/coucher %.1fh) "
                    "→ luminance ajustée à 80 (forçage CLAHE léger)",
                    filename, h_local, sunrise_h, sunset_h,
                )
                luminance = 80.0   # traitement crépuscule plutôt que nuit
                scene_mode = "twilight"
            elif scene_mode == "day" and in_night:
                # Lampadaire dans le ROI → luminance haute malgré la nuit
                logger.info(
                    "%s — garde astro : %.1fh en pleine nuit (lever %.1fh/coucher %.1fh) "
                    "→ luminance ajustée à 35 (forçage gamma nuit)",
                    filename, h_local, sunrise_h, sunset_h,
                )
                luminance = 35.0   # traitement nuit
                scene_mode = "night"

        # ── Phase 3 : détection visuelle YOLO adaptative ────────────────
        def on_segment_done(seg_idx: int, new_crossings_dicts: list):
            all_so_far = saved_crossings + new_crossings_dicts
            db.save_checkpoint(filename, segments, seg_idx + 1, all_so_far)

        def detection_progress(done: int, total: int):
            progress_tracker.set_phase("detection", frames_total=total)
            progress_tracker.update_frame(done)

        frame_sampler = FrameSampler(_DEBUG_FRAMES_DIR, filename)

        new_events = detector.process_video(
            video_path, segments, video_start_dt,
            on_progress=detection_progress,
            start_seg_idx=start_seg_idx,
            on_segment_done=on_segment_done,
            shutdown_event=_shutdown,
            frame_sampler=frame_sampler,
            preprocessor=preprocessor,
            scene_luminance=luminance,
            scene_glare_frac=glare_frac,
        )

        if _shutdown.is_set():
            logger.info("%s — traitement interrompu, reprise au prochain démarrage.", filename)
            progress_tracker.finish_file()
            return

        all_crossings = saved_crossings + [
            {
                "timestamp":    e.timestamp.isoformat(),
                "vehicle_type": e.vehicle_type,
                "direction":    e.direction,
                "confidence":   e.confidence,
                "source_file":  e.source_file,
            }
            for e in new_events
        ]

        # ── Confiance au son si visibilité mauvaise ──────────────────────
        # En nuit/contre-jour/crépuscule, un segment audio sans détection visuelle
        # est très probablement un vrai véhicule que YOLO n'a pas su voir.
        visual_ok = preprocessor.visual_reliable(luminance, glare_frac)
        audio_trusted = 0
        if config.audio_trust_low_visibility and not visual_ok and segments:
            synth = _synthesize_audio_crossings(segments, all_crossings, video_start_dt, filename)
            if synth:
                all_crossings.extend(synth)
                audio_trusted = len(synth)
                logger.info(
                    "%s — confiance au son : %d segment(s) sans détection visuelle "
                    "→ +%d véhicule(s) [lum=%.0f, glare=%.0f%%]",
                    filename, audio_trusted, audio_trusted, luminance, glare_frac * 100,
                )

        db.replace_crossings_for_file(filename, all_crossings)
        total_count = len(all_crossings)

        # ── Fallback audio global ──────────────────────────────────────
        audio_fallback = False
        if total_count == 0 and segments:
            total_count = len(segments)
            audio_fallback = True
            logger.info(
                "%s — YOLO=0 — fallback audio : %d segment(s) → %d véhicule(s)",
                filename, len(segments), total_count,
            )

        if frame_sampler.saved_frames:
            db.add_debug_frames(filename, frame_sampler.saved_frames)

        db.mark_file_done(
            filename,
            vehicle_count=total_count,
            duration_seconds=time.monotonic() - start_time,
            detection_mode="audio_fallback" if audio_fallback else scene_mode,
            vehicles_yolo=len(new_events),
            vehicles_night=None,
            audio_segments=len(segments),
        )
        db.clear_checkpoint(filename)
        progress_tracker.finish_file()
        logger.info(
            "%s — %d véhicule(s) [mode=%s, yolo=%d, synth=%d].",
            filename, total_count, scene_mode, len(new_events), audio_trusted,
        )

    except Exception as e:
        logger.exception("Erreur lors du traitement de %s : %s", filename, e)
        db.mark_file_error(filename, str(e))
        db.clear_checkpoint(filename)
        progress_tracker.finish_file()
    finally:
        _processing_start = None


def main():
    config_path = os.environ.get("CONFIG_PATH", "/app/data/config.yaml")
    config = load_config(config_path)

    log_dir = Path("/app/data/logs")
    setup_logging(
        config.log_level,
        log_dir,
        config.get("logging", "max_size_mb", default=10),
        config.get("logging", "backup_count", default=3),
    )

    logger.info("=== Démarrage — pipeline adaptatif unifié ===")
    logger.info("Dossier vidéos : %s", config.video_folder)
    logger.info("Base de données: %s", config.db_path)
    logger.info("Dashboard      : http://0.0.0.0:%d", config.dashboard_port)

    db = Database(config.db_path, timezone=config.timezone)
    _auto_purge(config, db)
    _track_detection_fingerprint(config, db)

    watcher      = FileWatcher(config, db)
    audio        = AudioDetector(config, db)
    detector     = VehicleDetector(config)
    preprocessor = AdaptivePreprocessor(config)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    threading.Thread(target=_watchdog_loop, daemon=True, name="watchdog").start()

    threading.Thread(
        target=bootstrap_calibration,
        args=(audio, config.video_folder, _shutdown),
        daemon=True,
        name="audio-bootstrap",
    ).start()

    threading.Thread(
        target=run_dashboard, args=(config, db, audio), daemon=True, name="dashboard"
    ).start()

    try:
        processing_loop(watcher, audio, detector, preprocessor, db)
    except KeyboardInterrupt:
        _shutdown.set()

    logger.info("=== Arrêt terminé ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
