"""Point d'entrée principal — orchestre tous les modules en continu."""
import logging
import logging.handlers
import os
import signal
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from .audio_filter import AudioFilter, bootstrap_calibration, union_segments
from .config import load_config
from .dashboard import run_dashboard
from .database import Database
from .detector import VehicleDetector
from .ingestion import FileWatcher
from .motion_filter import MotionFilter, Segment
from .progress import tracker as progress_tracker


def _check_video(path: Path) -> str | None:
    """Quick sanity check — returns an error string if the file can't be decoded, None if OK."""
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


def _run_motion_silent(path: Path, motion: MotionFilter) -> list[Segment]:
    """Run motion filter without progress tracking (used for background prefetch)."""
    return motion.analyze_video(path)


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
_PROCESSING_TIMEOUT_SEC = 900   # watchdog global : 15 min max par fichier
_MOTION_TIMEOUT_SEC = 600       # timeout spécifique au filtre mouvement : 10 min


def _handle_signal(signum, frame):
    logger.info("Signal %s reçu — arrêt en cours… (fichier en cours terminé avant fermeture)", signum)
    _shutdown.set()


def _watchdog_loop():
    """Daemon thread: force-kills the process if a single file takes too long."""
    while not _shutdown.is_set():
        _shutdown.wait(timeout=30)
        start = _processing_start
        if start is not None:
            elapsed = time.monotonic() - start
            if elapsed > _PROCESSING_TIMEOUT_SEC:
                logger.error(
                    "Timeout de traitement : bloqué depuis %d s — arrêt forcé.",
                    int(elapsed),
                )
                import os as _os
                _os.kill(_os.getpid(), signal.SIGTERM)
                return


def _auto_purge(config, db: Database):
    """Delete crossings older than data_retention_days if configured (> 0)."""
    days = config.data_retention_days
    if days <= 0:
        return
    count = db.delete_old_crossings(days)
    if count:
        logger.info("Rétention des données : %d franchissement(s) de plus de %d jours supprimé(s).", count, days)


def _check_and_reset_if_config_changed(config, db: Database):
    """Compare the stored detection fingerprint with the current one.
    If they differ, wipe processed_files + crossings so everything is re-processed."""
    current_fp = config.detection_fingerprint()
    stored_fp = db.get_state("detection_fingerprint")

    if stored_fp is None:
        # First run — just store the fingerprint, don't reset
        db.set_state("detection_fingerprint", current_fp)
        logger.info("Empreinte de config enregistrée : %s", current_fp)
        return

    if stored_fp != current_fp:
        logger.warning(
            "Paramètres de détection modifiés (ancienne empreinte=%s, nouvelle=%s) — "
            "suppression de l'historique pour retraiter tous les fichiers.",
            stored_fp, current_fp,
        )
        progress_tracker.set_resetting("Paramètres modifiés — réinitialisation en cours…")
        count = db.unmark_all_files()
        db.set_state("detection_fingerprint", current_fp)
        progress_tracker.clear_resetting()
        logger.info("Réinitialisation terminée — %d fichier(s) marqués à retraiter.", count)
    else:
        logger.info("Empreinte de config inchangée — pas de réinitialisation.")


def processing_loop(watcher: FileWatcher, motion: MotionFilter, audio: AudioFilter, detector: VehicleDetector, db: Database):
    logger.info("Boucle de traitement démarrée.")
    config = watcher.config
    _prev_imgsz = config.imgsz
    _prev_model_name = config.model_name
    _prev_log_level = config.log_level

    # Single background thread for motion prefetch (runs in parallel with AI detection)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="motion-prefetch") as prefetch_pool:
        while not _shutdown.is_set():
            # Hot-reload config from disk before each scan
            changed = config.reload()
            if changed:
                new_log_level = config.log_level
                if new_log_level != _prev_log_level:
                    logging.getLogger().setLevel(getattr(logging, new_log_level.upper(), logging.INFO))
                    logger.info("Niveau de log mis à jour : %s", new_log_level)
                    _prev_log_level = new_log_level

                new_imgsz = config.imgsz
                new_model_name = config.model_name
                if new_imgsz != _prev_imgsz or new_model_name != _prev_model_name:
                    logger.info(
                        "Modèle/imgsz modifié (%s imgsz=%d → %s imgsz=%d) — rechargement au prochain fichier.",
                        _prev_model_name, _prev_imgsz, new_model_name, new_imgsz,
                    )
                    detector._model = None
                    _prev_imgsz = new_imgsz
                    _prev_model_name = new_model_name

                _check_and_reset_if_config_changed(config, db)

            try:
                new_files = watcher.scan_new_files()
            except Exception as e:
                logger.error("Erreur lors du scan des fichiers : %s", e)
                _shutdown.wait(timeout=30)
                continue

            # Adaptive fps: inform config of current backlog size
            config._backlog_size = len(new_files)
            if len(new_files) > config.backlog_throttle_threshold > 0:
                logger.info(
                    "Backlog important (%d fichiers) — FPS réduit : motion=%.1f détection=%.1f",
                    len(new_files), config.effective_motion_fps, config.effective_detector_fps,
                )

            # Update queue size in progress tracker
            status = db.get_processing_status()
            queue_done = status.get("done", 0) + status.get("errors", 0)
            queue_total = queue_done + len(new_files)

            if new_files:
                motion_fp = config.motion_fingerprint()

                # Pre-submit motion filter for files not yet in cache (pipeline)
                prefetch: dict[str, Future] = {}
                if config.prefetch_motion:
                    for vp in new_files:
                        if db.get_motion_cache(vp.name, motion_fp) is None:
                            prefetch[vp.name] = prefetch_pool.submit(
                                _run_motion_silent, vp, motion
                            )
                    if prefetch:
                        logger.debug("Pré-calcul mouvement lancé pour %d fichier(s).", len(prefetch))

                for i, video_path in enumerate(new_files):
                    if _shutdown.is_set():
                        break
                    _process_file(
                        video_path, watcher, motion, audio, detector, db,
                        queue_done + i, queue_total,
                        motion_fp=motion_fp,
                        prefetch_future=prefetch.get(video_path.name),
                    )

            if not new_files:
                config._backlog_size = 0
                progress_tracker.set_idle()

            _shutdown.wait(timeout=watcher.config.scan_interval)

    logger.info("Boucle de traitement terminée.")


def _process_file(
    video_path: Path,
    watcher: FileWatcher,
    motion: MotionFilter,
    audio: AudioFilter,
    detector: VehicleDetector,
    db: Database,
    queue_done: int,
    queue_total: int,
    motion_fp: str | None = None,
    prefetch_future: "Future | None" = None,
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

    try:
        # ── Phase 1 : filtre mouvement ──────────────────────────────
        cached_data = db.get_motion_cache(filename, motion_fp) if motion_fp else None

        if cached_data is not None:
            segments = [Segment(start_sec=d["start_sec"], end_sec=d["end_sec"]) for d in cached_data]
            logger.debug("%s — %d segment(s) depuis le cache de mouvement.", filename, len(segments))
            progress_tracker.set_phase("motion", frames_total=0)
        elif prefetch_future is not None:
            # Motion already running in background — wait for result (with timeout)
            progress_tracker.set_phase("motion", frames_total=0)
            try:
                segments = prefetch_future.result(timeout=_MOTION_TIMEOUT_SEC)
            except TimeoutError:
                logger.error(
                    "Timeout filtre mouvement sur %s (>%ds) — fichier marqué en erreur.",
                    filename, _MOTION_TIMEOUT_SEC,
                )
                db.mark_file_error(filename, f"Filtre de mouvement bloqué (timeout >{_MOTION_TIMEOUT_SEC}s)")
                progress_tracker.finish_file()
                return
            if motion_fp:
                db.set_motion_cache(filename, motion_fp,
                                    [{"start_sec": s.start_sec, "end_sec": s.end_sec} for s in segments])
        else:
            # Fallback: run synchronously in a thread so we can apply a timeout
            def motion_progress(done: int, total: int):
                progress_tracker.set_phase("motion", frames_total=total)
                progress_tracker.update_frame(done)

            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="motion-sync") as _ex:
                _fut = _ex.submit(motion.analyze_video, video_path, motion_progress)
                try:
                    segments = _fut.result(timeout=_MOTION_TIMEOUT_SEC)
                except TimeoutError:
                    logger.error(
                        "Timeout filtre mouvement sur %s (>%ds) — fichier marqué en erreur.",
                        filename, _MOTION_TIMEOUT_SEC,
                    )
                    db.mark_file_error(filename, f"Filtre de mouvement bloqué (timeout >{_MOTION_TIMEOUT_SEC}s)")
                    progress_tracker.finish_file()
                    return
            if motion_fp:
                db.set_motion_cache(filename, motion_fp,
                                    [{"start_sec": s.start_sec, "end_sec": s.end_sec} for s in segments])

        # ── Phase 1b : filtre audio ────────────────────────────────────
        try:
            progress_tracker.set_phase("audio", frames_total=0)
            audio_segments = audio.analyze_video(video_path)
            if audio_segments:
                segments = union_segments(segments, audio_segments)
                logger.info("%s — %d segment(s) après fusion mouvement+audio.", filename, len(segments))
        except Exception as e:
            logger.warning("Filtre audio échoué sur %s : %s", filename, e)

        if not segments:
            logger.info("%s — aucun segment actif.", filename)
            db.mark_file_done(filename, vehicle_count=0, duration_seconds=time.monotonic() - start_time)
            progress_tracker.finish_file()
            return

        # ── Phase 2 : détection IA ──────────────────────────────────
        def detection_progress(done: int, total: int):
            progress_tracker.set_phase("detection", frames_total=total)
            progress_tracker.update_frame(done)

        events = detector.process_video(
            video_path, segments, video_start_dt, on_progress=detection_progress
        )

        if events:
            crossings = [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "vehicle_type": e.vehicle_type,
                    "direction": e.direction,
                    "confidence": e.confidence,
                    "source_file": e.source_file,
                }
                for e in events
            ]
            db.insert_crossings_batch(crossings)

        db.mark_file_done(filename, vehicle_count=len(events), duration_seconds=time.monotonic() - start_time)
        progress_tracker.finish_file()
        logger.info("%s — %d véhicule(s) compté(s).", filename, len(events))

    except Exception as e:
        logger.exception("Erreur lors du traitement de %s : %s", filename, e)
        db.mark_file_error(filename, str(e))
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

    logger.info("=== Démarrage du système de comptage de véhicules ===")
    logger.info("Dossier vidéos   : %s", config.video_folder)
    logger.info("Base de données  : %s", config.db_path)
    logger.info("Dashboard        : http://0.0.0.0:%d", config.dashboard_port)
    logger.info("Calibration      : http://0.0.0.0:%d/calibration/", config.dashboard_port)

    db = Database(config.db_path, timezone=config.timezone)
    _auto_purge(config, db)
    _check_and_reset_if_config_changed(config, db)
    watcher = FileWatcher(config, db)
    motion = MotionFilter(config)
    audio = AudioFilter(config, db)
    detector = VehicleDetector(config)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    threading.Thread(target=_watchdog_loop, daemon=True, name="watchdog").start()

    # Bootstrap calibration audio en arrière-plan sur les fichiers existants
    threading.Thread(
        target=bootstrap_calibration,
        args=(audio, config.video_folder, _shutdown),
        daemon=True,
        name="audio-bootstrap",
    ).start()

    # Un seul serveur web (dashboard + calibration)
    dashboard_thread = threading.Thread(
        target=run_dashboard, args=(config, db), daemon=True, name="dashboard"
    )
    dashboard_thread.start()

    try:
        processing_loop(watcher, motion, audio, detector, db)
    except KeyboardInterrupt:
        _shutdown.set()

    logger.info("=== Arrêt terminé ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
