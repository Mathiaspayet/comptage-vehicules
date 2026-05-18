"""Point d'entrée principal — orchestre tous les modules en continu."""
import logging
import logging.handlers
import os
import signal
import sys
import threading
import time
from pathlib import Path

from .config import load_config
from .dashboard import run_dashboard
from .database import Database
from .detector import VehicleDetector
from .ingestion import FileWatcher
from .motion_filter import MotionFilter
from .progress import tracker as progress_tracker


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


def _handle_signal(signum, frame):
    logger.info("Signal %s reçu — arrêt en cours…", signum)
    _shutdown.set()


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


def processing_loop(watcher: FileWatcher, motion: MotionFilter, detector: VehicleDetector, db: Database):
    logger.info("Boucle de traitement démarrée.")
    config = watcher.config
    _prev_imgsz = config.imgsz
    _prev_model_name = config.model_name
    _prev_log_level = config.log_level

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

        # Update queue size in progress tracker
        status = db.get_processing_status()
        queue_done = status.get("done", 0) + status.get("errors", 0)
        queue_total = queue_done + len(new_files)

        for i, video_path in enumerate(new_files):
            if _shutdown.is_set():
                break
            _process_file(video_path, watcher, motion, detector, db, queue_done + i, queue_total)

        if not new_files:
            progress_tracker.set_idle()

        _shutdown.wait(timeout=watcher.config.scan_interval)

    logger.info("Boucle de traitement terminée.")


def _process_file(video_path, watcher, motion, detector, db, queue_done: int, queue_total: int):
    filename = video_path.name
    logger.info("=== Traitement : %s ===", filename)
    video_start_dt = watcher.extract_datetime(filename)

    progress_tracker.start_file(filename, queue_done, queue_total)
    start_time = time.monotonic()

    try:
        # Phase 1 : filtre mouvement
        def motion_progress(done: int, total: int):
            progress_tracker.set_phase("motion", frames_total=total)
            progress_tracker.update_frame(done)

        segments = motion.analyze_video(video_path, on_progress=motion_progress)

        if not segments:
            logger.info("%s — aucun segment actif.", filename)
            db.mark_file_done(filename, vehicle_count=0, duration_seconds=time.monotonic() - start_time)
            progress_tracker.finish_file()
            return

        # Phase 2 : détection IA
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

    db = Database(config.db_path)
    _check_and_reset_if_config_changed(config, db)
    watcher = FileWatcher(config, db)
    motion = MotionFilter(config)
    detector = VehicleDetector(config)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # Un seul serveur web (dashboard + calibration)
    dashboard_thread = threading.Thread(
        target=run_dashboard, args=(config, db), daemon=True, name="dashboard"
    )
    dashboard_thread.start()

    try:
        processing_loop(watcher, motion, detector, db)
    except KeyboardInterrupt:
        _shutdown.set()

    logger.info("=== Arrêt terminé ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
