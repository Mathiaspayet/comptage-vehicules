"""Point d'entrée principal — orchestre tous les modules en continu."""
import logging
import logging.handlers
import os
import signal
import sys
import threading
import time
from pathlib import Path

from .calibration import run_calibration
from .config import load_config
from .dashboard import run_dashboard
from .database import Database
from .detector import VehicleDetector
from .ingestion import FileWatcher
from .motion_filter import MotionFilter


def setup_logging(level: str, log_dir: Path, max_mb: int, backup_count: int):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "comptage.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Quieten noisy third-party loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)

_shutdown = threading.Event()


def _handle_signal(signum, frame):
    logger.info("Signal %s reçu — arrêt en cours…", signum)
    _shutdown.set()


def processing_loop(watcher: FileWatcher, motion: MotionFilter, detector: VehicleDetector, db: Database):
    """Main processing loop — runs until _shutdown is set."""
    logger.info("Boucle de traitement démarrée.")
    while not _shutdown.is_set():
        try:
            new_files = watcher.scan_new_files()
        except Exception as e:
            logger.error("Erreur lors du scan des fichiers : %s", e)
            _shutdown.wait(timeout=30)
            continue

        for video_path in new_files:
            if _shutdown.is_set():
                break
            _process_file(video_path, watcher, motion, detector, db)

        # Wait for next scan cycle
        scan_interval = watcher.config.scan_interval
        logger.debug("Prochain scan dans %d secondes.", scan_interval)
        _shutdown.wait(timeout=scan_interval)

    logger.info("Boucle de traitement terminée.")


def _process_file(video_path, watcher, motion, detector, db):
    filename = video_path.name
    logger.info("=== Traitement : %s ===", filename)

    video_start_dt = watcher.extract_datetime(filename)

    try:
        # Step 1: motion filter
        segments = motion.analyze_video(video_path)
        if not segments:
            logger.info("%s — aucun segment actif, aucun véhicule.", filename)
            db.mark_file_done(filename, vehicle_count=0)
            return

        # Step 2: AI detection + counting
        events = detector.process_video(video_path, segments, video_start_dt)

        # Step 3: store in database
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

        db.mark_file_done(filename, vehicle_count=len(events))
        logger.info("%s — %d véhicule(s) compté(s).", filename, len(events))

    except Exception as e:
        logger.exception("Erreur lors du traitement de %s : %s", filename, e)
        db.mark_file_error(filename, str(e))


def main():
    config_path = os.environ.get("CONFIG_PATH", "/app/data/config.yaml")
    config = load_config(config_path)

    # Setup logging
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
    logger.info("Calibration      : http://0.0.0.0:%d", config.calibration_port)

    # Init components
    db = Database(config.db_path)
    watcher = FileWatcher(config, db)
    motion = MotionFilter(config)
    detector = VehicleDetector(config)

    # Graceful shutdown
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    # Start dashboard in background thread
    dashboard_thread = threading.Thread(
        target=run_dashboard,
        args=(config, db),
        daemon=True,
        name="dashboard",
    )
    dashboard_thread.start()

    # Start calibration tool in background thread
    calib_thread = threading.Thread(
        target=run_calibration,
        args=(config,),
        daemon=True,
        name="calibration",
    )
    calib_thread.start()

    # Run main processing loop (blocking)
    try:
        processing_loop(watcher, motion, detector, db)
    except KeyboardInterrupt:
        _shutdown.set()

    logger.info("=== Arrêt terminé ===")
    sys.exit(0)


if __name__ == "__main__":
    # Allow running as: python -m src.main  OR  python src/main.py
    main()
