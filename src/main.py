"""Point d'entrée principal — orchestre tous les modules en continu."""
import logging
import logging.handlers
import os
import signal
import sys
import threading
import time
from pathlib import Path

from .audio_filter import AudioFilter, bootstrap_calibration
from .config import load_config
from .dashboard import run_dashboard
from .database import Database
from .detector import CrossingEvent, VehicleDetector, sunrise_sunset_local
from .frame_sampler import FrameSampler, purge_old_frames
from .ingestion import FileWatcher
from .night_detector import NightDetector, measure_scene_lighting
from .progress import tracker as progress_tracker

_DEBUG_FRAMES_DIR = Path("/app/data/debug_frames")
_DEBUG_FRAMES_MAX_AGE_H = 48


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
_PROCESSING_TIMEOUT_SEC = 7200  # watchdog global : 2h max par fichier

# Adaptation dynamique du mode crépuscule :
# si le détecteur de phares (ou YOLO) ne trouve rien sur N fichiers consécutifs
# pendant une session crépusculaire, on le désactive pour les suivants.
_flash_dead_streak: int = 0  # flash=0 & YOLO>0 consécutifs en crépuscule
_yolo_dead_streak: int = 0   # YOLO=0 & flash>0 consécutifs en crépuscule
_ADAPT_STREAK: int = 2       # nombre de fichiers avant basculement


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
    """Delete crossings older than data_retention_days and debug frames older than 48h."""
    days = config.data_retention_days
    if days > 0:
        count = db.delete_old_crossings(days)
        if count:
            logger.info("Rétention des données : %d franchissement(s) de plus de %d jours supprimé(s).", count, days)

    # Purge des frames de debug (DB + disque)
    old_paths = db.delete_old_debug_frames(_DEBUG_FRAMES_MAX_AGE_H)
    if old_paths:
        import os
        for p in old_paths:
            try:
                os.remove(p)
            except OSError:
                pass
    purge_old_frames(_DEBUG_FRAMES_DIR, _DEBUG_FRAMES_MAX_AGE_H)


def _merge_crossing_events(
    events_yolo: list,
    events_night: list,
    merge_window_sec: float = 4.0,
) -> list:
    """Fusionne les événements de deux détecteurs en éliminant les doublons temporels.

    Si un événement YOLO et un événement nuit (phares) sont séparés de moins de
    merge_window_sec, c'est le même véhicule détecté par les deux méthodes.
    On garde celui avec la meilleure confiance. Les événements uniques à un seul
    détecteur sont conservés tels quels.
    """
    if not events_yolo and not events_night:
        return []

    tagged: list[tuple] = (
        [(e, "yolo") for e in events_yolo]
        + [(e, "night") for e in events_night]
    )
    tagged.sort(key=lambda x: x[0].timestamp)

    merged: list = []
    skip: set[int] = set()

    for i, (ev_i, src_i) in enumerate(tagged):
        if i in skip:
            continue
        best = ev_i
        for j in range(i + 1, len(tagged)):
            if j in skip:
                continue
            ev_j, src_j = tagged[j]
            dt = (ev_j.timestamp - ev_i.timestamp).total_seconds()
            if dt > merge_window_sec:
                break
            if src_j != src_i:
                # Même véhicule détecté par les deux méthodes — garder la meilleure
                if ev_j.confidence > best.confidence:
                    best = ev_j
                skip.add(j)
                break  # un seul doublon par événement
        merged.append(best)

    return merged


def _synthesize_audio_crossings(
    segments: list,
    existing: list[dict],
    video_start_dt,
    source_name: str,
) -> list[dict]:
    """Crée 1 véhicule par segment audio actif resté SANS détection visuelle.

    Utilisé en faible visibilité (contre-jour, crépuscule, nuit) où le son du moteur
    est fiable mais l'image trompe les détecteurs. On mappe chaque crossing visuel
    existant sur sa seconde dans la vidéo et on ne synthétise que pour les segments
    réellement vides. Le véhicule synthétisé est placé au milieu du segment, type
    « car » par défaut, confiance basse (0.2) pour rester identifiable.
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
            # Sans horodatage vidéo fiable, on ne peut pas mapper → ne synthétise que
            # si rien n'a été détecté du tout.
            covered = len(existing) > 0
        if covered:
            continue
        mid = (seg.start_sec + seg.end_sec) / 2.0
        ts = (video_start_dt + timedelta(seconds=mid)) if video_start_dt else datetime.utcnow()
        out.append({
            "timestamp": ts.isoformat(),
            "vehicle_type": "car",
            "direction": None,
            "confidence": 0.2,
            "source_file": source_name,
        })
    return out


def _track_detection_fingerprint(config, db: Database):
    """Track the detection fingerprint for informational purposes only.

    A config change NEVER wipes the history automatically: the new parameters
    apply to future files only. To re-process old files, the user selects them
    explicitly in the dashboard ("Remettre en attente")."""
    current_fp = config.detection_fingerprint()
    stored_fp = db.get_state("detection_fingerprint")
    db.set_state("detection_fingerprint", current_fp)

    if stored_fp is None:
        logger.info("Empreinte de config enregistrée : %s", current_fp)
    elif stored_fp != current_fp:
        logger.info(
            "Paramètres de détection modifiés (ancienne=%s, nouvelle=%s) — "
            "appliqués aux nouveaux fichiers uniquement. "
            "Pour retraiter d'anciens fichiers, utilisez « Remettre en attente » dans le tableau de bord.",
            stored_fp, current_fp,
        )


def processing_loop(watcher: FileWatcher, audio: AudioFilter, detector: VehicleDetector,
                    night: NightDetector, db: Database):
    logger.info("Boucle de traitement démarrée (pipeline audio-only).")
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

            _track_detection_fingerprint(config, db)

        try:
            new_files = watcher.scan_new_files()
        except Exception as e:
            logger.error("Erreur lors du scan des fichiers : %s", e)
            _shutdown.wait(timeout=30)
            continue

        config._backlog_size = len(new_files)
        if len(new_files) > config.backlog_throttle_threshold > 0:
            logger.info(
                "Backlog important (%d fichiers) — FPS détection réduit : %.1f",
                len(new_files), config.effective_detector_fps,
            )

        status = db.get_processing_status()
        queue_done = status.get("done", 0) + status.get("errors", 0)
        queue_total = queue_done + len(new_files)

        if new_files:
            for i, video_path in enumerate(new_files):
                if _shutdown.is_set():
                    break
                _process_file(
                    video_path, watcher, audio, detector, night, db,
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
    audio: AudioFilter,
    detector: VehicleDetector,
    night: NightDetector,
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

    try:
        # ── Vérifier checkpoint ──────────────────────────────────────
        checkpoint = db.get_checkpoint(filename)

        if checkpoint and checkpoint.get("segments"):
            from .motion_filter import Segment as _Seg
            segments = [_Seg(s["start_sec"], s["end_sec"]) for s in checkpoint["segments"]]
            start_seg_idx = checkpoint.get("cursor", 0)
            saved_crossings = checkpoint.get("crossings", [])
            logger.info(
                "%s — reprise depuis segment %d/%d (%d franchissement(s) déjà trouvés).",
                filename, start_seg_idx, len(segments), len(saved_crossings),
            )
            progress_tracker.set_phase("audio", frames_total=0)  # skip audio
        else:
            # ── Phase 1 : filtre audio (détecteur principal) ────────────
            progress_tracker.set_phase("audio", frames_total=0)
            try:
                segments = audio.analyze_video(video_path)
            except Exception as e:
                logger.warning("Filtre audio échoué sur %s : %s — fichier ignoré.", filename, e)
                db.mark_file_error(filename, f"Filtre audio échoué : {e}")
                db.clear_checkpoint(filename)
                progress_tracker.finish_file()
                return

            if not segments:
                if watcher.config.audio_enabled and audio.is_calibrated():
                    # Audio calibré → silence = pas de véhicule → terminé
                    logger.info("%s — aucun segment actif (audio calibré).", filename)
                    db.mark_file_done(filename, vehicle_count=0,
                                      duration_seconds=time.monotonic() - start_time,
                                      detection_mode="audio_only")
                    db.clear_checkpoint(filename)
                    progress_tracker.finish_file()
                    return
                else:
                    # Audio désactivé ou calibration incomplète → YOLO sur vidéo complète
                    reason = "désactivé" if not watcher.config.audio_enabled else "calibration en cours"
                    logger.info("%s — audio %s — YOLO sur vidéo complète.", filename, reason)
                    from .motion_filter import Segment as _FullSeg
                    segments = [_FullSeg(0.0, 9999.0)]

            # Sauvegarder le checkpoint audio (segments connus, détection pas encore commencée)
            db.save_checkpoint(filename, segments, 0, [])
            start_seg_idx = 0
            saved_crossings: list = []

        # ── Phase 2 : détection — YOLO (jour) ou phares (nuit) ──────
        def on_segment_done(seg_idx: int, new_crossings_dicts: list):
            all_so_far = saved_crossings + new_crossings_dicts
            db.save_checkpoint(filename, segments, seg_idx + 1, all_so_far)

        def detection_progress(done: int, total: int):
            progress_tracker.set_phase("detection", frames_total=total)
            progress_tracker.update_frame(done)

        # ── Choix du mode : nuit / crépuscule / jour ─────────────────
        global _flash_dead_streak, _yolo_dead_streak
        mode = "day"
        brightness = 128.0
        glare_frac = 0.0
        if night.config.night_detection_enabled:
            brightness, glare_frac = measure_scene_lighting(video_path, night.config)
            if brightness < night.config.night_brightness_threshold:
                mode = "night"
            elif brightness < night.config.night_twilight_threshold:
                mode = "twilight"
            # Contre-jour : soleil directement dans le viseur. Même si la luminosité
            # (ROI route) reste correcte, un fort glare rend YOLO peu fiable → on
            # bascule au moins en crépuscule pour activer les phares + le filet audio.
            if mode == "day" and glare_frac >= night.config.glare_twilight_fraction:
                mode = "twilight"
                logger.info(
                    "%s — contre-jour détecté (%.0f%% surexposé) → CRÉPUSCULE forcé",
                    filename, glare_frac * 100,
                )

            # ── Garde astronomique ───────────────────────────────────
            # La luminosité mesurée peut se tromper : route à l'ombre au lever
            # du soleil → "nuit" en plein jour (phares inutiles, YOLO jamais
            # lancé) ; lampadaire dans le ROI → "jour" en pleine nuit (filet
            # audio désactivé). L'heure solaire borne le mode : le crépuscule
            # (YOLO + phares + confiance audio) arbitre les cas ambigus.
            if video_start_dt is not None:
                sunrise_h, sunset_h = sunrise_sunset_local(
                    config.latitude, config.longitude, config.timezone_offset, video_start_dt,
                )
                h_local = video_start_dt.hour + video_start_dt.minute / 60.0
                if mode == "night" and (sunrise_h + 0.5) <= h_local <= (sunset_h - 0.5):
                    mode = "twilight"
                    logger.info(
                        "%s — garde astro : %.1fh est en plein jour (lever %.1fh / coucher %.1fh) "
                        "→ NUIT corrigé en CRÉPUSCULE",
                        filename, h_local, sunrise_h, sunset_h,
                    )
                elif mode == "day" and (h_local >= sunset_h + 0.75 or h_local <= sunrise_h - 0.75):
                    mode = "twilight"
                    logger.info(
                        "%s — garde astro : %.1fh est en pleine nuit (lever %.1fh / coucher %.1fh) "
                        "→ JOUR corrigé en CRÉPUSCULE",
                        filename, h_local, sunrise_h, sunset_h,
                    )

        # ── Adaptation dynamique de la session crépuscule ────────────
        # Hors crépuscule : on sort de la session → reset des compteurs
        brightness_mode = mode
        if brightness_mode != "twilight":
            _flash_dead_streak = 0
            _yolo_dead_streak = 0
        elif _flash_dead_streak >= _ADAPT_STREAK:
            # Flash inutile depuis N fichiers → on bascule en YOLO seul
            mode = "day"
            logger.info(
                "%s — crépuscule adaptatif → JOUR (flash=0 depuis %d fichier(s))",
                filename, _flash_dead_streak,
            )
        elif _yolo_dead_streak >= _ADAPT_STREAK:
            # YOLO inutile depuis N fichiers → on bascule en phares seul
            mode = "night"
            logger.info(
                "%s — crépuscule adaptatif → NUIT (YOLO=0 depuis %d fichier(s))",
                filename, _yolo_dead_streak,
            )

        MODE_LABELS = {
            "day": "JOUR (YOLO)",
            "twilight": "CRÉPUSCULE (YOLO + phares)",
            "night": "NUIT (phares)",
        }
        logger.info(
            "%s — luminosité ROI=%.1f surexposé=%.0f%% → mode %s",
            filename, brightness, glare_frac * 100, MODE_LABELS[mode],
        )

        if mode == "twilight":
            # En mode crépuscule on relance les deux détecteurs depuis 0 — le checkpoint
            # d'un run précédent (potentiellement dans un autre mode) ne s'applique pas.
            saved_crossings = []
            db.clear_checkpoint(filename)

        det_yolo: int | None = None   # nb véhicules détectés par YOLO
        det_night: int | None = None  # nb véhicules détectés par phares

        # ── Debug frames : une paire raw/proc par segment actif (max 3) ──
        frame_sampler = FrameSampler(_DEBUG_FRAMES_DIR, filename)

        if mode == "night":
            new_events = night.process_video(
                video_path, segments, video_start_dt,
                on_progress=detection_progress,
                start_seg_idx=start_seg_idx,
                on_segment_done=on_segment_done,
                shutdown_event=_shutdown,
                frame_sampler=frame_sampler,
            )
            det_night = len(saved_crossings) + len(new_events)
        elif mode == "day":
            new_events = detector.process_video(
                video_path, segments, video_start_dt,
                on_progress=detection_progress,
                start_seg_idx=start_seg_idx,
                on_segment_done=on_segment_done,
                shutdown_event=_shutdown,
                frame_sampler=frame_sampler,
            )
            det_yolo = len(saved_crossings) + len(new_events)
        else:
            # Mode crépuscule : les deux détecteurs, pas de checkpoint partiel
            # (on attend la fin complète des deux avant de sauvegarder)
            events_yolo = detector.process_video(
                video_path, segments, video_start_dt,
                on_progress=detection_progress,
                shutdown_event=_shutdown,
                frame_sampler=frame_sampler,
            )
            if _shutdown.is_set():
                logger.info("%s — traitement interrompu (YOLO crépuscule).", filename)
                progress_tracker.finish_file()
                return
            events_night = night.process_video(
                video_path, segments, video_start_dt,
                on_progress=detection_progress,
                shutdown_event=_shutdown,
            )
            new_events = _merge_crossing_events(
                events_yolo, events_night,
                merge_window_sec=night.config.night_merge_window_sec,
            )
            det_yolo = len(events_yolo)
            det_night = len(events_night)
            logger.info(
                "%s — crépuscule : YOLO=%d phares=%d après fusion=%d",
                filename, det_yolo, det_night, len(new_events),
            )

        if _shutdown.is_set():
            # Arrêt demandé en cours de détection — le checkpoint est déjà sauvegardé
            # segment par segment. On ne marque pas le fichier comme terminé.
            logger.info("%s — traitement interrompu, reprise au prochain démarrage.", filename)
            progress_tracker.finish_file()
            return

        all_crossings = saved_crossings + [
            {
                "timestamp": e.timestamp.isoformat(),
                "vehicle_type": e.vehicle_type,
                "direction": e.direction,
                "confidence": e.confidence,
                "source_file": e.source_file,
            }
            for e in new_events
        ]

        # ── Confiance au son par segment (conditions de faible visibilité) ──────────
        # Le bruit du moteur est une donnée très fiable. En contre-jour / crépuscule /
        # nuit, un segment audio actif SANS aucune détection visuelle est presque
        # certainement un véhicule que la caméra n'a pas su voir. On l'insiste : on
        # compte ce segment comme 1 véhicule (plutôt que de le traiter en faux positif).
        # En plein jour clair, on garde la prudence : un segment muet visuellement
        # reste un possible faux positif audio (porte, vent, voix) et n'est pas compté.
        visual_unreliable = (mode != "day") or (glare_frac >= night.config.glare_twilight_fraction)
        audio_trusted = 0
        if night.config.audio_trust_low_visibility and visual_unreliable and segments:
            synth = _synthesize_audio_crossings(segments, all_crossings, video_start_dt, filename)
            if synth:
                all_crossings.extend(synth)
                audio_trusted = len(synth)
                logger.info(
                    "%s — confiance au son : %d segment(s) actif(s) sans détection visuelle "
                    "→ +%d véhicule(s) (visibilité faible, mode=%s)",
                    filename, audio_trusted, audio_trusted, mode,
                )

        # Remplace les crossings existants du fichier (DELETE + INSERT atomiques)
        db.replace_crossings_for_file(filename, all_crossings)

        total_count = len(all_crossings)

        # ── Fallback audio global : dernier recours si TOUT est resté à 0 (vidéo
        # inexploitable + visuel et son par segment n'ont rien donné). Chaque segment
        # actif compte alors pour 1 véhicule.
        audio_fallback = False
        if total_count == 0 and segments:
            total_count = len(segments)
            audio_fallback = True
            logger.info(
                "%s — détecteur visuel : 0 véhicule(s) — fallback audio : %d segment(s) → %d véhicule(s)",
                filename, len(segments), total_count,
            )

        # Enregistrer les debug frames en DB
        if frame_sampler.saved_frames:
            db.add_debug_frames(filename, frame_sampler.saved_frames)

        db.mark_file_done(
            filename,
            vehicle_count=total_count,
            duration_seconds=time.monotonic() - start_time,
            detection_mode="audio_fallback" if audio_fallback else mode,
            vehicles_yolo=det_yolo,
            vehicles_night=det_night,
            audio_segments=len(segments),
        )
        db.clear_checkpoint(filename)
        progress_tracker.finish_file()
        logger.info("%s — %d véhicule(s) compté(s) [mode=%s].", filename, total_count,
                    "audio_fallback" if audio_fallback else mode)

        # ── Mise à jour du streak adaptatif (seulement si les deux détecteurs ont tourné)
        if brightness_mode == "twilight" and mode == "twilight":
            yolo_c  = det_yolo  if det_yolo  is not None else 0
            night_c = det_night if det_night is not None else 0
            if night_c == 0 and yolo_c > 0:
                _flash_dead_streak += 1
                _yolo_dead_streak = 0
            elif yolo_c == 0 and night_c > 0:
                _yolo_dead_streak += 1
                _flash_dead_streak = 0

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

    logger.info("=== Démarrage du système de comptage de véhicules ===")
    logger.info("Dossier vidéos   : %s", config.video_folder)
    logger.info("Base de données  : %s", config.db_path)
    logger.info("Dashboard        : http://0.0.0.0:%d", config.dashboard_port)
    logger.info("Calibration      : http://0.0.0.0:%d/calibration/", config.dashboard_port)

    db = Database(config.db_path, timezone=config.timezone)
    _auto_purge(config, db)
    _track_detection_fingerprint(config, db)
    watcher = FileWatcher(config, db)
    audio = AudioFilter(config, db)
    detector = VehicleDetector(config)
    night = NightDetector(config)

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
        target=run_dashboard, args=(config, db, audio), daemon=True, name="dashboard"
    )
    dashboard_thread.start()

    try:
        processing_loop(watcher, audio, detector, night, db)
    except KeyboardInterrupt:
        _shutdown.set()

    logger.info("=== Arrêt terminé ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
