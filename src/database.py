"""SQLite database operations — stores crossings and processed files."""
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS crossings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,   -- ISO-8601 with timezone
    vehicle_type    TEXT    NOT NULL,   -- car, truck, bus, motorcycle
    direction       TEXT,               -- left_to_right, right_to_left, or NULL
    confidence      REAL    NOT NULL,
    source_file     TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS processed_files (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT    NOT NULL UNIQUE,
    processed_at    TEXT    NOT NULL,   -- ISO-8601
    status          TEXT    NOT NULL,   -- done, error
    vehicle_count   INTEGER NOT NULL DEFAULT 0,
    error_message   TEXT
);

CREATE TABLE IF NOT EXISTS app_state (
    key             TEXT    PRIMARY KEY,
    value           TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS motion_cache (
    filename     TEXT PRIMARY KEY,
    motion_fp    TEXT NOT NULL,
    segments_json TEXT NOT NULL,
    cached_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audio_stats (
    filename   TEXT PRIMARY KEY,
    mean_db    REAL,
    median_db  REAL,
    std_db     REAL,
    p10_db     REAL,
    p90_db     REAL,
    video_hour INTEGER,          -- heure locale de la vidéo (0-23), pour calibration nuit
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_crossings_timestamp ON crossings(timestamp);
CREATE INDEX IF NOT EXISTS idx_crossings_type      ON crossings(vehicle_type);
CREATE INDEX IF NOT EXISTS idx_crossings_source    ON crossings(source_file);
CREATE INDEX IF NOT EXISTS idx_processed_filename  ON processed_files(filename);
"""


class Database:
    def __init__(self, db_path: Path, timezone: str = "UTC"):
        self.db_path = db_path
        self._timezone = timezone
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _tz_mod(self) -> str:
        """SQLite datetime modifier for crossing timestamps.

        Crossing timestamps are stored as naive LOCAL time (extracted from video
        filenames which encode the camera's local clock). No offset is needed —
        returning '+0 hours' lets SQLite date/hour functions operate correctly.
        """
        return "+0 hours"

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            # Migrations — ALTER TABLE ignore les colonnes déjà existantes
            for migration in [
                "ALTER TABLE processed_files ADD COLUMN processing_duration_seconds REAL",
                "ALTER TABLE audio_stats ADD COLUMN video_hour INTEGER",
                "ALTER TABLE processed_files ADD COLUMN checkpoint_json TEXT",
                "ALTER TABLE processed_files ADD COLUMN detection_mode TEXT",
                "ALTER TABLE processed_files ADD COLUMN vehicles_yolo INTEGER",
                "ALTER TABLE processed_files ADD COLUMN vehicles_night INTEGER",
            ]:
                try:
                    conn.execute(migration)
                except sqlite3.OperationalError:
                    pass  # colonne déjà présente
        logger.debug("Schéma base de données initialisé : %s", self.db_path)

    # ------------------------------------------------------------------ #
    # Processed files                                                      #
    # ------------------------------------------------------------------ #

    def is_file_processed(self, filename: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM processed_files WHERE filename = ? AND status IN ('done', 'error', 'skipped')",
                (filename,),
            ).fetchone()
            return row is not None

    def mark_file_done(
        self,
        filename: str,
        vehicle_count: int,
        duration_seconds: float | None = None,
        detection_mode: str | None = None,
        vehicles_yolo: int | None = None,
        vehicles_night: int | None = None,
    ):
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO processed_files
                    (filename, processed_at, status, vehicle_count, processing_duration_seconds,
                     detection_mode, vehicles_yolo, vehicles_night)
                VALUES (?, ?, 'done', ?, ?, ?, ?, ?)
                ON CONFLICT(filename) DO UPDATE SET
                    processed_at = excluded.processed_at,
                    status = 'done',
                    vehicle_count = excluded.vehicle_count,
                    error_message = NULL,
                    processing_duration_seconds = excluded.processing_duration_seconds,
                    detection_mode = excluded.detection_mode,
                    vehicles_yolo = excluded.vehicles_yolo,
                    vehicles_night = excluded.vehicles_night
                """,
                (filename, now, vehicle_count, duration_seconds,
                 detection_mode, vehicles_yolo, vehicles_night),
            )

    def mark_file_error(self, filename: str, error_message: str):
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO processed_files (filename, processed_at, status, vehicle_count, error_message)
                VALUES (?, ?, 'error', 0, ?)
                ON CONFLICT(filename) DO UPDATE SET
                    processed_at = excluded.processed_at,
                    status = 'error',
                    error_message = excluded.error_message
                """,
                (filename, now, error_message),
            )

    # ------------------------------------------------------------------ #
    # Crossings                                                            #
    # ------------------------------------------------------------------ #

    def insert_crossing(
        self,
        timestamp: datetime,
        vehicle_type: str,
        direction: str | None,
        confidence: float,
        source_file: str,
    ):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO crossings (timestamp, vehicle_type, direction, confidence, source_file)
                VALUES (?, ?, ?, ?, ?)
                """,
                (timestamp.isoformat(), vehicle_type, direction, confidence, source_file),
            )

    def insert_crossings_batch(self, crossings: list[dict]):
        if not crossings:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO crossings (timestamp, vehicle_type, direction, confidence, source_file)
                VALUES (:timestamp, :vehicle_type, :direction, :confidence, :source_file)
                """,
                crossings,
            )

    def delete_old_crossings(self, days: int) -> int:
        """Delete crossings older than `days` days. Returns count deleted."""
        if days <= 0:
            return 0
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM crossings WHERE timestamp < datetime('now', ?)",
                (f"-{days} days",),
            )
            return cur.rowcount

    def get_crossings_export(
        self,
        date_from: "str | None" = None,
        date_to: "str | None" = None,
        vehicle_type: str = "all",
    ) -> list[dict]:
        """Returns all crossings matching filters, ordered by timestamp (for CSV export)."""
        tzmod = self._tz_mod()
        clauses: list[str] = []
        params: dict = {"tzmod": tzmod}
        if date_from:
            clauses.append("date(datetime(timestamp, :tzmod)) >= :date_from")
            params["date_from"] = date_from
        if date_to:
            clauses.append("date(datetime(timestamp, :tzmod)) <= :date_to")
            params["date_to"] = date_to
        if vehicle_type != "all":
            clauses.append("vehicle_type = :vtype")
            params["vtype"] = vehicle_type
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        query = f"""
            SELECT timestamp, vehicle_type, direction, confidence, source_file
            FROM crossings
            {where}
            ORDER BY timestamp
        """
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Statistics queries                                                   #
    # ------------------------------------------------------------------ #

    def get_hourly_stats(self, date_str: str, vehicle_type: str = "all") -> list[dict]:
        """Returns list of {hour, count} for a given date (YYYY-MM-DD) in local time."""
        tzmod = self._tz_mod()
        type_filter = "" if vehicle_type == "all" else "AND vehicle_type = :vtype"
        query = f"""
            SELECT
                CAST(strftime('%H', datetime(timestamp, :tzmod)) AS INTEGER) AS hour,
                COUNT(*) AS count
            FROM crossings
            WHERE date(datetime(timestamp, :tzmod)) = :date
            {type_filter}
            GROUP BY hour
            ORDER BY hour
        """
        params = {"date": date_str, "vtype": vehicle_type, "tzmod": tzmod}
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        hourly = {r["hour"]: r["count"] for r in rows}
        return [{"hour": h, "count": hourly.get(h, 0)} for h in range(24)]

    def get_daily_stats(self, days: int = 30, vehicle_type: str = "all") -> list[dict]:
        """Returns list of {date, count} for the last N days (dates in local time)."""
        tzmod = self._tz_mod()
        type_filter = "" if vehicle_type == "all" else "AND vehicle_type = :vtype"
        query = f"""
            SELECT
                date(datetime(timestamp, :tzmod)) AS day,
                COUNT(*) AS count
            FROM crossings
            WHERE date(datetime(timestamp, :tzmod)) >= date(datetime('now', :tzmod), :offset)
            {type_filter}
            GROUP BY day
            ORDER BY day
        """
        params = {"offset": f"-{days} days", "vtype": vehicle_type, "tzmod": tzmod}
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [{"date": r["day"], "count": r["count"]} for r in rows]

    def get_summary(self, date_str: str, vehicle_type: str = "all") -> dict:
        """Returns total, peak_hour, avg_per_hour for a given date."""
        hourly = self.get_hourly_stats(date_str, vehicle_type)
        total = sum(h["count"] for h in hourly)
        active_hours = [h for h in hourly if h["count"] > 0]
        peak = max(hourly, key=lambda h: h["count"]) if hourly else {"hour": 0, "count": 0}
        avg = total / len(active_hours) if active_hours else 0
        return {
            "total": total,
            "peak_hour": peak["hour"],
            "peak_count": peak["count"],
            "avg_per_active_hour": round(avg, 1),
        }

    def get_vehicle_type_breakdown(self, date_str: str) -> list[dict]:
        tzmod = self._tz_mod()
        query = """
            SELECT vehicle_type, COUNT(*) AS count
            FROM crossings
            WHERE date(datetime(timestamp, :tzmod)) = :date
            GROUP BY vehicle_type
            ORDER BY count DESC
        """
        with self._connect() as conn:
            rows = conn.execute(query, {"date": date_str, "tzmod": tzmod}).fetchall()
        return [{"vehicle_type": r["vehicle_type"], "count": r["count"]} for r in rows]

    def get_available_dates(self) -> list[str]:
        tzmod = self._tz_mod()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT date(datetime(timestamp, :tzmod)) AS day "
                "FROM crossings ORDER BY day DESC LIMIT 90",
                {"tzmod": tzmod},
            ).fetchall()
        return [r["day"] for r in rows]

    def get_processed_files(
        self,
        limit: int = 500,
        offset: int = 0,
        status_filter: str = "all",
        max_age_days: int = 30,
        sort_by: str = "processed_at",
        sort_dir: str = "desc",
    ) -> list[dict]:
        """Returns processed files with optional age filter and sorting."""
        where_clauses = []
        params: list = []

        if status_filter != "all":
            where_clauses.append("status = ?")
            params.append(status_filter)
        if max_age_days > 0:
            where_clauses.append("processed_at >= datetime('now', ?)")
            params.append(f"-{max_age_days} days")

        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        _valid_sort = {"processed_at", "filename", "vehicle_count", "processing_duration_seconds"}
        if sort_by not in _valid_sort:
            sort_by = "processed_at"
        order = f"{sort_by} {'DESC' if sort_dir.lower() == 'desc' else 'ASC'}"

        query = f"""
            SELECT filename, processed_at, status, vehicle_count, error_message,
                   processing_duration_seconds, detection_mode, vehicles_yolo, vehicles_night
            FROM processed_files
            {where}
            ORDER BY {order}
            LIMIT ? OFFSET ?
        """
        params += [limit, offset]
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_all_processed_filenames(self) -> set[str]:
        """Returns the set of all processed filenames (for pending file detection)."""
        with self._connect() as conn:
            rows = conn.execute("SELECT filename FROM processed_files").fetchall()
        return {r["filename"] for r in rows}

    def get_files_missing_audio_stats(self, limit: int) -> list[str]:
        """Returns filenames of 'done' files that have no audio stats yet."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT pf.filename
                FROM processed_files pf
                LEFT JOIN audio_stats ast ON pf.filename = ast.filename
                WHERE pf.status = 'done' AND ast.filename IS NULL
                ORDER BY pf.processed_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [r["filename"] for r in rows]

    def delete_crossings_for_files(self, filenames: list[str]) -> int:
        """Delete all crossings whose source_file is in the given list. Returns count deleted."""
        if not filenames:
            return 0
        placeholders = ",".join("?" * len(filenames))
        with self._connect() as conn:
            cur = conn.execute(
                f"DELETE FROM crossings WHERE source_file IN ({placeholders})", filenames
            )
            return cur.rowcount

    # ------------------------------------------------------------------ #
    # Checkpoint (reprise sur crash)                                       #
    # ------------------------------------------------------------------ #

    def save_checkpoint(self, filename: str, segments: list, cursor: int, crossings: list) -> None:
        """Sauvegarde l'état de traitement intermédiaire."""
        import json
        data = {
            "segments": [{"start_sec": s.start_sec, "end_sec": s.end_sec} for s in segments],
            "cursor": cursor,
            "crossings": crossings,  # déjà des dicts sérialisables
        }
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO processed_files (filename, processed_at, status, vehicle_count) "
                "VALUES (?, '', 'processing', 0)",
                (filename,),
            )
            conn.execute(
                "UPDATE processed_files SET checkpoint_json = ? WHERE filename = ?",
                (json.dumps(data), filename),
            )

    def get_checkpoint(self, filename: str) -> "dict | None":
        """Retourne le checkpoint si disponible, None sinon."""
        import json
        with self._connect() as conn:
            row = conn.execute(
                "SELECT checkpoint_json FROM processed_files WHERE filename = ?",
                (filename,)
            ).fetchone()
        if row and row[0]:
            try:
                return json.loads(row[0])
            except Exception:
                return None
        return None

    def clear_checkpoint(self, filename: str) -> None:
        """Efface le checkpoint une fois le fichier terminé."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE processed_files SET checkpoint_json = NULL WHERE filename = ?",
                (filename,),
            )

    def unmark_files(self, filenames: list[str]) -> int:
        """Remove files from processed_files (and their crossings) so they get re-processed."""
        if not filenames:
            return 0
        self.delete_crossings_for_files(filenames)
        placeholders = ",".join("?" * len(filenames))
        with self._connect() as conn:
            cur = conn.execute(
                f"DELETE FROM processed_files WHERE filename IN ({placeholders})", filenames
            )
            return cur.rowcount

    def unmark_all_files(self) -> int:
        """Remove all non-skipped files from processed_files so they get re-queued.
        Crossings are preserved and will be replaced lazily when each file is re-processed."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM processed_files WHERE status != 'skipped'")
            return cur.rowcount

    def skip_files(self, filenames: list[str]) -> int:
        """Mark files as skipped so the processor ignores them even after a config reset."""
        if not filenames:
            return 0
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO processed_files
                    (filename, processed_at, status, vehicle_count, processing_duration_seconds)
                VALUES (?, ?, 'skipped', 0, 0.0)
                ON CONFLICT(filename) DO NOTHING
                """,
                [(f, now) for f in filenames],
            )
            return len(filenames)

    # ------------------------------------------------------------------ #
    # App state (config fingerprint)                                       #
    # ------------------------------------------------------------------ #

    def get_state(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM app_state WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else None

    def set_state(self, key: str, value: str):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO app_state (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # ------------------------------------------------------------------ #
    # Motion cache                                                         #
    # ------------------------------------------------------------------ #

    def get_motion_cache(self, filename: str, motion_fp: str) -> "list[dict] | None":
        """Returns cached segments as list of {start_sec, end_sec} or None on miss/stale."""
        import json as _json
        with self._connect() as conn:
            row = conn.execute(
                "SELECT segments_json FROM motion_cache WHERE filename = ? AND motion_fp = ?",
                (filename, motion_fp),
            ).fetchone()
        return _json.loads(row["segments_json"]) if row else None

    def set_motion_cache(self, filename: str, motion_fp: str, segments: list) -> None:
        """Stores serialised segments. segments is a list of {start_sec, end_sec} dicts."""
        import json as _json
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO motion_cache (filename, motion_fp, segments_json, cached_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(filename) DO UPDATE SET
                    motion_fp = excluded.motion_fp,
                    segments_json = excluded.segments_json,
                    cached_at = excluded.cached_at
                """,
                (filename, motion_fp, _json.dumps(segments), now),
            )

    def clear_motion_cache(self) -> int:
        """Deletes all cached motion segments. Returns number of rows deleted."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM motion_cache")
            return cur.rowcount

    def get_calendar_stats(self, year: int, month: int) -> list[dict]:
        """Returns daily counts for a calendar month (dates in local time)."""
        import calendar as _cal
        tzmod = self._tz_mod()
        last_day = _cal.monthrange(year, month)[1]
        date_from = f"{year:04d}-{month:02d}-01"
        date_to   = f"{year:04d}-{month:02d}-{last_day:02d}"
        query = """
            SELECT date(datetime(timestamp, :tzmod)) AS day, COUNT(*) AS count
            FROM crossings
            WHERE date(datetime(timestamp, :tzmod)) BETWEEN :from AND :to
            GROUP BY day
        """
        with self._connect() as conn:
            rows = conn.execute(query, {"from": date_from, "to": date_to, "tzmod": tzmod}).fetchall()
        return [{"date": r["day"], "count": r["count"]} for r in rows]

    def get_crossings_detail(self, day: str, hour: int | None = None) -> list[dict]:
        """Returns individual crossings for a day (and optionally an hour)."""
        with self._connect() as conn:
            if hour is not None:
                rows = conn.execute(
                    "SELECT timestamp, vehicle_type, direction, confidence, source_file "
                    "FROM crossings "
                    "WHERE date(timestamp) = ? AND cast(strftime('%H', timestamp) as integer) = ? "
                    "ORDER BY timestamp",
                    (day, hour)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT timestamp, vehicle_type, direction, confidence, source_file "
                    "FROM crossings WHERE date(timestamp) = ? ORDER BY timestamp",
                    (day,)
                ).fetchall()
        return [dict(r) for r in rows]

    def get_direction_stats(self, date_str: str) -> dict:
        """Returns direction counts for a given date (local time). Only non-NULL directions."""
        tzmod = self._tz_mod()
        query = """
            SELECT direction, COUNT(*) AS count
            FROM crossings
            WHERE date(datetime(timestamp, :tzmod)) = :date AND direction IS NOT NULL
            GROUP BY direction
        """
        with self._connect() as conn:
            rows = conn.execute(query, {"date": date_str, "tzmod": tzmod}).fetchall()
        result = {"left_to_right": 0, "right_to_left": 0}
        for r in rows:
            if r["direction"] in result:
                result[r["direction"]] = r["count"]
        result["total"] = result["left_to_right"] + result["right_to_left"]
        return result

    def get_processing_status(self) -> dict:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS n FROM processed_files").fetchone()["n"]
            done = conn.execute(
                "SELECT COUNT(*) AS n FROM processed_files WHERE status='done'"
            ).fetchone()["n"]
            errors = conn.execute(
                "SELECT COUNT(*) AS n FROM processed_files WHERE status='error'"
            ).fetchone()["n"]
            last = conn.execute(
                "SELECT filename, processed_at FROM processed_files ORDER BY processed_at DESC LIMIT 1"
            ).fetchone()
        return {
            "total_files": total,
            "done": done,
            "errors": errors,
            "last_file": dict(last) if last else None,
        }

    # ------------------------------------------------------------------ #
    # Audio stats                                                          #
    # ------------------------------------------------------------------ #

    def add_audio_stats(
        self, filename: str,
        mean_db: "float | None", median_db: "float | None", std_db: "float | None",
        p10_db: "float | None", p90_db: "float | None",
        video_hour: "int | None" = None,
    ):
        """Stocke les stats audio. Appeler avec des valeurs None pour marquer
        "analysé mais sans piste audio", afin d'éviter les nouvelles tentatives."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audio_stats
                    (filename, mean_db, median_db, std_db, p10_db, p90_db, video_hour)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(filename) DO UPDATE SET
                    mean_db=excluded.mean_db, median_db=excluded.median_db,
                    std_db=excluded.std_db, p10_db=excluded.p10_db,
                    p90_db=excluded.p90_db, video_hour=excluded.video_hour,
                    created_at=excluded.created_at
                """,
                (filename, mean_db, median_db, std_db, p10_db, p90_db, video_hour),
            )

    def get_audio_stats_count(self, night_only: bool = False,
                              night_start: int = 22, night_end: int = 6) -> int:
        """Compte les fichiers avec audio réel (p10_db non null).
        night_only=True : compte uniquement les fichiers heure nuit."""
        if night_only:
            with self._connect() as conn:
                return conn.execute(
                    """SELECT COUNT(*) FROM audio_stats
                       WHERE p10_db IS NOT NULL AND video_hour IS NOT NULL
                         AND (video_hour >= ? OR video_hour < ?)""",
                    (night_start, night_end),
                ).fetchone()[0]
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM audio_stats WHERE p10_db IS NOT NULL"
            ).fetchone()[0]

    def get_audio_calibration_aggregate(self, last_n: int = 50,
                                        night_only: bool = False,
                                        night_start: int = 22,
                                        night_end: int = 6) -> "dict | None":
        """Agrégats sur les N derniers fichiers avec audio pour calculer le seuil.
        night_only=True : n'utilise que les fichiers enregistrés de nuit."""
        night_filter = (
            "AND (video_hour >= :ns OR video_hour < :ne)" if night_only else ""
        )
        query = f"""
            SELECT
                COUNT(*)       AS count,
                AVG(p10_db)    AS avg_p10_db,
                AVG(std_db)    AS avg_std_db,
                AVG(median_db) AS avg_median_db,
                MIN(p10_db)    AS min_p10_db,
                MAX(p90_db)    AS max_p90_db
            FROM (
                SELECT p10_db, std_db, median_db, p90_db
                FROM audio_stats
                WHERE p10_db IS NOT NULL {night_filter}
                ORDER BY created_at DESC LIMIT :n
            )
        """
        with self._connect() as conn:
            row = conn.execute(query, {"n": last_n, "ns": night_start, "ne": night_end}).fetchone()
        if row and row["count"]:
            return dict(row)
        return None

    def get_audio_stats_history(self, limit: int = 30) -> list[dict]:
        """Historique des niveaux audio par fichier (pour le dashboard).
        N'inclut que les fichiers avec une vraie piste audio (p10_db non null)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT filename, mean_db, median_db, std_db, p10_db, p90_db, video_hour, created_at
                FROM audio_stats
                WHERE p10_db IS NOT NULL
                ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
