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

CREATE INDEX IF NOT EXISTS idx_crossings_timestamp ON crossings(timestamp);
CREATE INDEX IF NOT EXISTS idx_crossings_type      ON crossings(vehicle_type);
CREATE INDEX IF NOT EXISTS idx_processed_filename  ON processed_files(filename);
"""


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

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
        logger.debug("Schéma base de données initialisé : %s", self.db_path)

    # ------------------------------------------------------------------ #
    # Processed files                                                      #
    # ------------------------------------------------------------------ #

    def is_file_processed(self, filename: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM processed_files WHERE filename = ?", (filename,)
            ).fetchone()
            return row is not None

    def mark_file_done(self, filename: str, vehicle_count: int):
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO processed_files (filename, processed_at, status, vehicle_count)
                VALUES (?, ?, 'done', ?)
                ON CONFLICT(filename) DO UPDATE SET
                    processed_at = excluded.processed_at,
                    status = 'done',
                    vehicle_count = excluded.vehicle_count,
                    error_message = NULL
                """,
                (filename, now, vehicle_count),
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

    # ------------------------------------------------------------------ #
    # Statistics queries                                                   #
    # ------------------------------------------------------------------ #

    def get_hourly_stats(self, date_str: str, vehicle_type: str = "all") -> list[dict]:
        """Returns list of {hour, count} for a given date (YYYY-MM-DD)."""
        type_filter = "" if vehicle_type == "all" else "AND vehicle_type = :vtype"
        query = f"""
            SELECT
                CAST(strftime('%H', timestamp) AS INTEGER) AS hour,
                COUNT(*) AS count
            FROM crossings
            WHERE date(timestamp) = :date
            {type_filter}
            GROUP BY hour
            ORDER BY hour
        """
        params = {"date": date_str, "vtype": vehicle_type}
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        # Fill in all 24 hours
        hourly = {r["hour"]: r["count"] for r in rows}
        return [{"hour": h, "count": hourly.get(h, 0)} for h in range(24)]

    def get_daily_stats(self, days: int = 30, vehicle_type: str = "all") -> list[dict]:
        """Returns list of {date, count} for the last N days."""
        type_filter = "" if vehicle_type == "all" else "AND vehicle_type = :vtype"
        query = f"""
            SELECT
                date(timestamp) AS day,
                COUNT(*) AS count
            FROM crossings
            WHERE date(timestamp) >= date('now', :offset)
            {type_filter}
            GROUP BY day
            ORDER BY day
        """
        params = {"offset": f"-{days} days", "vtype": vehicle_type}
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
        query = """
            SELECT vehicle_type, COUNT(*) AS count
            FROM crossings
            WHERE date(timestamp) = :date
            GROUP BY vehicle_type
            ORDER BY count DESC
        """
        with self._connect() as conn:
            rows = conn.execute(query, {"date": date_str}).fetchall()
        return [{"vehicle_type": r["vehicle_type"], "count": r["count"]} for r in rows]

    def get_available_dates(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT date(timestamp) AS day FROM crossings ORDER BY day DESC LIMIT 90"
            ).fetchall()
        return [r["day"] for r in rows]

    def get_processed_files(self, limit: int = 500, offset: int = 0, status_filter: str = "all") -> list[dict]:
        """Returns processed files sorted by processing date descending."""
        where = "" if status_filter == "all" else f"WHERE status = '{status_filter}'"
        query = f"""
            SELECT filename, processed_at, status, vehicle_count, error_message
            FROM processed_files
            {where}
            ORDER BY processed_at DESC
            LIMIT ? OFFSET ?
        """
        with self._connect() as conn:
            rows = conn.execute(query, (limit, offset)).fetchall()
        return [dict(r) for r in rows]

    def unmark_files(self, filenames: list[str]) -> int:
        """Remove files from processed_files so they get re-processed. Returns count deleted."""
        if not filenames:
            return 0
        placeholders = ",".join("?" * len(filenames))
        with self._connect() as conn:
            cur = conn.execute(
                f"DELETE FROM processed_files WHERE filename IN ({placeholders})", filenames
            )
            return cur.rowcount

    def unmark_all_files(self) -> int:
        """Remove ALL files from processed_files. Returns count deleted."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM processed_files")
            return cur.rowcount

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
