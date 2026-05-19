"""Tests for src/database.py."""
from datetime import datetime, timezone


def test_mark_done_and_check(tmp_db):
    tmp_db.mark_file_done("video_001.mp4", vehicle_count=5)
    assert tmp_db.is_file_processed("video_001.mp4") is True


def test_mark_error(tmp_db):
    tmp_db.mark_file_error("broken.mp4", "codec error")
    assert tmp_db.is_file_processed("broken.mp4") is True
    status = tmp_db.get_processing_status()
    assert status["errors"] == 1


def test_skip_survives_unmark_all(tmp_db):
    tmp_db.skip_files(["important.mp4"])
    tmp_db.unmark_all_files()
    names = tmp_db.get_all_processed_filenames()
    assert "important.mp4" in names


def test_unmark_all_removes_done_not_skipped(tmp_db):
    tmp_db.mark_file_done("done_1.mp4", vehicle_count=1)
    tmp_db.mark_file_done("done_2.mp4", vehicle_count=2)
    tmp_db.skip_files(["skipped.mp4"])
    tmp_db.unmark_all_files()
    names = tmp_db.get_all_processed_filenames()
    assert "done_1.mp4" not in names
    assert "done_2.mp4" not in names
    assert "skipped.mp4" in names
    assert len(names) == 1


def test_insert_crossings_batch_and_stats(tmp_db):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    crossings = [
        {
            "timestamp": f"{today}T10:00:00",
            "vehicle_type": "car",
            "direction": "left_to_right",
            "confidence": 0.9,
            "source_file": "video.mp4",
        },
        {
            "timestamp": f"{today}T11:00:00",
            "vehicle_type": "truck",
            "direction": "right_to_left",
            "confidence": 0.8,
            "source_file": "video.mp4",
        },
        {
            "timestamp": f"{today}T11:30:00",
            "vehicle_type": "car",
            "direction": None,
            "confidence": 0.75,
            "source_file": "video.mp4",
        },
    ]
    tmp_db.insert_crossings_batch(crossings)
    stats = tmp_db.get_hourly_stats(today)
    total = sum(h["count"] for h in stats)
    assert total == 3


def test_state_set_get(tmp_db):
    tmp_db.set_state("my_key", "my_value")
    assert tmp_db.get_state("my_key") == "my_value"


def test_state_overwrite(tmp_db):
    tmp_db.set_state("key", "first")
    tmp_db.set_state("key", "second")
    assert tmp_db.get_state("key") == "second"


def test_motion_cache_miss(tmp_db):
    assert tmp_db.get_motion_cache("video.mp4", "fp_abc") is None


def test_motion_cache_set_get(tmp_db):
    segments = [{"start_sec": 1.0, "end_sec": 5.0}, {"start_sec": 10.0, "end_sec": 15.0}]
    tmp_db.set_motion_cache("video.mp4", "fp_abc", segments)
    result = tmp_db.get_motion_cache("video.mp4", "fp_abc")
    assert result == segments


def test_motion_cache_fp_mismatch(tmp_db):
    segments = [{"start_sec": 0.0, "end_sec": 3.0}]
    tmp_db.set_motion_cache("video.mp4", "fp_abc", segments)
    assert tmp_db.get_motion_cache("video.mp4", "fp_xyz") is None


def test_delete_old_crossings(tmp_db):
    from datetime import timedelta
    today = datetime.utcnow()
    old = (today - timedelta(days=100)).strftime("%Y-%m-%dT%H:%M:%S")
    recent = today.strftime("%Y-%m-%dT%H:%M:%S")
    tmp_db.insert_crossings_batch([
        {"timestamp": old,    "vehicle_type": "car", "direction": None, "confidence": 0.9, "source_file": "old.mp4"},
        {"timestamp": recent, "vehicle_type": "car", "direction": None, "confidence": 0.9, "source_file": "new.mp4"},
    ])
    deleted = tmp_db.delete_old_crossings(days=30)
    assert deleted == 1
    remaining = tmp_db.get_crossings_export()
    assert len(remaining) == 1
    assert remaining[0]["source_file"] == "new.mp4"


def test_timezone_aware_hourly(tmp_path):
    """Crossing timestamps are stored as naive local time — no UTC offset applied.

    A crossing stored as '2026-01-15T23:30:00' (local time) must appear
    in Jan 15 stats at hour 23, not shifted to Jan 16.
    """
    from src.database import Database
    db = Database(tmp_path / "tz_test.db", timezone="Europe/Paris")
    # Insert a crossing at 23:30 local time on 2026-01-15
    db.insert_crossings_batch([{
        "timestamp": "2026-01-15T23:30:00",
        "vehicle_type": "car", "direction": None,
        "confidence": 0.9, "source_file": "v.mp4",
    }])
    # Local date 2026-01-15 should have 1 hit (timestamp is already local)
    stats_15 = db.get_hourly_stats("2026-01-15")
    assert sum(h["count"] for h in stats_15) == 1
    # Hour 23 bucket should contain the crossing
    assert stats_15[23]["count"] == 1
    # Jan 16 should have 0 hits
    stats_16 = db.get_hourly_stats("2026-01-16")
    assert sum(h["count"] for h in stats_16) == 0


def test_crossings_export_filters(tmp_db):
    today = datetime.utcnow().strftime("%Y-%m-%d")
    tmp_db.insert_crossings_batch([
        {"timestamp": f"{today}T08:00:00", "vehicle_type": "car",   "direction": None, "confidence": 0.9, "source_file": "v1.mp4"},
        {"timestamp": f"{today}T09:00:00", "vehicle_type": "truck", "direction": None, "confidence": 0.8, "source_file": "v2.mp4"},
    ])
    all_rows = tmp_db.get_crossings_export()
    assert len(all_rows) == 2
    cars = tmp_db.get_crossings_export(vehicle_type="car")
    assert len(cars) == 1 and cars[0]["vehicle_type"] == "car"
