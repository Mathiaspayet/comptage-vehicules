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
