"""Tests for src/config.py — Config loading, merging, fingerprint, reload."""
import copy

import pytest

from src.config import Config, DEFAULTS, _deep_merge


# ------------------------------------------------------------------ #
# Defaults and deep merge                                             #
# ------------------------------------------------------------------ #

def test_defaults_loaded(base_config):
    assert base_config.model_name == "yolo11n"
    assert base_config.file_stable_delay == 120


def test_deep_merge_override():
    merged_data = _deep_merge(DEFAULTS, {"detector": {"imgsz": 640}})
    config = Config(merged_data)
    assert config.imgsz == 640
    # Other detector keys should be unchanged
    assert config.model_name == "yolo11n"
    assert config.confidence_threshold == 0.35


def test_deep_merge_does_not_pollute_base():
    original_imgsz = DEFAULTS["detector"]["imgsz"]
    _deep_merge(DEFAULTS, {"detector": {"imgsz": 999}})
    assert DEFAULTS["detector"]["imgsz"] == original_imgsz


# ------------------------------------------------------------------ #
# Fingerprint                                                          #
# ------------------------------------------------------------------ #

def test_fingerprint_changes_with_line():
    data1 = _deep_merge(DEFAULTS, {"counting": {"line_p1": [0, 100]}})
    data2 = _deep_merge(DEFAULTS, {"counting": {"line_p1": [0, 200]}})
    fp1 = Config(data1).detection_fingerprint()
    fp2 = Config(data2).detection_fingerprint()
    assert fp1 != fp2


def test_fingerprint_stable():
    fp1 = Config(DEFAULTS.copy()).detection_fingerprint()
    fp2 = Config(DEFAULTS.copy()).detection_fingerprint()
    assert fp1 == fp2


def test_fingerprint_includes_model_name():
    data1 = _deep_merge(DEFAULTS, {"detector": {"model_name": "yolo11n"}})
    data2 = _deep_merge(DEFAULTS, {"detector": {"model_name": "yolov8n"}})
    fp1 = Config(data1).detection_fingerprint()
    fp2 = Config(data2).detection_fingerprint()
    assert fp1 != fp2


# ------------------------------------------------------------------ #
# Reload                                                              #
# ------------------------------------------------------------------ #

def test_reload_updates_value(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("detector:\n  imgsz: 640\n", encoding="utf-8")
    from src.config import load_config
    config = load_config(config_file)
    assert config.imgsz == 640

    config_file.write_text("detector:\n  imgsz: 320\n", encoding="utf-8")
    config.reload()
    assert config.imgsz == 320


def test_reload_returns_changed_sections(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("detector:\n  imgsz: 640\n", encoding="utf-8")
    from src.config import load_config
    config = load_config(config_file)

    config_file.write_text("detector:\n  imgsz: 320\n", encoding="utf-8")
    changed = config.reload()
    assert "detector" in changed


def test_reload_no_file():
    config = Config(DEFAULTS.copy(), config_path=None)
    changed = config.reload()
    assert changed == set()
