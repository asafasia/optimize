from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


CONFIG_PATH = Path(__file__).resolve().parents[1] / "resources" / "resources.py"


def load_config_module():
    spec = importlib.util.spec_from_file_location("workbench_resources_config", CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_required_config_fails_fast_when_secret_is_missing(monkeypatch):
    for name in (
        "QTASKBOARD_USERNAME",
        "QTASKBOARD_PASSWORD",
        "QARAKAL_MONGO_URI",
        "QARAKAL_BLOB_CONN_STR",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="QTASKBOARD_USERNAME"):
        load_config_module()


def test_config_loads_from_environment(monkeypatch):
    monkeypatch.setenv("QTASKBOARD_USERNAME", "user")
    monkeypatch.setenv("QTASKBOARD_PASSWORD", "password")
    monkeypatch.setenv("QARAKAL_MONGO_URI", "mongodb://example")
    monkeypatch.setenv("QARAKAL_BLOB_CONN_STR", "blob")

    config = load_config_module()

    assert config.USERNAME == "user"
    assert config.PASSWORD == "password"
    assert config.MONGO_URI == "mongodb://example"
    assert config.BLOB_CONN_STR == "blob"
    assert config.DB_NAME == "QPUs"
    assert config.BLOB_CONTAINER == "profiles"
