from __future__ import annotations

import importlib.util
import sys
import types
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


def test_resources_package_import_does_not_require_taskboard_credentials(monkeypatch):
    for name in (
        "QTASKBOARD_USERNAME",
        "QTASKBOARD_PASSWORD",
        "QARAKAL_MONGO_URI",
        "QARAKAL_BLOB_CONN_STR",
    ):
        monkeypatch.delenv(name, raising=False)

    sys.modules.pop("resources", None)
    sys.modules.pop("resources.resources", None)

    resources = __import__("resources")

    assert hasattr(resources, "setup_workbench_environment")


def test_load_profile_module_import_does_not_require_taskboard_credentials(monkeypatch):
    for name in (
        "QTASKBOARD_USERNAME",
        "QTASKBOARD_PASSWORD",
        "QARAKAL_MONGO_URI",
        "QARAKAL_BLOB_CONN_STR",
    ):
        monkeypatch.delenv(name, raising=False)

    sys.modules.pop("resources", None)
    sys.modules.pop("resources.load_profile", None)

    from resources.load_profile import load_profile

    fake_profile = types.SimpleNamespace(name="local")
    fake_profile_class = types.SimpleNamespace(default=lambda: fake_profile)
    fake_profile_module = types.ModuleType("qratena.system.components_params.profile")
    fake_profile_module.Profile = fake_profile_class
    monkeypatch.setitem(
        sys.modules,
        "qratena.system.components_params.profile",
        fake_profile_module,
    )

    assert load_profile("main") is fake_profile
    assert fake_profile.name == "main"


def test_load_task_manager_requires_auth(monkeypatch):
    for name in (
        "QTASKBOARD_USERNAME",
        "QTASKBOARD_PASSWORD",
        "QTASKBOARD_TOKEN",
        "QIGEON_USERNAME",
        "QIGEON_PASSWORD",
        "QIGEON_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)

    sys.modules.pop("resources", None)
    sys.modules.pop("resources.load_profile", None)

    from resources.load_profile import load_task_manager

    with pytest.raises(RuntimeError, match="Missing qigeon taskboard credentials"):
        load_task_manager()


def test_load_task_manager_uses_default_uris_with_auth(monkeypatch):
    for name in (
        "QTASKBOARD_API_URI",
        "QTASKBOARD_REDIS_URI",
        "QIGEON_API_URI",
        "QIGEON_REDIS_URI",
        "QTASKBOARD_TOKEN",
        "QIGEON_TOKEN",
        "QTASKBOARD_USERNAME",
        "QTASKBOARD_PASSWORD",
        "QIGEON_USERNAME",
        "QIGEON_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QTASKBOARD_USERNAME", "user")
    monkeypatch.setenv("QTASKBOARD_PASSWORD", "password")

    sys.modules.pop("resources", None)
    sys.modules.pop("resources.load_profile", None)

    from resources.load_profile import load_task_manager

    class FakeTaskSubmitterAsync:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_task_submitter_module = types.ModuleType("qigeon.io.task_submitter")
    fake_task_submitter_module.TaskSubmitterAsync = FakeTaskSubmitterAsync
    monkeypatch.setitem(sys.modules, "qigeon.io.task_submitter", fake_task_submitter_module)

    task_manager = load_task_manager()

    assert task_manager.kwargs["api_uri"] == "http://172.16.0.104:41418"
    assert task_manager.kwargs["redis_uri"] == "redis://172.16.0.104:6379"
    assert task_manager.kwargs["username"] == "user"
    assert task_manager.kwargs["password"] == "password"


def test_load_task_manager_accepts_qigeon_credentials(monkeypatch):
    for name in (
        "QTASKBOARD_USERNAME",
        "QTASKBOARD_PASSWORD",
        "QIGEON_USERNAME",
        "QIGEON_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QIGEON_USERNAME", "user")
    monkeypatch.setenv("QIGEON_PASSWORD", "password")

    sys.modules.pop("resources", None)
    sys.modules.pop("resources.load_profile", None)

    from resources.load_profile import load_task_manager

    class FakeTaskSubmitterAsync:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_task_submitter_module = types.ModuleType("qigeon.io.task_submitter")
    fake_task_submitter_module.TaskSubmitterAsync = FakeTaskSubmitterAsync
    monkeypatch.setitem(sys.modules, "qigeon.io.task_submitter", fake_task_submitter_module)

    task_manager = load_task_manager()

    assert task_manager.kwargs["username"] == "user"
    assert task_manager.kwargs["password"] == "password"


def test_load_task_manager_accepts_token(monkeypatch):
    monkeypatch.setenv("QIGEON_TOKEN", "token")
    monkeypatch.setenv("QIGEON_USERNAME", "user")
    monkeypatch.setenv("QIGEON_PASSWORD", "password")

    sys.modules.pop("resources", None)
    sys.modules.pop("resources.load_profile", None)

    from resources.load_profile import load_task_manager

    class FakeTaskSubmitterAsync:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_task_submitter_module = types.ModuleType("qigeon.io.task_submitter")
    fake_task_submitter_module.TaskSubmitterAsync = FakeTaskSubmitterAsync
    monkeypatch.setitem(sys.modules, "qigeon.io.task_submitter", fake_task_submitter_module)

    task_manager = load_task_manager()

    assert task_manager.kwargs["token"] == "token"
    assert "username" not in task_manager.kwargs
    assert "password" not in task_manager.kwargs
