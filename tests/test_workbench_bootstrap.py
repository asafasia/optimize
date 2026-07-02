from __future__ import annotations

from pathlib import Path

import workbench_bootstrap
from workbench_bootstrap import load_workbench_dotenv


def test_load_workbench_dotenv_sets_missing_values(monkeypatch, tmp_path: Path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        """
        # ignored
        WORKBENCH_TEST_MONGO_URI=mongodb://example
        WORKBENCH_TEST_BLOB_CONN_STR="blob=value" # comment
        export WORKBENCH_TEST_DB_NAME='Test DB'
        """,
    )

    for name in (
        "WORKBENCH_TEST_MONGO_URI",
        "WORKBENCH_TEST_BLOB_CONN_STR",
        "WORKBENCH_TEST_DB_NAME",
    ):
        monkeypatch.delenv(name, raising=False)

    load_workbench_dotenv(dotenv)

    assert os_environ("WORKBENCH_TEST_MONGO_URI") == "mongodb://example"
    assert os_environ("WORKBENCH_TEST_BLOB_CONN_STR") == "blob=value"
    assert os_environ("WORKBENCH_TEST_DB_NAME") == "Test DB"


def test_load_workbench_dotenv_does_not_override_existing_values(monkeypatch, tmp_path: Path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("WORKBENCH_TEST_MONGO_URI=mongodb://from-file\n")
    monkeypatch.setenv("WORKBENCH_TEST_MONGO_URI", "mongodb://from-env")

    load_workbench_dotenv(dotenv)

    assert os_environ("WORKBENCH_TEST_MONGO_URI") == "mongodb://from-env"


def test_setup_workbench_environment_uses_writable_qratena_data_root(monkeypatch, tmp_path: Path):
    workbench_data = tmp_path / "data"
    package_data = tmp_path / "package-data"
    package_devices = package_data / "devices"
    package_devices.mkdir(parents=True)

    monkeypatch.setattr(workbench_bootstrap, "WORKBENCH_QRATENA_DATA_ROOT", workbench_data)
    monkeypatch.setattr(workbench_bootstrap, "QRATENA_DATA_ROOT", package_data)
    monkeypatch.delenv("QRATENA_DATA_DIR", raising=False)

    workbench_bootstrap.setup_workbench_environment()

    assert os_environ("QRATENA_DATA_DIR") == str(workbench_data)
    assert (workbench_data / "devices").is_symlink()
    assert (workbench_data / "devices").resolve() == package_devices


def os_environ(name: str) -> str:
    import os

    return os.environ[name]
