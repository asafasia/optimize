import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qigeon.io.task_submitter import TaskSubmitterAsync
    from qratena.system.components_params.profile import Profile


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def required_env_any(*names: str) -> str:
    value = env_any(*names)
    if value:
        return value
    joined_names = " or ".join(names)
    raise RuntimeError(f"Missing required environment variable: {joined_names}")


def env_any(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def load_profile(branch: str = "main") -> "Profile":
    connection_string = env_any("QARAKAL_MONGO_URI", "MONGO_CONNECTION_STRING")
    blob_connection_string = env_any("QARAKAL_BLOB_CONN_STR", "BLOBS_CONNECTION_STRING")
    if not connection_string or not blob_connection_string:
        from qratena.system.components_params.profile import Profile

        profile = Profile.default()
        profile.name = branch
        return profile

    database_name = os.environ.get("QARAKAL_DB_NAME") or os.environ.get("QRATENA_DATABASE_NAME", "QPUs")
    blob_container_name = os.environ.get("QARAKAL_BLOB_CONTAINER") or os.environ.get(
        "QRATENA_BLOBS_CONTAINER", "profiles"
    )

    from qratena.system.profile_manager import ProfileManager

    with ProfileManager(
        connection_string=connection_string,
        database_name=database_name,
        blob_connection_string=blob_connection_string,
        blob_container_name=blob_container_name,
    ) as pm:
        pv = pm.pull_profile(branch)

        return pv.data


def load_task_manager() -> "TaskSubmitterAsync":
    api_uri = os.environ.get("QTASKBOARD_API_URI") or os.environ.get(
        "QIGEON_API_URI", "http://172.16.0.104:41418"
    )
    redis_uri = os.environ.get("QTASKBOARD_REDIS_URI") or os.environ.get(
        "QIGEON_REDIS_URI", "redis://172.16.0.104:6379"
    )
    username = env_any("QTASKBOARD_USERNAME", "QIGEON_USERNAME")
    password = env_any("QTASKBOARD_PASSWORD", "QIGEON_PASSWORD")
    token = env_any("QTASKBOARD_TOKEN", "QIGEON_TOKEN")

    if not token and not (username and password):
        raise RuntimeError(
            "Missing qigeon taskboard credentials. Set QTASKBOARD_TOKEN, "
            "or set both QTASKBOARD_USERNAME and QTASKBOARD_PASSWORD in the "
            "environment or workbench .env before running hardware tasks. "
            "Use --do-emulation to run without submitting to the taskboard."
        )

    from qigeon.io.task_submitter import TaskSubmitterAsync

    config = {
        "api_uri": api_uri,
        "redis_uri": redis_uri,
    }
    if token:
        config["token"] = token
    elif username and password:
        config["username"] = username
        config["password"] = password

    return TaskSubmitterAsync(
        **config,
    )


def push_profile(profile: "Profile", branch: str = "main_asaf") -> None:
    connection_string = required_env_any("QARAKAL_MONGO_URI", "MONGO_CONNECTION_STRING")
    database_name = os.environ.get("QARAKAL_DB_NAME") or os.environ.get("QRATENA_DATABASE_NAME", "QPUs")
    blob_connection_string = required_env_any("QARAKAL_BLOB_CONN_STR", "BLOBS_CONNECTION_STRING")
    blob_container_name = os.environ.get("QARAKAL_BLOB_CONTAINER") or os.environ.get(
        "QRATENA_BLOBS_CONTAINER", "profiles"
    )

    from qratena.system.profile_manager import ProfileManager

    with ProfileManager(
        connection_string=connection_string,
        database_name=database_name,
        blob_connection_string=blob_connection_string,
        blob_container_name=blob_container_name,
    ) as pm:
        pm.push_profile(profile, branch, "Asaf")


if __name__ == "__main__":
    profile = load_profile("main_asaf")
    print(f"Loaded profile with {len(profile.qubits)} qubits.")
    
    
    
    print(profile.qubits['q3'].pulses['readout']['const'].readout_duration)
