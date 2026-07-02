

from qigeon.io.task_submitter import TaskSubmitterAsync
from qratena.system.components_params.profile import Profile
from qratena.system.profile_manager import ProfileManager
from resources.resources import (
    API_URI,
    BLOB_CONN_STR,
    BLOB_CONTAINER,
    DB_NAME,
    MONGO_URI,
    PASSWORD,
    REDIS_URI,
    USERNAME,
)


def load_profile(branch: str = "main") -> Profile:

    with ProfileManager(
        connection_string=MONGO_URI,
        database_name=DB_NAME,
        blob_connection_string=BLOB_CONN_STR,
        blob_container_name=BLOB_CONTAINER,
    ) as pm:
        pv = pm.pull_profile(branch)

        return pv.data


def load_task_manager() -> TaskSubmitterAsync:
    return TaskSubmitterAsync(
        api_uri=API_URI,
        redis_uri=REDIS_URI,
        username=USERNAME,
        password=PASSWORD,
    )



def push_profile(profile: Profile, branch: str = "main_asaf") -> None:
    with ProfileManager(
        connection_string=MONGO_URI,
        database_name=DB_NAME,
        blob_connection_string=BLOB_CONN_STR,
        blob_container_name=BLOB_CONTAINER,
    ) as pm:
        pm.push_profile(profile, branch, "Asaf")



if __name__ == "__main__":
    profile = load_profile("main_asaf")
    print(f"Loaded profile with {len(profile.qubits)} qubits.")
