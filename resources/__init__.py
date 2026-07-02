from .bootstrap import setup_workbench_environment

setup_workbench_environment()

__all__ = [
    "API_URI",
    "BLOB_CONN_STR",
    "BLOB_CONTAINER",
    "DB_NAME",
    "load_profile",
    "load_task_manager",
    "MONGO_URI",
    "PASSWORD",
    "push_profile",
    "REDIS_URI",
    "setup_workbench_environment",
    "USERNAME",
]

_RESOURCE_NAMES = {
    "API_URI",
    "BLOB_CONN_STR",
    "BLOB_CONTAINER",
    "DB_NAME",
    "MONGO_URI",
    "PASSWORD",
    "REDIS_URI",
    "USERNAME",
}


def __getattr__(name: str):
    if name in {"load_profile", "load_task_manager", "push_profile"}:
        from .load_profile import load_profile, load_task_manager, push_profile

        return {
            "load_profile": load_profile,
            "load_task_manager": load_task_manager,
            "push_profile": push_profile,
        }[name]

    if name in _RESOURCE_NAMES:
        from . import resources

        return getattr(resources, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
