import os


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


USERNAME = required_env("QTASKBOARD_USERNAME")
PASSWORD = required_env("QTASKBOARD_PASSWORD")

API_URI = os.environ.get("QTASKBOARD_API_URI", "http://172.16.0.104:41418")
REDIS_URI = os.environ.get("QTASKBOARD_REDIS_URI", "redis://172.16.0.104:6379")

MONGO_URI = required_env("QARAKAL_MONGO_URI")
BLOB_CONN_STR = required_env("QARAKAL_BLOB_CONN_STR")
DB_NAME = os.environ.get("QARAKAL_DB_NAME", "QPUs")
BLOB_CONTAINER = os.environ.get("QARAKAL_BLOB_CONTAINER", "profiles")
