from typing import Any, List

from ..extensions import ensure_gcs_fs


def list_objects(bucket_path: str, detail: bool = True) -> List[Any]:
    fs = ensure_gcs_fs()
    if not fs:
        return []
    return fs.ls(bucket_path, detail=detail)


def read_text(path: str) -> str:
    fs = ensure_gcs_fs()
    if not fs:
        raise RuntimeError("GCS filesystem is not initialized")
    with fs.open(path, "r") as f:
        return f.read()


__all__ = ["list_objects", "read_text"]
