from __future__ import annotations

import base64
import mimetypes

from .fs import rel, resolve
from .types import Tool


MAX_MEDIA_BYTES = 8 * 1024 * 1024
IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
DOCUMENT_TYPES = {"application/pdf"}


def run(args: dict):
    path = resolve(args["path"])
    if not path.exists():
        raise FileNotFoundError(rel(path))
    if not path.is_file():
        raise IsADirectoryError(rel(path))
    data = path.read_bytes()
    if len(data) > MAX_MEDIA_BYTES:
        raise ValueError(f"media file too large ({len(data)} bytes, max {MAX_MEDIA_BYTES})")
    media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    encoded = base64.b64encode(data).decode("ascii")
    metadata = f"media: {rel(path)}\ntype: {media_type}\nbytes: {len(data)}"
    if media_type in IMAGE_TYPES:
        block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": encoded,
            },
        }
    elif media_type in DOCUMENT_TYPES:
        block = {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": encoded,
            },
        }
    else:
        raise ValueError(f"unsupported media type: {media_type}")
    return {
        "__crypt_tool_result__": True,
        "display": metadata,
        "content": [
            {"type": "text", "text": metadata},
            block,
        ],
    }


def summary(args: dict) -> str:
    return str(args.get("path", ""))


TOOL = Tool(
    "read_media",
    "Read an image or PDF inside the workspace as a native model-visible media block.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
        },
        "required": ["path"],
    },
    "auto",
    run,
    priority=25,
    summary=summary,
    parallel_safe=True,
)
