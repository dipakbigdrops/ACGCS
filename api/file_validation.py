import io

MAGIC = {
    "pdf": (b"%PDF", 0),
    "png": (b"\x89PNG\r\n\x1a\n", 0),
    "jpeg": (b"\xff\xd8\xff", 0),
    "jpg": (b"\xff\xd8\xff", 0),
    "gif": (b"GIF87a", 0),
    "gif2": (b"GIF89a", 0),
    "bmp": (b"BM", 0),
    "webp": (b"RIFF", 0),
}

WEBP_OFFSET = (b"WEBP", 8)

def _read_head(stream: io.BytesIO, length: int = 32) -> bytes:
    stream.seek(0)
    return stream.read(length)

def validate_file_content(extension: str, content: bytes) -> bool:
    if not content:
        return False
    head = content[:32]
    ext = extension.lower().strip()
    if ext == "pdf":
        return head.startswith(b"%PDF")
    if ext in ("png", "jpg", "jpeg"):
        if ext in ("jpg", "jpeg"):
            return head.startswith(b"\xff\xd8\xff")
        return head.startswith(b"\x89PNG\r\n\x1a\n")
    if ext == "gif":
        return head.startswith(b"GIF87a") or head.startswith(b"GIF89a")
    if ext == "bmp":
        return head.startswith(b"BM")
    if ext == "webp":
        if len(head) < 12:
            return False
        return head.startswith(b"RIFF") and head[8:12] == b"WEBP"
    if ext in ("html", "htm"):
        return b"<" in head or head.lstrip().startswith(b"<!") or head.lstrip().startswith(b"<html")
    return True

def validate_upload_extension_and_content(extension: str, content: bytes, allowed: set) -> tuple[bool, str]:
    if extension not in allowed:
        return False, f"Unsupported file type: {extension}"
    if not validate_file_content(extension, content):
        return False, f"File content does not match extension .{extension}"
    return True, ""
