import os

MAX_FILE_SIZE = int(os.environ.get("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024
MAX_ZIP_SIZE = int(os.environ.get("MAX_ZIP_SIZE_MB", "50")) * 1024 * 1024
MAX_FILES_IN_ZIP = int(os.environ.get("MAX_FILES_IN_ZIP", "100"))
DEFAULT_GUIDELINES_ID = "default"
ANALYZE_TIMEOUT = float(os.environ.get("ANALYZE_TIMEOUT_SEC", "150"))
ZIP_ANALYZE_TIMEOUT = float(os.environ.get("ZIP_ANALYZE_TIMEOUT_SEC", "300"))
MAX_CONCURRENT_ANALYZE = int(os.environ.get("MAX_CONCURRENT_ANALYZE", "1"))
ANALYZE_QUEUE_LIMIT = int(os.environ.get("ANALYZE_QUEUE_LIMIT", "0"))

_origins = os.environ.get("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _origins.split(",")] if _origins else ["*"]

SEMANTIC_MODEL_VERSION = os.environ.get("SEMANTIC_MODEL_VERSION", "v1")
SEMANTIC_CONFIDENCE_THRESHOLD = float(os.environ.get("SEMANTIC_CONFIDENCE_THRESHOLD", "0.5"))
EMBEDDING_CACHE_MAX_SIZE = int(os.environ.get("EMBEDDING_CACHE_MAX_SIZE", "5000"))

OCR_THREAD_POOL_WORKERS = int(os.environ.get("OCR_THREAD_POOL_WORKERS", "1"))
IMAGE_DOWNLOAD_RETRIES = int(os.environ.get("IMAGE_DOWNLOAD_RETRIES", "2"))
IMAGE_DOWNLOAD_TIMEOUT = float(os.environ.get("IMAGE_DOWNLOAD_TIMEOUT", "20.0"))
IMAGE_DOWNLOAD_CAP = int(os.environ.get("IMAGE_DOWNLOAD_CAP_MB", "25")) * 1024 * 1024
IMAGE_MAX_DIMENSION = int(os.environ.get("IMAGE_MAX_DIMENSION", "1920"))
IMAGE_JPEG_QUALITY = int(os.environ.get("IMAGE_JPEG_QUALITY", "85"))
CIRCUIT_BREAKER_FAILURES = int(os.environ.get("CIRCUIT_BREAKER_FAILURES", "5"))
CIRCUIT_BREAKER_RESET_SEC = float(os.environ.get("CIRCUIT_BREAKER_RESET_SEC", "60.0"))

GUIDELINES_PERSISTENCE_DIR = os.environ.get("GUIDELINES_PERSISTENCE_DIR", "")
RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW_SEC = int(os.environ.get("RATE_LIMIT_WINDOW_SEC", "60"))
RATE_LIMIT_HEADER = os.environ.get("RATE_LIMIT_HEADER", "X-API-Key")

ENABLE_METRICS = os.environ.get("ENABLE_METRICS", "true").lower() in ("1", "true", "yes")
MODEL_HOT_RELOAD_POLL_SEC = float(os.environ.get("MODEL_HOT_RELOAD_POLL_SEC", "0"))

LOW_MEMORY_MODE = os.environ.get("LOW_MEMORY_MODE", "false").lower() in ("1", "true", "yes")
LAZY_LOAD_MODELS = os.environ.get("LAZY_LOAD_MODELS", "true").lower() in ("1", "true", "yes")
ENABLE_PLAYWRIGHT = os.environ.get("ENABLE_PLAYWRIGHT", "false").lower() in ("1", "true", "yes")
ENABLE_OCR = os.environ.get("ENABLE_OCR", "true").lower() in ("1", "true", "yes")
