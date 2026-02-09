import asyncio
import io
import logging
import os
import threading
import time
import uuid
import warnings

warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)

try:
    import torch.utils._pytree as _torch_pytree
    if not hasattr(_torch_pytree, "register_pytree_node") and hasattr(_torch_pytree, "_register_pytree_node"):
        def _register_pytree_node_compat(typ, flatten_fn, unflatten_fn, **kwargs):
            _torch_pytree._register_pytree_node(typ, flatten_fn, unflatten_fn)
        _torch_pytree.register_pytree_node = _register_pytree_node_compat
except ImportError:
    pass

try:
    from PIL import Image
    if not hasattr(Image, "ANTIALIAS"):
        Image.ANTIALIAS = Image.LANCZOS
except ImportError:
    pass

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import (
    ALLOWED_ORIGINS,
    ANALYZE_QUEUE_LIMIT,
    DEFAULT_GUIDELINES_ID,
    ENABLE_METRICS,
    GUIDELINES_PERSISTENCE_DIR,
    MAX_CONCURRENT_ANALYZE,
    MODEL_HOT_RELOAD_POLL_SEC,
    RATE_LIMIT_HEADER,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SEC,
    LAZY_LOAD_MODELS,
)
from api.guidelines_store import GuidelinesStore
from api.metrics import get_metrics, record_error, record_latency, prometheus_export
from api.pdf_processor import PDFProcessor
from api.rate_limit import SlidingWindowRateLimiter
from api.routers import v1
from api.semantic_analyzer import SemanticAnalyzer
from api.text_extractor import TextExtractor
from api.rule_engine import RuleEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GUIDELINES_PATH = os.path.join(BASE_DIR, "default_guidelines.pdf")


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = GuidelinesStore(
        persistence_dir=GUIDELINES_PERSISTENCE_DIR,
        default_guidelines_path=DEFAULT_GUIDELINES_PATH,
        default_id=DEFAULT_GUIDELINES_ID,
    )
    pdf_processor = PDFProcessor()
    text_extractor = TextExtractor()
    semantic_analyzer = SemanticAnalyzer()
    rule_engine = RuleEngine()
    
    if not LAZY_LOAD_MODELS:
        logger.info("Loading models at startup (LAZY_LOAD_MODELS=false)")
        semantic_analyzer.load_model()
    else:
        logger.info("Lazy loading enabled - models will load on first use")
    
    store.cleanup_expired()

    app.state.guidelines_store = store
    app.state.default_guidelines_path = DEFAULT_GUIDELINES_PATH
    app.state.pdf_processor = pdf_processor
    app.state.text_extractor = text_extractor
    app.state.semantic_analyzer = semantic_analyzer
    app.state.rule_engine = rule_engine
    app.state.analyze_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ANALYZE)
    app.state.enable_metrics = ENABLE_METRICS
    app.state.analyze_pending = 0
    app.state.analyze_pending_lock = threading.Lock()

    if MODEL_HOT_RELOAD_POLL_SEC > 0:
        from api.config import SEMANTIC_MODEL_VERSION
        current_version = [SEMANTIC_MODEL_VERSION]
        version_file = os.path.join(BASE_DIR, "models", "semantic_model", ".version")

        async def poll_reload():
            while True:
                await asyncio.sleep(MODEL_HOT_RELOAD_POLL_SEC)
                try:
                    if os.path.exists(version_file):
                        with open(version_file, "r") as f:
                            new_ver = f.read().strip()
                        if new_ver and new_ver != current_version[0]:
                            semantic_analyzer.load_model(version=new_ver)
                            current_version[0] = new_ver
                            logger.info("Hot-reloaded semantic model to version %s", new_ver)
                except Exception as e:
                    logger.warning("Hot-reload check failed: %s", e)

        task = asyncio.create_task(poll_reload())
    else:
        task = None

    logger.info("API ready")
    yield
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    await text_extractor.close_async()
    logger.info("Shutting down...")


app = FastAPI(
    title="Automated Creative Guideline Compliance API",
    description="Real-time text-based compliance checking for marketing creatives",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rate_limiter = SlidingWindowRateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SEC)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())[:12]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    key = request.headers.get(RATE_LIMIT_HEADER, "").strip()
    if key:
        allowed, retry_after = rate_limiter.is_allowed(key)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Retry later."},
                headers={"Retry-After": str(retry_after)},
            )
    return await call_next(request)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    path = request.url.path or "/"
    response = await call_next(request)
    if ENABLE_METRICS:
        try:
            record_latency(path, time.perf_counter() - start)
            record_error(path, response.status_code)
        except Exception:
            pass
    return response


app.include_router(v1.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    from fastapi import HTTPException
    if isinstance(exc, HTTPException):
        raise exc
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    if ENABLE_METRICS:
        try:
            record_error(request.url.path, 500)
        except Exception:
            pass
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
async def health_root(request: Request):
    return {
        "status": "healthy",
        "model_loaded": request.app.state.semantic_analyzer.model_loaded,
        "model_version": getattr(request.app.state.semantic_analyzer, "model_version", "v1"),
        "version": "1.0.0",
    }


@app.get("/metrics")
async def metrics_json(request: Request):
    if not ENABLE_METRICS:
        return JSONResponse(status_code=404, content={"detail": "Metrics disabled"})
    return get_metrics()


@app.get("/metrics/prometheus")
async def metrics_prometheus(request: Request):
    if not ENABLE_METRICS:
        return JSONResponse(status_code=404, content={"detail": "Metrics disabled"})
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(prometheus_export(), media_type="text/plain")


@app.get("/")
async def root():
    return {
        "message": "Automated Creative Guideline Compliance API",
        "version": "1.0.0",
        "docs": "/docs",
        "v1": "/v1",
    }
