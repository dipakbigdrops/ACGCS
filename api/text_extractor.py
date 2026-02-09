import io
import logging
import os
import re
import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import httpx

from api.models import ExtractedText
from api.config import (
    OCR_THREAD_POOL_WORKERS,
    IMAGE_DOWNLOAD_TIMEOUT,
    IMAGE_DOWNLOAD_RETRIES,
    CIRCUIT_BREAKER_FAILURES,
    CIRCUIT_BREAKER_RESET_SEC,
    IMAGE_DOWNLOAD_CAP,
    IMAGE_MAX_DIMENSION,
    IMAGE_JPEG_QUALITY,
    LAZY_LOAD_MODELS,
    ENABLE_PLAYWRIGHT,
    ENABLE_OCR,
)

logger = logging.getLogger(__name__)

MIN_OCR_CONFIDENCE = float(os.environ.get("MIN_OCR_CONFIDENCE", "0.25"))
EASYOCR_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".EasyOCR",
    "model"
)

try:
    from PIL import Image
    if not hasattr(Image, 'ANTIALIAS'):
        Image.ANTIALIAS = Image.LANCZOS
except ImportError:
    pass


class TextExtractor:
    def __init__(self):
        self.ocr_reader = None
        self._ocr_initialized = False
        self._ocr_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=OCR_THREAD_POOL_WORKERS, thread_name_prefix="playwright")
        self._http_client = None
        self._circuit: Dict[str, Tuple[int, float]] = {}
        self._circuit_lock = threading.Lock()
        if not LAZY_LOAD_MODELS and ENABLE_OCR:
            self._init_ocr()
    
    def __del__(self):
        """Cleanup thread pool executor on deletion."""
        if hasattr(self, '_executor') and self._executor:
            self._executor.shutdown(wait=False)
    
    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=IMAGE_DOWNLOAD_TIMEOUT)
        return self._http_client

    def _host_from_url(self, url: str) -> str:
        try:
            return urlparse(url).netloc or url
        except Exception:
            return url

    def _circuit_open(self, host: str) -> bool:
        with self._circuit_lock:
            if host not in self._circuit:
                return False
            failures, opened_at = self._circuit[host]
            if failures < CIRCUIT_BREAKER_FAILURES:
                return False
            if time.monotonic() - opened_at >= CIRCUIT_BREAKER_RESET_SEC:
                self._circuit.pop(host, None)
                return False
            return True

    def _circuit_record(self, host: str, success: bool) -> None:
        with self._circuit_lock:
            if success:
                self._circuit.pop(host, None)
                return
            failures, opened_at = self._circuit.get(host, (0, time.monotonic()))
            failures += 1
            if failures >= CIRCUIT_BREAKER_FAILURES:
                self._circuit[host] = (failures, opened_at)
                logger.warning("Circuit breaker open for host %s after %s failures", host, failures)
            else:
                self._circuit[host] = (failures, opened_at)

    def _downgrade_image(self, image_bytes: bytes, max_bytes: Optional[int] = None) -> bytes:
        try:
            from PIL import Image
            if not hasattr(Image, "ANTIALIAS"):
                Image.ANTIALIAS = Image.LANCZOS
        except ImportError:
            return image_bytes

        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.load()
        except Exception as e:
            logger.warning("Could not open image for downgrade: %s", e)
            return image_bytes

        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            try:
                img = img.convert("RGB")
            except Exception:
                pass

        w, h = img.size
        if max(w, h) > IMAGE_MAX_DIMENSION:
            ratio = IMAGE_MAX_DIMENSION / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            resample = getattr(Image, "LANCZOS", Image.ANTIALIAS)
            img = img.resize(new_size, resample)

        out = io.BytesIO()
        quality = IMAGE_JPEG_QUALITY
        img.save(out, format="JPEG", quality=quality, optimize=True)
        result = out.getvalue()

        if max_bytes and len(result) > max_bytes:
            for q in (70, 50, 35, 25):
                if q >= quality:
                    continue
                out = io.BytesIO()
                img.save(out, format="JPEG", quality=q, optimize=True)
                result = out.getvalue()
                if len(result) <= max_bytes:
                    break
            if len(result) > max_bytes:
                scale = (max_bytes / len(result)) ** 0.5
                new_w = max(320, int(img.size[0] * scale))
                new_h = max(240, int(img.size[1] * scale))
                img = img.resize((new_w, new_h), getattr(Image, "LANCZOS", Image.ANTIALIAS))
                out = io.BytesIO()
                img.save(out, format="JPEG", quality=35, optimize=True)
                result = out.getvalue()

        return result

    async def close_async(self):
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
    
    def _init_ocr(self):
        """Initialize OCR reader using EasyOCR. Models are cached in EASYOCR_MODEL_DIR and downloaded at most once."""
        if self._ocr_initialized:
            return
        
        with self._ocr_lock:
            if self._ocr_initialized:
                return
            
            if not ENABLE_OCR:
                logger.warning("OCR is disabled via ENABLE_OCR config")
                self._ocr_initialized = True
                return
            
            use_gpu = False
            try:
                os.makedirs(EASYOCR_MODEL_DIR, exist_ok=True)
            except OSError:
                pass
            try:
                from PIL import Image
                if not hasattr(Image, 'ANTIALIAS'):
                    Image.ANTIALIAS = Image.LANCZOS
                import easyocr
                import torch
                torch.set_num_threads(1)
                os.environ['OMP_NUM_THREADS'] = '1'
                os.environ['MKL_NUM_THREADS'] = '1'
                
                self.ocr_reader = easyocr.Reader(
                    ['en'],
                    gpu=False,
                    model_storage_directory=EASYOCR_MODEL_DIR,
                    download_enabled=True,
                    quantize=True,
                    verbose=False
                )
                logger.info(f"Initialized EasyOCR with CPU (memory-optimized), model dir: {EASYOCR_MODEL_DIR}")
            except ImportError:
                logger.error("EasyOCR not found. Please install easyocr: pip install easyocr")
                self.ocr_reader = None
            
            self._ocr_initialized = True
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for OCR processing."""
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
                return True
        except ImportError:
            pass
        
        logger.info("No GPU detected, using CPU")
        return False
    
    def extract_from_image(self, image_file: io.BytesIO) -> List[ExtractedText]:
        """Extract text from image using OCR (sync, may block)."""
        if LAZY_LOAD_MODELS and not self._ocr_initialized:
            logger.info("Lazy loading OCR model on first use...")
            self._init_ocr()
        
        if not ENABLE_OCR:
            logger.warning("OCR is disabled. Returning empty results.")
            return []
        
        if not self.ocr_reader:
            raise RuntimeError("OCR not initialized. Install easyocr: pip install easyocr")
        
        image_file.seek(0)
        image_bytes = image_file.read()

        if len(image_bytes) == 0:
            logger.warning("Empty image file provided")
            return []

        image_bytes = self._downgrade_image(image_bytes)

        extracted_texts = []

        try:
            results = self.ocr_reader.readtext(image_bytes)
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise RuntimeError(f"Failed to extract text from image: {str(e)}")
        for result in results:
            bbox, text, confidence = result
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            x = min(x_coords)
            y = min(y_coords)
            width = max(x_coords) - x
            height = max(y_coords) - y
            
            if text.strip() and float(confidence) >= MIN_OCR_CONFIDENCE:
                extracted_texts.append(ExtractedText(
                    text=text.strip(),
                    source="ocr",
                    bounding_box=[float(x), float(y), float(width), float(height)],
                    confidence=float(confidence)
                ))
        return self._normalize_texts(extracted_texts)

    async def extract_from_image_async(self, image_bytes: bytes) -> List[ExtractedText]:
        """Extract text from image using OCR without blocking the event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return await loop.run_in_executor(
            self._executor,
            self.extract_from_image,
            io.BytesIO(image_bytes)
        )
    
    def _extract_text_from_html_fallback(self, html_content: str) -> List[ExtractedText]:
        segments = re.findall(r">([^<]+)<", html_content)
        extracted = []
        seen = set()
        for i, seg in enumerate(segments):
            seg = seg.strip()
            if seg and seg.lower() not in seen:
                seen.add(seg.lower())
                extracted.append(ExtractedText(
                    text=seg,
                    source="dom",
                    bounding_box=[0.0, float(i * 20), 100.0, 16.0],
                    confidence=1.0
                ))
        return extracted

    def _extract_image_urls(self, html_content: str) -> List[str]:
        """Extract image URLs from HTML content."""
        img_pattern = r'<img[^>]+src=["\']([^"\']+)["\']'
        urls = re.findall(img_pattern, html_content, re.IGNORECASE)
        
        absolute_urls = []
        seen_urls = set()
        for url in urls:
            url = url.strip()
            if not url:
                continue
            
            if url.startswith('http://') or url.startswith('https://'):
                if url not in seen_urls:
                    absolute_urls.append(url)
                    seen_urls.add(url)
            elif url.startswith('//'):
                full_url = 'https:' + url
                if full_url not in seen_urls:
                    absolute_urls.append(full_url)
                    seen_urls.add(full_url)
        
        return absolute_urls
    
    async def _download_image_once(self, url: str, download_limit: int) -> Optional[io.BytesIO]:
        client = self._get_http_client()
        async with client.stream("GET", url, timeout=IMAGE_DOWNLOAD_TIMEOUT) as response:
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > download_limit:
                        logger.warning("Image from %s exceeds download cap: %s bytes", url, content_length)
                        return None
                except (ValueError, TypeError):
                    pass
            content = b""
            async for chunk in response.aiter_bytes():
                content += chunk
                if len(content) > download_limit:
                    logger.warning("Image from %s exceeds download cap during download", url)
                    return None
            return io.BytesIO(content)

    async def _download_image(self, url: str, download_limit: Optional[int] = None) -> Optional[io.BytesIO]:
        limit = download_limit if download_limit is not None else IMAGE_DOWNLOAD_CAP
        host = self._host_from_url(url)
        if self._circuit_open(host):
            logger.warning("Skipping download for %s (circuit open)", url)
            return None
        last_exc = None
        for attempt in range(IMAGE_DOWNLOAD_RETRIES):
            try:
                result = await self._download_image_once(url, limit)
                self._circuit_record(host, result is not None)
                return result
            except Exception as e:
                last_exc = e
                if attempt < IMAGE_DOWNLOAD_RETRIES - 1:
                    await asyncio.sleep(min(2 ** attempt, 10))
        logger.warning("Failed to download image from %s after %s attempts: %s", url, IMAGE_DOWNLOAD_RETRIES, last_exc)
        self._circuit_record(host, False)
        return None
    
    async def _extract_from_html_images(self, html_content: str, max_image_size: int = 10 * 1024 * 1024) -> List[ExtractedText]:
        """Extract text from images found in HTML by downloading and processing them."""
        extracted_texts = []
        
        image_urls = self._extract_image_urls(html_content)
        if not image_urls:
            logger.info("No image URLs found in HTML")
            return extracted_texts
        
        logger.info(f"Found {len(image_urls)} image URL(s) in HTML, downloading in parallel...")
        download_tasks = [self._download_image(url, download_limit=IMAGE_DOWNLOAD_CAP) for url in image_urls]
        results = await asyncio.gather(*download_tasks, return_exceptions=True)
        for img_url, result in zip(image_urls, results):
            if isinstance(result, Exception):
                logger.warning(f"Download failed for {img_url}: {result}")
                continue
            if result is None:
                continue
            try:
                loop = asyncio.get_running_loop()
                raw = result.getvalue()
                downgraded = await loop.run_in_executor(
                    self._executor,
                    lambda r=raw, m=max_image_size: self._downgrade_image(r, m),
                )
                image_texts = await loop.run_in_executor(
                    self._executor,
                    self.extract_from_image,
                    io.BytesIO(downgraded),
                )
                extracted_texts.extend(image_texts)
            except Exception as e:
                logger.warning(f"OCR failed for {img_url}: {e}")
        logger.info(f"Extracted {len(extracted_texts)} text element(s) from {len(image_urls)} image(s)")
        return extracted_texts
    
    async def extract_from_html(self, html_content: str, max_image_size: int = 10 * 1024 * 1024) -> List[ExtractedText]:
        extracted_texts = []
        
        if ENABLE_PLAYWRIGHT:
            try:
                dom_texts = await asyncio.wait_for(self._extract_dom_text(html_content), timeout=30.0)
                extracted_texts.extend(dom_texts)
                logger.info(f"Extracted {len(dom_texts)} text element(s) from DOM")
            except asyncio.TimeoutError:
                logger.warning("DOM extraction timed out, using fallback")
                extracted_texts.extend(self._extract_text_from_html_fallback(html_content))
            except Exception as e:
                logger.warning(f"DOM extraction failed: {e}, using fallback")
                extracted_texts.extend(self._extract_text_from_html_fallback(html_content))
        else:
            logger.info("Playwright disabled, using regex-based HTML parsing")
            extracted_texts.extend(self._extract_text_from_html_fallback(html_content))

        if ENABLE_OCR:
            image_texts = await self._extract_from_html_images(html_content, max_image_size=max_image_size)
            extracted_texts.extend(image_texts)
            logger.info(f"Extracted {len(image_texts)} text element(s) from HTML images")

        if not extracted_texts and ENABLE_PLAYWRIGHT and ENABLE_OCR:
            try:
                ocr_texts = await asyncio.wait_for(self._extract_html_ocr(html_content), timeout=60.0)
                extracted_texts.extend(ocr_texts)
                logger.info(f"Extracted {len(ocr_texts)} text element(s) from HTML screenshot OCR")
            except asyncio.TimeoutError:
                logger.warning("HTML OCR timed out")
                extracted_texts.extend(self._extract_text_from_html_fallback(html_content))
            except Exception as e:
                logger.warning(f"HTML OCR failed: {e}")
                extracted_texts.extend(self._extract_text_from_html_fallback(html_content))
        elif not extracted_texts:
            logger.info("No text extracted, using regex fallback")
            extracted_texts.extend(self._extract_text_from_html_fallback(html_content))
        else:
            logger.info("Skipping HTML screenshot OCR (DOM/image text already available)")

        return self._deduplicate_texts(extracted_texts)
    
    def _run_playwright_dom_extraction(self, html_content: str) -> List[ExtractedText]:
        """Run Playwright DOM extraction in a separate thread using sync API."""
        extracted_texts = []
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        '--disable-dev-shm-usage',
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--single-process',
                        '--disable-gpu',
                        '--disable-software-rasterizer',
                        '--disable-extensions',
                        '--disable-background-networking',
                        '--disable-sync',
                        '--disable-translate',
                        '--metrics-recording-only',
                        '--no-first-run',
                        '--mute-audio',
                        '--disable-breakpad'
                    ]
                )
                page = browser.new_page(viewport={'width': 1280, 'height': 720})
                
                page.set_content(html_content, wait_until='domcontentloaded', timeout=10000)
                
                text_elements = page.evaluate("""
                    () => {
                        const elements = [];
                        const walker = document.createTreeWalker(
                            document.body,
                            NodeFilter.SHOW_TEXT,
                            null,
                            false
                        );
                        
                        let node;
                        while (node = walker.nextNode()) {
                            const text = node.textContent.trim();
                            if (text && text.length > 0) {
                                const parent = node.parentElement;
                                if (parent) {
                                    const rect = parent.getBoundingClientRect();
                                    elements.push({
                                        text: text,
                                        x: rect.x,
                                        y: rect.y,
                                        width: rect.width,
                                        height: rect.height
                                    });
                                }
                            }
                        }
                        return elements;
                    }
                """)
                
                for elem in text_elements:
                    if elem['text'].strip():
                        extracted_texts.append(ExtractedText(
                            text=elem['text'].strip(),
                            source="dom",
                            bounding_box=[
                                float(elem['x']),
                                float(elem['y']),
                                float(elem['width']),
                                float(elem['height'])
                            ],
                            confidence=1.0
                        ))
                
                browser.close()
        except PlaywrightTimeoutError:
            logger.warning("Playwright timeout, continuing with partial extraction")
        except Exception as e:
            logger.error(f"DOM extraction error: {e}")
        
        return extracted_texts
    
    async def _extract_dom_text(self, html_content: str) -> List[ExtractedText]:
        """Extract text from DOM using Playwright."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        extracted_texts = await loop.run_in_executor(
            self._executor,
            self._run_playwright_dom_extraction,
            html_content
        )
        return extracted_texts
    
    def _run_playwright_ocr_extraction(self, html_content: str) -> List[ExtractedText]:
        """Run Playwright OCR extraction in a separate thread using sync API."""
        if not self.ocr_reader:
            return []
        
        extracted_texts = []
        screenshot_bytes = None
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        '--disable-dev-shm-usage',
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--single-process',
                        '--disable-gpu',
                        '--disable-software-rasterizer',
                        '--disable-extensions',
                        '--disable-background-networking',
                        '--disable-sync',
                        '--disable-translate',
                        '--metrics-recording-only',
                        '--no-first-run',
                        '--mute-audio',
                        '--disable-breakpad'
                    ]
                )
                page = browser.new_page(viewport={'width': 1280, 'height': 720})
                
                page.set_content(html_content, wait_until='domcontentloaded', timeout=10000)
                
                screenshot_bytes = page.screenshot(type='png', full_page=True)
                browser.close()
        except Exception as e:
            logger.warning(f"HTML OCR extraction error: {e}")
        
        if screenshot_bytes:
            image_file = io.BytesIO(screenshot_bytes)
            ocr_texts = self.extract_from_image(image_file)
            extracted_texts.extend(ocr_texts)
        
        return extracted_texts
    
    async def _extract_html_ocr(self, html_content: str) -> List[ExtractedText]:
        """Extract text from HTML screenshot using OCR."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        extracted_texts = await loop.run_in_executor(
            self._executor,
            self._run_playwright_ocr_extraction,
            html_content
        )
        return extracted_texts
    
    def _normalize_texts(self, texts: List[ExtractedText]) -> List[ExtractedText]:
        """Normalize extracted texts (lowercase, strip, dedupe)."""
        normalized = []
        seen_texts = set()
        
        for text in texts:
            normalized_text = text.text.lower().strip()
            if normalized_text and normalized_text not in seen_texts:
                seen_texts.add(normalized_text)
                text.text = normalized_text
                normalized.append(text)
        
        return normalized
    
    def _deduplicate_texts(self, texts: List[ExtractedText]) -> List[ExtractedText]:
        """Remove duplicate texts based on content and bounding box overlap."""
        if not texts:
            return []
        
        deduplicated = []
        
        for i, text1 in enumerate(texts):
            is_duplicate = False
            
            for j, text2 in enumerate(texts):
                if i >= j:
                    continue
                
                if self._texts_overlap(text1, text2):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(text1)
        
        return self._normalize_texts(deduplicated)
    
    def _texts_overlap(self, text1: ExtractedText, text2: ExtractedText) -> bool:
        """Check if two text extractions overlap significantly."""
        if text1.text.lower() == text2.text.lower():
            return True
        
        if len(text1.bounding_box) < 4 or len(text2.bounding_box) < 4:
            return False
        
        x1, y1, w1, h1 = text1.bounding_box[:4]
        x2, y2, w2, h2 = text2.bounding_box[:4]
        
        overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        overlap_area = overlap_x * overlap_y
        
        area1 = w1 * h1
        area2 = w2 * h2
        
        if area1 == 0 or area2 == 0:
            return False
        
        overlap_ratio = overlap_area / min(area1, area2)
        return overlap_ratio > 0.5

