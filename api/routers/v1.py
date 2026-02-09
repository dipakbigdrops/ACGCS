import asyncio
import io
import logging
import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from api.config import (
    ANALYZE_QUEUE_LIMIT,
    ANALYZE_TIMEOUT,
    ZIP_ANALYZE_TIMEOUT,
    DEFAULT_GUIDELINES_ID,
    MAX_FILE_SIZE,
    MAX_ZIP_SIZE,
    MAX_FILES_IN_ZIP,
)
from api.file_validation import validate_upload_extension_and_content
from api.metrics import record_error, record_latency, record_ocr_confidence, record_semantic_confidence
from api.models import ComplianceResponse, ZipComplianceResponse, FileAnalysisResult, Violation
from api.pdf_processor import PDFProcessor
from api.zip_processor import ZipProcessor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["v1"])


def get_deps(request: Request):
    return request.app.state


ALLOWED_CREATIVE_EXTENSIONS = {"html", "htm", "png", "jpg", "jpeg", "gif", "bmp", "webp"}
ALLOWED_PDF_EXTENSIONS = {"pdf"}
ALLOWED_ZIP_EXTENSIONS = {"zip"}


@router.post("/upload-guidelines")
async def upload_guidelines(
    request: Request,
    guidelines_pdf: Optional[UploadFile] = File(None),
):
    deps = get_deps(request)
    store = deps.guidelines_store
    pdf_processor = deps.pdf_processor
    store.cleanup_expired()
    pdf_bytes = None
    filename = None
    if guidelines_pdf and guidelines_pdf.filename:
        if not guidelines_pdf.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="File must be a PDF file")
        pdf_bytes = await guidelines_pdf.read()
        if len(pdf_bytes) == 0:
            raise HTTPException(status_code=400, detail="PDF file is empty")
        if len(pdf_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / (1024*1024):.1f}MB",
            )
        ok, err = validate_upload_extension_and_content("pdf", pdf_bytes, ALLOWED_PDF_EXTENSIONS)
        if not ok:
            raise HTTPException(status_code=400, detail=err)
        filename = guidelines_pdf.filename
    if pdf_bytes is None:
        default_path = deps.default_guidelines_path
        if not default_path or not __import__("os").path.exists(default_path):
            raise HTTPException(
                status_code=404,
                detail="No PDF file provided and default_guidelines.pdf not found.",
            )
        with open(default_path, "rb") as f:
            pdf_bytes = f.read()
        filename = "default_guidelines.pdf"
        if len(pdf_bytes) == 0:
            raise HTTPException(status_code=400, detail="default_guidelines.pdf is empty")
    rules = pdf_processor.extract_rules(io.BytesIO(pdf_bytes))
    if not rules:
        raise HTTPException(
            status_code=400,
            detail="No rules extracted from PDF. Please ensure the PDF contains readable guideline text.",
        )
    guidelines_id = str(uuid.uuid4())
    store.set(guidelines_id, rules, filename or "guidelines.pdf", ttl_hours=24, pdf_bytes=pdf_bytes)
    logger.info("Guidelines uploaded. ID: %s, rules: %s, file: %s", guidelines_id, len(rules), filename)
    return {
        "guidelines_id": guidelines_id,
        "rules_count": len(rules),
        "filename": filename,
        "message": "Guidelines uploaded and processed successfully",
    }


@router.post("/analyze")
async def analyze_compliance(
    request: Request,
    guidelines_id: Optional[str] = Form(None),
    creative_file: UploadFile = File(...),
):
    start = time.perf_counter()
    endpoint = "/v1/analyze"
    deps = get_deps(request)
    store = deps.guidelines_store
    semantic_analyzer = deps.semantic_analyzer
    rule_engine = deps.rule_engine
    text_extractor = deps.text_extractor
    analyze_semaphore = deps.analyze_semaphore
    store.cleanup_expired()
    if not guidelines_id or not str(guidelines_id).strip():
        store.ensure_default_loaded()
        guidelines_id = DEFAULT_GUIDELINES_ID

    guidelines_data = store.get(guidelines_id)
    if not guidelines_data:
        record_error(endpoint, 404)
        raise HTTPException(
            status_code=404,
            detail=f"Guidelines ID '{guidelines_id}' not found. Upload guidelines first via /v1/upload-guidelines.",
        )
    if __import__("datetime").datetime.now() > guidelines_data["expires_at"]:
        store.delete(guidelines_id)
        record_error(endpoint, 410)
        raise HTTPException(status_code=410, detail="Guidelines have expired. Please upload again.")
    rules = guidelines_data["rules"]

    if not creative_file.filename:
        record_error(endpoint, 400)
        raise HTTPException(status_code=400, detail="Creative file must have a filename")
    file_extension = creative_file.filename.lower().split(".")[-1] if "." in creative_file.filename else ""
    
    if file_extension == "zip":
        logger.info("ZIP file detected, processing with batch analyzer")
        return await _analyze_zip_file(request, guidelines_id, creative_file, start)

    if ANALYZE_QUEUE_LIMIT > 0 and hasattr(deps, "analyze_pending"):
        with deps.analyze_pending_lock:
            if deps.analyze_pending >= ANALYZE_QUEUE_LIMIT:
                record_error(endpoint, 503)
                raise HTTPException(status_code=503, detail="Analysis queue full. Try again shortly.")
            deps.analyze_pending += 1

    async def do_analyze():
        try:
            async with analyze_semaphore:
                if file_extension in ("html", "htm"):
                    html_bytes = await creative_file.read()
                    if len(html_bytes) > MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / (1024*1024):.1f}MB",
                        )
                    ok, err = validate_upload_extension_and_content(
                        file_extension, html_bytes, ALLOWED_CREATIVE_EXTENSIONS
                    )
                    if not ok:
                        raise HTTPException(status_code=400, detail=err)
                    
                    # Try multiple encodings to handle different HTML file formats
                    html_content = None
                    encodings_to_try = ['utf-8', 'utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1', 'windows-1252', 'iso-8859-1']
                    
                    for encoding in encodings_to_try:
                        try:
                            html_content = html_bytes.decode(encoding)
                            logger.info(f"Successfully decoded HTML with {encoding} encoding")
                            break
                        except (UnicodeDecodeError, LookupError):
                            continue
                    
                    if html_content is None:
                        raise HTTPException(
                            status_code=400, 
                            detail="Unable to decode HTML file. Please ensure it's a valid HTML file with standard encoding (UTF-8, Latin-1, or Windows-1252)"
                        )
                    
                    extracted_texts = await text_extractor.extract_from_html(html_content, max_image_size=MAX_FILE_SIZE)
                elif file_extension in ("png", "jpg", "jpeg", "gif", "bmp", "webp"):
                    image_bytes = await creative_file.read()
                    if len(image_bytes) == 0:
                        raise HTTPException(status_code=400, detail="Image file is empty")
                    if len(image_bytes) > MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / (1024*1024):.1f}MB",
                        )
                    ok, err = validate_upload_extension_and_content(
                        file_extension, image_bytes, ALLOWED_CREATIVE_EXTENSIONS
                    )
                    if not ok:
                        raise HTTPException(status_code=400, detail=err)
                    extracted_texts = await text_extractor.extract_from_image_async(image_bytes)
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file type: {file_extension or 'unknown'}. Use HTML or image (PNG/JPG/etc.).",
                    )

                for t in extracted_texts:
                    if t.confidence is not None and deps.enable_metrics:
                        try:
                            record_ocr_confidence(t.confidence)
                        except Exception:
                            pass

                if not extracted_texts:
                    raise HTTPException(
                        status_code=400,
                        detail="No text could be extracted from the creative file.",
                    )

                semantic_results = {}
                all_banned_categories = set()
                category_thresholds = {}
                for rule in rules:
                    if rule.rule_type == "prohibited_semantic_claim":
                        banned_categories = rule.params.get("banned_categories", [])
                        all_banned_categories.update(banned_categories)
                        th = rule.params.get("confidence_threshold")
                        if th is not None:
                            for c in banned_categories:
                                category_thresholds[c] = float(th)

                if all_banned_categories and extracted_texts:
                    texts = [t.text for t in extracted_texts]
                    classifications = semantic_analyzer.batch_classify(
                        texts, list(all_banned_categories), category_thresholds=category_thresholds or None
                    )
                    for text_obj, (category, confidence) in zip(extracted_texts, classifications):
                        text_key = text_obj.text
                        semantic_results[text_key] = (category, confidence)
                        if text_key.lower() != text_key:
                            semantic_results[text_key.lower()] = (category, confidence)
                        if deps.enable_metrics and confidence > 0:
                            try:
                                record_semantic_confidence(confidence)
                            except Exception:
                                pass

                violations = rule_engine.evaluate(rules, extracted_texts, semantic_results)
                if len(violations) == 0:
                    return ComplianceResponse(message="All guidelines are followed")
                return ComplianceResponse(message=violations)
        finally:
            if ANALYZE_QUEUE_LIMIT > 0 and hasattr(deps, "analyze_pending"):
                with deps.analyze_pending_lock:
                    deps.analyze_pending = max(0, deps.analyze_pending - 1)

    try:
        result = await __import__("asyncio").wait_for(do_analyze(), timeout=ANALYZE_TIMEOUT)
        if deps.enable_metrics:
            try:
                record_latency(endpoint, time.perf_counter() - start)
            except Exception:
                pass
        return result
    except __import__("asyncio").TimeoutError:
        logger.warning("Analyze request timed out after %s s", ANALYZE_TIMEOUT)
        record_error(endpoint, 504)
        raise HTTPException(
            status_code=504,
            detail=f"Analysis timed out after {int(ANALYZE_TIMEOUT)}s.",
        )
    except HTTPException as exc:
        record_error(endpoint, exc.status_code)
        raise


@router.api_route("/health", methods=["GET", "HEAD"])
async def health(request: Request):
    deps = get_deps(request)
    return {
        "status": "healthy",
        "model_loaded": deps.semantic_analyzer.model_loaded,
        "model_version": getattr(deps.semantic_analyzer, "model_version", "v1"),
        "version": "1.0.0",
    }


@router.get("/guidelines")
async def list_guidelines(
    request: Request,
    page: int = 1,
    page_size: int = 20,
):
    deps = get_deps(request)
    if page < 1:
        page = 1
    if page_size < 1 or page_size > 100:
        page_size = 20
    ids, total = deps.guidelines_store.list_ids(page=page, page_size=page_size)
    return {"guidelines_ids": ids, "total": total, "page": page, "page_size": page_size}


async def _analyze_zip_file(
    request: Request,
    guidelines_id: str,
    creative_zip: UploadFile,
    start: float,
) -> ZipComplianceResponse:
    """Internal function to handle ZIP file analysis."""
    endpoint = "/v1/analyze"
    deps = get_deps(request)
    store = deps.guidelines_store
    semantic_analyzer = deps.semantic_analyzer
    rule_engine = deps.rule_engine
    text_extractor = deps.text_extractor
    analyze_semaphore = deps.analyze_semaphore
    
    guidelines_data = store.get(guidelines_id)
    rules = guidelines_data["rules"]
    
    if not creative_zip.filename or not creative_zip.filename.lower().endswith('.zip'):
        record_error(endpoint, 400)
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")
    
    zip_bytes = await creative_zip.read()
    
    if len(zip_bytes) == 0:
        record_error(endpoint, 400)
        raise HTTPException(status_code=400, detail="ZIP file is empty")
    
    if len(zip_bytes) > MAX_ZIP_SIZE:
        record_error(endpoint, 413)
        raise HTTPException(
            status_code=413,
            detail=f"ZIP file size exceeds maximum allowed size of {MAX_ZIP_SIZE / (1024*1024):.0f}MB",
        )
    
    zip_processor = ZipProcessor(max_file_size=MAX_FILE_SIZE)
    
    is_valid, error_msg, stats = zip_processor.validate_zip_structure(zip_bytes)
    if not is_valid:
        record_error(endpoint, 400)
        raise HTTPException(status_code=400, detail=error_msg)
    
    logger.info(
        f"ZIP validation passed: {stats['total_files']} files "
        f"({stats['image_files']} images, {stats['html_files']} HTML), "
        f"nested_folders={stats['nested_folders']}"
    )
    
    if stats['total_files'] > MAX_FILES_IN_ZIP:
        record_error(endpoint, 400)
        raise HTTPException(
            status_code=400,
            detail=f"ZIP contains too many files ({stats['total_files']}). Maximum allowed: {MAX_FILES_IN_ZIP}",
        )
    
    if ANALYZE_QUEUE_LIMIT > 0 and hasattr(deps, "analyze_pending"):
        with deps.analyze_pending_lock:
            if deps.analyze_pending >= ANALYZE_QUEUE_LIMIT:
                record_error(endpoint, 503)
                raise HTTPException(status_code=503, detail="Analysis queue full. Try again shortly.")
            deps.analyze_pending += 1
    
    async def process_zip_files():
        try:
            results = []
            files_processed = 0
            
            async with analyze_semaphore:
                all_banned_categories = set()
                category_thresholds = {}
                for rule in rules:
                    if rule.rule_type == "prohibited_semantic_claim":
                        banned_categories = rule.params.get("banned_categories", [])
                        all_banned_categories.update(banned_categories)
                        th = rule.params.get("confidence_threshold")
                        if th is not None:
                            for c in banned_categories:
                                category_thresholds[c] = float(th)
                
                async for file_path, file_type, file_bytes in zip_processor.extract_files_sequential(zip_bytes):
                    try:
                        if file_type == 'html':
                            # Try multiple encodings to handle different HTML file formats
                            html_content = None
                            encodings_to_try = ['utf-8', 'utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1', 'windows-1252', 'iso-8859-1']
                            
                            for encoding in encodings_to_try:
                                try:
                                    html_content = file_bytes.decode(encoding)
                                    break
                                except (UnicodeDecodeError, LookupError):
                                    continue
                            
                            if html_content is None:
                                logger.warning(f"Skipping HTML file with unrecognized encoding: {file_path}")
                                continue
                            
                            extracted_texts = await text_extractor.extract_from_html(
                                html_content, 
                                max_image_size=MAX_FILE_SIZE
                            )
                        
                        elif file_type == 'image':
                            extracted_texts = await text_extractor.extract_from_image_async(file_bytes)
                        
                        else:
                            logger.warning(f"Skipping unsupported file type: {file_path}")
                            continue
                        
                        del file_bytes
                        
                        for t in extracted_texts:
                            if t.confidence is not None and deps.enable_metrics:
                                try:
                                    record_ocr_confidence(t.confidence)
                                except Exception:
                                    pass
                        
                        if not extracted_texts:
                            logger.warning(f"No text extracted from {file_path}")
                            results.append(FileAnalysisResult(
                                file_path=file_path,
                                status="pass",
                                violations=[],
                                processed_at=datetime.now().isoformat()
                            ))
                            files_processed += 1
                            continue
                        
                        semantic_results = {}
                        if all_banned_categories and extracted_texts:
                            texts = [t.text for t in extracted_texts]
                            classifications = semantic_analyzer.batch_classify(
                                texts, 
                                list(all_banned_categories), 
                                category_thresholds=category_thresholds or None
                            )
                            for text_obj, (category, confidence) in zip(extracted_texts, classifications):
                                text_key = text_obj.text
                                semantic_results[text_key] = (category, confidence)
                                if text_key.lower() != text_key:
                                    semantic_results[text_key.lower()] = (category, confidence)
                                if deps.enable_metrics and confidence > 0:
                                    try:
                                        record_semantic_confidence(confidence)
                                    except Exception:
                                        pass
                        
                        violations = rule_engine.evaluate(rules, extracted_texts, semantic_results)
                        
                        file_status = "fail" if len(violations) > 0 else "pass"
                        
                        results.append(FileAnalysisResult(
                            file_path=file_path,
                            status=file_status,
                            violations=violations,
                            processed_at=datetime.now().isoformat()
                        ))
                        
                        files_processed += 1
                        logger.info(f"Processed {files_processed}/{stats['total_files']}: {file_path} - {file_status}")
                        
                        await asyncio.sleep(0)
                        
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
                        continue
            
            passed_files = sum(1 for r in results if r.status == "pass")
            failed_files = sum(1 for r in results if r.status == "fail")
            overall_status = "pass" if failed_files == 0 else "fail"
            
            summary = (
                f"Processed {len(results)} files: "
                f"{passed_files} passed, {failed_files} failed compliance checks"
            )
            
            return ZipComplianceResponse(
                overall_status=overall_status,
                total_files=len(results),
                passed_files=passed_files,
                failed_files=failed_files,
                results=results,
                summary=summary
            )
        
        finally:
            if ANALYZE_QUEUE_LIMIT > 0 and hasattr(deps, "analyze_pending"):
                with deps.analyze_pending_lock:
                    deps.analyze_pending = max(0, deps.analyze_pending - 1)
    
    try:
        result = await asyncio.wait_for(process_zip_files(), timeout=ZIP_ANALYZE_TIMEOUT)
        if deps.enable_metrics:
            try:
                record_latency(endpoint, time.perf_counter() - start)
            except Exception:
                pass
        return result
    except asyncio.TimeoutError:
        logger.warning("ZIP analyze request timed out after %s s", ZIP_ANALYZE_TIMEOUT)
        record_error(endpoint, 504)
        raise HTTPException(
            status_code=504,
            detail=f"ZIP analysis timed out after {int(ZIP_ANALYZE_TIMEOUT)}s.",
        )
    except HTTPException as exc:
        record_error(endpoint, exc.status_code)
        raise
    except Exception as e:
        logger.error(f"Unexpected error in ZIP analysis: {e}")
        record_error(endpoint, 500)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/analyze-zip", response_model=ZipComplianceResponse)
async def analyze_zip_compliance(
    request: Request,
    guidelines_id: Optional[str] = Form(None),
    creative_zip: UploadFile = File(...),
):
    """
    Analyze multiple creative files in a ZIP archive.
    Processes files sequentially to minimize memory usage.
    Supports nested folders and mixed HTML/image files.
    
    Note: You can also use /v1/analyze with a ZIP file and it will automatically detect and process it.
    """
    start = time.perf_counter()
    deps = get_deps(request)
    store = deps.guidelines_store
    
    store.cleanup_expired()
    
    if not guidelines_id or not str(guidelines_id).strip():
        store.ensure_default_loaded()
        guidelines_id = DEFAULT_GUIDELINES_ID
    
    guidelines_data = store.get(guidelines_id)
    if not guidelines_data:
        raise HTTPException(
            status_code=404,
            detail=f"Guidelines ID '{guidelines_id}' not found. Upload guidelines first via /v1/upload-guidelines.",
        )
    
    if __import__("datetime").datetime.now() > guidelines_data["expires_at"]:
        store.delete(guidelines_id)
        raise HTTPException(status_code=410, detail="Guidelines have expired. Please upload again.")
    
    return await _analyze_zip_file(request, guidelines_id, creative_zip, start)


@router.get("/")
async def root_v1():
    return {"message": "Automated Creative Guideline Compliance API", "version": "1.0.0", "docs": "/docs"}
