import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from api.models import Rule
from api.pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)


class GuidelinesStore:
    def __init__(self, persistence_dir: str = "", default_guidelines_path: str = "", default_id: str = "default"):
        self._memory: Dict[str, dict] = {}
        self._lock = __import__("threading").RLock()
        self._persistence_dir = persistence_dir
        self._default_path = default_guidelines_path
        self._default_id = default_id
        self._pdf_processor = PDFProcessor()

    def _meta_path(self, guidelines_id: str) -> str:
        return os.path.join(self._persistence_dir, f"{guidelines_id}.meta.json")

    def _pdf_path(self, guidelines_id: str) -> str:
        return os.path.join(self._persistence_dir, f"{guidelines_id}.pdf")

    def _rules_to_json(self, rules: List[Rule]) -> List[dict]:
        return [r.model_dump() for r in rules]

    def _rules_from_json(self, data: List[dict]) -> List[Rule]:
        return [Rule(**item) for item in data]

    def _load_one(self, guidelines_id: str) -> Optional[dict]:
        if not self._persistence_dir:
            return self._memory.get(guidelines_id)
        meta_path = self._meta_path(guidelines_id)
        if not os.path.exists(meta_path):
            return self._memory.get(guidelines_id)
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if datetime.fromisoformat(meta["expires_at"]) < datetime.now():
                self.delete(guidelines_id)
                return None
            pdf_path = self._pdf_path(guidelines_id)
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                rules = self._pdf_processor.extract_rules(__import__("io").BytesIO(pdf_bytes))
                meta["rules"] = rules
            else:
                meta["rules"] = self._rules_from_json(meta.get("rules_data", []))
            return meta
        except Exception as e:
            logger.warning(f"Failed to load persisted guidelines {guidelines_id}: {e}")
            return self._memory.get(guidelines_id)

    def get(self, guidelines_id: str) -> Optional[dict]:
        with self._lock:
            if guidelines_id in self._memory:
                data = self._memory[guidelines_id]
                if data["expires_at"] < datetime.now():
                    self.delete(guidelines_id)
                    return None
                return data
            return self._load_one(guidelines_id)

    def set(
        self,
        guidelines_id: str,
        rules: List[Rule],
        filename: str,
        ttl_hours: int = 24,
        pdf_bytes: Optional[bytes] = None,
    ) -> None:
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        data = {
            "rules": rules,
            "filename": filename,
            "uploaded_at": datetime.now(),
            "expires_at": expires_at,
        }
        with self._lock:
            self._memory[guidelines_id] = data
        if self._persistence_dir:
            try:
                os.makedirs(self._persistence_dir, exist_ok=True)
                meta_path = self._meta_path(guidelines_id)
                meta = {
                    "filename": filename,
                    "uploaded_at": data["uploaded_at"].isoformat(),
                    "expires_at": data["expires_at"].isoformat(),
                    "rules_data": self._rules_to_json(rules),
                }
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=0)
                if pdf_bytes:
                    with open(self._pdf_path(guidelines_id), "wb") as f:
                        f.write(pdf_bytes)
            except Exception as e:
                logger.warning("Failed to persist guidelines %s: %s", guidelines_id, e)

    def delete(self, guidelines_id: str) -> None:
        with self._lock:
            self._memory.pop(guidelines_id, None)
        if self._persistence_dir:
            for p in (self._meta_path(guidelines_id), self._pdf_path(guidelines_id)):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass

    def list_ids(self, page: int = 1, page_size: int = 20) -> Tuple[List[str], int]:
        with self._lock:
            now = datetime.now()
            valid = [gid for gid, d in self._memory.items() if d["expires_at"] > now]
            if self._persistence_dir:
                for f in os.listdir(self._persistence_dir):
                    if f.endswith(".meta.json"):
                        gid = f.replace(".meta.json", "")
                        if gid not in self._memory and self._load_one(gid):
                            valid.append(gid)
            valid = list(dict.fromkeys(valid))
            total = len(valid)
            start = (page - 1) * page_size
            end = start + page_size
            return valid[start:end], total

    def cleanup_expired(self, keep_default: Optional[str] = None) -> int:
        removed = 0
        keep = keep_default if keep_default is not None else self._default_id
        with self._lock:
            now = datetime.now()
            to_remove = [gid for gid, d in self._memory.items() if d["expires_at"] <= now and gid != keep]
            for gid in to_remove:
                self.delete(gid)
                removed += 1
        if self._persistence_dir:
            for f in os.listdir(self._persistence_dir):
                if f.endswith(".meta.json"):
                    gid = f.replace(".meta.json", "")
                    if gid == keep:
                        continue
                    m = self._load_one(gid)
                    if m is None:
                        removed += 1
        if removed:
            logger.info(f"Cleaned up {removed} expired guideline(s)")
        return removed

    def ensure_default_loaded(self) -> None:
        from fastapi import HTTPException
        with self._lock:
            if self._memory.get(self._default_id):
                if self._memory[self._default_id]["expires_at"] > datetime.now():
                    return
        if not self._default_path or not os.path.exists(self._default_path):
            raise HTTPException(status_code=404, detail="Default guidelines file not found.")
        with open(self._default_path, "rb") as f:
            pdf_bytes = f.read()
        if len(pdf_bytes) == 0:
            raise HTTPException(status_code=400, detail="Default guidelines file is empty.")
        rules = self._pdf_processor.extract_rules(__import__("io").BytesIO(pdf_bytes))
        if not rules:
            raise HTTPException(status_code=400, detail="No rules extracted from default guidelines.")
        self.set(self._default_id, rules, "default_guidelines.pdf", ttl_hours=24)
        logger.info(f"Loaded default guidelines: {len(rules)} rules")
