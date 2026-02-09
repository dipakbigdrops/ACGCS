import logging
import os
import pickle
import threading
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from api.config import SEMANTIC_MODEL_VERSION, SEMANTIC_CONFIDENCE_THRESHOLD, EMBEDDING_CACHE_MAX_SIZE, LAZY_LOAD_MODELS

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=".*resume_download.*",
    category=FutureWarning
)


class EmbeddingCache:
    def __init__(self, max_size: int):
        self._max_size = max_size
        self._cache: Dict[str, np.ndarray] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()

    def get(self, text: str) -> Optional[np.ndarray]:
        with self._lock:
            return self._cache.get(text)

    def set(self, text: str, embedding: np.ndarray) -> None:
        with self._lock:
            if text in self._cache:
                self._order.remove(text)
            elif len(self._cache) >= self._max_size and self._order:
                old = self._order.pop(0)
                self._cache.pop(old, None)
            self._cache[text] = embedding
            self._order.append(text)


class SemanticAnalyzer:
    def __init__(self, model_version: Optional[str] = None):
        self.model = None
        self.model_loaded = False
        self.model_version = model_version or SEMANTIC_MODEL_VERSION
        self.id2label = None
        self.category_embeddings = None
        self._embedding_cache = EmbeddingCache(EMBEDDING_CACHE_MAX_SIZE)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(base_dir, "models", "semantic_model", self.model_version)
        self._load_lock = threading.Lock()

    def load_model(self, version: Optional[str] = None) -> None:
        with self._load_lock:
            if version is not None:
                self.model_version = version
                self.model_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "models", "semantic_model", self.model_version
                )
            try:
                logger.info("Loading sentence-transformers model: all-MiniLM-L6-v2 (version=%s)", self.model_version)
                
                import torch
                torch.set_num_threads(1)
                os.environ['OMP_NUM_THREADS'] = '1'
                os.environ['MKL_NUM_THREADS'] = '1'
                
                self.model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')

                id2label_path = os.path.join(self.model_path, "id2label.pkl")
                base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "semantic_model")
                if not os.path.exists(id2label_path) and base_path != self.model_path:
                    fallback = os.path.join(base_path, "id2label.pkl")
                    if os.path.exists(fallback):
                        id2label_path = fallback
                if os.path.exists(id2label_path):
                    with open(id2label_path, "rb") as f:
                        self.id2label = pickle.load(f)
                    logger.info("Loaded category mappings: %s categories", len(self.id2label))
                else:
                    logger.warning("id2label.pkl not found at %s, using default categories", id2label_path)
                    self.id2label = self._get_default_categories()

                self._initialize_category_embeddings()
                self.model_loaded = True
                logger.info("Semantic model loaded successfully (version=%s, memory-optimized)", self.model_version)
            except Exception as e:
                logger.error("Error loading semantic model: %s", e)
                raise

    def _get_default_categories(self) -> Dict[int, str]:
        return {
            0: "savings_claim",
            1: "urgency",
            2: "government_association",
            3: "guaranteed_outcome",
            4: "misleading_language",
            5: "neutral"
        }

    def _initialize_category_embeddings(self) -> None:
        category_keywords = {
            "savings_claim": [
                "save money", "discount", "cheaper", "lower price", "reduce cost",
                "affordable", "budget", "savings", "cut expenses", "less expensive"
            ],
            "urgency": [
                "limited time", "act now", "expires soon", "hurry", "urgent",
                "deadline", "don't miss", "last chance", "ending soon", "quick"
            ],
            "government_association": [
                "government", "federal", "irs", "official", "authorized",
                "approved by", "endorsed", "certified", "licensed", "regulated"
            ],
            "guaranteed_outcome": [
                "guarantee", "guaranteed", "promise", "assure", "certain",
                "definitely", "will work", "proven", "verified", "certified result"
            ],
            "misleading_language": [
                "free", "no cost", "risk-free", "no obligation", "hidden",
                "secret", "exclusive", "limited", "one-time", "special offer"
            ]
        }
        self.category_embeddings = {}
        for category, keywords in category_keywords.items():
            keyword_texts = " ".join(keywords)
            embedding = self.model.encode(keyword_texts, convert_to_numpy=True)
            self.category_embeddings[category] = embedding

    def _encode_cached_simple(self, texts: List[str]) -> np.ndarray:
        cached_embeddings = {}
        to_encode = []
        indices = []
        for i, t in enumerate(texts):
            cached = self._embedding_cache.get(t)
            if cached is not None:
                cached_embeddings[i] = cached
            else:
                to_encode.append(t)
                indices.append(i)
        if not to_encode:
            dim = next(iter(cached_embeddings.values())).shape[0]
            out = np.zeros((len(texts), dim), dtype=np.float32)
            for i, emb in cached_embeddings.items():
                out[i] = emb
            return out
        encoded = self.model.encode(to_encode, convert_to_numpy=True, show_progress_bar=False)
        for idx, j in enumerate(indices):
            self._embedding_cache.set(to_encode[idx], encoded[idx])
        if len(to_encode) == len(texts):
            return encoded
        dim = encoded.shape[1]
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for idx, j in enumerate(indices):
            out[j] = encoded[idx]
        for i, emb in cached_embeddings.items():
            out[i] = emb
        return out

    def classify_text(
        self,
        text: str,
        banned_categories: List[str],
        category_thresholds: Optional[Dict[str, float]] = None
    ) -> Tuple[Optional[str], float]:
        if LAZY_LOAD_MODELS and not self.model_loaded:
            logger.info("Lazy loading semantic model on first use...")
            self.load_model()
        
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        default_threshold = SEMANTIC_CONFIDENCE_THRESHOLD
        thresholds = category_thresholds or {}
        text_embedding = self._embedding_cache.get(text)
        if text_embedding is None:
            text_embedding = self.model.encode(text, convert_to_numpy=True)
            self._embedding_cache.set(text, text_embedding)
        best_category = None
        best_confidence = 0.0
        for category in banned_categories:
            if category in self.category_embeddings:
                sim = self._cosine_similarity(text_embedding, self.category_embeddings[category])
                if sim > best_confidence:
                    best_confidence = sim
                    best_category = category
        th = thresholds.get(best_category, default_threshold) if best_category else default_threshold
        if best_category and best_confidence > th:
            return best_category, float(best_confidence)
        return None, 0.0

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    def batch_classify(
        self,
        texts: List[str],
        banned_categories: List[str],
        category_thresholds: Optional[Dict[str, float]] = None
    ) -> List[Tuple[Optional[str], float]]:
        if LAZY_LOAD_MODELS and not self.model_loaded:
            logger.info("Lazy loading semantic model on first use...")
            self.load_model()
        
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if not texts:
            return []
        default_threshold = SEMANTIC_CONFIDENCE_THRESHOLD
        thresholds = category_thresholds or {}
        seen = {}
        unique_texts = []
        for t in texts:
            if t not in seen:
                seen[t] = len(unique_texts)
                unique_texts.append(t)
        if not unique_texts:
            return [(None, 0.0)] * len(texts)
        text_embeddings = self._encode_cached_simple(unique_texts)
        banned = list(banned_categories)
        unique_results = []
        for text_embedding in text_embeddings:
            best_category = None
            best_confidence = 0.0
            for category in banned:
                if category in self.category_embeddings:
                    sim = self._cosine_similarity(text_embedding, self.category_embeddings[category])
                    if sim > best_confidence:
                        best_confidence = sim
                        best_category = category
            th = thresholds.get(best_category, default_threshold) if best_category else default_threshold
            if best_category and best_confidence > th:
                unique_results.append((best_category, float(best_confidence)))
            else:
                unique_results.append((None, 0.0))
        return [unique_results[seen[t]] for t in texts]
