import json
import logging
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    """Collapse tabs / odd whitespace for better BM25 tokenization."""
    if not text:
        return ""
    text = text.replace("\t", " ").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    text = _normalize_text(text).lower()
    # Keep alphanumerics and simple tech tokens (c++, rag, llm)
    return re.findall(r"[a-z0-9][a-z0-9+.#/-]*", text)


class HybridRetriever:
    """
    Component B: Hybrid Retrieval, Reranking & Feature Vector Extraction.

    Pipeline:
      1. Dense semantic search (Chroma embeddings)
      2. Sparse BM25 lexical search over the same corpus
      3. Reciprocal Rank Fusion (RRF) to merge both ranked lists
      4. MMR diversification on the fused candidate pool
      5. Cross-Encoder rerank → top K
      6. State vector S_t = [E_query (384) | Score_avg | Score_max]
    """

    def __init__(self, ingestor, config: Optional[PipelineConfig] = None):
        self.ingestor = ingestor
        self.config = config or PipelineConfig()
        self.cross_encoder = None
        self._init_reranker()

        # BM25 corpus cache (rebuilt when Chroma count / hash changes)
        self._bm25 = None
        self._bm25_ids: List[str] = []
        self._bm25_docs: List[str] = []
        self._bm25_metas: List[Dict[str, Any]] = []
        self._bm25_fingerprint: Optional[int] = None

    def _init_reranker(self):
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading CrossEncoder reranker: {self.config.reranker_model_name}")
            self.cross_encoder = CrossEncoder(self.config.reranker_model_name)
        except Exception as e:
            logger.warning(f"CrossEncoder load warning: {e}. Attempting Flashrank fallback...")
            try:
                from flashrank import Ranker
                self.cross_encoder = Ranker()
            except Exception as e2:
                logger.warning(f"Flashrank fallback error: {e2}. Using cosine/hybrid scores.")
                self.cross_encoder = None

    def invalidate_bm25(self):
        """Call after ingest / reset so the BM25 index rebuilds."""
        self._bm25 = None
        self._bm25_fingerprint = None

    def _corpus_fingerprint(self) -> int:
        try:
            return int(self.ingestor.collection.count())
        except Exception:
            return -1

    def _ensure_bm25(self) -> bool:
        """Load all Chroma documents and build / refresh BM25 index."""
        fp = self._corpus_fingerprint()
        if self._bm25 is not None and self._bm25_fingerprint == fp and fp > 0:
            return True
        if fp <= 0:
            self._bm25 = None
            self._bm25_ids, self._bm25_docs, self._bm25_metas = [], [], []
            self._bm25_fingerprint = fp
            return False

        try:
            data = self.ingestor.collection.get(include=["documents", "metadatas"])
            ids = data.get("ids") or []
            documents = data.get("documents") or []
            metadatas = data.get("metadatas") or [{}] * len(ids)
            if not ids or not documents:
                return False

            tokenized = []
            clean_docs = []
            for doc in documents:
                cleaned = _normalize_text(doc or "")
                clean_docs.append(cleaned)
                tokenized.append(_tokenize(cleaned))

            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(tokenized)
            self._bm25_ids = list(ids)
            self._bm25_docs = clean_docs
            self._bm25_metas = list(metadatas)
            self._bm25_fingerprint = fp
            logger.info(f"BM25 index ready over {len(ids)} chunks.")
            return True
        except Exception as e:
            logger.warning(f"BM25 index build failed: {e}")
            self._bm25 = None
            return False

    def _bm25_search(self, query: str, top_n: int) -> List[Dict[str, Any]]:
        if not self._ensure_bm25() or self._bm25 is None:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)
        if scores.size == 0:
            return []
        top_n = min(top_n, len(scores))
        # Argpartition for speed, then sort the top slice
        if top_n < len(scores):
            candidate_idx = np.argpartition(scores, -top_n)[-top_n:]
            candidate_idx = candidate_idx[np.argsort(scores[candidate_idx])[::-1]]
        else:
            candidate_idx = np.argsort(scores)[::-1]

        results = []
        max_score = float(scores[candidate_idx[0]]) if top_n else 1.0
        max_score = max(max_score, 1e-9)
        for rank, idx in enumerate(candidate_idx):
            raw = float(scores[idx])
            if raw <= 0:
                continue
            results.append({
                "chunk_id": self._bm25_ids[idx],
                "text": self._bm25_docs[idx],
                "metadata": self._bm25_metas[idx] if idx < len(self._bm25_metas) else {},
                "bm25_score": raw,
                "bm25_norm": raw / max_score,
                "bm25_rank": rank + 1,
                "source": "bm25",
            })
        return results

    def _semantic_search(self, query_emb: np.ndarray, top_n: int) -> List[Dict[str, Any]]:
        try:
            count = self.ingestor.collection.count()
            if count == 0:
                return []
            n_results = min(max(top_n, 1), count)
            res = self.ingestor.collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=n_results,
                include=["documents", "metadatas", "embeddings", "distances"],
            )
        except Exception as e:
            logger.error(f"Chroma semantic query error: {e}")
            return []

        ids = (res.get("ids") or [[]])[0]
        documents = (res.get("documents") or [[]])[0]
        metadatas = (res.get("metadatas") or [[]])[0]
        embeddings = (res.get("embeddings") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]

        results = []
        for rank, (cid, doc, meta, emb, dist) in enumerate(
            zip(ids, documents, metadatas, embeddings, distances)
        ):
            # Chroma cosine space: distance ≈ 1 - cosine_sim
            cosine = 1.0 - float(dist) if dist is not None else 0.0
            cosine = max(-1.0, min(1.0, cosine))
            results.append({
                "chunk_id": cid,
                "text": _normalize_text(doc or ""),
                "metadata": meta or {},
                "embedding": np.asarray(emb, dtype=np.float64) if emb is not None else None,
                "cosine_sim": cosine,
                "semantic_rank": rank + 1,
                "source": "semantic",
            })
        return results

    def _rrf_fuse(
        self,
        semantic_hits: List[Dict[str, Any]],
        bm25_hits: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion:
          score(d) = w_sem / (k + rank_sem) + w_bm25 / (k + rank_bm25)
        """
        k = self.config.rrf_k
        w_sem = self.config.semantic_weight
        w_bm25 = self.config.bm25_weight
        fused: Dict[str, Dict[str, Any]] = {}

        for hit in semantic_hits:
            cid = hit["chunk_id"]
            entry = fused.setdefault(cid, {
                "chunk_id": cid,
                "text": hit.get("text", ""),
                "metadata": hit.get("metadata", {}),
                "embedding": hit.get("embedding"),
                "cosine_sim": hit.get("cosine_sim", 0.0),
                "bm25_score": 0.0,
                "bm25_norm": 0.0,
                "rrf_score": 0.0,
                "sources": [],
            })
            rank = hit.get("semantic_rank", 999)
            entry["rrf_score"] += w_sem / (k + rank)
            entry["cosine_sim"] = hit.get("cosine_sim", entry["cosine_sim"])
            if hit.get("embedding") is not None:
                entry["embedding"] = hit["embedding"]
            if hit.get("text"):
                entry["text"] = hit["text"]
            if "semantic" not in entry["sources"]:
                entry["sources"].append("semantic")

        for hit in bm25_hits:
            cid = hit["chunk_id"]
            entry = fused.setdefault(cid, {
                "chunk_id": cid,
                "text": hit.get("text", ""),
                "metadata": hit.get("metadata", {}),
                "embedding": None,
                "cosine_sim": 0.0,
                "bm25_score": 0.0,
                "bm25_norm": 0.0,
                "rrf_score": 0.0,
                "sources": [],
            })
            rank = hit.get("bm25_rank", 999)
            entry["rrf_score"] += w_bm25 / (k + rank)
            entry["bm25_score"] = hit.get("bm25_score", 0.0)
            entry["bm25_norm"] = hit.get("bm25_norm", 0.0)
            if hit.get("text") and not entry.get("text"):
                entry["text"] = hit["text"]
            if hit.get("metadata") and not entry.get("metadata"):
                entry["metadata"] = hit["metadata"]
            if "bm25" not in entry["sources"]:
                entry["sources"].append("bm25")

        # Fill missing embeddings for BM25-only hits (needed for MMR)
        missing = [cid for cid, e in fused.items() if e.get("embedding") is None]
        if missing:
            try:
                got = self.ingestor.collection.get(ids=missing, include=["embeddings", "documents", "metadatas"])
                for cid, emb, doc, meta in zip(
                    got.get("ids") or [],
                    got.get("embeddings") or [],
                    got.get("documents") or [],
                    got.get("metadatas") or [],
                ):
                    if cid in fused:
                        fused[cid]["embedding"] = np.asarray(emb, dtype=np.float64) if emb is not None else None
                        if doc:
                            fused[cid]["text"] = _normalize_text(doc)
                        if meta:
                            fused[cid]["metadata"] = meta
            except Exception as e:
                logger.warning(f"Could not backfill embeddings for BM25 hits: {e}")

        ranked = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
        return ranked

    def _compute_mmr(
        self,
        query_embedding: np.ndarray,
        candidates: List[Dict[str, Any]],
        top_n: int,
        lambda_param: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        MMR on hybrid-fused candidates using dense embeddings for diversity:
        MMR = λ · Sim(d, q) − (1−λ) · max_{s∈S} Sim(d, s)
        Relevance uses a blend of cosine + RRF when available.
        """
        if not candidates:
            return []

        usable = [c for c in candidates if c.get("embedding") is not None]
        if not usable:
            return candidates[:top_n]

        top_n = min(top_n, len(usable))
        doc_embeddings = np.vstack([np.asarray(c["embedding"], dtype=np.float64) for c in usable])

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        doc_norms = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-9)
        query_sims = np.dot(doc_norms, q_norm)
        doc_sims = np.dot(doc_norms, doc_norms.T)

        # Blend dense similarity with normalized RRF for relevance signal
        rrf_scores = np.array([c.get("rrf_score", 0.0) for c in usable], dtype=np.float64)
        rrf_max = float(rrf_scores.max()) if rrf_scores.size else 1.0
        rrf_norm = rrf_scores / max(rrf_max, 1e-9)
        relevance = 0.55 * query_sims + 0.45 * rrf_norm

        selected: List[int] = []
        remaining = list(range(len(usable)))
        first = int(np.argmax(relevance))
        selected.append(first)
        remaining.remove(first)

        while len(selected) < top_n and remaining:
            best_idx = None
            best_score = -float("inf")
            for c_idx in remaining:
                redundancy = max(doc_sims[c_idx, s_idx] for s_idx in selected)
                mmr = lambda_param * relevance[c_idx] - (1.0 - lambda_param) * redundancy
                if mmr > best_score:
                    best_score = mmr
                    best_idx = c_idx
            if best_idx is None:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)

        out = []
        for idx in selected:
            cand = dict(usable[idx])
            cand["cosine_sim"] = float(query_sims[idx])
            cand["hybrid_relevance"] = float(relevance[idx])
            # Deserialize payload for downstream synthesis
            raw_meta = cand.get("metadata") or {}
            json_payload = {}
            if "json_payload" in raw_meta:
                try:
                    json_payload = json.loads(raw_meta["json_payload"])
                except Exception:
                    json_payload = {}
            cand["payload"] = json_payload
            out.append(cand)
        return out

    def hybrid_retrieve(self, query: str, top_k: Optional[int] = None) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
        """
        BM25 ∪ Semantic → RRF → MMR candidate set (size ≈ multiplier * K).
        """
        k = top_k or self.config.top_k
        n_mmr = max(k * self.config.mmr_candidates_multiplier, k)

        query_emb = self.ingestor.embedding_model.encode(
            _normalize_text(query), convert_to_numpy=True
        ).astype(np.float64)

        try:
            if self.ingestor.collection.count() == 0:
                logger.warning("ChromaDB collection is empty. Please ingest documents first.")
                return [], query_emb, {"bm25": 0, "semantic": 0, "fused": 0}
        except Exception:
            pass

        semantic_hits = self._semantic_search(query_emb, self.config.semantic_top_n)
        bm25_hits = self._bm25_search(query, self.config.bm25_top_n) if self.config.hybrid_enabled else []

        if self.config.hybrid_enabled and bm25_hits:
            fused = self._rrf_fuse(semantic_hits, bm25_hits)
        else:
            # Dense-only fallback
            fused = []
            for hit in semantic_hits:
                fused.append({
                    **hit,
                    "rrf_score": 1.0 / (self.config.rrf_k + hit.get("semantic_rank", 1)),
                    "bm25_score": 0.0,
                    "bm25_norm": 0.0,
                    "sources": ["semantic"],
                })

        # Cap fused pool before MMR
        fused = fused[: max(self.config.bm25_top_n, self.config.semantic_top_n)]
        candidates = self._compute_mmr(
            query_embedding=query_emb,
            candidates=fused,
            top_n=n_mmr,
            lambda_param=self.config.mmr_lambda,
        )

        stats = {
            "bm25": len(bm25_hits),
            "semantic": len(semantic_hits),
            "fused": len(fused),
            "mmr": len(candidates),
        }
        logger.info(
            f"Hybrid retrieve: BM25={stats['bm25']} semantic={stats['semantic']} "
            f"fused={stats['fused']} mmr={stats['mmr']}"
        )
        return candidates, query_emb, stats

    def mmr_retrieve(self, query: str, top_k: Optional[int] = None) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """Backward-compatible alias → hybrid_retrieve. """
        candidates, query_emb, _ = self.hybrid_retrieve(query, top_k=top_k)
        return candidates, query_emb

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        k = top_k or self.config.top_k
        if not candidates:
            return []

        pairs = [[query, cand.get("text", "")] for cand in candidates]
        scores: List[float]

        if self.cross_encoder is not None:
            try:
                if hasattr(self.cross_encoder, "predict"):
                    raw_scores = self.cross_encoder.predict(pairs)
                    scores = (1.0 / (1.0 + np.exp(-np.array(raw_scores, dtype=np.float64)))).tolist()
                elif hasattr(self.cross_encoder, "rerank"):
                    from flashrank import RerankRequest
                    passages = [{"id": i, "text": cand.get("text", "")} for i, cand in enumerate(candidates)]
                    req = RerankRequest(query=query, passages=passages)
                    res = self.cross_encoder.rerank(req)
                    scores_dict = {item["id"]: float(item["score"]) for item in res}
                    scores = [scores_dict.get(i, 0.5) for i in range(len(candidates))]
                else:
                    scores = [float(c.get("hybrid_relevance", c.get("cosine_sim", 0.0))) for c in candidates]
            except Exception as e:
                logger.warning(f"Reranker scoring error: {e}. Falling back to hybrid scores.")
                scores = [float(c.get("hybrid_relevance", c.get("cosine_sim", 0.0))) for c in candidates]
        else:
            scores = [
                0.5 * float(c.get("cosine_sim", 0.0)) + 0.5 * float(c.get("bm25_norm", c.get("rrf_score", 0.0)))
                for c in candidates
            ]

        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = float(score)

        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:k]

    def build_state_vector(self, query_emb: np.ndarray, reranked_docs: List[Dict[str, Any]]) -> np.ndarray:
        e_q = np.asarray(query_emb, dtype=np.float64).flatten()
        if len(e_q) < 384:
            e_q = np.pad(e_q, (0, 384 - len(e_q)))
        elif len(e_q) > 384:
            e_q = e_q[:384]

        if reranked_docs:
            scores = [doc.get("rerank_score", 0.0) for doc in reranked_docs]
            score_avg = float(np.mean(scores))
            score_max = float(np.max(scores))
        else:
            score_avg = 0.0
            score_max = 0.0

        return np.concatenate([e_q, np.array([score_avg, score_max], dtype=np.float64)])

    def expand_to_parents(
        self,
        ranked_children: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Parent–child expansion: keep best child hit per parent_id, replace
        synthesis text with the full parent section while preserving child scores.
        """
        k = top_k or self.config.top_k
        if not ranked_children:
            return []
        if not getattr(self.config, "parent_child_enabled", True):
            return ranked_children[:k]

        expanded: List[Dict[str, Any]] = []
        seen_parents = set()

        for child in ranked_children:
            meta = child.get("metadata") or {}
            parent_id = meta.get("parent_id") or child.get("parent_id") or child.get("chunk_id")
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

            parent_text = (
                meta.get("parent_text")
                or child.get("parent_text")
                or child.get("text")
                or ""
            )
            section_title = meta.get("section_title") or child.get("section_title") or ""
            child_text = meta.get("child_text") or child.get("child_text") or child.get("text") or ""

            doc = dict(child)
            doc["text"] = parent_text  # LLM sees full section
            doc["child_text"] = child_text
            doc["parent_id"] = parent_id
            doc["section_title"] = section_title
            doc["expanded_parent"] = True
            # Keep metadata informative for UI
            new_meta = dict(meta)
            new_meta["parent_id"] = parent_id
            new_meta["section_title"] = section_title
            new_meta["matched_child_id"] = child.get("chunk_id", "")
            doc["metadata"] = new_meta
            expanded.append(doc)
            if len(expanded) >= k:
                break

        return expanded

    def retrieve_and_score(self, query: str) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, float]]:
        """
        Full hybrid pipeline:
          BM25 + Semantic → RRF → MMR → Cross-Encoder → Parent expansion → S_t
        """
        mmr_candidates, query_emb, hybrid_stats = self.hybrid_retrieve(query)
        # Rerank children (fetch a bit more so parent dedupe still fills top_k)
        child_pool = self.rerank(query, mmr_candidates, top_k=max(self.config.top_k * 3, 8))
        top_docs = self.expand_to_parents(child_pool, top_k=self.config.top_k)
        state_vec = self.build_state_vector(query_emb, top_docs)

        metrics = {
            "score_avg": float(state_vec[-2]),
            "score_max": float(state_vec[-1]),
            "num_candidates_mmr": len(mmr_candidates),
            "num_reranked": len(top_docs),
            "num_bm25": hybrid_stats.get("bm25", 0),
            "num_semantic": hybrid_stats.get("semantic", 0),
            "num_fused": hybrid_stats.get("fused", 0),
            "num_children_reranked": len(child_pool),
            "retrieval_mode": (
                "hybrid_bm25_semantic_mmr_parent_child"
                if self.config.hybrid_enabled
                else "semantic_mmr_parent_child"
            ),
        }
        return top_docs, state_vec, metrics
