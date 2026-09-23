import json
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from .config import PipelineConfig

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Component B: Retrieval, Reranking & Feature Vector Extraction.
    
    1. MMR Vector Search: Queries Chroma DB using Maximal Marginal Relevance (MMR)
       to retrieve 2 * K diverse candidates.
    2. Reranking: Scores [query, document] pairs using Cross-Encoder
       'cross-encoder/ms-marco-MiniLM-L-6-v2' (or Flashrank). Keeps top K documents.
    3. State Feature Vector (S_t):
       S_t = [ E_query (384 dims), Score_avg, Score_max ]
       Total dimension = 386.
    """

    def __init__(self, ingestor, config: Optional[PipelineConfig] = None):
        """
        Initialize retriever with reference to ingestor (for Chroma collection and embedding model).
        """
        self.ingestor = ingestor
        self.config = config or PipelineConfig()
        
        # Load Cross-Encoder reranker
        self.cross_encoder = None
        self._init_reranker()

    def _init_reranker(self):
        """
        Initialize Cross-Encoder or FlashRank reranker.
        """
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
                logger.warning(f"Flashrank fallback error: {e2}. Using cosine reranker.")
                self.cross_encoder = None

    def _compute_mmr(
        self,
        query_embedding: np.ndarray,
        doc_embeddings: np.ndarray,
        doc_ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        top_n: int,
        lambda_param: float = 0.6
    ) -> List[Dict[str, Any]]:
        r"""
        Maximal Marginal Relevance (MMR) selection algorithm:
        MMR = argmax_{d in R \ S} [ lambda * Sim(d, q) - (1 - lambda) * max_{s in S} Sim(d, s) ]
        """
        if len(doc_ids) == 0:
            return []
            
        n_docs = len(doc_ids)
        top_n = min(top_n, n_docs)
        
        # Normalize embeddings for cosine similarity
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        doc_norms = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-9)
        
        # Similarity to query
        query_sims = np.dot(doc_norms, q_norm)
        
        # Document-to-document similarity matrix
        doc_sims = np.dot(doc_norms, doc_norms.T)
        
        selected_indices = []
        candidate_indices = list(range(n_docs))
        
        # Pick top document by query similarity first
        first_pick = int(np.argmax(query_sims))
        selected_indices.append(first_pick)
        candidate_indices.remove(first_pick)
        
        while len(selected_indices) < top_n and candidate_indices:
            best_idx = None
            best_score = -float("inf")
            
            for c_idx in candidate_indices:
                sim_to_query = query_sims[c_idx]
                max_sim_to_selected = max([doc_sims[c_idx, s_idx] for s_idx in selected_indices])
                
                # MMR formula
                mmr_score = lambda_param * sim_to_query - (1.0 - lambda_param) * max_sim_to_selected
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = c_idx
                    
            if best_idx is not None:
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)
            else:
                break
                
        # Build candidate objects
        selected_candidates = []
        for idx in selected_indices:
            raw_meta = metadatas[idx] if metadatas and idx < len(metadatas) else {}
            # Deserialize JSON payload
            json_payload = {}
            if "json_payload" in raw_meta:
                try:
                    json_payload = json.loads(raw_meta["json_payload"])
                except Exception:
                    json_payload = {}
                    
            selected_candidates.append({
                "chunk_id": doc_ids[idx],
                "text": documents[idx],
                "metadata": raw_meta,
                "payload": json_payload,
                "cosine_sim": float(query_sims[idx]),
                "embedding": doc_embeddings[idx]
            })
            
        return selected_candidates

    def mmr_retrieve(self, query: str, top_k: Optional[int] = None) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Execute MMR vector search to retrieve 2 * K candidate chunks.
        
        Returns:
            Tuple of (mmr_candidates: List[Dict], query_embedding: np.ndarray)
        """
        k = top_k or self.config.top_k
        n_candidates = k * self.config.mmr_candidates_multiplier  # 2 * K
        
        # 1. Encode query
        query_emb = self.ingestor.embedding_model.encode(query, convert_to_numpy=True).astype(np.float64)
        
        # 2. Query Chroma collection for an initial candidate pool (up to 4 * K items)
        initial_fetch = max(n_candidates * 2, 10)
        
        # Clamp to actual collection size to avoid ChromaDB errors with small collections
        try:
            collection_count = self.ingestor.collection.count()
            if collection_count == 0:
                logger.warning("ChromaDB collection is empty. Please ingest documents first.")
                return [], query_emb
            initial_fetch = min(initial_fetch, collection_count)
        except Exception:
            pass
            
        try:
            res = self.ingestor.collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=initial_fetch,
                include=["documents", "metadatas", "embeddings"]
            )
        except Exception as e:
            logger.error(f"ChromaDB query error: {e}")
            return [], query_emb
            
        doc_ids = res.get("ids", [[]])[0]
        documents = res.get("documents", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        embeddings = res.get("embeddings", [[]])[0]
        
        if not doc_ids or not documents:
            return [], query_emb
            
        doc_embeddings = np.array(embeddings, dtype=np.float64)
        
        # 3. Apply MMR
        candidates = self._compute_mmr(
            query_embedding=query_emb,
            doc_embeddings=doc_embeddings,
            doc_ids=doc_ids,
            documents=documents,
            metadatas=metadatas,
            top_n=n_candidates,
            lambda_param=0.65
        )
        
        return candidates, query_emb

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank MMR candidates using Cross-Encoder ms-marco-MiniLM-L-6-v2.
        Keep top K documents.
        """
        k = top_k or self.config.top_k
        if not candidates:
            return []
            
        pairs = [[query, cand["text"]] for cand in candidates]
        
        # Cross-Encoder scoring
        if self.cross_encoder is not None:
            try:
                # If sentence_transformers CrossEncoder
                if hasattr(self.cross_encoder, "predict"):
                    raw_scores = self.cross_encoder.predict(pairs)
                    # Convert logits to probability via sigmoid for stable 0.0 - 1.0 confidence
                    scores = 1.0 / (1.0 + np.exp(-np.array(raw_scores, dtype=np.float64)))
                # If FlashRank
                elif hasattr(self.cross_encoder, "rerank"):
                    from flashrank import RerankRequest
                    passages = [{"id": i, "text": cand["text"]} for i, cand in enumerate(candidates)]
                    req = RerankRequest(query=query, passages=passages)
                    res = self.cross_encoder.rerank(req)
                    scores_dict = {item["id"]: float(item["score"]) for item in res}
                    scores = [scores_dict.get(i, 0.5) for i in range(len(candidates))]
                else:
                    scores = [cand["cosine_sim"] for cand in candidates]
            except Exception as e:
                logger.warning(f"Reranker scoring error: {e}. Falling back to cosine similarity.")
                scores = [cand["cosine_sim"] for cand in candidates]
        else:
            # Fallback to normalized cosine similarity
            scores = [(cand["cosine_sim"] + 1.0) / 2.0 for cand in candidates]
            
        # Attach rerank score to candidates
        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = float(score)
            
        # Sort by rerank_score descending
        ranked_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked_candidates[:k]

    def build_state_vector(self, query_emb: np.ndarray, reranked_docs: List[Dict[str, Any]]) -> np.ndarray:
        """
        Construct State Feature Vector S_t:
        S_t = [ E_query (384 dims), Score_avg, Score_max ]
        Total dimension = 386.
        """
        # Ensure query embedding is 384 dims
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
            
        # Total dimension: 384 + 1 + 1 = 386
        state_vector = np.concatenate([e_q, np.array([score_avg, score_max], dtype=np.float64)])
        return state_vector

    def retrieve_and_score(self, query: str) -> Tuple[List[Dict[str, Any]], np.ndarray, Dict[str, float]]:
        """
        Full retrieval pipeline:
        1. MMR vector search (2*K)
        2. Cross-Encoder reranking (top K)
        3. Assemble S_t (386 dims)
        
        Returns:
            Tuple of (top_k_docs: List[Dict], state_vector: np.ndarray, confidence_metrics: Dict)
        """
        # 1. MMR Candidates
        mmr_candidates, query_emb = self.mmr_retrieve(query)
        
        # 2. Rerank
        top_docs = self.rerank(query, mmr_candidates)
        
        # 3. State Vector S_t
        state_vec = self.build_state_vector(query_emb, top_docs)
        
        metrics = {
            "score_avg": float(state_vec[-2]),
            "score_max": float(state_vec[-1]),
            "num_candidates_mmr": len(mmr_candidates),
            "num_reranked": len(top_docs)
        }
        
        return top_docs, state_vec, metrics
