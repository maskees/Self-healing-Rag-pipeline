import os
from dataclasses import dataclass, field
from typing import Dict

# Load .env file automatically if it exists (supports GROQ_API_KEY etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)  # Does NOT override existing env vars
except ImportError:
    pass  # python-dotenv not installed; rely on system env vars


@dataclass
class PipelineConfig:
    """
    Configuration parameters for RL-Enhanced Self-Healing CRAG Architecture.
    """
    # Vector DB & Embeddings
    chroma_db_dir: str = os.path.join(os.path.expanduser("~"), ".cache", "crag_chroma_db")
    # Project-local JSONL output so Docling nodes are visible next to the app
    jsonl_output_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output",
        "jsonl"
    )
    collection_name: str = "crag_docling_collection"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    # Chroma metadata value size soft limit (bytes) — keep payloads under this
    max_metadata_bytes: int = 3500
    
    # Reranker
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 3
    mmr_candidates_multiplier: int = 2  # 2 * K = 6 candidates
    
    # LinUCB Contextual Bandit
    # State dimension S_t = [Query Embeddings (384) + Score_avg (1) + Score_max (1)] = 386
    state_dim: int = 386
    alpha: float = 0.5  # Exploration rate in LinUCB
    
    # Action Space
    ACTION_DIRECT_GEN: int = 0
    ACTION_QUERY_REWRITE: int = 1
    ACTION_EXTERNAL_SEARCH: int = 2
    
    # Action penalties: R_t = Score_judge - Penalty(A_t)
    # Action 0 = 0.0, Action 1 = -0.1, Action 2 = -0.25
    action_penalties: Dict[int, float] = field(default_factory=lambda: {
        0: 0.0,
        1: 0.10,
        2: 0.25,
    })
    
    # LLM Settings (Groq)
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_llm_model: str = "llama-3.3-70b-versatile"
    groq_vision_model: str = "llama-3.2-11b-vision-instruct"
    
    # Search Engine
    max_search_results: int = 4
