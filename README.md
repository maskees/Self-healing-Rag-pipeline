# RL-Driven Self-Healing CRAG Architecture
## Native Docling JSON Ingestion · Groq Vision · LinUCB Contextual Bandit · Interactive Web Interface

---

## Architecture Overview

```
Document Ingestion (Docling PDF/Img → JSON)
          ↓
Multimodal Analysis & Embed (Groq Vision + SentenceTransformers)
          ↓
ChromaDB Vector Storage (JSON Payload Metadata)
          ↓
Hybrid Retrieval: MMR (2K candidates) → Cross-Encoder Reranking (top K)
          ↓
State Vector S_t [E_query (384) | Score_avg | Score_max] = 386 dims
          ↓
LinUCB Contextual Bandit → Action {0, 1, 2}
    ├── Action 0: Direct Generation
    ├── Action 1: Query Rewrite & Re-Retrieve
    └── Action 2: External DuckDuckGo Search Fallback
          ↓
Groq llama-3.3-70b-versatile Response Synthesis
          ↓
LLM-as-a-Judge Score (0.0–1.0) → R_t = Judge − Penalty(A_t)
          ↓
Closed-Loop LinUCB Policy Update: A_a ← A_a + xxᵀ, b_a ← b_a + R_t·x
```

---

## Component Modules

| File | Class | Responsibility |
|------|-------|---------------|
| `pipeline/config.py` | `PipelineConfig` | All hyperparameters (α=0.5, d=386, K=3, penalties) |
| `pipeline/ingestor.py` | `DoclingJSONIngestor` | Docling parsing, Groq Vision enrichment, ChromaDB |
| `pipeline/retriever.py` | `HybridRetriever` | MMR retrieval, Cross-Encoder reranking, S_t vector |
| `pipeline/bandit_policy.py` | `LinUCBBanditPolicy` | LinUCB disjoint bandit arms, select_action, update |
| `pipeline/external_search.py` | `ExternalSearchTool` | DuckDuckGo search fallback |
| `pipeline/agent.py` | `SelfHealingRAGAgent` | Full CRAG orchestration loop |
| `app.py` | FastAPI App | REST API server + static frontend serving |
| `main.py` | `__main__` | CLI demonstrator of all 3 action paths |

---

## Mathematical Formulation

### LinUCB Action Selection
```
p_{t,a} = θ̂_a^T x_{t,a} + α √(x_{t,a}^T A_a^{-1} x_{t,a})

where:  θ̂_a = A_a^{-1} b_a
        A_a  ← A_a + x x^T     (covariance update)
        b_a  ← b_a + R_t · x   (reward update)
```

### Net Reward
```
R_t = Score_judge − Penalty(A_t)
    Action 0 (Direct Gen):     Penalty = 0.00
    Action 1 (Query Rewrite):  Penalty = 0.10
    Action 2 (Ext. Search):    Penalty = 0.25
```

### State Feature Vector
```
S_t = [E_query (384 dims) | Score_avg | Score_max]  →  d = 386
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Set Groq API Key
```bash
# Windows PowerShell
$env:GROQ_API_KEY = "gsk_your_key_here"

# Or create a .env file
echo "GROQ_API_KEY=gsk_your_key_here" > .env
```
> **Note:** The system runs fully without an API key using built-in fallback completions.

### 3. Run CLI Demonstrator (All 3 Action Paths)
```bash
python main.py
```

### 4. Launch Interactive Web Application
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Open your browser at **http://127.0.0.1:8000**

### Ingestion → Embed → Retrieve Flow
1. Upload / paste text / load sample → Docling JSON nodes extracted  
2. **Parent–child chunking** — children = fine nodes (section-prefixed); parents = full sections (e.g. Work Experience)  
3. Each child written as one line in `output/jsonl/*.jsonl`; parents saved under `output/parents/`  
4. Children embedded with MiniLM and stored in **ChromaDB**  
5. Query runs **BM25 + dense semantic → RRF → MMR → Cross-Encoder → parent expansion** → LinUCB → answer  

---

## REST API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check + Chroma/embedding status |
| `GET` | `/api/documents` | List indexed documents |
| `GET` | `/api/jsonl` | List written JSONL artifact files |
| `GET` | `/api/jsonl/{filename}` | Load JSONL records for the UI viewer |
| `GET` | `/api/chroma/stats` | Chroma chunk count + embedding readiness |
| `POST` | `/api/ingest/text` | Ingest raw text / markdown |
| `POST` | `/api/ingest/upload` | Upload PDF, DOCX, JSON, TXT |
| `POST` | `/api/query` | Execute full CRAG pipeline |
| `GET` | `/api/bandit/stats` | Live LinUCB arm telemetry |
| `POST` | `/api/bandit/reset` | Reset bandit parameters |
| `POST` | `/api/reset` | Reset DB + bandit |

### Query Endpoint Example
```json
POST /api/query
{
  "query": "What coolant is used in the Helios Mark-IV reactor?",
  "forced_action": null
}
```
Response includes: `action`, `state_vector_summary`, `bandit_telemetry` (UCB scores for all 3 arms), `retrieved_docs`, `response`, `judge_score`, `net_reward`, `policy_stats`.

---

## Web Interface Features

- **Document Ingestion Studio** — Drag-and-drop upload, raw text input, and pre-loaded sample Docling specification (Helios Fusion Reactor)
- **Docling JSONL Output Panel** — Live view of one-JSON-object-per-node files under `output/jsonl/`, then stored in ChromaDB
- **CRAG Execution Console** — Free-form query, policy mode selector (Auto LinUCB / Force Action 0/1/2), and quick preset scenarios
- **Pipeline State Machine Visualizer** — Live step-by-step glow animation: MMR → Cross-Encoder → S_t → LinUCB → Branch Execution → Reward
- **Synthesized Answer + Reward Strip** — LLM Judge Score, Action Penalty, Net Reward R_t shown after every query
- **LinUCB Math Breakdown** — Exploitation vs. exploration bonus for all 3 arms with chosen arm highlighted
- **Execution Trace JSON Viewer** — Full serialized telemetry for every query

---

## Models Used

| Model | Role |
|-------|------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384-dim document & query embeddings |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranking candidate documents |
| `groq/llama-3.3-70b-versatile` | Query rewrite, response synthesis, LLM judge |
| `groq/llama-3.2-11b-vision-instruct` | Multimodal picture node enrichment |
| **LinUCB Contextual Bandit** (custom) | RL self-healing policy, d=386, α=0.5 |

