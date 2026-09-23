import os
import sys
import json
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pipeline.config import PipelineConfig
from pipeline.ingestor import DoclingJSONIngestor
from pipeline.retriever import HybridRetriever
from pipeline.bandit_policy import LinUCBBanditPolicy
from pipeline.external_search import ExternalSearchTool
from pipeline.agent import SelfHealingRAGAgent

logger = logging.getLogger("CRAG-API")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="RL-Enhanced Self-Healing CRAG API",
    description="Production-grade CRAG with native Docling ingestion and LinUCB Contextual Bandit",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline singletons
config = PipelineConfig()
ingestor = DoclingJSONIngestor(config=config)
retriever = HybridRetriever(ingestor=ingestor, config=config)
bandit_policy = LinUCBBanditPolicy(dimension=config.state_dim, alpha=config.alpha)
search_tool = ExternalSearchTool(max_results=config.max_search_results)
agent = SelfHealingRAGAgent(
    retriever=retriever,
    bandit_policy=bandit_policy,
    search_tool=search_tool,
    config=config
)

# Pre-seed sample document if DB is empty
def ensure_seed_data():
    try:
        docs = ingestor.get_indexed_documents()
        if not docs:
            from main import create_sample_docling_document
            sample_doc = create_sample_docling_document()
            res = ingestor.ingest_docling_dict(
                sample_doc,
                doc_name="Project_Helios_Fusion_Spec.pdf",
                doc_id="seed_doc_001",
            )
            logger.info(
                f"Pre-seeded sample Docling document into ChromaDB "
                f"({res.get('num_chunks')} chunks, embedded={res.get('embedded')}, "
                f"jsonl={res.get('jsonl_path')})."
            )
    except Exception as e:
        logger.warning(f"Seed data error: {e}", exc_info=True)

ensure_seed_data()

# Static files directory
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    query: str
    forced_action: Optional[int] = None

class TextInputRequest(BaseModel):
    text: str
    title: Optional[str] = "Manual_Entry.txt"

@app.get("/api/health")
async def health_check():
    stats = ingestor.get_collection_stats()
    return {
        "status": "ok",
        "service": "RL-CRAG Pipeline",
        "version": "1.0.0",
        **stats,
    }

@app.get("/api/documents")
async def list_documents():
    """
    List all ingested documents and their chunk counts.
    """
    try:
        docs = ingestor.get_indexed_documents()
        stats = ingestor.get_collection_stats()
        return {"documents": docs, "total": len(docs), "chroma": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jsonl")
async def list_jsonl():
    """List Docling JSONL artifact files written during ingestion."""
    try:
        files = ingestor.list_jsonl_files()
        return {"files": files, "total": len(files), "output_dir": config.jsonl_output_dir}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jsonl/{filename}")
async def get_jsonl(filename: str):
    """Return full JSONL records for the UI viewer."""
    try:
        return ingestor.read_jsonl_file(filename)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chroma/stats")
async def chroma_stats():
    """ChromaDB collection + embedding readiness."""
    return ingestor.get_collection_stats()

@app.post("/api/ingest/text")
async def ingest_text(payload: TextInputRequest):
    """
    Ingest plain text or markdown by converting into Docling JSON schema and ChromaDB.
    """
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text content cannot be empty.")
    try:
        res = ingestor.ingest(payload.text, is_raw_text=True, doc_name=payload.title)
        return res
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest/upload")
async def ingest_file(file: UploadFile = File(...)):
    """
    Ingest uploaded PDF, JSON, DOCX, TXT, or image files via Docling DocumentConverter.
    """
    try:
        temp_dir = os.path.join(os.path.expanduser("~"), ".cache", "crag_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        res = ingestor.ingest(file_path, is_raw_text=False)
        return res
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def run_query(req: QueryRequest):
    """
    Execute complete RL-driven CRAG query pipeline.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        result = agent.process_query(req.query, forced_action=req.forced_action)
        # Cleanly serialize to JSON-safe dictionary
        safe_result = json.loads(json.dumps(result, default=str))
        return JSONResponse(content=safe_result)
    except Exception as e:
        logger.error(f"CRAG execution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/bandit/stats")
async def bandit_stats():
    """
    Get live LinUCB contextual bandit telemetry.
    """
    return bandit_policy.get_stats()

@app.post("/api/bandit/reset")
async def reset_bandit():
    """
    Reset LinUCB bandit parameters and history.
    """
    bandit_policy.reset()
    return {"status": "success", "message": "Bandit policy reset successfully."}

@app.post("/api/reset")
async def reset_all():
    """
    Reset ChromaDB and bandit policy.
    """
    ingestor.clear_database()
    bandit_policy.reset()
    ensure_seed_data()
    return {"status": "success", "message": "Pipeline and database reset."}

# Mount static folder
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def root():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "RL-Driven Self-Healing CRAG API is running. Build frontend in static/."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
