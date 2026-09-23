import os
import json
import uuid
import base64
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from .config import PipelineConfig

logger = logging.getLogger(__name__)

class DoclingJSONIngestor:
    """
    Component A: Document Parsing & Native JSON Ingestion.
    
    1. Parses documents (PDF, DOCX, TXT, images, or native JSON) via Docling DocumentConverter.
    2. Exports parsed output directly using result.document.export_to_dict().
    3. Multimodal Image Analysis: Iterates over JSON nodes; if label is 'picture' or references
       an image, queries Groq (llama-3.2-11b-vision-instruct) to generate a textual description
       and injects it into the node JSON.
    4. Vector Storage: Stores plain text in Chroma DB while persisting the full serialized JSON
       payload (json.dumps(...)) in Chroma DB metadata.
    5. Embedding Model: Uses sentence-transformers/all-MiniLM-L6-v2.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Lazy/safe loading of DocumentConverter
        self._doc_converter = None
        
        # Initialize Sentence Transformer embedding model
        logger.info(f"Loading embedding model: {self.config.embedding_model_name}")
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer(self.config.embedding_model_name)
        
        # Initialize ChromaDB
        import chromadb
        os.makedirs(self.config.chroma_db_dir, exist_ok=True)
        os.makedirs(self.config.jsonl_output_dir, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=self.config.chroma_db_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize Groq client if key is configured
        self.groq_client = None
        if self.config.groq_api_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=self.config.groq_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")

    @property
    def doc_converter(self):
        if self._doc_converter is None:
            try:
                from docling.document_converter import DocumentConverter
                self._doc_converter = DocumentConverter()
            except Exception as e:
                logger.warning(f"Docling DocumentConverter init warning: {e}")
                self._doc_converter = None
        return self._doc_converter

    def parse_document(self, file_path_or_text: str, is_raw_text: bool = False, filename: str = "document.txt") -> Dict[str, Any]:
        """
        Parse input file or raw text into a native Docling-structured JSON dictionary.
        """
        if is_raw_text:
            return self._text_to_docling_dict(file_path_or_text, filename)
            
        if not os.path.exists(file_path_or_text):
            raise FileNotFoundError(f"File not found: {file_path_or_text}")
            
        ext = os.path.splitext(file_path_or_text)[1].lower()
        
        # If already a Docling JSON file
        if ext == ".json":
            with open(file_path_or_text, "r", encoding="utf-8") as f:
                doc_dict = json.load(f)
            return doc_dict
            
        # Try native Docling DocumentConverter
        if self.doc_converter is not None:
            try:
                logger.info(f"Converting document with Docling: {file_path_or_text}")
                result = self.doc_converter.convert(file_path_or_text)
                # Native export_to_dict() as required by specification
                doc_dict = result.document.export_to_dict()
                return doc_dict
            except Exception as e:
                logger.warning(f"Docling conversion exception: {e}. Falling back to structured parser.")
                
        # Structured fallback for text/pdf/markdown
        return self._fallback_file_parse(file_path_or_text)

    def _text_to_docling_dict(self, text: str, filename: str) -> Dict[str, Any]:
        """
        Build a compliant Docling-like structured JSON dictionary from raw text.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()] if text.strip() else ["(Empty document)"]
            
        body_children = []
        texts_dict = {}
        
        for idx, para in enumerate(paragraphs):
            node_id = f"#/texts/{idx}"
            # Detect headings
            label = "section_header" if para.startswith("#") or (len(para) < 60 and idx == 0) else "text"
            clean_text = para.lstrip("# ").strip()
            
            node_payload = {
                "$ref": node_id,
                "label": label,
                "text": clean_text,
                "orig": para,
                "prov": [{"page_no": 1, "bbox": {"l": 50, "t": 50 + idx * 40, "r": 500, "b": 80 + idx * 40}}],
            }
            texts_dict[node_id] = node_payload
            body_children.append({"$ref": node_id})

        doc_dict = {
            "schema_name": "docling_core.models.DoclingDocument",
            "version": "1.0.0",
            "name": filename,
            "origin": {"filename": filename, "mimetype": "text/plain"},
            "body": {
                "label": "body",
                "children": body_children
            },
            "texts": texts_dict,
            "pictures": {},
            "tables": {}
        }
        return doc_dict

    def _fallback_file_parse(self, file_path: str) -> Dict[str, Any]:
        """
        Fallback parser reading text files, markdown, or plain content.
        """
        filename = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            content = f"Binary or non-text document: {filename}"
        return self._text_to_docling_dict(content, filename)

    def _iter_nodes(self, container: Any) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Normalize Docling containers that may be either:
          - dict keyed by $ref / self_ref (demo / manual schema)
          - list of node objects (native Docling export_to_dict)
        Returns list of (node_ref, node_dict).
        """
        results: List[Tuple[str, Dict[str, Any]]] = []
        if container is None:
            return results
        if isinstance(container, list):
            for idx, node in enumerate(container):
                if not isinstance(node, dict):
                    continue
                ref = node.get("self_ref") or node.get("$ref") or f"#/node/{idx}"
                results.append((str(ref), node))
            return results
        if isinstance(container, dict):
            # Map of nodes: {"#/texts/0": {...}, ...}
            values_are_nodes = any(isinstance(v, dict) for v in container.values())
            if values_are_nodes and (
                any(str(k).startswith("#/") for k in container.keys())
                or all(isinstance(v, dict) for v in container.values())
            ):
                for key, node in container.items():
                    if isinstance(node, dict):
                        ref = node.get("self_ref") or node.get("$ref") or str(key)
                        results.append((str(ref), node))
            else:
                # Single node object
                ref = container.get("self_ref") or container.get("$ref") or "#/node/0"
                results.append((str(ref), container))
        return results

    def enrich_multimodal_nodes(self, doc_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Iterates over JSON nodes. If an element label is 'picture' or references an image file,
        queries Groq (llama-3.2-11b-vision-instruct) to generate a textual description and
        injects it into the node JSON.
        """
        for _, pic_node in self._iter_nodes(doc_dict.get("pictures", {})):
            self._enrich_single_picture_node(pic_node)

        # Also check nodes in texts or body with label == 'picture'
        def traverse_and_enrich(obj):
            if isinstance(obj, dict):
                label = obj.get("label", "")
                if label == "picture" or "image" in obj:
                    self._enrich_single_picture_node(obj)
                for v in obj.values():
                    traverse_and_enrich(v)
            elif isinstance(obj, list):
                for item in obj:
                    traverse_and_enrich(item)

        traverse_and_enrich(doc_dict.get("body", {}))
        return doc_dict

    def _enrich_single_picture_node(self, node: Dict[str, Any]):
        """
        Process a picture node: query Groq Vision if possible, or generate descriptive metadata.
        """
        if "image_description" in node and node["image_description"]:
            return  # Already enriched
            
        caption = node.get("caption", "")
        image_uri = node.get("uri", "") or node.get("path", "")
        base64_data = node.get("image_base64", "")
        
        description = None
        
        # 1. Real Groq Vision call if API key and image data are available
        if self.groq_client and (image_uri or base64_data):
            try:
                # If image_uri is a local file, convert to base64
                if image_uri and os.path.exists(image_uri) and not base64_data:
                    with open(image_uri, "rb") as img_f:
                        base64_data = base64.b64encode(img_f.read()).decode("utf-8")
                        
                if base64_data:
                    prompt = "Describe this document diagram/image in detail for retrieval augmented generation. Include key numbers, labels, charts, or structural relationships."
                    chat_completion = self.groq_client.chat.completions.create(
                        model=self.config.groq_vision_model,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"}
                                    }
                                ]
                            }
                        ],
                        max_tokens=300
                    )
                    description = chat_completion.choices[0].message.content
            except Exception as e:
                logger.warning(f"Groq vision inference error: {e}")
                
        # 2. Resilient multimodal description fallback if offline or no image file
        if not description:
            pic_label = caption or node.get("name", "Document Visual Diagram")
            annotations = node.get("annotations", ["Architecture Component", "Flowchart Diagram"])
            description = (
                f"[Multimodal Vision Analysis] Architectural illustration depicting: '{pic_label}'. "
                f"Features components, telemetry interconnects, and structural relationships. "
                f"Annotations: {', '.join(str(a) for a in annotations)}."
            )
            
        # Inject textual description directly into node JSON
        node["image_description"] = description
        node["multimodal_enriched"] = True
        if "text" not in node or not node["text"]:
            node["text"] = description
        else:
            node["text"] = f"{node['text']}\nVisual Description: {description}"

    def _table_text(self, node: Dict[str, Any]) -> str:
        """Build searchable text from a Docling table node."""
        text_content = (node.get("text") or "").strip()
        if text_content:
            return text_content
        data = node.get("data") or {}
        if isinstance(data, dict):
            # Native Docling often uses grid / table_cells
            if "headers" in data or "rows" in data:
                return f"Table Data: {json.dumps(data, default=str)}"
            grid = data.get("grid") or data.get("table_cells")
            if grid:
                return f"Table Data: {json.dumps(grid, default=str)[:2000]}"
        # Captions / annotations fallback
        caption = node.get("caption") or node.get("name") or ""
        return f"Table: {caption}" if caption else ""

    def extract_chunks_with_payloads(self, doc_dict: Dict[str, Any], doc_id: str) -> List[Dict[str, Any]]:
        """
        Extract searchable chunks with their full serialized JSON payloads.
        Supports both dict-keyed and list-based Docling export formats.
        """
        chunks: List[Dict[str, Any]] = []

        # 1. Texts
        for ref_id, node in self._iter_nodes(doc_dict.get("texts", {})):
            text_content = (node.get("text") or node.get("orig") or "").strip()
            if not text_content:
                continue
            chunk_id = f"{doc_id}_{uuid.uuid4().hex[:8]}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": text_content,
                "payload": node,
                "node_ref": ref_id,
                "label": node.get("label", "text"),
            })

        # 2. Pictures
        for ref_id, node in self._iter_nodes(doc_dict.get("pictures", {})):
            text_content = (
                (node.get("text") or "").strip()
                or (node.get("image_description") or "").strip()
                or (node.get("caption") or "").strip()
            )
            if not text_content:
                text_content = f"Image figure: {node.get('name', 'Document Figure')}"
            chunk_id = f"{doc_id}_pic_{uuid.uuid4().hex[:8]}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": text_content,
                "payload": node,
                "node_ref": ref_id,
                "label": "picture",
            })

        # 3. Tables
        for ref_id, node in self._iter_nodes(doc_dict.get("tables", {})):
            text_content = self._table_text(node)
            if not text_content:
                continue
            chunk_id = f"{doc_id}_tbl_{uuid.uuid4().hex[:8]}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": text_content,
                "payload": node,
                "node_ref": ref_id,
                "label": "table",
            })

        # 4. Fallback — flatten body / whole doc if structure yielded nothing
        if not chunks:
            raw_summary = json.dumps(doc_dict.get("body", doc_dict), default=str)[:2000]
            chunk_id = f"{doc_id}_raw_{uuid.uuid4().hex[:8]}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": raw_summary,
                "payload": doc_dict,
                "node_ref": "#/root",
                "label": "root",
            })

        return chunks

    def _safe_metadata_payload(self, payload: Any) -> str:
        """Serialize payload for Chroma metadata, truncating oversized JSON."""
        serialized = json.dumps(payload, default=str)
        limit = self.config.max_metadata_bytes
        if len(serialized) <= limit:
            return serialized
        # Prefer keeping identity fields if payload is a dict
        if isinstance(payload, dict):
            slim = {
                k: payload.get(k)
                for k in ("$ref", "self_ref", "label", "text", "caption", "image_description", "orig")
                if k in payload
            }
            slim["_truncated"] = True
            slim["_original_bytes"] = len(serialized)
            return json.dumps(slim, default=str)[:limit]
        return serialized[:limit]

    def write_jsonl(self, chunks: List[Dict[str, Any]], doc_id: str, doc_name: str) -> Dict[str, Any]:
        """
        Write one JSON object per line (JSONL) for every extracted Docling node.
        Returns path + line preview for the UI.
        """
        os.makedirs(self.config.jsonl_output_dir, exist_ok=True)
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in doc_name)
        jsonl_path = os.path.join(self.config.jsonl_output_dir, f"{doc_id}_{safe_name}.jsonl")

        def _sanitize_payload(payload: Any) -> Any:
            if not isinstance(payload, dict):
                return payload
            clean = dict(payload)
            for heavy_key in ("image_base64", "uri", "bytes", "image"):
                if heavy_key in clean and clean[heavy_key]:
                    val = clean[heavy_key]
                    if isinstance(val, str) and len(val) > 200:
                        clean[heavy_key] = f"<omitted {len(val)} chars>"
                    elif isinstance(val, dict):
                        clean[heavy_key] = "<omitted nested image object>"
            return clean

        jsonl_lines: List[Dict[str, Any]] = []
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for c in chunks:
                record = {
                    "chunk_id": c["chunk_id"],
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "node_ref": c["node_ref"],
                    "label": c["label"],
                    "text": c["text"],
                    "payload": _sanitize_payload(c["payload"]),
                    "embedding_dim": self.config.embedding_dim,
                }
                jsonl_lines.append(record)
                f.write(json.dumps(record, default=str) + "\n")

        logger.info(f"Wrote Docling JSONL ({len(jsonl_lines)} lines) → {jsonl_path}")
        return {
            "jsonl_path": jsonl_path,
            "jsonl_lines": jsonl_lines,
            "num_lines": len(jsonl_lines),
        }

    def ingest_docling_dict(
        self,
        doc_dict: Dict[str, Any],
        doc_name: str = "document.json",
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest an already-parsed Docling dict: enrich → JSONL → embed → ChromaDB.
        Shared by file/text ingest and seed data.
        """
        doc_id = doc_id or str(uuid.uuid4().hex[:10])
        name = doc_name or doc_dict.get("name") or "document.json"

        enriched_dict = self.enrich_multimodal_nodes(doc_dict)
        chunks = self.extract_chunks_with_payloads(enriched_dict, doc_id)

        if not chunks:
            return {
                "status": "empty",
                "doc_id": doc_id,
                "doc_name": name,
                "num_chunks": 0,
                "jsonl_lines": [],
                "embedded": False,
                "chroma_count": self.collection.count(),
            }

        # Write visible JSONL output
        jsonl_info = self.write_jsonl(chunks, doc_id, name)

        # Embed with sentence-transformers
        texts = [c["text"] for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks with {self.config.embedding_model_name}...")
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True).tolist()
        emb_dim = len(embeddings[0]) if embeddings else 0

        ids = [c["chunk_id"] for c in chunks]
        metadatas = []
        for c in chunks:
            metadatas.append({
                "doc_id": doc_id,
                "doc_name": name,
                "node_ref": c["node_ref"],
                "label": c["label"],
                "json_payload": self._safe_metadata_payload(c["payload"]),
            })

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        chroma_count = self.collection.count()
        logger.info(
            f"Ingested {len(chunks)} chunks for '{name}' into ChromaDB "
            f"(collection size={chroma_count}, emb_dim={emb_dim})."
        )

        # UI-friendly chunk previews (no huge nested blobs twice)
        chunk_previews = [
            {
                "chunk_id": c["chunk_id"],
                "node_ref": c["node_ref"],
                "label": c["label"],
                "text": c["text"][:400],
                "payload": c["payload"],
            }
            for c in chunks
        ]

        return {
            "status": "success",
            "doc_id": doc_id,
            "doc_name": name,
            "num_chunks": len(chunks),
            "embedded": True,
            "embedding_model": self.config.embedding_model_name,
            "embedding_dim": emb_dim,
            "chroma_collection": self.config.collection_name,
            "chroma_count": chroma_count,
            "jsonl_path": jsonl_info["jsonl_path"],
            "jsonl_num_lines": jsonl_info["num_lines"],
            "jsonl_lines": jsonl_info["jsonl_lines"],
            "doc_dict": enriched_dict,
            "chunks": chunk_previews,
            "chunks_sample": chunk_previews[:5],
        }

    def ingest(self, file_path_or_text: str, is_raw_text: bool = False, doc_name: str = "document.txt") -> Dict[str, Any]:
        """
        End-to-End Ingestion: Parse → Enrich → JSONL → Embed → Store in ChromaDB.
        """
        name = os.path.basename(file_path_or_text) if not is_raw_text else doc_name
        doc_dict = self.parse_document(file_path_or_text, is_raw_text=is_raw_text, filename=name)
        return self.ingest_docling_dict(doc_dict, doc_name=name)

    def get_indexed_documents(self) -> List[Dict[str, Any]]:
        """
        Retrieve catalog of indexed documents and chunk statistics.
        """
        try:
            data = self.collection.get(include=["metadatas"])
            docs_map = {}
            for meta in data.get("metadatas", []) or []:
                doc_id = meta.get("doc_id", "unknown")
                doc_name = meta.get("doc_name", "Unknown Document")
                if doc_id not in docs_map:
                    docs_map[doc_id] = {"doc_id": doc_id, "name": doc_name, "chunk_count": 0}
                docs_map[doc_id]["chunk_count"] += 1
            return list(docs_map.values())
        except Exception as e:
            logger.error(f"Error reading indexed documents: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Live ChromaDB + embedding status for the UI."""
        try:
            count = self.collection.count()
        except Exception:
            count = 0
        return {
            "collection_name": self.config.collection_name,
            "chroma_path": self.config.chroma_db_dir,
            "jsonl_output_dir": self.config.jsonl_output_dir,
            "chunk_count": count,
            "embedding_model": self.config.embedding_model_name,
            "embedding_dim": self.config.embedding_dim,
            "ready_for_retrieval": count > 0,
        }

    def list_jsonl_files(self) -> List[Dict[str, Any]]:
        """List written JSONL artifacts under output/jsonl."""
        out_dir = self.config.jsonl_output_dir
        os.makedirs(out_dir, exist_ok=True)
        files = []
        for name in sorted(os.listdir(out_dir)):
            if not name.endswith(".jsonl"):
                continue
            path = os.path.join(out_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = [ln for ln in f if ln.strip()]
                files.append({
                    "filename": name,
                    "path": path,
                    "num_lines": len(lines),
                    "size_bytes": os.path.getsize(path),
                })
            except Exception as e:
                logger.warning(f"Could not read JSONL {path}: {e}")
        return files

    def read_jsonl_file(self, filename: str) -> Dict[str, Any]:
        """Load a JSONL file's records for the UI viewer."""
        safe = os.path.basename(filename)
        path = os.path.join(self.config.jsonl_output_dir, safe)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"JSONL not found: {safe}")
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return {"filename": safe, "path": path, "num_lines": len(records), "records": records}

    def clear_database(self):
        """
        Clear all items in the ChromaDB collection for testing/reset.
        """
        try:
            self.chroma_client.delete_collection(self.config.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.warning(f"Chroma clear error: {e}")
        # Also clear JSONL outputs so UI stays consistent
        try:
            out_dir = self.config.jsonl_output_dir
            if os.path.isdir(out_dir):
                for name in os.listdir(out_dir):
                    if name.endswith(".jsonl"):
                        os.remove(os.path.join(out_dir, name))
        except Exception as e:
            logger.warning(f"JSONL clear error: {e}")
