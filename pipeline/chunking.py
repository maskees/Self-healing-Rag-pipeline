"""
Parent–child section chunking for Docling documents.

Children are fine-grained nodes used for BM25 + dense retrieval.
Parents are section-level contexts returned to the LLM after expansion.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple


def normalize_ws(text: str) -> str:
    if not text:
        return ""
    text = str(text).replace("\t", " ").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def is_section_header(node: Dict[str, Any], label: str, text: str) -> bool:
    if label in ("section_header", "title", "page_header"):
        return True
    if len(text) <= 80 and text.upper() == text and any(
        k in text.upper()
        for k in ("EXPERIENCE", "EDUCATION", "SKILL", "PROJECT", "PROFILE", "CERTIFIC", "INTERNSHIP")
    ):
        return True
    return False


def build_parent_child_chunks(
    text_nodes: List[Tuple[str, Dict[str, Any]]],
    picture_nodes: List[Tuple[str, Dict[str, Any]]],
    table_nodes: List[Tuple[str, Dict[str, Any]]],
    doc_id: str,
    max_chars: int = 900,
    table_text_fn=None,
    doc_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Returns (child_chunks, parents_index).
    Child `text` is section-prefixed for retrieval; `parent_text` holds full section.
    """
    sections: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {
        "title": "Document",
        "header_node": None,
        "header_ref": "#/body",
        "items": [],
    }

    for ref_id, node in text_nodes:
        text = normalize_ws(node.get("text") or node.get("orig") or "")
        if not text:
            continue
        label = node.get("label", "text")
        if is_section_header(node, label, text):
            if current["items"] or current["title"] != "Document":
                sections.append(current)
            current = {
                "title": text,
                "header_node": node,
                "header_ref": ref_id,
                "items": [],
            }
        else:
            current["items"].append({
                "ref": ref_id,
                "node": node,
                "text": text,
                "label": label,
            })

    if current["items"] or current["title"] != "Document":
        sections.append(current)

    chunks: List[Dict[str, Any]] = []
    parents_index: Dict[str, Dict[str, Any]] = {}

    for sec_i, section in enumerate(sections):
        title = section["title"]
        items = section["items"]

        windows: List[List[Dict[str, Any]]] = []
        window: List[Dict[str, Any]] = []
        running = len(title)
        for item in items:
            add_len = len(item["text"]) + 2
            if window and running + add_len > max_chars:
                windows.append(window)
                window = []
                running = len(title)
            window.append(item)
            running += add_len
        if window or not items:
            windows.append(window)

        for win_i, win_items in enumerate(windows):
            parent_id = f"{doc_id}_parent_{sec_i:03d}_{win_i:02d}"
            body_lines = [it["text"] for it in win_items]
            parent_text = title if not body_lines else f"{title}\n\n" + "\n".join(body_lines)
            parents_index[parent_id] = {
                "parent_id": parent_id,
                "section_title": title,
                "text": parent_text,
                "child_ids": [],
                "node_ref": section.get("header_ref", f"#/section/{sec_i}"),
            }

            if not win_items:
                child_id = f"{doc_id}_child_{sec_i:03d}_{win_i:02d}_hdr_{uuid.uuid4().hex[:6]}"
                parents_index[parent_id]["child_ids"].append(child_id)
                chunks.append({
                    "chunk_id": child_id,
                    "text": title,
                    "child_text": title,
                    "parent_id": parent_id,
                    "parent_text": parent_text,
                    "section_title": title,
                    "chunk_role": "child",
                    "payload": section.get("header_node") or {"label": "section_header", "text": title},
                    "node_ref": section.get("header_ref", f"#/section/{sec_i}"),
                    "label": "section_header",
                })
                continue

            for j, item in enumerate(win_items):
                child_id = f"{doc_id}_child_{sec_i:03d}_{win_i:02d}_{j:03d}_{uuid.uuid4().hex[:6]}"
                parents_index[parent_id]["child_ids"].append(child_id)
                search_text = f"{title}: {item['text']}"
                chunks.append({
                    "chunk_id": child_id,
                    "text": search_text,
                    "child_text": item["text"],
                    "parent_id": parent_id,
                    "parent_text": parent_text,
                    "section_title": title,
                    "chunk_role": "child",
                    "payload": item["node"],
                    "node_ref": item["ref"],
                    "label": item["label"],
                })

    for ref_id, node in picture_nodes:
        text_content = normalize_ws(
            (node.get("text") or "")
            or (node.get("image_description") or "")
            or (node.get("caption") or "")
            or f"Image figure: {node.get('name', 'Document Figure')}"
        )
        parent_id = f"{doc_id}_parent_pic_{uuid.uuid4().hex[:8]}"
        child_id = f"{doc_id}_child_pic_{uuid.uuid4().hex[:8]}"
        parents_index[parent_id] = {
            "parent_id": parent_id,
            "section_title": "Figure",
            "text": text_content,
            "child_ids": [child_id],
            "node_ref": ref_id,
        }
        chunks.append({
            "chunk_id": child_id,
            "text": f"Figure: {text_content}",
            "child_text": text_content,
            "parent_id": parent_id,
            "parent_text": text_content,
            "section_title": "Figure",
            "chunk_role": "child",
            "payload": node,
            "node_ref": ref_id,
            "label": "picture",
        })

    for ref_id, node in table_nodes:
        text_content = normalize_ws(table_text_fn(node) if table_text_fn else (node.get("text") or ""))
        if not text_content:
            continue
        if text_content.startswith("Table Data:") and len(text_content) > 1200:
            text_content = text_content[:1200] + "…"
        parent_id = f"{doc_id}_parent_tbl_{uuid.uuid4().hex[:8]}"
        child_id = f"{doc_id}_child_tbl_{uuid.uuid4().hex[:8]}"
        parents_index[parent_id] = {
            "parent_id": parent_id,
            "section_title": "Table",
            "text": text_content,
            "child_ids": [child_id],
            "node_ref": ref_id,
        }
        chunks.append({
            "chunk_id": child_id,
            "text": f"Table: {text_content}",
            "child_text": text_content,
            "parent_id": parent_id,
            "parent_text": text_content,
            "section_title": "Table",
            "chunk_role": "child",
            "payload": node,
            "node_ref": ref_id,
            "label": "table",
        })

    if not chunks:
        raw_summary = json.dumps((doc_dict or {}).get("body", doc_dict or {}), default=str)[:2000]
        parent_id = f"{doc_id}_parent_root"
        child_id = f"{doc_id}_child_root_{uuid.uuid4().hex[:8]}"
        parents_index[parent_id] = {
            "parent_id": parent_id,
            "section_title": "Document",
            "text": raw_summary,
            "child_ids": [child_id],
            "node_ref": "#/root",
        }
        chunks.append({
            "chunk_id": child_id,
            "text": raw_summary,
            "child_text": raw_summary,
            "parent_id": parent_id,
            "parent_text": raw_summary,
            "section_title": "Document",
            "chunk_role": "child",
            "payload": doc_dict or {},
            "node_ref": "#/root",
            "label": "root",
        })

    return chunks, parents_index
