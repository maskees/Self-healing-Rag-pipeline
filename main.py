"""
RL-Driven Self-Healing CRAG Architecture with Native Docling JSON Ingestion
Complete End-to-End CLI & Action Path Demonstrator
"""

import sys
import os
import json
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CRAG-Main")

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pipeline.config import PipelineConfig
from pipeline.ingestor import DoclingJSONIngestor
from pipeline.retriever import HybridRetriever
from pipeline.bandit_policy import LinUCBBanditPolicy
from pipeline.external_search import ExternalSearchTool
from pipeline.agent import SelfHealingRAGAgent

def create_sample_docling_document() -> dict:
    """
    Constructs a production-like native Docling JSON document structure
    including text paragraphs, tables, and a multimodal picture node.
    """
    return {
        "schema_name": "docling_core.models.DoclingDocument",
        "version": "1.0.0",
        "name": "quantum_fusion_crag_spec.pdf",
        "origin": {
            "filename": "quantum_fusion_crag_spec.pdf",
            "mimetype": "application/pdf"
        },
        "body": {
            "label": "body",
            "children": [
                {"$ref": "#/texts/0"},
                {"$ref": "#/texts/1"},
                {"$ref": "#/tables/0"},
                {"$ref": "#/pictures/0"},
                {"$ref": "#/texts/2"}
            ]
        },
        "texts": {
            "#/texts/0": {
                "$ref": "#/texts/0",
                "label": "section_header",
                "text": "Project Helios: High-Confinement Magnetohydrodynamic Fusion Reactor",
                "orig": "# Project Helios: High-Confinement Magnetohydrodynamic Fusion Reactor"
            },
            "#/texts/1": {
                "$ref": "#/texts/1",
                "label": "text",
                "text": "The Helios Mark-IV reactor utilizes dual deuterium-tritium plasma injectors with a magnetic field strength of 14.8 Tesla. The primary cooling cycle employs liquid lithium-lead eutectic (Li17Pb83) circulating at 650 degrees Celsius to achieve a target thermal conversion efficiency of 48.5 percent.",
                "orig": "The Helios Mark-IV reactor utilizes dual deuterium-tritium plasma injectors with a magnetic field strength of 14.8 Tesla..."
            },
            "#/texts/2": {
                "$ref": "#/texts/2",
                "label": "text",
                "text": "Self-healing divertor tiles are engineered using tungsten-carbide nano-lattice composites designed to resist neutron embrittlement under intense flux exceeding 15 MW per square meter.",
                "orig": "Self-healing divertor tiles are engineered using tungsten-carbide nano-lattice composites..."
            }
        },
        "tables": {
            "#/tables/0": {
                "$ref": "#/tables/0",
                "label": "table",
                "text": "Table 1: Operational Parameters of Helios Mark-IV | Magnetic Field: 14.8 T | Coolant: Li17Pb83 | Plasma Injectors: Dual D-T | Divertor Heat Flux: 15 MW/m^2 | Q-factor: 12.4",
                "data": {
                    "headers": ["Parameter", "Specification"],
                    "rows": [
                        ["Magnetic Field", "14.8 Tesla"],
                        ["Coolant Eutectic", "Liquid Lithium-Lead (Li17Pb83)"],
                        ["Divertor Peak Flux", "15 MW/m^2"],
                        ["Target Q-factor", "12.4"]
                    ]
                }
            }
        },
        "pictures": {
            "#/pictures/0": {
                "$ref": "#/pictures/0",
                "label": "picture",
                "caption": "Figure 3: Cross-sectional schematics of the toroidal magnetic containment chamber",
                "image_description": "Multimodal analysis: Toroidal vacuum vessel surrounded by 18 superconducting D-shaped coils. Dual deuterium injectors enter at 45-degree angles, while lower divertor plates channel exhausted alpha particles.",
                "text": "Toroidal magnetic containment schematic showing 18 superconducting coils and dual injector ports."
            }
        }
    }


def main():
    print("=" * 80)
    print("RL-DRIVEN SELF-HEALING CRAG PIPELINE INITIALIZATION")
    print("Native Docling Ingestion + Groq Vision + MMR + Cross-Encoder + LinUCB Bandit")
    print("=" * 80)
    
    # 1. Initialize Configuration and Modules
    config = PipelineConfig()
    ingestor = DoclingJSONIngestor(config=config)
    ingestor.clear_database()  # Clean baseline for demo
    
    retriever = HybridRetriever(ingestor=ingestor, config=config)
    bandit_policy = LinUCBBanditPolicy(dimension=config.state_dim, alpha=config.alpha)
    search_tool = ExternalSearchTool(max_results=3)
    agent = SelfHealingRAGAgent(
        retriever=retriever,
        bandit_policy=bandit_policy,
        search_tool=search_tool,
        config=config
    )
    
    # 2. Ingest native Docling document
    print("\n[Step 1] Ingesting Native Docling JSON Specification...")
    sample_doc = create_sample_docling_document()
    ingest_result = ingestor.ingest_docling_dict(
        sample_doc,
        doc_name="quantum_fusion_crag_spec.pdf",
        doc_id="docling_helios_001",
    )
    print(
        f"-> Successfully ingested {ingest_result['num_chunks']} nodes into ChromaDB "
        f"(embedded={ingest_result.get('embedded')}, dim={ingest_result.get('embedding_dim')})"
    )
    print(f"-> JSONL written to: {ingest_result.get('jsonl_path')}")
    print(f"-> Chroma collection size: {ingest_result.get('chroma_count')}")

    # 3. Demonstrate All 3 Action Paths
    test_cases = [
        {
            "title": "PATH 0: Direct Generation (High-confidence local match)",
            "query": "What is the magnetic field strength and coolant used in the Helios Mark-IV reactor?",
            "forced_action": 0
        },
        {
            "title": "PATH 1: Query Expansion & Rewrite (Ambiguous / colloquial query)",
            "query": "tell me about that fusion chamber picture and coils",
            "forced_action": 1
        },
        {
            "title": "PATH 2: External Fallback Search (Out-of-corpus query)",
            "query": "What was the latest James Webb Space Telescope discovery regarding exoplanet atmospheres in 2024?",
            "forced_action": 2
        }
    ]
    
    for idx, test in enumerate(test_cases, start=1):
        print("\n" + "=" * 80)
        print(f"EXECUTION {idx}/3: {test['title']}")
        print(f"Query: \"{test['query']}\"")
        print("=" * 80)
        
        result = agent.process_query(test["query"], forced_action=test["forced_action"])
        
        # Display State Vector S_t Breakdown
        state_summary = result["state_vector_summary"]
        print("\n--- 1. State Vector Assembly (S_t) ---")
        print(f"Dimension: {state_summary['dimension']} (384 query embed + Score_avg + Score_max)")
        print(f"Score Avg: {state_summary['score_avg']:.4f} | Score Max: {state_summary['score_max']:.4f}")
        print(f"Query Embedding L2 Norm: {state_summary['embedding_norm']:.4f}")
        
        # Display LinUCB Contextual Bandit Telemetry
        b_tel = result["bandit_telemetry"]
        print("\n--- 2. LinUCB Bandit Action Selection ---")
        print(f"Selected Arm: Action {result['action']} -> '{result['action_name']}'")
        for arm_id in [0, 1, 2]:
            score = b_tel["arm_scores"][arm_id]
            mean_p = b_tel["arm_mean_payoff"][arm_id]
            bonus = b_tel["arm_variance_bonus"][arm_id]
            star = " <--- CHOSEN" if arm_id == result["action"] else ""
            print(f"  Arm {arm_id} ({bandit_policy.ACTION_NAMES[arm_id]}): "
                  f"UCB={score:.4f} [Payoff={mean_p:.4f} + Bonus={bonus:.4f}]{star}")
                  
        # Branch Execution Info
        print("\n--- 3. Self-Healing Action Branch Execution ---")
        if result["rewritten_query"]:
            print(f"[Action 1 Triggered] Groq Query Rewriter: '{result['rewritten_query']}'")
        if result["external_search_results"]:
            print(f"[Action 2 Triggered] DuckDuckGo Search retrieved {len(result['external_search_results'])} live web sources.")
            for s in result["external_search_results"][:2]:
                print(f"  * {s['title']} ({s['source'][:40]}...)")
                
        print(f"Local Context Nodes Retrieved: {len(result['retrieved_docs'])}")
        for d in result["retrieved_docs"][:2]:
            ref = d.get("metadata", {}).get("node_ref", "N/A")
            label = d.get("metadata", {}).get("label", "N/A")
            r_score = d.get("rerank_score", d.get("cosine_sim", 0.0))
            print(f"  - [{label} {ref}] Rerank Score: {r_score:.4f} | {d['text'][:75]}...")

        # Response Synthesis
        print("\n--- 4. Synthesized Answer ---")
        print(result["response"].strip())
        
        # Closed-loop Reward and Weight Update
        print("\n--- 5. Closed-Loop Reward & Bandit Policy Update ---")
        print(f"LLM-as-a-Judge Score: {result['judge_score']:.4f}")
        print(f"Action Latency/Cost Penalty: -{result['action_penalty']:.4f}")
        print(f"Net Reward (R_t = Judge - Penalty): {result['net_reward']:.4f}")
        print(f"Policy Update Log: {result['update_log']}")

    # Final Bandit Summary
    print("\n" + "=" * 80)
    print("FINAL LINUCB BANDIT ARMS SUMMARY & CONVERGENCE TELEMETRY")
    print("=" * 80)
    stats = bandit_policy.get_stats()
    for arm_id, arm_data in stats["arms"].items():
        print(f"Action {arm_id} ({arm_data['name']}):")
        print(f"  Count: {arm_data['count']} | Cumulative Reward: {arm_data['total_reward']:.4f} | Avg Reward: {arm_data['avg_reward']:.4f} | Parameter Norm ||theta||: {arm_data['theta_norm']:.4f}")
        
    print("\n[SUCCESS] Pipeline executed all action branches with native Docling JSON metadata persistence and closed-loop LinUCB updates.")

if __name__ == "__main__":
    main()
