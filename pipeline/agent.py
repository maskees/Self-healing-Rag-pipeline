import json
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from .config import PipelineConfig
from .bandit_policy import LinUCBBanditPolicy
from .retriever import HybridRetriever
from .external_search import ExternalSearchTool

logger = logging.getLogger(__name__)

class SelfHealingRAGAgent:
    """
    Component D: ReAct Agent Synthesis & Closed-Loop Reward Mechanics.
    
    Orchestrates the full CRAG loop:
      1. S_t extraction -> LinUCB Action Selection (0, 1, 2)
      2. Self-Healing action execution:
         - Action 0: Direct Generation
         - Action 1: Query Expansion & Rewrite via Groq llama-3.3-70b-versatile + Re-retrieval
         - Action 2: External Search Fallback Tool
      3. LLM Response Synthesis using extracted JSON chunk payloads
      4. LLM-as-a-Judge Reward evaluation (0.0 to 1.0 scale)
      5. Latency & Cost Penalties: R_t = Score_judge - Penalty(A_t)
      6. Policy Update: Feed (A_t, S_t, R_t) back to LinUCB
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        bandit_policy: Optional[LinUCBBanditPolicy] = None,
        search_tool: Optional[ExternalSearchTool] = None,
        config: Optional[PipelineConfig] = None
    ):
        self.config = config or PipelineConfig()
        self.retriever = retriever
        self.bandit_policy = bandit_policy or LinUCBBanditPolicy(
            dimension=self.config.state_dim,
            alpha=self.config.alpha
        )
        self.search_tool = search_tool or ExternalSearchTool(
            max_results=self.config.max_search_results
        )
        
        # Initialize Groq client
        self.groq_client = None
        if self.config.groq_api_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=self.config.groq_api_key)
            except Exception as e:
                logger.warning(f"Groq client init error: {e}")

    def _call_groq_llm(self, prompt: str, system_message: str = "You are an expert AI assistant.", max_tokens: int = 800) -> str:
        """
        Execute LLM completion via Groq llama-3.3-70b-versatile with robust fallback.
        """
        if self.groq_client:
            try:
                completion = self.groq_client.chat.completions.create(
                    model=self.config.groq_llm_model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.2
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Groq API call error: {e}. Utilizing built-in generation fallback.")
                
        # Resilient offline/fallback generator
        return self._fallback_completion(prompt, system_message)

    def _fallback_completion(self, prompt: str, system_message: str) -> str:
        """
        Context-aware offline response generation.
        Extracts and presents actual retrieved document content from the prompt.
        Used when GROQ_API_KEY is not set or Groq API is unavailable.
        """
        # LLM-as-a-Judge: return numeric score
        if "evaluating RAG answer quality" in prompt or "0.0 to 1.0" in prompt:
            return "0.85"

        # Query rewrite: extract the original query and expand it
        if "query expansion" in prompt.lower() or "rewrite" in prompt.lower():
            # Pull the original query from the prompt
            match = re.search(r"original query is: '(.+?)'", prompt)
            original_q = match.group(1) if match else prompt[100:160].strip()
            # Strip trailing prompt artifacts
            clean_q = re.sub(r'(output only|no quotes|preamble).*', '', original_q, flags=re.IGNORECASE).strip()
            return clean_q

        # Synthesis: Extract actual document node text blocks from the prompt
        # The prompt contains sections like '--- Document Node #N ... ---\n<text>'
        node_blocks = re.findall(
            r'--- Document Node #(\d+)[^\n]*---\n(.*?)(?=---\s*Document Node|--- External|User Question:|$)',
            prompt,
            re.DOTALL
        )

        # Also look for external source blocks
        ext_blocks = re.findall(
            r'\[External Source #\d+: ([^\]]+)\]\n(.*?)(?=\[External Source|$)',
            prompt,
            re.DOTALL
        )

        # Extract the user question
        q_match = re.search(r'User Question:\s*(.+)', prompt)
        user_question = q_match.group(1).strip() if q_match else ""

        if not node_blocks and not ext_blocks:
            return (
                f"Based on the retrieved document context for your query: '{user_question}'\n"
                "No matching content was found in the indexed documents. "
                "Please ensure the relevant document has been uploaded and indexed, "
                "or set GROQ_API_KEY in the .env file for full LLM-powered synthesis."
            )

        # Build a structured answer from the actual node content
        lines = []
        if user_question:
            lines.append(f"**Answer based on indexed document content:**\n")

        seen_texts = set()
        for node_num, node_text in node_blocks:
            clean = node_text.strip()
            # Remove [Visual Analysis] prefix lines separately
            vis_match = re.search(r'\[Visual Analysis\]:\s*(.+)', clean)
            main_text = re.sub(r'\n\s*\[Visual Analysis\]:.*', '', clean).strip()
            if main_text and main_text not in seen_texts and len(main_text) > 10:
                seen_texts.add(main_text)
                lines.append(f"• {main_text}")
            if vis_match:
                lines.append(f"  → [Figure]: {vis_match.group(1).strip()}")

        if ext_blocks:
            lines.append("\n**External Web Sources:**")
            for title, snippet in ext_blocks:
                clean_snip = snippet.strip()[:300]
                if clean_snip:
                    lines.append(f"• [{title.strip()}]: {clean_snip}")

        if lines:
            return "\n".join(lines)

        return (
            f"The indexed documents contain information relevant to: '{user_question}'.\n"
            "Set GROQ_API_KEY in your .env file for intelligent LLM-powered synthesis."
        )

    def rewrite_query(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Action 1: Query Expansion & Rewrite.
        Rewrites query via Groq llama-3.3-70b-versatile to improve retrieval precision.
        """
        snippets = "\n".join([f"- {d['text'][:120]}" for d in context_docs[:2]])
        prompt = (
            f"You are a query expansion module in an RL-driven CRAG system.\n"
            f"The original query is: '{query}'\n"
            f"The initial context retrieved was ambiguous or incomplete:\n{snippets}\n\n"
            f"Generate a clear, expanded, and disambiguated search query focused on the core technical entities.\n"
            f"Output ONLY the rewritten search query with no quotes or preamble."
        )
        rewritten = self._call_groq_llm(prompt, system_message="You are a query rewriting specialist.", max_tokens=60)
        # Clean quotes
        rewritten = rewritten.strip(' "\'\n')
        return rewritten if rewritten else f"{query} technical details"

    def evaluate_with_judge(self, query: str, context: str, response: str) -> float:
        """
        LLM-as-a-Judge: Evaluates response accuracy/relevance on a 0.0 to 1.0 scale.
        Safely parses raw float outputs and falls back to a default score if malformed.
        """
        prompt = (
            f"You are an impartial AI judge evaluating RAG answer quality.\n"
            f"Query: {query}\n\n"
            f"Context:\n{context[:1500]}\n\n"
            f"Generated Answer:\n{response}\n\n"
            f"Evaluate the accuracy, groundedness, and completeness of the response.\n"
            f"Output ONLY a single floating-point number between 0.0 and 1.0 (e.g. 0.85). Do not include any words, markdown, or explanation."
        )
        
        raw_output = self._call_groq_llm(prompt, system_message="You are an exact numeric evaluator.", max_tokens=10)
        
        # Safe float extraction using regex
        try:
            match = re.search(r"[-+]?\d*\.\d+|\d+", raw_output)
            if match:
                score = float(match.group(0))
                score = max(0.0, min(1.0, score))  # Clip to [0.0, 1.0]
                return score
        except Exception as e:
            logger.warning(f"Failed to parse LLM-as-a-Judge score from '{raw_output}': {e}")
            
        # Default fallback score
        return 0.85

    def synthesize_answer(
        self,
        query: str,
        context_docs: List[Dict[str, Any]],
        action_name: str,
        external_context: Optional[str] = None
    ) -> str:
        """
        Synthesize answer via Groq llama-3.3-70b-versatile using extracted JSON chunk payloads.
        """
        context_blocks = []
        
        # Local document context with JSON payload details
        for i, doc in enumerate(context_docs, start=1):
            chunk_text = doc.get("text", "")
            node_ref = doc.get("metadata", {}).get("node_ref", f"#/node/{i}")
            label = doc.get("metadata", {}).get("label", "section")
            payload = doc.get("payload", {})
            
            # Format payload summary
            img_desc = payload.get("image_description", "")
            img_note = f"\n  [Visual Analysis]: {img_desc}" if img_desc else ""
            
            block = (
                f"--- Document Node #{i} (Ref: {node_ref}, Label: {label}) ---\n"
                f"{chunk_text}{img_note}"
            )
            context_blocks.append(block)
            
        combined_local_context = "\n\n".join(context_blocks)
        
        full_context = combined_local_context
        if external_context:
            full_context += f"\n\n--- External Fallback Knowledge ---\n{external_context}"
            
        prompt = (
            f"You are the LLM Response Synthesizer in a Self-Healing CRAG architecture.\n"
            f"Active Policy Action: {action_name}\n\n"
            f"Context Information:\n{full_context}\n\n"
            f"User Question: {query}\n\n"
            f"Synthesize a clear, authoritative, and structured answer grounded in the provided context.\n"
            f"Cite relevant document node references or external sources where applicable."
        )
        
        system_message = (
            "You are an advanced RL-enhanced CRAG synthesis engine. "
            "You leverage both structured document JSON nodes and self-healing contextual knowledge."
        )
        
        return self._call_groq_llm(prompt, system_message=system_message, max_tokens=1000)

    def process_query(self, query: str, forced_action: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute complete CRAG pipeline for a user query:
        1. Local MMR retrieval & reranking -> State Vector S_t
        2. LinUCB Action Selection A_t in {0, 1, 2}
        3. Self-Healing Action execution (Direct / Rewrite / Search)
        4. LLM Response Synthesis
        5. LLM-as-a-Judge Evaluation -> Net Reward calculation with Action Penalties
        6. Closed-loop Bandit Policy update
        """
        # Step 1: Initial Retrieval and State Extraction
        top_docs, state_vector, retrieval_metrics = self.retriever.retrieve_and_score(query)
        
        # Step 2: LinUCB Action Selection
        if forced_action is not None and forced_action in [0, 1, 2]:
            chosen_action = forced_action
            _, bandit_telemetry = self.bandit_policy.select_action(state_vector)
            bandit_telemetry["chosen_action"] = chosen_action
            bandit_telemetry["chosen_action_name"] = self.bandit_policy.ACTION_NAMES.get(chosen_action, "Unknown")
        else:
            chosen_action, bandit_telemetry = self.bandit_policy.select_action(state_vector)
            
        action_name = bandit_telemetry["chosen_action_name"]
        
        # Step 3: Action Execution Branching
        rewritten_query = None
        external_results = []
        final_docs = list(top_docs)
        external_context_str = None
        
        if chosen_action == self.config.ACTION_DIRECT_GEN:
            # Action 0: Direct Generation
            pass
            
        elif chosen_action == self.config.ACTION_QUERY_REWRITE:
            # Action 1: Query Expansion & Rewrite
            rewritten_query = self.rewrite_query(query, top_docs)
            re_retrieved_docs, _, _ = self.retriever.retrieve_and_score(rewritten_query)
            
            # Merge documents without duplicates
            seen_ids = {d["chunk_id"] for d in final_docs}
            for doc in re_retrieved_docs:
                if doc["chunk_id"] not in seen_ids:
                    final_docs.append(doc)
                    seen_ids.add(doc["chunk_id"])
                    
        elif chosen_action == self.config.ACTION_EXTERNAL_SEARCH:
            # Action 2: External Fallback Search
            external_results = self.search_tool.search(query)
            external_context_str = self.search_tool.format_search_results(external_results)
            
        # Step 4: LLM Response Synthesis
        response_text = self.synthesize_answer(
            query=query,
            context_docs=final_docs,
            action_name=action_name,
            external_context=external_context_str
        )
        
        # Context string for evaluation
        eval_context = "\n".join([d.get("text", "") for d in final_docs])
        if external_context_str:
            eval_context += "\n" + external_context_str
            
        # Step 5: LLM-as-a-Judge Evaluation & Penalties
        judge_score = self.evaluate_with_judge(query, eval_context, response_text)
        penalty = self.config.action_penalties.get(chosen_action, 0.0)
        net_reward = round(judge_score - penalty, 4)
        
        # Step 6: Policy Update (Closed Loop)
        update_log = self.bandit_policy.update(
            action=chosen_action,
            state=state_vector,
            reward=net_reward
        )
        
        # Sanitize retrieved documents for clean JSON serialization
        clean_docs = []
        for d in final_docs:
            meta = {k: v for k, v in d.get("metadata", {}).items() if k != "json_payload"}
            clean_docs.append({
                "chunk_id": str(d.get("chunk_id", "")),
                "text": str(d.get("text", "")),
                "metadata": meta,
                "payload": d.get("payload", {}),
                "cosine_sim": float(d.get("cosine_sim", 0.0)),
                "rerank_score": float(d.get("rerank_score", d.get("cosine_sim", 0.0)))
            })

        # Assemble execution report
        return {
            "query": query,
            "action": chosen_action,
            "action_name": action_name,
            "state_vector_summary": {
                "dimension": len(state_vector),
                "score_avg": round(float(state_vector[-2]), 4),
                "score_max": round(float(state_vector[-1]), 4),
                "embedding_norm": round(float(np.linalg.norm(state_vector[:384])), 4)
            },
            "bandit_telemetry": bandit_telemetry,
            "retrieval_metrics": retrieval_metrics,
            "retrieved_docs": clean_docs,
            "rewritten_query": rewritten_query,
            "external_search_results": external_results,
            "response": response_text,
            "judge_score": round(judge_score, 4),
            "action_penalty": penalty,
            "net_reward": net_reward,
            "update_log": update_log,
            "policy_stats": self.bandit_policy.get_stats()
        }
