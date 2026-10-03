"""Prompt templates and system instructions for ResearchMate RAG pipeline.

Enforces strict academic grounding, verifiable citations, anti-hallucination rules,
and conversation reference resolution.
"""

from __future__ import annotations

from typing import Dict, List, Optional


ACADEMIC_RAG_SYSTEM_PROMPT = """You are ResearchMate, an expert academic AI research assistant.
Your job is to answer research questions with strict factual accuracy based ONLY on the provided research paper passages.

CRITICAL RULES:
1. GROUNDING ONLY: Answer the question using ONLY the facts and data directly mentioned in the provided [Source X] passages. Do NOT use outside knowledge or extrapolate beyond the text.
2. INSUFFICIENT EVIDENCE: If the supplied text does not contain enough information to answer the question accurately, you MUST explicitly state:
   "The provided papers do not contain enough information to answer this question."
   Do NOT attempt to guess or make up plausible answers.
3. CITATIONS: Every factual claim, finding, method, metric, or statement derived from the papers MUST include an inline citation corresponding to the source number, for example [1], [2], or [1, 3].
4. CITATION INTEGRITY: Only cite sources that actually support the specific claim. Never fabricate citation numbers (e.g. do not cite [9] if only 3 sources are provided).
5. MULTI-PAPER SYNTHESIS: When the question compares multiple papers or the context contains passages from different papers, clearly identify which findings belong to which paper/author.
6. CONFLICTS & DISAGREEMENTS: When different papers present conflicting findings, architectures, or metrics, explicitly highlight the disagreement and cite the relevant sources.
7. UNCERTAINTY: Preserve scientific uncertainty where authors express hedging or limitations.
8. FORMATTING:
   - Provide a clear, well-structured academic response.
   - Use headings or bullet points where appropriate for readability.
   - Conclude with a 'Sources' summary mapping the citations used (e.g. Sources:\n[1] filename — Page X — Section).
"""


REFERENCE_RESOLUTION_SYSTEM_PROMPT = """You are an academic query disambiguation assistant.
Your task is to rewrite conversational follow-up questions into standalone search queries.
Use the conversation history ONLY to resolve ambiguous pronouns (e.g., "it", "they", "the second paper", "the first method", "that architecture").
Do NOT answer the question. Only output the rewritten standalone search query. If the question is already standalone, return it unchanged.
"""


def build_rag_user_prompt(
    query: str,
    context: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Construct the complete user prompt incorporating research context and query.

    Args:
        query: Academic user question.
        context: Formatted string of retrieved [Source X] passages.
        chat_history: Optional list of previous chat messages [{"role": "user"|"assistant", "content": "..."}].

    Returns:
        Structured user prompt string.
    """
    prompt_parts: List[str] = []

    # 1. Retrieved Research Context
    prompt_parts.append("### RETRIEVED RESEARCH PAPER CONTEXT ###\n")
    if context.strip():
        prompt_parts.append(context.strip())
    else:
        prompt_parts.append("[No relevant context passages found]")

    prompt_parts.append("\n\n### END OF CONTEXT ###\n")

    # 2. Optional Brief Chat History (up to 3 turns)
    if chat_history:
        history_snippet = []
        # Take up to last 4 messages
        recent_history = chat_history[-4:]
        for msg in recent_history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "").strip()
            if len(content) > 300:
                content = content[:297] + "..."
            history_snippet.append(f"{role}: {content}")

        if history_snippet:
            prompt_parts.append("### RECENT CONVERSATION CONTEXT (FOR REFERENCE ONLY) ###\n")
            prompt_parts.append("\n".join(history_snippet))
            prompt_parts.append("\n### END OF CONVERSATION HISTORY ###\n\n")

    # 3. User Question
    prompt_parts.append(f"### RESEARCH QUESTION ###\n{query.strip()}\n\n")
    prompt_parts.append(
        "Please provide a grounded, rigorous academic response based ONLY on the retrieved research context above, including inline citations [1], [2], etc."
    )

    return "".join(prompt_parts)


def build_reference_resolution_prompt(
    current_query: str,
    chat_history: List[Dict[str, str]],
) -> str:
    """Build prompt for conversational reference resolution.

    Args:
        current_query: Follow-up question from user.
        chat_history: Past conversation messages.

    Returns:
        Prompt string.
    """
    history_lines = []
    for msg in chat_history[-4:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").strip()
        history_lines.append(f"{role}: {content}")

    return (
        f"Conversation History:\n"
        f"{chr(10).join(history_lines)}\n\n"
        f"Follow-up Question: {current_query}\n\n"
        f"Standalone Search Query:"
    )
