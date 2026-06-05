# -*- coding: utf-8 -*-
"""Specialist registry. Each sub-agent is its own module (sql_analyst, doc_researcher);
add a new specialist by dropping a module here and registering it in SPECIALISTS.

Agents are built lazily (no LLM calls at import) and cached.
"""
from __future__ import annotations

from fahmai.agents.llm import current_tool_model
from fahmai.agents.specialists import doc_researcher, rag_researcher, sql_analyst
from fahmai.agents.specialists.base import run_specialist_async

# kind -> module (each module exposes build(), KIND, RECURSION)
SPECIALISTS = {
    sql_analyst.KIND: sql_analyst,
    doc_researcher.KIND: doc_researcher,
    rag_researcher.KIND: rag_researcher,
}

# Cache keyed by (kind, tool_model). The model is part of the key so a per-request model
# override (e.g. /agent/thaillm) builds & reuses a specialist on the overridden model rather
# than the build-time-frozen default — without rebuilding on every call.
_BUILT: dict = {}


def _agent_for(kind: str):
    """Get (build + cache) the specialist for `kind` on the currently-resolved tool model."""
    mod = SPECIALISTS.get(kind, sql_analyst)
    key = (mod.KIND, current_tool_model())   # build() reads the override via make_tool_llm()
    agent = _BUILT.get(key)
    if agent is None:
        agent = mod.build()
        _BUILT[key] = agent
    return agent, mod


def build_specialists() -> dict:
    """Build & cache every specialist agent on the current tool model. Returns {kind: agent}."""
    return {kind: _agent_for(kind)[0] for kind in SPECIALISTS}


async def run(kind: str, subq: str) -> str:
    """Route a sub-question to a specialist (defaults to 'sql' for unknown kinds)."""
    agent, mod = _agent_for(kind)
    return await run_specialist_async(agent, subq, mod.RECURSION)


__all__ = ["SPECIALISTS", "build_specialists", "run", "run_specialist_async"]
