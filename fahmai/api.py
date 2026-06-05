# -*- coding: utf-8 -*-
"""FahMai API.

POST /agent/local     {"question": "..."}  -> agent on the local Gemma-31B
POST /agent/thaillm   {"question": "..."}  -> agent on the Thai small LLM
    both return       {"id": "<uuid>", "answer": "...", "total_output_token": N}

POST /ocr             {"id","header","transaction":[...]}  (base64 image/pdf)
    returns           {"id", "answer": {"account", "opening_balance_row", "transactions":[], "total_output_token"}}

total_output_token counts ALL tokens consumed across every LLM call in one request
(classify, plan, workers, sql_verify, compute, synth, guard) via a LangChain callback.

Run:
    uv run uvicorn fahmai.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import datetime
import os
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from pydantic import BaseModel

from fahmai.agents.config import THAI_MODEL
from fahmai.agents.graph import aanswer, get_team
from fahmai.agents.llm import reset_model_override, set_model_override
from fahmai.ocr import run_ocr_request_structured


# ---------------------------------------------------------------------------
# Token-counting callback
# ---------------------------------------------------------------------------

class _TokenCounter(BaseCallbackHandler):
    """Accumulates token usage reported by every LLM call in the graph."""

    def __init__(self):
        super().__init__()
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        usage = (response.llm_output or {}).get("token_usage") or {}
        self.prompt_tokens     += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.total_tokens      += usage.get("total_tokens", 0)


# ---------------------------------------------------------------------------
# Per-request LLM audit logger
# ---------------------------------------------------------------------------

# Where the <request_id>.txt audit files are written (override with FAHMAI_AUDIT_DIR).
AUDIT_DIR = Path(os.getenv("FAHMAI_AUDIT_DIR", "audit_logs"))


class _LLMAuditLogger(BaseCallbackHandler):
    """Append every LLM call (prompt, response, token usage) to <request_id>.txt.

    The file name is the request UUID, which is the same value returned as the
    response `id`. Covers every LLM call in the graph (orchestration nodes AND
    specialists) because callbacks propagate to child runs. Each call is written
    on completion so a partial log survives a crash mid-request.
    """

    def __init__(self, request_id: str, question: str):
        super().__init__()
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        self.path = AUDIT_DIR / f"{request_id}.txt"
        self._prompts: dict[str, str] = {}   # run_id -> rendered prompt
        self._lock = threading.Lock()
        self._n = 0
        header = (
            f"{'#' * 80}\n"
            f"# FahMai agent LLM audit log\n"
            f"# request_id : {request_id}\n"
            f"# started    : {datetime.datetime.now().isoformat(timespec='seconds')}\n"
            f"# question   : {question}\n"
            f"{'#' * 80}\n"
        )
        self.path.write_text(header, encoding="utf-8")

    @staticmethod
    def _render_messages(messages) -> str:
        parts: list[str] = []
        for batch in messages:
            for m in batch:
                role = getattr(m, "type", None) or m.__class__.__name__
                parts.append(f"[{role}]\n{getattr(m, 'content', '')}")
        return "\n".join(parts)

    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs) -> None:
        self._prompts[str(run_id)] = self._render_messages(messages)

    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs) -> None:
        self._prompts[str(run_id)] = "\n".join(prompts)

    def on_llm_end(self, response: LLMResult, *, run_id=None, **kwargs) -> None:
        prompt = self._prompts.pop(str(run_id), "(prompt not captured)")
        texts: list[str] = []
        for batch in response.generations:
            for g in batch:
                msg = getattr(g, "message", None)
                texts.append(getattr(g, "text", "") or (getattr(msg, "content", "") if msg else ""))
        usage = (response.llm_output or {}).get("token_usage") or {}
        model = (response.llm_output or {}).get("model_name") or ""
        self._write(model, prompt, "\n".join(texts), usage)

    def on_llm_error(self, error: BaseException, *, run_id=None, **kwargs) -> None:
        prompt = self._prompts.pop(str(run_id), "(prompt not captured)")
        self._write("", prompt, f"ERROR: {error!r}", {})

    def _write(self, model: str, prompt: str, response: str, usage: dict) -> None:
        with self._lock:
            self._n += 1
            n = self._n
            block = (
                f"\n{'=' * 80}\n"
                f"CALL #{n} | {datetime.datetime.now().isoformat(timespec='seconds')} | model={model}\n"
                f"tokens: prompt={usage.get('prompt_tokens', 0)} "
                f"completion={usage.get('completion_tokens', 0)} "
                f"total={usage.get('total_tokens', 0)}\n"
                f"{'-' * 80}\nPROMPT:\n{prompt}\n"
                f"{'-' * 80}\nRESPONSE:\n{response}\n"
            )
            with self.path.open("a", encoding="utf-8") as f:
                f.write(block)

    def finalize(self, answer: str, total_tokens: int) -> None:
        with self._lock:
            footer = (
                f"\n{'#' * 80}\n"
                f"# FINAL ANSWER ({self._n} LLM calls, total_output_token={total_tokens})\n"
                f"{'#' * 80}\n{answer}\n"
            )
            with self.path.open("a", encoding="utf-8") as f:
                f.write(footer)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_team()          # warm up the graph + specialist agents at startup
    yield


app = FastAPI(
    title="FahMai Answer API",
    description="LangGraph multi-agent QA over the FahMai Supabase warehouse + RAG corpus.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    id: str
    answer: str
    total_output_token: int


class OCRRequest(BaseModel):
    id: str
    header: str                   # base64-encoded header image/pdf (data-URL prefix optional)
    transaction: list[str] = []   # base64-encoded transaction images/pdfs


# --- OCR structured response schema ---

class OCRAccount(BaseModel):
    account_number: str | None = None
    owner_branch: str | None = None


class OCROpeningBalance(BaseModel):
    label: str | None = None
    date: str | None = None
    balance_text: str | None = None


class OCRTransactionRow(BaseModel):
    row_index: int
    date: str | None = None
    item: str | None = None
    debit_text: str | None = None
    credit_text: str | None = None
    amount_direction: str | None = None
    balance_text: str | None = None
    details: str | None = None


class OCRAnswer(BaseModel):
    account: OCRAccount = OCRAccount()
    opening_balance_row: OCROpeningBalance = OCROpeningBalance()
    transactions: list[OCRTransactionRow] = []
    total_output_token: int = 0


class OCRResponse(BaseModel):
    id: str
    answer: OCRAnswer


# ---------------------------------------------------------------------------
# Agent endpoints
# ---------------------------------------------------------------------------

async def _run_agent(question: str, model: str | None) -> AnswerResponse:
    """Run the agent; `model` (if given) overrides the orchestration LLM for this request."""
    if not question.strip():
        raise HTTPException(status_code=422, detail="question must not be blank")

    req_id = str(uuid.uuid4())          # same value is the response id AND the audit filename
    counter = _TokenCounter()
    audit = _LLMAuditLogger(req_id, question)
    token = set_model_override(model) if model else None
    try:
        ans = await aanswer(question, callbacks=[counter, audit])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if token is not None:
            reset_model_override(token)

    audit.finalize(ans, counter.total_tokens)
    return AnswerResponse(
        id=req_id,
        answer=ans,
        total_output_token=counter.total_tokens,
    )


@app.post("/agent/local", response_model=AnswerResponse)
async def agent_local(req: QuestionRequest) -> AnswerResponse:
    """Agent powered by the local Gemma-31B (default orchestration model)."""
    return await _run_agent(req.question, model=None)


@app.post("/agent/thaillm", response_model=AnswerResponse)
async def agent_thaillm(req: QuestionRequest) -> AnswerResponse:
    """Agent with orchestration nodes running on the Thai small LLM."""
    return await _run_agent(req.question, model=THAI_MODEL)


@app.post("/ocr", response_model=OCRResponse)
async def ocr(req: OCRRequest) -> OCRResponse:
    if not req.header.strip():
        raise HTTPException(status_code=422, detail="header must not be blank")
    try:
        answer = await run_ocr_request_structured(req.header, req.transaction)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return OCRResponse(id=req.id, answer=answer)


@app.get("/health")
async def health():
    return {"status": "ok"}
