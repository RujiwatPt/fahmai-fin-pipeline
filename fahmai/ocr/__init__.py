# -*- coding: utf-8 -*-
"""FahMai OCR package.

Public API (used by fahmai.api):
    run_ocr(data_b64)                     sync, single image/PDF → {result, pages, total_output_token}
    run_ocr_async(data_b64)               async equivalent
    run_ocr_request(header_b64, txn_b64s) async full /ocr request, concurrent pages

Batch pipeline (standalone scripts, unchanged):
    fahmai/ocr/pipeline.py, bank_statement.py, ocr.py, etc.
"""
from __future__ import annotations

import asyncio
import base64
import os

import httpx
from fahmai.db import ROOT  # noqa: F401 — triggers .env load
from fahmai.ocr.bank_statement import run_bank_statement_structured

# OCR endpoint config.
# Env var priority: OCR_OCR_* (new) → FAHMAI_OCR_* (legacy) → LLM defaults.
_LLM_BASE_URL = os.getenv("FAHMAI_LLM_BASE_URL", "https://openrouter.ai/api/v1")
_LLM_API_KEY_ENV = "FAHMAI_LLM_API_KEY" if os.getenv("FAHMAI_LLM_BASE_URL") else "OPEN_ROUTER"
OCR_MODEL = (os.getenv("OCR_OCR_MODEL")
             or os.getenv("FAHMAI_OCR_MODEL")
             or "typhoon-ocr-preview")
OCR_BASE_URL = (os.getenv("OCR_OCR_BASE_URL")
                or os.getenv("FAHMAI_OCR_BASE_URL")
                or _LLM_BASE_URL)
if os.getenv("OCR_OCR_API_KEY"):
    OCR_API_KEY_ENV = "OCR_OCR_API_KEY"
elif os.getenv("FAHMAI_OCR_API_KEY"):
    OCR_API_KEY_ENV = "FAHMAI_OCR_API_KEY"
else:
    OCR_API_KEY_ENV = _LLM_API_KEY_ENV

# LLM (text stage) config — uses orchestration LLM for structured extraction
LLM_MODEL = os.getenv("FAHMAI_THAI_MODEL", "typhoon-ai/typhoon-s-thaillm-8b-instruct-research-preview")
LLM_BASE_URL = _LLM_BASE_URL
LLM_API_KEY_ENV = _LLM_API_KEY_ENV

_PROMPT = (
    "Extract ALL text visible in this image exactly as it appears, preserving reading order, "
    "line breaks, numbers, and Thai/English characters. Return only the extracted text."
)

# Structured extraction prompts for the bank-statement schema
_HEADER_PROMPT = """\
You are extracting structured data from a Thai bank statement header image.

Return ONLY valid JSON with exactly this structure — no markdown fences, no explanation:
{
  "account": {
    "account_number": "<account number exactly as printed>",
    "owner_branch": "<branch name in Thai>"
  },
  "opening_balance_row": {
    "label": "<e.g. ยอดยกมา>",
    "date": "<DD-MM-YY>",
    "balance_text": "<balance amount as string, e.g. 330.10>"
  }
}

Use null for any field not found in the image.\
"""

_TXN_PROMPT = """\
You are extracting transaction rows from a Thai bank statement page image.

Return ONLY a valid JSON array — no markdown fences, no explanation.
Each element must have exactly these keys:
{
  "row_index": <integer, 1-based sequential within this page>,
  "date": "<DD-MM-YY>",
  "item": "<transaction description, Thai or English>",
  "debit_text": "<debit amount string, or null if not a debit>",
  "credit_text": "<credit amount string, or null if not a credit>",
  "amount_direction": "<'debit' or 'credit'>",
  "balance_text": "<running balance as string>",
  "details": "<additional details / reference text, or null>"
}

Include every transaction row visible. Use null for missing fields.\
"""

_TIMEOUT = 360
_MAX_SIDE = int(os.getenv("FAHMAI_OCR_MAX_SIDE", "2000"))


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _detect(raw: bytes) -> tuple[str, bool]:
    """Return (mime, is_pdf). Sniffs magic bytes."""
    if raw[:5] == b"%PDF-":
        return "application/pdf", True
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png", False
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg", False
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp", False
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif", False
    return "image/png", False  # default; let the model try


def _cap_image(raw: bytes) -> tuple[bytes, str]:
    """Downscale image so its longest side <= _MAX_SIDE. Returns (bytes, mime)."""
    import fitz  # lazy — only needed when resizing

    mime, _ = _detect(raw)
    doc = fitz.open(stream=raw, filetype=None)
    try:
        page = doc[0]
        w, h = page.rect.width, page.rect.height
        if max(w, h) <= _MAX_SIDE:
            return raw, mime
        scale = _MAX_SIDE / max(w, h)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        return pix.tobytes("png"), "image/png"
    finally:
        doc.close()


def _pdf_pages_to_png(raw: bytes, dpi: int = 200) -> list[bytes]:
    """Rasterize each PDF page to PNG bytes, capping longest side at _MAX_SIDE."""
    import fitz  # lazy

    pages: list[bytes] = []
    doc = fitz.open(stream=raw, filetype="pdf")
    try:
        zoom = dpi / 72.0
        for page in doc:
            w, h = page.rect.width * zoom, page.rect.height * zoom
            longest = max(w, h)
            z = zoom * (_MAX_SIDE / longest) if longest > _MAX_SIDE else zoom
            pix = page.get_pixmap(matrix=fitz.Matrix(z, z))
            pages.append(pix.tobytes("png"))
    finally:
        doc.close()
    return pages


def _image_to_png(raw: bytes) -> bytes:
    """Rasterize a single image to PNG bytes, capping longest side at _MAX_SIDE.

    Always returns real PNG bytes so the downstream `.png` tempfile / data-URL
    mime is correct even when the input was JPEG/WEBP/GIF.
    """
    import fitz  # lazy

    doc = fitz.open(stream=raw, filetype=None)
    try:
        page = doc[0]
        w, h = page.rect.width, page.rect.height
        scale = _MAX_SIDE / max(w, h) if max(w, h) > _MAX_SIDE else 1.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        return pix.tobytes("png")
    finally:
        doc.close()


def _to_page_pngs(raw: bytes) -> list[bytes]:
    """Convert an image or PDF payload into a list of capped PNG page images.

    PDFs (single- or multi-page) are rasterized one PNG per page; images become
    a single-element list. Raises ValueError on empty/invalid input.
    """
    if not raw:
        raise ValueError("empty input")
    try:
        _, is_pdf = _detect(raw)
        if is_pdf:
            pages = _pdf_pages_to_png(raw)
            if not pages:
                raise ValueError("PDF has no pages")
            return pages
        return [_image_to_png(raw)]
    except ValueError:
        raise
    except Exception as e:  # fitz failure on a corrupt payload
        raise ValueError(str(e)) from e


# ---------------------------------------------------------------------------
# Core async caller (shared httpx.AsyncClient across concurrent requests)
# ---------------------------------------------------------------------------

async def _call_ocr(
    client: httpx.AsyncClient,
    img_b64: str,
    mime: str,
    prompt: str | None = None,
) -> tuple[str, int]:
    """POST one base64 image to the OCR model. Returns (text, total_tokens)."""
    key = os.environ.get(OCR_API_KEY_ENV, "EMPTY")
    r = await client.post(
        OCR_BASE_URL.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": OCR_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                    {"type": "text", "text": prompt or _PROMPT},
                ],
            }],
            "max_tokens": 4096,
            "temperature": 0.0,
        },
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    j = r.json()
    text = j["choices"][0]["message"]["content"]
    total = (j.get("usage") or {}).get("total_tokens", 0)
    return text, total


def _strip_json_fences(text: str) -> str:
    """Remove ```json … ``` or ``` … ``` fences that the model may wrap around output."""
    import re
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


async def _call_ocr_json(
    client: httpx.AsyncClient,
    img_b64: str,
    mime: str,
    prompt: str,
) -> tuple[object, int]:
    """Call OCR with a structured prompt; parse and return (parsed_json, total_tokens).
    Returns (None, tokens) if the response is not valid JSON."""
    import json
    text, tok = await _call_ocr(client, img_b64, mime, prompt=prompt)
    try:
        return json.loads(_strip_json_fences(text)), tok
    except (json.JSONDecodeError, ValueError):
        return None, tok


async def _ocr_raw(client: httpx.AsyncClient, raw: bytes) -> tuple[str, int, int]:
    """OCR raw bytes with a shared client. Returns (text, pages, total_tokens)."""
    mime, is_pdf = _detect(raw)
    if is_pdf:
        page_pngs = await asyncio.to_thread(_pdf_pages_to_png, raw)
        results = await asyncio.gather(*[
            _call_ocr(client, base64.b64encode(png).decode(), "image/png")
            for png in page_pngs
        ])
        texts = [f"--- page {i + 1} ---\n{t}" for i, (t, _) in enumerate(results)]
        return "\n\n".join(texts), len(page_pngs), sum(tk for _, tk in results)
    capped, cap_mime = await asyncio.to_thread(_cap_image, raw)
    text, tok = await _call_ocr(client, base64.b64encode(capped).decode(), cap_mime)
    return text, 1, tok


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _strip_data_url(b64: str) -> str:
    if b64.startswith("data:") and "," in b64:
        return b64.split(",", 1)[1]
    return b64


async def run_ocr_async(data_b64: str) -> dict:
    """Async OCR of a single base64 image or PDF.

    Returns {"result": str, "pages": int, "total_output_token": int}.
    """
    try:
        raw = base64.b64decode(data_b64, validate=False)
    except Exception as e:
        raise ValueError(f"invalid base64: {e}") from e
    if not raw:
        raise ValueError("empty input")
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        text, pages, tok = await _ocr_raw(client, raw)
    return {"result": text, "pages": pages, "total_output_token": tok}


def run_ocr(data_b64: str) -> dict:
    """Sync wrapper around run_ocr_async — for legacy / non-async callers."""
    return asyncio.run(run_ocr_async(data_b64))


async def run_ocr_request(header_b64: str, transactions: list[str]) -> dict:
    """Process a full /ocr request concurrently: header + N transaction images.

    Returns {"header": str, "transaction": [str], "total_output_token": int}.
    Strips data-URL prefixes automatically.
    """
    try:
        header_raw = base64.b64decode(_strip_data_url(header_b64), validate=False)
    except Exception as e:
        raise ValueError(f"invalid header base64: {e}") from e

    txn_raws: list[bytes] = []
    for i, b in enumerate(transactions):
        if not b or not b.strip():
            continue
        try:
            txn_raws.append(base64.b64decode(_strip_data_url(b), validate=False))
        except Exception as e:
            raise ValueError(f"invalid transaction[{i}] base64: {e}") from e

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        results = await asyncio.gather(
            _ocr_raw(client, header_raw),
            *[_ocr_raw(client, r) for r in txn_raws],
        )

    header_text, _, header_tok = results[0]
    txn_texts = [text for text, _, _ in results[1:]]
    total_tok = header_tok + sum(tok for _, _, tok in results[1:])

    return {
        "header": header_text,
        "transaction": txn_texts,
        "total_output_token": total_tok,
    }


async def run_ocr_request_structured(header_b64: str, transactions: list[str]) -> dict:
    """Process a bank-statement /ocr request via the two-stage bank_statement pipeline.

    Header and transaction payloads may each be a base64 image OR PDF (data-URL
    prefix optional). PDFs are rasterized to per-page PNGs: the header uses its
    first page, and multi-page transaction PDFs are flattened into the transaction
    page list in page order.

    Returns:
    {
      "account": {"account_number": ..., "owner_branch": ...},
      "opening_balance_row": {"label": ..., "date": ..., "balance_text": ...},
      "transactions": [ {row_index, date, item, debit_text, credit_text,
                          amount_direction, balance_text, details}, ... ],
      "total_output_token": int
    }
    """
    try:
        header_raw = base64.b64decode(_strip_data_url(header_b64), validate=False)
    except Exception as e:
        raise ValueError(f"invalid header base64: {e}") from e

    txn_raws: list[bytes] = []
    for i, b in enumerate(transactions):
        if not b or not b.strip():
            continue
        try:
            txn_raws.append(base64.b64decode(_strip_data_url(b), validate=False))
        except Exception as e:
            raise ValueError(f"invalid transaction[{i}] base64: {e}") from e

    # Convert image/PDF payloads to per-page PNGs before the sync pipeline.
    try:
        header_pages = await asyncio.to_thread(_to_page_pngs, header_raw)
    except ValueError as e:
        raise ValueError(f"invalid header document: {e}") from e

    txn_page_groups = await asyncio.gather(
        *[asyncio.to_thread(_to_page_pngs, r) for r in txn_raws],
        return_exceptions=True,
    )
    txn_page_bytes: list[bytes] = []
    for i, group in enumerate(txn_page_groups):
        if isinstance(group, Exception):
            raise ValueError(f"invalid transaction[{i}] document: {group}") from group
        txn_page_bytes.extend(group)

    ocr_key = os.environ.get(OCR_API_KEY_ENV, "EMPTY")
    llm_key = os.environ.get(LLM_API_KEY_ENV, "EMPTY")
    result = await asyncio.to_thread(
        run_bank_statement_structured,
        header_pages[0],
        txn_page_bytes,
        OCR_MODEL, ocr_key, OCR_BASE_URL,
        LLM_MODEL, llm_key, LLM_BASE_URL,
    )

    result["total_output_token"] = 0
    return result
