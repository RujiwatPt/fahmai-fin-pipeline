"""Shared configuration, constants, and infrastructure used by all category modules."""
from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("ocr")


def setup_logging(level: int = logging.DEBUG) -> None:
    """Call once from the entry point to configure log format and level."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root = logging.getLogger("ocr")
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def load_env_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


ENV_FILE = (BASE_DIR / "../.env").resolve()
load_env_file(ENV_FILE)


class Config:
    """Centralized configuration management."""

    OCR_ENGINE = os.getenv("OCR_ENGINE", "remote").lower()
    LLM_ENGINE = os.getenv("LLM_ENGINE", "remote").lower()
    DOC_TYPE = os.getenv("OCR_DOC_TYPE", "all").lower()
    DEBUG_LLM = os.getenv("OCR_DEBUG_LLM", "1").strip().lower() in {"1", "true", "yes", "on"}
    REMOTE_OCR_COOLDOWN = float(os.getenv("REMOTE_OCR_COOLDOWN", "2"))
    REMOTE_LLM_COOLDOWN = float(os.getenv("REMOTE_LLM_COOLDOWN", "1"))

    ENDPOINTS = {
        "ocr": {
            "local": {
                "base_url": os.getenv("LOCAL_OCR_BASE_URL") or "http://localhost:8000",
                "model": os.getenv("LOCAL_OCR_MODEL") or "ocr-local",
                "api_key": os.getenv("LOCAL_OCR_API_KEY") or os.getenv("OPENAI_API_KEY"),
            },
            "remote": {
                "base_url": os.getenv("REMOTE_OCR_BASE_URL") or "https://api.openai.com/v1",
                "model": os.getenv("REMOTE_OCR_MODEL") or os.getenv("OPENAI_MODEL_OCR") or "gpt-4-vision",
                "api_key": os.getenv("REMOTE_OCR_API_KEY") or os.getenv("OPENAI_API_KEY"),
            },
        },
        "llm": {
            "local": {
                "base_url": os.getenv("LOCAL_LLM_BASE_URL") or "http://localhost:8000",
                "model": os.getenv("LOCAL_LLM_MODEL") or "llm-local",
                "api_key": os.getenv("LOCAL_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            },
            "remote": {
                "base_url": os.getenv("REMOTE_LLM_BASE_URL") or "https://api.openai.com/v1",
                "model": os.getenv("REMOTE_LLM_MODEL") or os.getenv("OPENAI_MODEL_LLM") or "gpt-3.5-turbo",
                "api_key": os.getenv("REMOTE_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            },
        },
    }

    DOC_TYPE_CONFIG = {
        "t3": {
            "type": "image",
            "label_dir": DATA_DIR / "per_artifact/t3_doc",
            "output_dir": DATA_DIR / "ocr_output/t3_doc",
        },
        "t2": {
            "type": "pdf",
            "pdf_dir": DATA_DIR / "renders/t2_doc",
            "output_dir": DATA_DIR / "pdf_output/t2_doc",
        },
        "bank_statement": {
            "type": "bank_statement",
            "label_dir": DATA_DIR / "per_artifact/bank_statement",
            "output_dir": DATA_DIR / "ocr_output/bank_statement",
        },
        "all": {
            "type": "all",
            "renders_dir": DATA_DIR / "renders",
            "per_artifact_dir": DATA_DIR / "per_artifact",
            "output_dir": DATA_DIR / "ocr_output",
            # Optional comma-separated filter, e.g. OCR_CATEGORIES=bank_statement,receipt
            "categories_filter": [
                c.strip() for c in os.getenv("OCR_CATEGORIES", "").split(",") if c.strip()
            ],
            # Optional bank template filter: kbank, scb, bbl (maps to renderer_template_id prefix)
            # e.g. OCR_BANK_TEMPLATES=kbank,scb
            "bank_templates_filter": [
                t.strip().lower() for t in os.getenv("OCR_BANK_TEMPLATES", "").split(",") if t.strip()
            ],
        },
    }

    @classmethod
    def validate(cls) -> None:
        valid_engines = {"local", "remote"}
        if cls.OCR_ENGINE not in valid_engines:
            logger.critical("Invalid OCR_ENGINE '%s'.", cls.OCR_ENGINE)
            sys.exit(1)
        if cls.LLM_ENGINE not in valid_engines:
            logger.critical("Invalid LLM_ENGINE '%s'.", cls.LLM_ENGINE)
            sys.exit(1)
        if cls.DOC_TYPE not in cls.DOC_TYPE_CONFIG:
            logger.critical(
                "Unknown OCR_DOC_TYPE '%s'. Available: %s",
                cls.DOC_TYPE, ", ".join(cls.DOC_TYPE_CONFIG.keys()),
            )
            sys.exit(1)

    @classmethod
    def get_ocr_config(cls) -> dict[str, str]:
        return cls.ENDPOINTS["ocr"][cls.OCR_ENGINE]

    @classmethod
    def get_ocr_base_url(cls) -> str:
        return cls.get_ocr_config()["base_url"]

    @classmethod
    def get_ocr_model(cls) -> str:
        return cls.get_ocr_config()["model"]

    @classmethod
    def get_ocr_api_key(cls) -> str | None:
        return cls.get_ocr_config()["api_key"]

    @classmethod
    def get_llm_config(cls) -> dict[str, str]:
        return cls.ENDPOINTS["llm"][cls.LLM_ENGINE]

    @classmethod
    def get_llm_base_url(cls) -> str:
        return cls.get_llm_config()["base_url"]

    @classmethod
    def get_llm_model(cls) -> str:
        return cls.get_llm_config()["model"]

    @classmethod
    def get_llm_api_key(cls) -> str | None:
        return cls.get_llm_config()["api_key"]

    @classmethod
    def get_doc_config(cls) -> dict[str, Any]:
        return cls.DOC_TYPE_CONFIG[cls.DOC_TYPE]


Config.validate()

BASE_URL = Config.get_ocr_base_url()
MODEL = Config.get_ocr_model()
API_KEY = Config.get_ocr_api_key()

LLM_BASE_URL = Config.get_llm_base_url()
LLM_MODEL = Config.get_llm_model()
LLM_API_KEY = Config.get_llm_api_key()

DEBUG_LLM = Config.DEBUG_LLM
OCR_ENGINE = Config.OCR_ENGINE
LLM_ENGINE = Config.LLM_ENGINE
DOC_TYPE = Config.DOC_TYPE

OCR_COOLDOWN = Config.REMOTE_OCR_COOLDOWN if OCR_ENGINE == "remote" else 0.0
LLM_COOLDOWN = Config.REMOTE_LLM_COOLDOWN if LLM_ENGINE == "remote" else 0.0

CONFIG = Config.get_doc_config()
LABEL_DIR = CONFIG.get("label_dir")
OUTPUT_DIR = CONFIG["output_dir"]
PDF_RENDER_DIR = CONFIG.get("pdf_dir")


def file_to_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or "application/octet-stream"
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def debug_log_llm(stage: str, payload: str) -> None:
    logger.debug("[%s] %s", stage, payload)


def resolve_data_path(path_value: str) -> Path:
    normalized = path_value.strip().replace("\\", "/").lstrip("/")
    if normalized.startswith("data/"):
        normalized = normalized[len("data/"):]
    return DATA_DIR / normalized


def strip_json_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ``` or ``` ... ```) from LLM output."""
    import re
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def repair_json(text: str) -> str:
    """Fix common LLM JSON errors: trailing commas before } or ]."""
    import re
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def extract_json_objects(text: str) -> list[dict]:
    """Extract every valid JSON object from text using bracket matching.

    Robust fallback when the overall array is malformed — skips corrupt rows
    and returns all rows that parse cleanly.
    """
    import re
    objects: list[dict] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                candidate = text[start:i + 1]
                try:
                    obj = json.loads(repair_json(candidate))
                    if isinstance(obj, dict):
                        objects.append(obj)
                except (json.JSONDecodeError, ValueError):
                    pass
                start = -1
    return objects


def recover_truncated_json_array(text: str) -> str:
    """Recover a truncated JSON array by closing at the last complete object."""
    text = text.strip()
    last_brace = text.rfind("}")
    if last_brace == -1:
        return text
    recovered = text[:last_brace + 1].rstrip().rstrip(",")
    # Ensure the result is wrapped in an array
    if not recovered.lstrip().startswith("["):
        recovered = "[" + recovered
    if not recovered.rstrip().endswith("]"):
        recovered = recovered + "\n]"
    return recovered


LLM_RETRY_DELAY = float(os.getenv("LLM_RETRY_DELAY", "5"))
OCR_BATCH_SIZE = int(os.getenv("OCR_BATCH_SIZE", "6"))          # concurrent pages per artifact
OCR_ARTIFACT_RETRIES = int(os.getenv("OCR_ARTIFACT_RETRIES", "3"))  # re-runs if validation fails
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "10"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8192"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))      # seconds per request
LLM_CONNECT_TIMEOUT = float(os.getenv("LLM_CONNECT_TIMEOUT", "10"))  # seconds to establish connection

# Date range filter (Thai Buddhist Era year; None = open bound)
_to_int_or_none = lambda v: int(v) if v else None
DATE_YEAR_START: int | None = _to_int_or_none(os.getenv("OCR_YEAR_START", ""))
DATE_YEAR_END: int | None = _to_int_or_none(os.getenv("OCR_YEAR_END", ""))
DATE_MONTH_START: int | None = _to_int_or_none(os.getenv("OCR_MONTH_START", ""))
DATE_MONTH_END: int | None = _to_int_or_none(os.getenv("OCR_MONTH_END", ""))


def call_llm_vision(
    image_path: Path,
    user_prompt: str,
    model: str,
    api_key: str,
    base_url: str = None,
    system_prompt: str = None,
    timeout: float = None,
    max_tokens: int = None,
) -> str:
    """Send image + prompt to a multimodal LLM, retrying with exponential backoff up to LLM_MAX_RETRIES times."""
    try:
        import httpx
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: install with `pip install openai httpx`.") from exc

    effective_timeout = httpx.Timeout(
        timeout or LLM_TIMEOUT,
        connect=LLM_CONNECT_TIMEOUT,
    )
    client = OpenAI(api_key=api_key, base_url=base_url or BASE_URL, timeout=effective_timeout)
    image_data_url = file_to_data_url(image_path)

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ],
    })

    last_exc: Exception | None = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        logger.info("LLM attempt %d/%d | model=%s | image=%s", attempt, LLM_MAX_RETRIES, model, image_path.name)
        t0 = time.time()
        try:
            resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens or LLM_MAX_TOKENS)
            elapsed = time.time() - t0
            text = (resp.choices[0].message.content or "").strip()
            logger.info("LLM OK (%.1fs) | %d chars | image=%s", elapsed, len(text), image_path.name)
            return text
        except Exception as exc:
            last_exc = exc
            elapsed = time.time() - t0
            if hasattr(exc, "status_code") and exc.status_code == 400:
                logger.error("LLM bad request (no retry) | %s: %s | image=%s", type(exc).__name__, exc, image_path.name)
                raise
            delay = min(LLM_RETRY_DELAY * (2 ** (attempt - 1)), 60.0)
            logger.warning("LLM FAILED (%.1fs) | %s: %s | retry in %.1fs | image=%s", elapsed, type(exc).__name__, exc, delay, image_path.name)
            if attempt < LLM_MAX_RETRIES:
                time.sleep(delay)
    logger.critical("LLM gave up after %d attempts | image=%s | last error: %s", LLM_MAX_RETRIES, image_path.name, last_exc)
    raise RuntimeError(f"LLM vision call failed after {LLM_MAX_RETRIES} attempts: {last_exc}") from last_exc


def call_llm_text(
    user_prompt: str,
    model: str,
    api_key: str,
    base_url: str = None,
    system_prompt: str = None,
    timeout: float = None,
    max_tokens: int = None,
) -> str:
    """Send text-only prompt to LLM (no image), retrying with exponential backoff."""
    try:
        import httpx
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: install with `pip install openai httpx`.") from exc

    effective_timeout = httpx.Timeout(
        timeout or LLM_TIMEOUT,
        connect=LLM_CONNECT_TIMEOUT,
    )
    client = OpenAI(api_key=api_key, base_url=base_url or LLM_BASE_URL, timeout=effective_timeout)

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    last_exc: Exception | None = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        logger.info("LLM-text attempt %d/%d | model=%s", attempt, LLM_MAX_RETRIES, model)
        t0 = time.time()
        try:
            resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens or LLM_MAX_TOKENS)
            elapsed = time.time() - t0
            text = (resp.choices[0].message.content or "").strip()
            logger.info("LLM-text OK (%.1fs) | %d chars", elapsed, len(text))
            return text
        except Exception as exc:
            last_exc = exc
            elapsed = time.time() - t0
            if hasattr(exc, "status_code") and exc.status_code == 400:
                logger.error("LLM-text bad request (no retry) | %s: %s", type(exc).__name__, exc)
                raise
            delay = min(LLM_RETRY_DELAY * (2 ** (attempt - 1)), 60.0)
            logger.warning("LLM-text FAILED (%.1fs) | %s: %s | retry in %.1fs", elapsed, type(exc).__name__, exc, delay)
            if attempt < LLM_MAX_RETRIES:
                time.sleep(delay)
    logger.critical("LLM-text gave up after %d attempts | last error: %s", LLM_MAX_RETRIES, last_exc)
    raise RuntimeError(f"LLM text call failed after {LLM_MAX_RETRIES} attempts: {last_exc}") from last_exc


def sleep_if_needed(cooldown: float) -> None:
    if cooldown > 0:
        time.sleep(cooldown)


async def _run_with_semaphore(sem: "asyncio.Semaphore", fn, *args, **kwargs):
    import asyncio
    async with sem:
        return await asyncio.to_thread(fn, *args, **kwargs)


def run_batch(tasks: list, batch_size: int = None) -> list:
    """Run a list of (fn, *args) tuples concurrently, capped at batch_size.

    Each task is a callable (no args) that runs in a thread.
    Returns results in the same order as tasks.
    """
    import asyncio

    sem = asyncio.Semaphore(batch_size or OCR_BATCH_SIZE)

    async def _gather():
        return await asyncio.gather(*[
            _run_with_semaphore(sem, fn) for fn in tasks
        ])

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already inside an event loop (e.g. Jupyter) — create a new one
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                return pool.submit(asyncio.run, _gather()).result()
        return loop.run_until_complete(_gather())
    except RuntimeError:
        return asyncio.run(_gather())


async def async_call_llm_vision(
    image_path: Path,
    user_prompt: str,
    model: str,
    api_key: str,
    base_url: str = None,
    system_prompt: str = None,
    timeout: float = None,
    max_tokens: int = None,
) -> str:
    """Async version of call_llm_vision using AsyncOpenAI client."""
    import asyncio
    try:
        import httpx
        from openai import AsyncOpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: install with `pip install openai httpx`.") from exc

    effective_timeout = httpx.Timeout(timeout or LLM_TIMEOUT, connect=LLM_CONNECT_TIMEOUT)
    client = AsyncOpenAI(api_key=api_key, base_url=base_url or BASE_URL, timeout=effective_timeout)
    image_data_url = file_to_data_url(image_path)

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ],
    })

    last_exc: Exception | None = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        logger.info("LLM attempt %d/%d | model=%s | image=%s", attempt, LLM_MAX_RETRIES, model, image_path.name)
        t0 = time.time()
        try:
            resp = await client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens or LLM_MAX_TOKENS)
            elapsed = time.time() - t0
            text = (resp.choices[0].message.content or "").strip()
            logger.info("LLM OK (%.1fs) | %d chars | image=%s", elapsed, len(text), image_path.name)
            return text
        except Exception as exc:
            last_exc = exc
            elapsed = time.time() - t0
            if hasattr(exc, "status_code") and exc.status_code == 400:
                logger.error("LLM bad request (no retry) | %s: %s | image=%s", type(exc).__name__, exc, image_path.name)
                raise
            delay = min(LLM_RETRY_DELAY * (2 ** (attempt - 1)), 60.0)
            logger.warning("LLM FAILED (%.1fs) | %s: %s | retry in %.1fs | image=%s", elapsed, type(exc).__name__, exc, delay, image_path.name)
            if attempt < LLM_MAX_RETRIES:
                await asyncio.sleep(delay)
    logger.critical("LLM gave up after %d attempts | image=%s | last error: %s", LLM_MAX_RETRIES, image_path.name, last_exc)
    raise RuntimeError(f"LLM vision call failed after {LLM_MAX_RETRIES} attempts: {last_exc}") from last_exc
