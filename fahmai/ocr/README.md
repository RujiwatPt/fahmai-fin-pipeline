# OCR Pipeline — Setup Guide

## Requirements

- Python 3.10+
- Access to an OpenAI-compatible OCR endpoint (e.g. Typhoon OCR)
- Access to an OpenAI-compatible LLM endpoint (e.g. Typhoon LLM)

---

## 1. Create a virtual environment

```bash
cd fahmaifinal
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

---

## 2. Install dependencies

```bash
pip install openai httpx tqdm Pillow
```

| Package | Purpose |
|---------|---------|
| `openai` | OpenAI-compatible API client |
| `httpx` | HTTP client with timeout support |
| `tqdm` | Progress bars |
| `Pillow` | Image resizing (optional but recommended) |

---

## 3. Configure `.env`

Create a `.env` file in the **project root** (`fahmaifinal/`) — one level above the `ocr/` folder:

```env
# ── Engine selection ──────────────────────────────────────
OCR_ENGINE=remote
LLM_ENGINE=remote

# ── Cooldowns (seconds between requests) ─────────────────
REMOTE_OCR_COOLDOWN=0
REMOTE_LLM_COOLDOWN=4

# ── OCR endpoint (vision-capable model) ──────────────────
REMOTE_OCR_BASE_URL=https://<your-ocr-host>/v1
REMOTE_OCR_MODEL=typhoon-ocr
REMOTE_OCR_API_KEY=<your-api-key>

# ── LLM endpoint (text-only model) ───────────────────────
REMOTE_LLM_BASE_URL=https://<your-llm-host>/v1
REMOTE_LLM_MODEL=typhoon-ai/typhoon-s-thaillm-8b-instruct-research-preview
REMOTE_LLM_API_KEY=<your-api-key>

# ── Token / output limits ─────────────────────────────────
# Must be less than the LLM model's total context window
LLM_MAX_TOKENS=1024

# ── Legacy fallback ───────────────────────────────────────
OPENAI_API_KEY=<your-api-key>
```

---

## 4. Configure `ocr/ocr.py`

Edit the `RUN_CONFIG` block at the top of `ocr/ocr.py`:

```python
RUN_CONFIG = {
    # Pipeline: "all" | "bank_statement" | "t3" | "t2"
    "doc_type": "all",

    # Categories to process (empty = all)
    # Options: "bank_statement", "e7_banner", "receipt",
    #          "t2_doc", "t3_doc", "vendor_invoice", "warranty_form"
    "categories": ["bank_statement"],

    # Bank template filter (empty = all banks)
    # Options: "kbank", "scb", "bbl"
    "bank_templates": ["scb", "kbank"],

    # Date range — Thai Buddhist Era year (e.g. 2567 = 2024 CE)
    # Set None to disable that bound
    "date_filter": {
        "year_start": 2567,
        "year_end":   2568,
        "month_start": 1,
        "month_end":  12,
    },
}
```

---

## 5. Prepare input data

Place input files under `ocr/data/`:

```
ocr/data/
├── renders/
│   └── bank_statement/
│       └── 2024-01/
│           ├── BS-KBANK-OPER-2567-01_header.png
│           └── BS-KBANK-OPER-2567-01_transactions_p1.png
└── per_artifact/
    └── bank_statement/
        └── BS-KBANK-OPER-2567-01.json
```

---

## 6. Run

```bash
cd fahmaifinal
python ocr/ocr.py
```

Output JSON files are written to `ocr/data/ocr_output/`.

---

## Pipeline overview

```
Image (.png)
    │
    ▼
[Stage 1 — OCR]  typhoon-ocr   →  raw text / HTML table
    │
    ▼
[Stage 2 — LLM]  typhoon-llm   →  structured JSON fields
    │
    ▼
ocr/data/ocr_output/<artifact_id>.json
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `400 — 0 input tokens available` | `LLM_MAX_TOKENS` equals the model's total context | Lower `LLM_MAX_TOKENS` (e.g. `1024`) |
| `400 — Input length exceeds context` | Image too large for OCR model | Reduce image resolution before processing |
| `404 — model does not exist` | Wrong model name in `.env` | Check `REMOTE_LLM_MODEL` / `REMOTE_OCR_MODEL` |
| `Missing OCR API key` | `.env` not found or key empty | Confirm `.env` is in `fahmaifinal/` root |
