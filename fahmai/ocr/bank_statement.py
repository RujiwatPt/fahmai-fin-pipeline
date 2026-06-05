"""Bank statement pipeline — prompts, rule-based extraction, vision extraction, and folder processor."""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import tempfile

from .shared import (
    MODEL as OCR_MODEL, API_KEY as OCR_API_KEY, BASE_URL as OCR_BASE_URL, OCR_COOLDOWN,
    LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, LLM_COOLDOWN, OCR_BATCH_SIZE, OCR_ARTIFACT_RETRIES,
    DATE_YEAR_START, DATE_YEAR_END, DATE_MONTH_START, DATE_MONTH_END,
    debug_log_llm, resolve_data_path, call_llm_vision, call_llm_text, strip_json_fences,
    recover_truncated_json_array, repair_json, extract_json_objects, run_batch, logger,
)


# ---------------------------------------------------------------------------
# OCR prompts (kept for reference; main pipeline now uses vision directly)
# ---------------------------------------------------------------------------

def build_ocr_header_prompt(template_id: str) -> str:
    if template_id == "bank_statement_scb":
        layout = (
            "This is an SCB (ไทยพาณิชย์) bank statement header page.\n"
            "Purple header bar at top with 'SCB ไทยพาณิชย์' logo.\n"
            "The page contains labeled fields:\n"
            "  ชื่อบัญชี / Account Name\n"
            "  เลขที่บัญชี / Account No.  → the account number (e.g. 002-1-XXXXX-2)\n"
            "  ประเภทบัญชี / Account Type  → e.g. Savings\n"
            "  สกุลเงิน / Currency  → e.g. THB\n"
        )
    elif template_id == "bank_statement_kbank":
        layout = (
            "This is a KBANK (กสิกรไทย) bank statement header page.\n"
            "Logo 'ธนาคารกสิกรไทย KASIKORNBANK' top-right.\n"
            "A box on the right contains:\n"
            "  เลขที่อ้างอิง  → reference code (e.g. KBANK-OPER-20250901)\n"
            "  เลขที่บัญชีเงินฝาก  → account number (e.g. 001-1-00001-1)\n"
            "  ระยะเวลา  → statement period\n"
            "  สาขาจำหน่าย  → branch\n"
        )
    elif template_id == "bank_statement_bbl":
        layout = (
            "This is a BBL (Bangkok Bank / ธนาคารกรุงเทพ) bank statement header page.\n"
            "Blue logo 'Bangkok Bank ธนาคารกรุงเทพ' top-left.\n"
            "Note: the account number is NOT on this cover page — it appears on the transaction page.\n"
        )
    else:
        layout = f"This is a {template_id} bank statement header page with labeled account fields.\n"

    return (
        f"{layout}\n"
        "Extract all labeled field values from this page.\n"
        "Output each field as 'Label: Value' on its own line, exactly as printed.\n\n"
        "Rules:\n"
        "- One 'Label: Value' pair per line.\n"
        "- Do NOT include addresses or long paragraphs — only labeled fields.\n"
        'Return JSON only with exactly this key: ["full_text"]\n'
        '- "full_text": all label-value pairs as a string, separated by \\n\n'
        "- No explanation, only JSON."
    )


def build_ocr_table_prompt(template_id: str) -> str:
    if template_id == "bank_statement_scb":
        col_names = "วัน/เวลา, รายการ, ช่องทาง, เลขที่เช็ค, Withdrawal(Debit), Deposit(Credit), Balance, Description"
    elif template_id == "bank_statement_kbank":
        col_names = "วันที่, เวลา, รายการ, Amount(THB), Balance(THB), ช่องทาง, รายละเอียด"
    elif template_id == "bank_statement_bbl":
        col_names = "วันที่, รายละเอียด, Chq.No., Withdrawal, Deposit, Balance, Via"
    else:
        col_names = "Date, Description, Withdrawal, Deposit, Balance"

    return (
        "This image is a bank statement transaction table.\n"
        f"The table has these columns (in order): {col_names}\n\n"
        "Extract ALL rows from the table and return as a valid HTML table.\n\n"
        "Rules:\n"
        f"- The table has exactly these columns: {col_names}\n"
        "- Output a complete <table> with <thead> and <tbody>.\n"
        "- One <tr> per row — do NOT skip or merge any rows.\n"
        "- CRITICAL: One <td> per column in EVERY row — NEVER use colspan or rowspan.\n"
        "- Each <td> must contain only the value for that specific column.\n"
        "- If a cell is empty for that row, use empty <td></td>.\n"
        "- Keep every cell value exactly as printed — no reformatting.\n"
        'Return JSON only with exactly this key: ["full_text"]\n'
        '- "full_text": the complete HTML table as a string\n'
        "- No explanation, only JSON."
    )


# ---------------------------------------------------------------------------
# HTML table utilities
# ---------------------------------------------------------------------------

def extract_html_table_block(html: str) -> str:
    m = re.search(r"<table[^>]*>.*?</table>", (html or ""), re.DOTALL | re.IGNORECASE)
    return m.group(0) if m else ""


def parse_html_table_to_markdown(html: str) -> str:
    html = (html or "").strip()
    if not html:
        return ""

    def clean_cell(raw: str) -> str:
        text = re.sub(r"<[^>]+>", " ", raw)
        return re.sub(r"\s+", " ", text).strip()

    row_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
    cell_pattern = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)

    md_rows: list[str] = []
    for row_match in row_pattern.finditer(html):
        cells = [clean_cell(m.group(1)) for m in cell_pattern.finditer(row_match.group(1))]
        if not any(cells):
            continue
        md_rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(md_rows)


# ---------------------------------------------------------------------------
# Vision prompts
# ---------------------------------------------------------------------------

def build_llm_vision_header_prompt(visible_fields: list[str], template_id: str) -> tuple[str, str]:
    def _field(n: int, name: str, rule: str) -> str:
        return f"{n}. {name}: {rule}\n" if name in visible_fields else ""

    blank_output = "{\n" + "".join(f'  "{f}": null,\n' for f in visible_fields).rstrip(",\n") + "\n}"
    fields_list = json.dumps(visible_fields, ensure_ascii=False)

    if template_id == "bank_statement_scb":
        layout = (
            "You are reading an SCB (ไทยพาณิชย์) bank statement header page.\n"
            "Visual layout: Purple header bar at top, 'SCB ไทยพาณิชย์' logo top-right, "
            "title 'รายการเดินบัญชีย้อนหลัง'. Labeled fields appear in the body.\n\n"
        )
        field_rules = (
            "FIELD RULES:\n"
            + _field(1, "account_id", "Not on this page — always null.")
            + _field(2, "bank", "Always 'ไทยพาณิชย์'.")
            + _field(3, "account_number", "After 'เลขที่บัญชี:' — keep dashes (e.g. '002-1-00002-2').")
            + _field(4, "account_role", "After 'ประเภทบัญชี:' — e.g. 'Savings' or 'ออมทรัพย์'.")
            + _field(5, "currency", "After 'สกุลเงิน:' — e.g. 'THB'.")
        )
        few_shot = json.dumps(
            {f: (None if f == "account_id" else {"bank": "ไทยพาณิชย์", "account_number": "002-1-00002-2", "account_role": "Savings", "currency": "THB"}.get(f))
             for f in visible_fields}, ensure_ascii=False, indent=2)
        few_shot_block = f"EXAMPLE output for a typical SCB header:\n{few_shot}\n\n"

    elif template_id == "bank_statement_kbank":
        layout = (
            "You are reading a KBANK (กสิกรไทย) bank statement header page.\n"
            "Visual layout: Logo 'ธนาคารกสิกรไทย KASIKORNBANK' top-right. "
            "A box on the right contains: เลขที่อ้างอิง, เลขที่บัญชีเงินฝาก, ระยะเวลา, สาขาจำหน่าย.\n\n"
        )
        field_rules = (
            "FIELD RULES:\n"
            + _field(1, "account_id", "After 'เลขที่อ้างอิง:' — strip date suffix (e.g. 'KBANK-OPER-20250901' → 'KBANK-OPER').")
            + _field(2, "bank", "Always 'กสิกรไทย'.")
            + _field(3, "account_number", "After 'เลขที่บัญชีเงินฝาก:' — keep dashes (e.g. '001-1-00001-1').")
            + _field(4, "account_role", "From document title: 'ออมทรัพย์' or 'กระแสรายวัน'.")
            + _field(5, "currency", "Default 'THB' if not shown.")
        )
        few_shot = json.dumps(
            {f: {"account_id": "KBANK-OPER", "bank": "กสิกรไทย", "account_number": "001-1-00001-1", "account_role": "ออมทรัพย์", "currency": "THB"}.get(f)
             for f in visible_fields}, ensure_ascii=False, indent=2)
        few_shot_block = f"EXAMPLE output for a typical KBANK header:\n{few_shot}\n\n"

    elif template_id == "bank_statement_bbl":
        layout = (
            "You are reading a BBL (Bangkok Bank / ธนาคารกรุงเทพ) bank statement header page.\n"
            "Visual layout: Blue logo top-left, account holder name/address on left, "
            "statement period and date on right.\n\n"
        )
        field_rules = (
            "FIELD RULES:\n"
            + _field(1, "account_id", "Not on this page — always null.")
            + _field(2, "bank", "Always 'กรุงเทพ'.")
            + _field(3, "account_number", "Look for เลขที่บัญชี / Account No. — keep dashes. null if not found.")
            + _field(4, "account_role", "Look for ประเภทบัญชี / account type → 'ออมทรัพย์' / 'Savings' / 'กระแสรายวัน'. null if not found.")
            + _field(5, "currency", "Look for สกุลเงิน / Currency. Default 'THB' if bank is confirmed but field not shown.")
        )
        few_shot = json.dumps(
            {f: {"account_id": None, "bank": "กรุงเทพ", "account_number": "001-1-00001-1", "account_role": "ออมทรัพย์", "currency": "THB"}.get(f)
             for f in visible_fields}, ensure_ascii=False, indent=2)
        few_shot_block = f"EXAMPLE output for a typical BBL header:\n{few_shot}\n\n"

    else:
        layout = f"You are reading a {template_id.upper()} bank statement header page.\n\n"
        field_rules = (
            "FIELD RULES:\n"
            + _field(1, "account_id", "Reference code on statement.")
            + _field(2, "bank", "Bank name from logo.")
            + _field(3, "account_number", "After 'เลขที่บัญชี' or 'Account No.' — keep dashes.")
            + _field(4, "account_role", "'ออมทรัพย์' / 'กระแสรายวัน' / 'Savings' / 'Current'.")
            + _field(5, "currency", "e.g. 'THB'.")
        )
        few_shot_block = ""

    system_prompt = (
        f"{layout}"
        f"Extract ONLY these fields: {fields_list}\n\n"
        f"{field_rules}\n"
        f"{few_shot_block}"
        "OUTPUT RULES:\n"
        "- Return ONLY valid JSON with exactly the fields listed — no extra keys.\n"
        "- Values must be SHORT strings — no addresses.\n"
        "- account_number: preserve dashes exactly.\n"
        "- Use null for any field not found.\n"
        f"JSON shape:\n{blank_output}"
    )
    user_prompt = (
        f"Look at this bank statement header image and extract these fields: {fields_list}\n"
        f"Return JSON with exactly these keys:\n{blank_output}"
    )
    return system_prompt, user_prompt


def build_llm_vision_transaction_prompt(visible_fields: list[str], template_id: str) -> tuple[str, str]:
    non_account_fields = [f for f in visible_fields if f != "account_id"]
    blank_row = "{\n" + "".join(f'    "{f}": null,\n' for f in non_account_fields).rstrip(",\n") + "\n  }"
    row_fields_list = json.dumps(non_account_fields, ensure_ascii=False)

    if template_id == "bank_statement_scb":
        layout = (
            "You are reading an SCB (ไทยพาณิชย์) bank statement transaction page.\n"
            "The table has these columns (left→right): "
            "วัน/เวลา | รายการ | ช่องทาง | เลขที่เช็ค | Withdrawal(Debit) | Deposit(Credit) | Balance | Description\n\n"
        )
        field_rules = (
            "FIELD RULES per row:\n"
            + ("- business_event_date: 'วัน/เวลา' column (e.g. '05-09-25').\n" if "business_event_date" in non_account_fields else "")
            + ("- transaction_type: Withdrawal filled → 'debit'; Deposit filled → 'credit'.\n" if "transaction_type" in non_account_fields else "")
            + ("- amount_thb: Withdrawal or Deposit value — plain positive number, no commas.\n" if "amount_thb" in non_account_fields else "")
            + ("- balance_after_thb: Balance column — plain number, no commas.\n" if "balance_after_thb" in non_account_fields else "")
            + ("- description: Description + Channel combined.\n" if "description" in non_account_fields else "")
        )
        few_shot_output = json.dumps([
            {"business_event_date": "03-09-25", "transaction_type": "credit", "amount_thb": 120000.0, "balance_after_thb": 5120000.0, "description": "VENDOR-PAY-001 K PLUS"},
            {"business_event_date": "05-09-25", "transaction_type": "debit",  "amount_thb": 300000.0, "balance_after_thb": 4820000.0, "description": "KBANK-OPER transfer K PLUS"},
            {"business_event_date": "10-09-25", "transaction_type": "credit", "amount_thb": 50000.0,  "balance_after_thb": 4870000.0, "description": "CUST-PAY-002 K PLUS"},
        ], ensure_ascii=False, indent=2)

    elif template_id == "bank_statement_kbank":
        layout = (
            "You are reading a KBANK (กสิกรไทย) bank statement transaction page.\n"
            "The table has these columns (left→right): "
            "วันที่ | เวลา | รายการ | Amount(THB) | Balance(THB) | ช่องทาง | รายละเอียด\n\n"
        )
        field_rules = (
            "FIELD RULES per row:\n"
            + ("- business_event_date: 'วันที่' + 'เวลา' combined (e.g. '01-09-25 09:00').\n" if "business_event_date" in non_account_fields else "")
            + ("- transaction_type: 'รับโอนเงิน'/'ฝากเงิน' → 'credit'; 'โอนเงิน'/'ถอนเงิน' → 'debit'.\n" if "transaction_type" in non_account_fields else "")
            + ("- amount_thb: Amount column — plain positive number, no commas.\n" if "amount_thb" in non_account_fields else "")
            + ("- balance_after_thb: Balance column — plain number, no commas.\n" if "balance_after_thb" in non_account_fields else "")
            + ("- description: รายละเอียด + ช่องทาง combined.\n" if "description" in non_account_fields else "")
        )
        few_shot_output = json.dumps([
            {"business_event_date": "01-09-25 09:00", "transaction_type": "credit", "amount_thb": 80600.0,  "balance_after_thb": 10080600.0, "description": "CUST-L3-B2B-015800 TXN-001 K PLUS"},
            {"business_event_date": "02-09-25 10:30", "transaction_type": "debit",  "amount_thb": 500000.0, "balance_after_thb": 9719300.0,  "description": "VENDOR-PAY-A K PLUS"},
            {"business_event_date": "03-09-25 14:00", "transaction_type": "credit", "amount_thb": 250000.0, "balance_after_thb": 9969300.0,  "description": "CUST-PAY-B K PLUS"},
        ], ensure_ascii=False, indent=2)

    elif template_id == "bank_statement_bbl":
        layout = (
            "You are reading a BBL (Bangkok Bank / ธนาคารกรุงเทพ) bank statement transaction page.\n"
            "The table has these columns (left→right): "
            "วันที่/Date | รายละเอียด/Particulars | Chq.No. | Withdrawal | Deposit | Balance | Via\n\n"
        )
        field_rules = (
            "FIELD RULES per row:\n"
            + ("- business_event_date: Date column (e.g. '05-09-25').\n" if "business_event_date" in non_account_fields else "")
            + ("- transaction_type: Withdrawal filled → 'debit'; Deposit filled → 'credit'.\n" if "transaction_type" in non_account_fields else "")
            + ("- amount_thb: Withdrawal or Deposit value — plain positive number, no commas.\n" if "amount_thb" in non_account_fields else "")
            + ("- balance_after_thb: Balance column — plain number, no commas.\n" if "balance_after_thb" in non_account_fields else "")
            + ("- description: Particulars + Via combined (e.g. 'โอนเงิน K PLUS').\n" if "description" in non_account_fields else "")
        )
        few_shot_output = json.dumps([
            {"business_event_date": "03-09-25", "transaction_type": "credit", "amount_thb": 150000.0, "balance_after_thb": 3150000.0, "description": "รับโอนเงิน K PLUS"},
            {"business_event_date": "05-09-25", "transaction_type": "debit",  "amount_thb": 500000.0, "balance_after_thb": 2650000.0, "description": "โอนเงิน K PLUS"},
            {"business_event_date": "10-09-25", "transaction_type": "credit", "amount_thb": 80000.0,  "balance_after_thb": 2730000.0, "description": "รับโอนเงิน K PLUS"},
        ], ensure_ascii=False, indent=2)

    else:
        layout = f"You are reading a {template_id.upper()} bank statement transaction page.\n\n"
        field_rules = (
            "FIELD RULES per row:\n"
            + ("- business_event_date: Transaction date.\n" if "business_event_date" in non_account_fields else "")
            + ("- transaction_type: 'debit' (withdrawal) or 'credit' (deposit).\n" if "transaction_type" in non_account_fields else "")
            + ("- amount_thb: Plain positive number, no commas.\n" if "amount_thb" in non_account_fields else "")
            + ("- balance_after_thb: Plain number, no commas.\n" if "balance_after_thb" in non_account_fields else "")
            + ("- description: Transaction memo.\n" if "description" in non_account_fields else "")
        )
        few_shot_output = ""

    few_shot_block = f"EXAMPLE output:\n{few_shot_output}\n\n" if few_shot_output else ""
    blank_array = f"[\n  {blank_row},\n  {blank_row}\n]"

    system_prompt = (
        f"{layout}"
        f"Extract ONLY these fields per row: {row_fields_list}\n\n"
        f"{field_rules}\n"
        "SKIP: 'ยอดยกมา' (opening balance), 'ยอดยกไป' (closing balance), and summary/total rows.\n\n"
        f"{few_shot_block}"
        "OUTPUT RULES:\n"
        "- IMPORTANT: Extract EVERY real transaction row — a page may have 1 to 50+ rows.\n"
        "- Do NOT stop early. Do NOT summarise. Output one JSON object per row.\n"
        f"- Each object must have exactly these keys: {row_fields_list}\n"
        "- amount_thb and balance_after_thb must be plain numbers (e.g. 300000.0) — no commas.\n"
        "- Use null for any field not visible in a row.\n"
        "- OUTPUT MUST be a JSON array [...], even if there is only one row.\n"
        f"Array shape (one object per row):\n{blank_array}"
    )
    user_prompt = (
        f"Extract ALL real transaction rows from this bank statement image.\n"
        f"Return ONLY a JSON array. Each element must have exactly these keys: {row_fields_list}\n"
        f"Do NOT return an empty array. If you see rows in the table, include them.\n"
        f"Output the JSON array only — no explanation, no markdown fences."
    )
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Text-based LLM prompts (Stage 2 of the two-stage OCR → LLM pipeline)
# ---------------------------------------------------------------------------

def build_llm_text_header_prompt(ocr_text: str, visible_fields: list[str], template_id: str) -> tuple[str, str]:
    base_system, _ = build_llm_vision_header_prompt(visible_fields, template_id)
    fields_list = json.dumps(visible_fields, ensure_ascii=False)

    # Template-specific reasoning steps
    if template_id == "bank_statement_kbank":
        reasoning_hint = (
            "Scan the OCR text step by step:\n"
            "  1. account_number — find เลขที่บัญชีเงินฝาก → keep dashes (e.g. 001-1-00001-1)\n"
            "  2. account_role   — find document title: ออมทรัพย์ or กระแสรายวัน\n"
            "  3. currency       — look for สกุลเงิน or default THB\n"
            "  4. bank           — always กสิกรไทย\n"
            "  5. account_id     — find เลขที่อ้างอิง, strip date suffix (e.g. KBANK-OPER-20250901 → KBANK-OPER)\n"
        )
    elif template_id == "bank_statement_scb":
        reasoning_hint = (
            "Scan the OCR text step by step:\n"
            "  1. account_number — find เลขที่บัญชี → keep dashes (e.g. 002-1-00002-2)\n"
            "  2. account_role   — find ประเภทบัญชี → e.g. Savings / ออมทรัพย์\n"
            "  3. currency       — find สกุลเงิน → usually THB\n"
            "  4. bank           — always ไทยพาณิชย์\n"
            "  5. account_id     — not on this page, always null\n"
        )
    elif template_id == "bank_statement_bbl":
        reasoning_hint = (
            "Scan the OCR text step by step:\n"
            "  1. account_number — find เลขที่บัญชี / Account No → keep dashes; null if absent\n"
            "  2. account_role   — find ประเภทบัญชี / account type → ออมทรัพย์ / Savings; null if absent\n"
            "  3. currency       — find สกุลเงิน / Currency → default THB if bank confirmed\n"
            "  4. bank           — always กรุงเทพ\n"
            "  5. account_id     — not on this page, always null\n"
        )
    else:
        reasoning_hint = (
            "Scan the OCR text step by step:\n"
            "  1. account_number — find เลขที่บัญชี / Account No → keep dashes exactly\n"
            "  2. account_role   — find account type → ออมทรัพย์ / กระแสรายวัน / Savings / Current\n"
            "  3. currency       — find สกุลเงิน / Currency → usually THB\n"
            "  4. bank           — identify bank name\n"
            "  5. account_id     — find reference code\n"
        )
    reasoning_hint += "Write your findings concisely in _reasoning, then fill the other fields."
    blank_output = (
        '{\n'
        '  "_reasoning": "<your step-by-step findings here>",\n'
        + "".join(f'  "{f}": null,\n' for f in visible_fields).rstrip(",\n") + "\n}"
    )

    system_prompt = (
        f"{base_system}\n\n"
        f"REASONING INSTRUCTIONS:\n{reasoning_hint}\n\n"
        "OUTPUT: Return a single JSON object. Put _reasoning first, then the data fields.\n"
        "Do NOT include _reasoning in any other key. Output JSON only — no markdown fences."
    )
    user_prompt = (
        f"Here is the OCR-extracted text from a bank statement header page:\n\n"
        f"{ocr_text}\n\n"
        f"Fields to extract: {fields_list}\n\n"
        f"Think step by step (account_number → account_role → currency → bank → account_id), "
        f"write your reasoning in _reasoning, then return JSON:\n{blank_output}"
    )
    return system_prompt, user_prompt


def build_llm_text_transaction_prompt(ocr_text: str, visible_fields: list[str], template_id: str) -> tuple[str, str]:
    non_account_fields = [f for f in visible_fields if f != "account_id"]
    row_fields_list = json.dumps(non_account_fields, ensure_ascii=False)
    base_system, _ = build_llm_vision_transaction_prompt(visible_fields, template_id)

    reasoning_hint = (
        "For EACH row, reason through:\n"
        "  1. business_event_date — read the date (and time if present) exactly as printed\n"
        "  2. transaction_type    — if Withdrawal/Debit column has a value → 'debit'; "
        "if Deposit/Credit column has a value → 'credit'\n"
        "  3. amount_thb          — take the non-empty amount column; strip commas; positive number\n"
        "  4. balance_after_thb   — take the Balance column value; strip commas\n"
        "  5. description         — combine channel + memo columns\n"
        "SKIP opening/closing balance rows (ยอดยกมา / ยอดยกไป / Opening Balance / Closing Balance)."
    )

    system_prompt = (
        f"{base_system}\n\n"
        f"REASONING INSTRUCTIONS:\n{reasoning_hint}\n\n"
        "OUTPUT: Return ONLY a valid JSON array — no extra keys, no markdown fences."
    )
    user_prompt = (
        f"Here is the OCR-extracted table from a bank statement transaction page:\n\n"
        f"{ocr_text}\n\n"
        f"For each row: reason through transaction_type (debit vs credit) and amount before writing the value.\n"
        f"Return ONLY a JSON array. Each element must have exactly these keys: {row_fields_list}\n"
        f"Do NOT return an empty array. Output JSON only — no explanation, no markdown fences."
    )
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Vision extractors
# ---------------------------------------------------------------------------

def extract_bank_statement_header_vision(
    image_path: Path,
    visible_fields: list[str],
    template_id: str,
    model: str,
    api_key: str,
    base_url: str = None,
) -> dict[str, Any] | None:
    from shared import LLM_MAX_RETRIES, LLM_RETRY_DELAY

    # Stage 1: OCR — extract raw text from image
    ocr_prompt = build_ocr_header_prompt(template_id)
    logger.info("Header OCR start | image=%s | template=%s | fields=%s", image_path.name, template_id, visible_fields)
    try:
        ocr_raw = call_llm_vision(
            image_path=image_path, user_prompt=ocr_prompt,
            model=OCR_MODEL, api_key=OCR_API_KEY, base_url=OCR_BASE_URL,
        )
        if OCR_COOLDOWN > 0:
            time.sleep(OCR_COOLDOWN)
        debug_log_llm("bs_header_ocr", ocr_raw)
        try:
            ocr_json = json.loads(strip_json_fences(ocr_raw))
            ocr_text = (ocr_json.get("full_text") or ocr_raw) if isinstance(ocr_json, dict) else ocr_raw
        except (json.JSONDecodeError, ValueError):
            ocr_text = ocr_raw
    except Exception as exc:
        logger.error("Header OCR failed | image=%s | %s: %s", image_path.name, type(exc).__name__, exc)
        return None

    # Stage 2: LLM — structure the OCR text into JSON fields
    system_prompt, user_prompt = build_llm_text_header_prompt(ocr_text, visible_fields, template_id)
    logger.info("Header LLM start | image=%s", image_path.name)
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            raw_text = call_llm_text(
                user_prompt=user_prompt, system_prompt=system_prompt,
                model=model, api_key=api_key, base_url=base_url,
            )
            if LLM_COOLDOWN > 0:
                time.sleep(LLM_COOLDOWN)
            debug_log_llm("bs_header_llm", raw_text)
            if not raw_text:
                delay = min(LLM_RETRY_DELAY * (2 ** (attempt - 1)), 60.0)
                logger.warning("Header empty response | attempt=%d/%d | image=%s | retry in %.1fs", attempt, LLM_MAX_RETRIES, image_path.name, delay)
                if attempt < LLM_MAX_RETRIES:
                    time.sleep(delay)
                continue
            clean = strip_json_fences(raw_text)
            try:
                parsed = json.loads(repair_json(clean))
            except json.JSONDecodeError:
                try:
                    recovered = recover_truncated_json_array(repair_json(clean))
                    parsed = json.loads(recovered)
                    debug_log_llm("bs_header_llm_recover", "truncation recovery succeeded")
                except json.JSONDecodeError:
                    objs = extract_json_objects(clean)
                    parsed = objs[0] if objs else {}
                    debug_log_llm("bs_header_llm_obj_extract", f"extracted {len(objs)} object(s)")
            if isinstance(parsed, dict):
                result = {f: parsed.get(f) for f in visible_fields}
                logger.info("Header extracted | image=%s | %s", image_path.name, {k: v for k, v in result.items() if v is not None})
                return result
            logger.warning("Header bad shape | expected dict got %s | image=%s", type(parsed).__name__, image_path.name)
            debug_log_llm("bs_header_llm_bad_shape", f"expected dict, got {type(parsed).__name__}")
        except json.JSONDecodeError as exc:
            debug_log_llm("bs_header_llm_json_error", str(exc))
            logger.error("Header JSON parse error | attempt=%d/%d | image=%s | %s", attempt, LLM_MAX_RETRIES, image_path.name, exc)
        except Exception as exc:
            debug_log_llm("bs_header_llm_error", f"{type(exc).__name__}: {exc}")
            logger.error("Header LLM error (no retry) | image=%s | %s: %s", image_path.name, type(exc).__name__, exc)
            return None

        delay = min(LLM_RETRY_DELAY * (2 ** (attempt - 1)), 60.0)
        if attempt < LLM_MAX_RETRIES:
            time.sleep(delay)

    logger.critical("Header gave up after %d attempts | image=%s", LLM_MAX_RETRIES, image_path.name)
    return None


def extract_bank_statement_transactions_vision(
    image_path: Path,
    visible_fields: list[str],
    template_id: str,
    account_id: str | None,
    model: str,
    api_key: str,
    base_url: str = None,
) -> list[dict[str, Any]]:
    from shared import LLM_MAX_RETRIES, LLM_RETRY_DELAY
    non_account_fields = [f for f in visible_fields if f != "account_id"]

    # Stage 1: OCR — extract raw table text from image
    ocr_prompt = build_ocr_table_prompt(template_id)
    logger.info("Transaction OCR start | image=%s | template=%s", image_path.name, template_id)
    try:
        ocr_raw = call_llm_vision(
            image_path=image_path, user_prompt=ocr_prompt,
            model=OCR_MODEL, api_key=OCR_API_KEY, base_url=OCR_BASE_URL,
        )
        if OCR_COOLDOWN > 0:
            time.sleep(OCR_COOLDOWN)
        debug_log_llm("bs_tx_ocr", ocr_raw)
        try:
            ocr_json = json.loads(strip_json_fences(ocr_raw))
            ocr_text = (ocr_json.get("full_text") or ocr_raw) if isinstance(ocr_json, dict) else ocr_raw
        except (json.JSONDecodeError, ValueError):
            ocr_text = ocr_raw
    except Exception as exc:
        logger.error("Transaction OCR failed | image=%s | %s: %s", image_path.name, type(exc).__name__, exc)
        return []

    # Stage 2: LLM — structure the OCR text into transaction rows
    system_prompt, user_prompt = build_llm_text_transaction_prompt(ocr_text, visible_fields, template_id)
    logger.info("Transaction LLM start | image=%s", image_path.name)
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            raw_text = call_llm_text(
                user_prompt=user_prompt, system_prompt=system_prompt,
                model=model, api_key=api_key, base_url=base_url,
            )
            if LLM_COOLDOWN > 0:
                time.sleep(LLM_COOLDOWN)
            debug_log_llm("bs_tx_llm", raw_text)
            clean = strip_json_fences(raw_text)
            try:
                parsed = json.loads(repair_json(clean))
            except json.JSONDecodeError:
                try:
                    logger.warning("Transaction JSON malformed, attempting truncation recovery | image=%s", image_path.name)
                    recovered = recover_truncated_json_array(repair_json(clean))
                    debug_log_llm("bs_tx_llm_recover", recovered[-200:])
                    parsed = json.loads(recovered)
                    logger.info("Transaction JSON recovered via truncation fix | image=%s", image_path.name)
                except json.JSONDecodeError:
                    logger.warning("Transaction JSON unrecoverable, extracting objects individually | image=%s", image_path.name)
                    parsed = extract_json_objects(clean)
                    logger.info("Transaction JSON extracted %d objects individually | image=%s", len(parsed), image_path.name)
            if isinstance(parsed, dict):
                logger.warning("Transaction LLM returned dict, wrapping in list | image=%s", image_path.name)
                debug_log_llm("bs_tx_llm_wrap", "LLM returned dict, wrapping in list")
                parsed = [parsed]
            if isinstance(parsed, list):
                rows = []
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    row: dict[str, Any] = {}
                    if "account_id" in visible_fields:
                        row["account_id"] = account_id
                    for f in non_account_fields:
                        row[f] = item.get(f)
                    rows.append(row)
                if rows:
                    logger.info("Transaction extracted %d rows | image=%s", len(rows), image_path.name)
                    return rows
                delay = min(LLM_RETRY_DELAY * (2 ** (attempt - 1)), 60.0)
                logger.warning("Transaction empty rows | attempt=%d/%d | image=%s | retry in %.1fs", attempt, LLM_MAX_RETRIES, image_path.name, delay)
                if attempt < LLM_MAX_RETRIES:
                    time.sleep(delay)
                continue
            logger.warning("Transaction bad shape | expected list got %s | image=%s", type(parsed).__name__, image_path.name)
            debug_log_llm("bs_tx_llm_bad_shape", f"expected list, got {type(parsed).__name__}")
        except json.JSONDecodeError as exc:
            debug_log_llm("bs_tx_llm_json_error", str(exc))
            logger.error("Transaction JSON unrecoverable | attempt=%d/%d | image=%s | %s", attempt, LLM_MAX_RETRIES, image_path.name, exc)
        except Exception as exc:
            debug_log_llm("bs_tx_llm_error", f"{type(exc).__name__}: {exc}")
            logger.error("Transaction LLM error (no retry) | image=%s | %s: %s", image_path.name, type(exc).__name__, exc)
            return []

        delay = min(LLM_RETRY_DELAY * (2 ** (attempt - 1)), 60.0)
        if attempt < LLM_MAX_RETRIES:
            time.sleep(delay)

    logger.critical("Transaction gave up after %d attempts | image=%s", LLM_MAX_RETRIES, image_path.name)
    return []


# ---------------------------------------------------------------------------
# Rule-based extraction (fallback / reference)
# ---------------------------------------------------------------------------

_SKIP_ROW = re.compile(
    r"ยอดยกมา|ยอดยกไป|Opening Balance|Closing Balance"
    r"|Total Debit|Total Credit|จำนวนเดบิต|จำนวนเครดิต"
    r"|รวมเดบิต|รวมเครดิต|ยอดรวม",
    re.IGNORECASE,
)
_DEBIT_TYPE = re.compile(r"โอนเงิน|ถอนเงิน|debit|withdrawal", re.IGNORECASE)
_CREDIT_TYPE = re.compile(r"รับโอนเงิน|ฝากเงิน|credit|deposit", re.IGNORECASE)
_ACCT_NO = re.compile(r"\b(\d{3}-\d-\d{4,6}-\d)\b")


def _parse_number(s: str) -> float | None:
    if not s:
        return None
    cleaned = re.sub(r"[,\s]", "", s.strip())
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_markdown_table(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    header_seen = False
    for line in (text or "").splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        if all(re.match(r"^[-:]+$", c) or c == "" for c in cells):
            continue
        if not header_seen:
            header_seen = True
            continue
        rows.append(cells)
    return rows


def _cell(row: list[str], idx: int) -> str:
    try:
        return row[idx].strip()
    except IndexError:
        return ""


def extract_bank_statement_header_rulebase(
    full_text: str,
    visible_fields: list[str],
    template_id: str,
) -> dict[str, Any]:
    text = full_text or ""
    vf = set(visible_fields)
    result: dict[str, Any] = {f: None for f in visible_fields}

    if "bank" in vf:
        result["bank"] = {
            "bank_statement_scb": "ไทยพาณิชย์",
            "bank_statement_kbank": "กสิกรไทย",
            "bank_statement_bbl": "กรุงเทพ",
        }.get(template_id)

    if "account_number" in vf:
        m = _ACCT_NO.search(text)
        if m:
            result["account_number"] = m.group(1)

    if "account_id" in vf and template_id == "bank_statement_kbank":
        m = re.search(r"เลขที่อ้างอิง[:\s]+([A-Z0-9]+-[A-Z0-9]+-\d{8})", text)
        if m:
            result["account_id"] = re.sub(r"-\d{8}$", "", m.group(1))

    if "account_role" in vf:
        if re.search(r"ออมทรัพย์|Savings|SAVINGS", text):
            result["account_role"] = "ออมทรัพย์"
        elif re.search(r"กระแสรายวัน|Current|CURRENT", text):
            result["account_role"] = "กระแสรายวัน"
        else:
            result["account_role"] = "ออมทรัพย์"

    if "currency" in vf:
        result["currency"] = "THB"

    return result


def extract_bank_statement_transactions_rulebase(
    full_text: str,
    visible_fields: list[str],
    template_id: str,
    account_id: str | None,
) -> list[dict[str, Any]]:
    vf = set(visible_fields)
    table_rows = _parse_markdown_table(full_text or "")
    rows: list[dict[str, Any]] = []

    for cells in table_rows:
        if _SKIP_ROW.search(" ".join(cells)):
            continue
        if not any(c.strip() for c in cells):
            continue

        row: dict[str, Any] = {f: None for f in visible_fields}
        if "account_id" in vf:
            row["account_id"] = account_id

        if template_id == "bank_statement_scb":
            date = _cell(cells, 0)
            if not date:
                continue
            withdrawal = _cell(cells, 4)
            deposit = _cell(cells, 5)
            if "business_event_date" in vf:
                row["business_event_date"] = date
            if withdrawal:
                if "transaction_type" in vf:
                    row["transaction_type"] = "debit"
                if "amount_thb" in vf:
                    row["amount_thb"] = _parse_number(withdrawal)
            elif deposit:
                if "transaction_type" in vf:
                    row["transaction_type"] = "credit"
                if "amount_thb" in vf:
                    row["amount_thb"] = _parse_number(deposit)
            if "balance_after_thb" in vf:
                row["balance_after_thb"] = _parse_number(_cell(cells, 6))
            if "description" in vf:
                row["description"] = " ".join(filter(None, [_cell(cells, 1), _cell(cells, 2), _cell(cells, 7)])).strip() or None

        elif template_id == "bank_statement_kbank":
            date = _cell(cells, 0)
            if not date:
                continue
            tx_type = _cell(cells, 2)
            if "business_event_date" in vf:
                time_val = _cell(cells, 1)
                row["business_event_date"] = f"{date} {time_val}".strip() if time_val else date
            if "transaction_type" in vf:
                if _CREDIT_TYPE.search(tx_type):
                    row["transaction_type"] = "credit"
                elif _DEBIT_TYPE.search(tx_type):
                    row["transaction_type"] = "debit"
                else:
                    row["transaction_type"] = tx_type or None
            if "amount_thb" in vf:
                row["amount_thb"] = _parse_number(_cell(cells, 3))
            if "balance_after_thb" in vf:
                row["balance_after_thb"] = _parse_number(_cell(cells, 4))
            if "description" in vf:
                row["description"] = " ".join(filter(None, [_cell(cells, 6), _cell(cells, 5)])).strip() or None

        elif template_id == "bank_statement_bbl":
            date = _cell(cells, 0)
            if not date:
                continue
            withdrawal = _cell(cells, 3)
            deposit = _cell(cells, 4)
            if "business_event_date" in vf:
                row["business_event_date"] = date
            if withdrawal:
                if "transaction_type" in vf:
                    row["transaction_type"] = "debit"
                if "amount_thb" in vf:
                    row["amount_thb"] = _parse_number(withdrawal)
            elif deposit:
                if "transaction_type" in vf:
                    row["transaction_type"] = "credit"
                if "amount_thb" in vf:
                    row["amount_thb"] = _parse_number(deposit)
            if "balance_after_thb" in vf:
                row["balance_after_thb"] = _parse_number(_cell(cells, 5))
            if "description" in vf:
                row["description"] = " ".join(filter(None, [_cell(cells, 1), _cell(cells, 6)])).strip() or None

        else:
            date = _cell(cells, 0)
            if not date:
                continue
            if "business_event_date" in vf:
                row["business_event_date"] = date
            if "description" in vf:
                row["description"] = " ".join(c for c in cells[1:] if c).strip() or None

        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Folder pipeline
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"-(\d{4})-(\d{2})$")


def _artifact_in_date_range(artifact_id: str) -> bool:
    """Return True if artifact_id's year/month falls within the configured date filter."""
    if not any(x is not None for x in (DATE_YEAR_START, DATE_YEAR_END, DATE_MONTH_START, DATE_MONTH_END)):
        return True
    m = _DATE_RE.search(artifact_id)
    if not m:
        return True  # can't parse — include by default
    year, month = int(m.group(1)), int(m.group(2))
    if DATE_YEAR_START is not None and year < DATE_YEAR_START:
        return False
    if DATE_YEAR_END is not None and year > DATE_YEAR_END:
        return False
    # Apply month_start only on the start-year boundary
    if DATE_MONTH_START is not None and DATE_YEAR_START is not None and year == DATE_YEAR_START and month < DATE_MONTH_START:
        return False
    # Apply month_end only on the end-year boundary
    if DATE_MONTH_END is not None and DATE_YEAR_END is not None and year == DATE_YEAR_END and month > DATE_MONTH_END:
        return False
    return True


def process_bank_statement_label_folder(
    label_dir: Path,
    output_dir: Path,
    model: str,
) -> dict[str, Any]:
    all_label_files = sorted(label_dir.glob("*.json"))
    label_files = [f for f in all_label_files if _artifact_in_date_range(f.stem)]
    skipped = len(all_label_files) - len(label_files)
    if skipped:
        logger.info("Date filter: skipping %d/%d artifacts outside range", skipped, len(all_label_files))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "label_dir": str(label_dir),
        "output_dir": str(output_dir),
        "model": model,
        "total_artifacts": len(label_files),
        "artifacts": [],
    }

    for label_file in label_files:
        label_data = json.loads(label_file.read_text(encoding="utf-8"))
        artifact_id = label_data.get("artifact_id", label_file.stem)
        template_id = label_data.get("renderer_template_id", "bank_statement_unknown")
        pages = label_data.get("pages", [])
        output_pages: list[dict[str, Any]] = []
        header_account_id: str | None = None

        logger.info("Processing artifact | id=%s | template=%s | pages=%d | batch=%d", artifact_id, template_id, len(pages), OCR_BATCH_SIZE)
        output_pages = [None] * len(pages)

        def _process_bs_page(idx: int, page: dict, acct_id: str | None) -> dict[str, Any]:
            output_rel = page.get("output_path", "")
            image_path = resolve_data_path(output_rel)
            visible_fields = page.get("visible_fields", [])
            page_kind = page.get("page_kind", "page")
            page_result: dict[str, Any] = {
                "output_path": output_rel,
                "page_kind": page_kind,
                "visible_fields": visible_fields,
                "ocr_fields": None,
                "error": None,
            }
            if not image_path.exists():
                logger.error("Image not found | path=%s", image_path)
                page_result["error"] = f"Image not found: {image_path}"
                return page_result
            try:
                logger.info("Page start | artifact=%s | kind=%s | image=%s", artifact_id, page_kind, output_rel)
                debug_log_llm("bs_page_start", f"artifact={artifact_id} kind={page_kind} image={output_rel}")
                if page_kind == "header":
                    ocr_fields = extract_bank_statement_header_vision(
                        image_path=image_path, visible_fields=visible_fields,
                        template_id=template_id, model=model,
                        api_key=LLM_API_KEY, base_url=LLM_BASE_URL,
                    )
                    page_result["ocr_fields"] = ocr_fields or {}
                else:
                    rows = extract_bank_statement_transactions_vision(
                        image_path=image_path, visible_fields=visible_fields,
                        template_id=template_id, account_id=acct_id,
                        model=model, api_key=LLM_API_KEY, base_url=LLM_BASE_URL,
                    )
                    page_result["ocr_fields"] = rows
            except Exception as exc:  # noqa: BLE001
                page_result["error"] = str(exc)
            return page_result

        # Process header(s) first sequentially, then batch transactions in parallel
        header_pages = [(i, p) for i, p in enumerate(pages) if p.get("page_kind") == "header"]
        tx_pages = [(i, p) for i, p in enumerate(pages) if p.get("page_kind") != "header"]
        for idx, page in header_pages:
            result = _process_bs_page(idx, page, None)
            if isinstance(result.get("ocr_fields"), dict) and result["ocr_fields"].get("account_id"):
                header_account_id = result["ocr_fields"]["account_id"]
            output_pages[idx] = result
        if tx_pages:
            tasks = [lambda idx=idx, page=page: (idx, _process_bs_page(idx, page, header_account_id)) for idx, page in tx_pages]
            for idx, result in run_batch(tasks):
                output_pages[idx] = result

        artifact_result = {
            "artifact_id": artifact_id,
            "renderer_template_id": template_id,
            "template_version": label_data.get("template_version"),
            "source_label_file": str(label_file),
            "pages": output_pages,
        }
        out_file = output_dir / f"{artifact_id}.json"
        out_file.write_text(json.dumps(artifact_result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Artifact saved | id=%s | file=%s", artifact_id, out_file)

        from validate_output import validate_artifact
        retries_exhausted = False
        for retry in range(OCR_ARTIFACT_RETRIES):
            issues = validate_artifact(out_file)
            empty_paths = {i["output_path"] for i in issues if i["kind"] == "empty"}
            if not empty_paths:
                break
            logger.warning("Artifact rerun %d/%d | id=%s | %d empty pages", retry + 1, OCR_ARTIFACT_RETRIES, artifact_id, len(empty_paths))
            failed_tx = [(i, p) for i, p in enumerate(pages)
                         if p.get("output_path") in empty_paths and p.get("page_kind") != "header"]
            if failed_tx:
                tasks = [lambda idx=idx, page=page: (idx, _process_bs_page(idx, page, header_account_id)) for idx, page in failed_tx]
                for idx, result in run_batch(tasks):
                    output_pages[idx] = result
            artifact_result["pages"] = output_pages
            out_file.write_text(json.dumps(artifact_result, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Artifact re-saved after rerun %d | id=%s", retry + 1, artifact_id)
            if retry == OCR_ARTIFACT_RETRIES - 1:
                retries_exhausted = True

        if retries_exhausted:
            remaining = [i["output_path"] for i in validate_artifact(out_file) if i["kind"] == "empty"]
            if remaining:
                logger.error("Artifact still has %d empty pages after %d reruns | id=%s", len(remaining), OCR_ARTIFACT_RETRIES, artifact_id)

        for issue in validate_artifact(out_file):
            if issue["kind"] != "empty":
                logger.warning("Output validation | [%s] %s | %s", issue["kind"].upper(), issue["output_path"], issue["detail"])

        summary["artifacts"].append({"artifact_id": artifact_id, "output_file": str(out_file)})

    summary_file = output_dir / "_summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# API-spec extraction (called by fahmai.api /ocr endpoint)
# ---------------------------------------------------------------------------

_API_OCR_HEADER_PROMPT = (
    "Extract ALL text visible in this bank statement header image exactly as it appears, "
    "preserving reading order, numbers, and Thai/English characters. Return only the extracted text."
)

_API_LLM_HEADER_SYSTEM = """\
You are extracting structured data from a Thai bank statement header image.

Return ONLY valid JSON with exactly this structure — no markdown fences, no explanation:
{
  "account": {
    "account_number": "<account number exactly as printed, e.g. 001-1-00001-1>",
    "owner_branch": "<branch name in Thai>"
  },
  "opening_balance_row": {
    "label": "<e.g. ยอดยกมา>",
    "date": "<DD-MM-YY>",
    "balance_text": "<balance amount as string, e.g. 330.10>"
  }
}

Use null for any field not found in the text.\
"""

_API_OCR_TXN_PROMPT = (
    "Extract ALL text from this bank statement transaction table image exactly as it appears, "
    "including all rows, dates, amounts, and descriptions. Return only the extracted text."
)

_API_LLM_TXN_SYSTEM = """\
You are extracting transaction rows from a Thai bank statement page.

Return ONLY a valid JSON array — no markdown fences, no explanation.
Each element must have exactly these keys:
{
  "date": "<DD-MM-YY>",
  "item": "<transaction description, Thai or English>",
  "debit_text": "<debit amount string, or null if not a debit>",
  "credit_text": "<credit amount string, or null if not a credit>",
  "amount_direction": "<debit or credit>",
  "balance_text": "<running balance as string>",
  "details": "<additional details / reference text, or null>"
}

Skip ยอดยกมา (opening balance) and ยอดยกไป (closing balance) rows.
Include every real transaction row visible. Use null for missing fields.\
"""


def _extract_api_header(
    raw: bytes,
    ocr_model: str, ocr_key: str, ocr_url: str,
    llm_model: str, llm_key: str, llm_url: str,
) -> dict:
    """Two-stage header extraction → API spec format."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(raw)
        tmp = Path(f.name)
    try:
        try:
            ocr_text = call_llm_vision(
                image_path=tmp, user_prompt=_API_OCR_HEADER_PROMPT,
                model=ocr_model, api_key=ocr_key, base_url=ocr_url,
            )
        except Exception as exc:
            logger.warning("API header OCR failed: %s", exc)
            ocr_text = ""

        user_prompt = (
            f"Here is the OCR-extracted text from a bank statement header page:\n\n{ocr_text}\n\n"
            "Extract account_number, owner_branch, and opening_balance_row. Return JSON only."
        )
        try:
            raw_text = call_llm_text(
                user_prompt=user_prompt, system_prompt=_API_LLM_HEADER_SYSTEM,
                model=llm_model, api_key=llm_key, base_url=llm_url,
            )
            parsed = json.loads(repair_json(strip_json_fences(raw_text)))
            return parsed if isinstance(parsed, dict) else {}
        except Exception as exc:
            logger.warning("API header LLM failed: %s", exc)
            return {}
    finally:
        tmp.unlink(missing_ok=True)


def _extract_api_transactions(
    raw: bytes,
    ocr_model: str, ocr_key: str, ocr_url: str,
    llm_model: str, llm_key: str, llm_url: str,
) -> list[dict]:
    """Two-stage transaction extraction → API spec format."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(raw)
        tmp = Path(f.name)
    try:
        try:
            ocr_text = call_llm_vision(
                image_path=tmp, user_prompt=_API_OCR_TXN_PROMPT,
                model=ocr_model, api_key=ocr_key, base_url=ocr_url,
            )
        except Exception as exc:
            logger.warning("API transaction OCR failed: %s", exc)
            ocr_text = ""

        user_prompt = (
            f"Here is the OCR-extracted text from a bank statement transaction page:\n\n{ocr_text}\n\n"
            "Extract ALL transaction rows. Return JSON array only."
        )
        try:
            raw_text = call_llm_text(
                user_prompt=user_prompt, system_prompt=_API_LLM_TXN_SYSTEM,
                model=llm_model, api_key=llm_key, base_url=llm_url,
            )
            clean = strip_json_fences(raw_text)
            try:
                parsed = json.loads(repair_json(clean))
            except json.JSONDecodeError:
                try:
                    parsed = json.loads(recover_truncated_json_array(repair_json(clean)))
                except json.JSONDecodeError:
                    parsed = extract_json_objects(clean)
            return parsed if isinstance(parsed, list) else []
        except Exception as exc:
            logger.warning("API transaction LLM failed: %s", exc)
            return []
    finally:
        tmp.unlink(missing_ok=True)


def run_bank_statement_structured(
    header_raw: bytes,
    txn_raws: list[bytes],
    ocr_model: str,
    ocr_key: str,
    ocr_url: str,
    llm_model: str,
    llm_key: str,
    llm_url: str,
) -> dict:
    """Process bank statement images → API spec format (synchronous).

    Returns:
    {
      "account": {"account_number": ..., "owner_branch": ...},
      "opening_balance_row": {"label": ..., "date": ..., "balance_text": ...},
      "transactions": [{row_index, date, item, debit_text, credit_text,
                        amount_direction, balance_text, details}, ...]
    }
    """
    header_data = _extract_api_header(header_raw, ocr_model, ocr_key, ocr_url, llm_model, llm_key, llm_url)

    all_txns: list[dict] = []
    for raw in txn_raws:
        rows = _extract_api_transactions(raw, ocr_model, ocr_key, ocr_url, llm_model, llm_key, llm_url)
        for row in rows:
            if not isinstance(row, dict):
                continue
            all_txns.append({
                "row_index": len(all_txns) + 1,
                "date": row.get("date"),
                "item": row.get("item"),
                "debit_text": row.get("debit_text"),
                "credit_text": row.get("credit_text"),
                "amount_direction": row.get("amount_direction"),
                "balance_text": row.get("balance_text"),
                "details": row.get("details"),
            })

    return {
        "account": header_data.get("account") or {},
        "opening_balance_row": header_data.get("opening_balance_row") or {},
        "transactions": all_txns,
    }
