# extractor.py — FIXED: Per-document extraction calls, doc-specific page scoring
import fitz
import io
import os
import re
import json
import time
import pytesseract
from PIL import Image
from groq import Groq
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
from typing import Any

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
client = Groq(api_key=GROQ_API_KEY)

# ── TOKEN / CHAR LIMITS ───────────────────────────────────────────────────────
# Each doc type now gets its OWN char budget (no sharing between P&L and BS)
PER_DOC_CHAR_LIMIT = 10_000   # chars fed per doc-type extraction call
NOTES_TEXT_LIMIT   = 8_000
MAX_OUTPUT_TOKENS  = 8000     # llama-3.3-70b supports up to 8192 output tokens

# ── KEYWORD SETS ─────────────────────────────────────────────────────────────
FINANCIAL_KEYWORDS = [
    "revenue", "income", "expenditure", "expenses", "profit", "loss",
    "balance sheet", "assets", "liabilities", "equity", "capital",
    "cash", "receivable", "payable", "depreciation", "surplus",
    "fixed assets", "current assets", "borrowings", "turnover",
    "sales", "purchases", "gross", "net", "ebitda", "reserves",
    "investments", "provisions", "creditors", "debtors", "schedule",
    "intangible", "trade receivable", "advance", "prepaid", "inventory"
]

# Doc-specific keyword sets for targeted page selection
PNL_KEYWORDS = [
    "revenue", "sales", "income", "turnover", "other income",
    "expenses", "expenditure", "profit", "loss", "ebitda",
    "gross profit", "operating", "cost of goods", "cost of sales",
    "employee", "employee benefit", "depreciation", "amortisation",
    "finance cost", "finance charges", "interest expense",
    "tax expense", "current tax", "deferred tax",
    "purchases", "changes in inventories", "manufacturing",
    "statement of profit", "profit and loss", "income statement"
]

BS_KEYWORDS = [
    "balance sheet", "financial position", "assets", "liabilities", "equity",
    "fixed assets", "property plant", "tangible assets", "intangible assets",
    "current assets", "non-current assets", "non current assets",
    "borrowings", "long-term borrowings", "short-term borrowings",
    "reserves", "surplus", "share capital", "shareholders fund",
    "trade payables", "trade receivables", "inventory", "inventories",
    "cash and bank", "cash and cash equivalents",
    "investments", "provisions", "deferred tax liability",
    "creditors", "debtors", "current liabilities", "non-current liabilities"
]

CF_KEYWORDS = [
    "cash flow", "cash flows", "statement of cash",
    "operating activities", "investing activities", "financing activities",
    "net cash", "opening cash", "closing cash", "cash equivalents",
    "proceeds from", "repayment of", "dividend paid",
    "purchase of fixed assets", "acquisition", "capital expenditure"
]

NOTES_KEYWORDS = [
    "note", "notes to", "schedule", "accounting policy", "significant",
    "depreciation method", "inventories", "taxation", "related party",
    "contingent", "commitments", "auditor", "basis of preparation",
    "revenue recognition", "lease", "segment", "earnings per share"
]

INDIAN_NUMBER_RE = re.compile(r'\d{1,2}(,\d{2})*,\d{3}(\.\d+)?|\d[\d,]+\.\d{2}')


class ExtractResponse(BaseModel):
    mapped: Any
    raw: str
    confidence: Any


# ── CLEAN NUMERIC VALUE ───────────────────────────────────────────────────────
def clean_val(v):
    """Remove Indian-format commas and convert to float."""
    try:
        return float(str(v).replace(',', '').replace(' ', '').strip())
    except Exception:
        return 0.0


# ── COMPANY TYPE DETECTION ────────────────────────────────────────────────────
def detect_company_type(text: str) -> str:
    lower = text.lower()
    if "proprietorship" in lower or "sole proprietor" in lower:
        return "proprietorship"
    if "partnership" in lower:
        return "partnership"
    if "trust" in lower or "ngo" in lower or "society" in lower:
        return "trust"
    if "llp" in lower or "limited liability" in lower:
        return "llp"
    if "private limited" in lower or "pvt" in lower or "ltd" in lower:
        return "pvt_ltd"
    return "pvt_ltd"


# ── TEXT EXTRACTION ───────────────────────────────────────────────────────────
def extract_page_text_structured(page) -> str:
    """Coordinate-sorted extraction — preserves table column alignment."""
    try:
        raw_dict  = page.get_text("dict")
        all_spans = []
        for block in raw_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text:
                        continue
                    x, y = span["origin"][0], span["origin"][1]
                    all_spans.append((y, x, text))
        if not all_spans:
            return page.get_text().strip()
        all_spans.sort(key=lambda s: (s[0], s[1]))
        rows, current_row = [], [all_spans[0]]
        for i in range(1, len(all_spans)):
            if abs(all_spans[i][0] - current_row[-1][0]) <= 4:
                current_row.append(all_spans[i])
            else:
                rows.append(current_row)
                current_row = [all_spans[i]]
        rows.append(current_row)
        lines = ["  |  ".join(s[2] for s in sorted(row, key=lambda s: s[1])) for row in rows]
        return "\n".join(lines)
    except Exception:
        return page.get_text().strip()


# ── PAGE SCORERS ──────────────────────────────────────────────────────────────
def _base_financial_score(text: str, keywords: list) -> int:
    """Shared scoring base: numbers + pipes + year hits + keyword hits."""
    score = 0
    lower = text.lower()
    score += len(INDIAN_NUMBER_RE.findall(text)) * 4
    score += len(re.findall(r'\d[\d,]{4,}', text)) * 2
    for kw in keywords:
        if kw in lower:
            score += 4
    score += text.count("|") * 3
    year_hits = re.findall(
        r'(FY\s*\d{2,4}|20\d{2}[-–]\d{2,4}|31[.\-/]\d{2}[.\-/]\d{2,4})', text
    )
    score += len(year_hits) * 5
    words     = len(text.split())
    num_count = len(re.findall(r'\d+', text))
    if words > 80 and words > 0 and (num_count / words) < 0.05:
        score = int(score * 0.3)
    return score


def score_pnl_page(text: str) -> int:
    return _base_financial_score(text, PNL_KEYWORDS)

def score_bs_page(text: str) -> int:
    return _base_financial_score(text, BS_KEYWORDS)

def score_cf_page(text: str) -> int:
    return _base_financial_score(text, CF_KEYWORDS)

def score_financial_page(text: str) -> int:
    return _base_financial_score(text, FINANCIAL_KEYWORDS)

def score_notes_page(text: str) -> int:
    score = 0
    lower = text.lower()
    for kw in NOTES_KEYWORDS:
        if kw in lower:
            score += 5
    note_hits = re.findall(
        r'\bnote\s*\d+\b|\b\d+\.\s+[A-Z]|\bschedule\s+[IVX\d]+\b',
        text, re.IGNORECASE
    )
    score += len(note_hits) * 8
    score += len(INDIAN_NUMBER_RE.findall(text)) * 2
    score += text.count("|") * 2
    words     = len(text.split())
    num_count = len(re.findall(r'\d+', text))
    if words > 30 and words > 0 and (num_count / words) > 0.5:
        score = int(score * 0.4)
    return score


# Map doc key → its specific scorer
DOC_SCORER_MAP = {
    "pnl": score_pnl_page,
    "bs":  score_bs_page,
    "cf":  score_cf_page,
}

def get_doc_scorer(doc_key: str):
    """Return doc-specific scorer, fallback to generic."""
    for key_fragment, scorer in DOC_SCORER_MAP.items():
        if key_fragment in doc_key.lower():
            return scorer
    return score_financial_page


# ── SMART PAGE SELECTOR ───────────────────────────────────────────────────────
def select_pages(page_texts: list, scorer_fn, char_limit: int, top_n: int = 20) -> str:
    """Score all pages, pick top N, re-sort into document order, concatenate."""
    scored = []
    for i, text in enumerate(page_texts):
        s = scorer_fn(text)
        if s > 0:
            scored.append((s, i, text))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = sorted(scored[:top_n], key=lambda x: x[1])  # restore page order

    result = ""
    for _, _, text in top:
        if len(result) + len(text) + 1 > char_limit:
            remaining = char_limit - len(result)
            if remaining > 800:
                result += text[:remaining] + "\n"
            break
        result += text + "\n"
    return result.strip()


# ── CONFIDENCE SCORER ─────────────────────────────────────────────────────────
def compute_item_confidence(item_name: str, val: float, full_text: str, model_conf: dict) -> int:
    score = 0
    model_score = model_conf.get(item_name)
    if model_score is None:
        lk = item_name.lower().strip()
        model_score = next((v for k, v in model_conf.items() if k.lower().strip() == lk), None)
    score += min(20, int(model_score * 0.2)) if model_score is not None else 10
    if val and val != 0:         score += 20
    if val and abs(val) > 10000: score += 20
    if item_name.lower() in full_text.lower(): score += 40
    return min(score, 100)


# ── GROQ CALL WITH RATE-LIMIT RETRY ──────────────────────────────────────────
def groq_call(messages: list, max_tokens: int = MAX_OUTPUT_TOKENS, retries: int = 2) -> dict:
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                temperature=0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            err_str = str(e).lower()
            if ("rate_limit" in err_str or "429" in err_str) and attempt < retries:
                wait_secs = 20 * (attempt + 1)
                print(f"[Groq] Rate limit hit — waiting {wait_secs}s before retry {attempt + 1}...")
                time.sleep(wait_secs)
                continue
            raise e


# ── SINGLE-DOC EXTRACTION PROMPT ─────────────────────────────────────────────
def build_doc_prompt(doc_key: str, sections: list, doc_text: str,
                     company_type: str, detected_periods: list) -> str:
    section_desc    = "\n".join([f'  "{s["key"]}": {s["title"]}' for s in sections])
    periods_hint    = json.dumps(detected_periods) if detected_periods else '["FY2024","FY2023"]'
    data_hint       = f'"{doc_key}": {{ "section_key": {{ "FY2024": {{ "Line Item": 1234.00 }} }} }}'

    return f"""You are a senior financial analyst extraction engine specialising in Indian statutory audit reports (Schedule III format).
Company Type Detected: {company_type}
Document Type: {doc_key.upper()} only — extract ONLY {doc_key.upper()} data.
Expected periods (already detected from PDF): {periods_hint}

The document text has been extracted with table columns separated by " | ".
Each row: Line Item Name | Note No | VALUE_YEAR1 | VALUE_YEAR2

Return ONLY valid JSON:

{{
  "company_name": "string",
  "period": "FY2024",
  "all_periods": ["FY2024", "FY2023"],
  "confidence": {{ "Exact Item Name": 95 }},
  "data": {{
    {data_hint}
  }}
}}

Section keys for {doc_key.upper()} (use EXACTLY these keys):
{section_desc}

CRITICAL EXTRACTION RULES (India Schedule III):

1. YEAR DETECTION: First value column = most recent year, second = prior year. Normalise → "FY20XX".
   - Only create FY entry if ACTUAL data exists. No zero-filled years.
   - Read year headers independently; NEVER copy values between years.

2. COMPANY TYPE ADJUSTMENTS:
   - {company_type.upper()}: {'No Non-Current sections. Skip any Non-Current Assets/Liabilities.' if company_type == 'proprietorship' else 'Include all Schedule III sections as present.'}
   - Proprietorship/Partnership: Equity = Opening Capital + Profit - Withdrawals (NOT single value).
   - Trust: Use "Capital Fund" not "Equity"; include "Grants" and "Donations" separately.

3. NUMBER EXTRACTION (CRITICAL):
   - READ EXACT VALUES FROM PDF. Match each digit character-by-character.
   - Indian format: "12,34,567.89" = 1234567.89 (NOT 123456789)
   - Remove commas ONLY: "1,26,44,429.00" → 12644429.00
   - If decimal places differ (1.70 vs 1.72), re-check source PDF twice. Use EXACT source value.
   - Flag confidence < 70 for any digit misread risk.

4. ASSET BREAKDOWN:
   - Current Assets MUST include: Trade Receivables + Inventory + Cash + Prepayments + Advances + Other Current Assets
   - Non-Current Assets MUST include: PPE + Intangible Assets + Investments + Long-term Receivables + Other NCAs
   - If section empty or field missing in source, use 0 — do NOT merge with other items.

5. LIABILITY BREAKDOWN:
   - Current Liabilities: Trade Payables + Short-term Borrowings + Advances + Other CL + Provision (as separate items)
   - Non-Current Liabilities: Long-term Borrowings + Deferred Tax + Other NCL
   - Do NOT aggregate; keep item-level detail.

6. EXPENSE DETAIL:
   - Employee Cost: Salary + Wages + Benefits (separate lines if available)
   - Finance Costs: Interest + Bank Charges (keep separate)
   - Tax: Base Tax + Cess + Interest on Tax (keep EACH component separate)
   - Depreciation: Exact line value (not inferred from difference)
   - Do NOT group expenses into single totals unless source shows "Total Expenses".

7. CASH FLOW MOVEMENT:
   - Cash Flow = Closing Balance MINUS Opening Balance (yearly movement)
   - NOT just closing balance alone.
   - If movement data missing, extract opening and closing separately.

8. CONFIDENCE SCORING:
   - 90-100: Value clearly visible in table, exact match to source text
   - 70-89: Inferred or slightly unclear
   - <70: Hallucinated or cannot verify in source

9. EDGE CASES:
   - "-", blank, "Nil", "N/A" → 0.0 (not null)
   - Multiple years in single cell → use most recent year for FY2024 column
   - Text merged with numbers → extract numbers only

10. RETURN ONLY JSON. No markdown. No explanation. No code blocks.

11. NO MATH EXPRESSIONS (CRITICAL): All values MUST be a single, final float number (e.g., 1500.0).
    NEVER output mathematical expressions like "1000 + 200". Perform the math yourself and output only the final number.

Document text ({doc_key.upper()} section):
{doc_text}"""


# ── MAIN EXTRACT ENDPOINT ─────────────────────────────────────────────────────
@app.post("/extract", response_model=ExtractResponse)
async def extract_pdf(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files allowed")

    # Parse *_schema form fields
    form_data   = await request.form()
    doc_schemas = {}
    for field_name, field_value in form_data.items():
        if field_name.endswith("_schema"):
            doc_key = field_name[:-7]
            try:
                doc_schemas[doc_key] = json.loads(field_value)
            except Exception:
                doc_schemas[doc_key] = []

    if not doc_schemas:
        doc_schemas = {
            "pnl": [{"key": "revenue", "title": "Revenue"}],
            "bs":  [{"key": "currentAssets", "title": "Current Assets"}]
        }

    # ── Extract all page texts ────────────────────────────────────
    contents   = await file.read()
    pdf_doc    = fitz.open(stream=contents, filetype="pdf")
    page_texts = []

    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]
        text = extract_page_text_structured(page)
        if not text.strip():
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang='eng').strip()
        page_texts.append(text)

    pdf_doc.close()

    full_text = "\n".join(page_texts)
    if not full_text.strip():
        raise HTTPException(400, "No extractable text found in PDF.")

    company_type = detect_company_type(full_text)

    # ── Pre-detect fiscal year periods from full text ─────────────
    raw_hits = re.findall(
        r'FY\s*(\d{2,4})|(?:20(\d{2}))[-–](?:20)?(\d{2,4})',
        full_text
    )
    detected_periods = []
    for h in raw_hits:
        if h[0]:
            yr = h[0] if len(h[0]) == 4 else "20" + h[0]
            p  = f"FY{yr}"
        elif h[1]:
            p = f"FY20{h[1]}"
        else:
            continue
        if p not in detected_periods:
            detected_periods.append(p)
    detected_periods = sorted(set(detected_periods), reverse=True)[:3]
    if not detected_periods:
        detected_periods = ["FY2024", "FY2023"]

    # ── PER-DOC extraction ────────────────────────────────────────
    # KEY FIX: each doc type gets its own page selection + its own LLM call
    # This prevents BS pages from crowding out P&L pages in a shared char budget
    final_data     = {}
    all_confidence = {}
    company_name   = "Unknown"
    primary_period = detected_periods[0] if detected_periods else "FY2024"
    all_periods    = list(detected_periods)

    for doc_key, sections in doc_schemas.items():
        print(f"[Extract] Processing doc: {doc_key}")

        # Use doc-specific scorer to pick the most relevant pages for THIS doc
        scorer   = get_doc_scorer(doc_key)
        doc_text = select_pages(page_texts, scorer, PER_DOC_CHAR_LIMIT, top_n=18)

        # Fallback: if doc-specific scorer found nothing, use generic
        if not doc_text.strip():
            print(f"[Extract] {doc_key}: doc-specific scorer empty, falling back to generic")
            doc_text = select_pages(page_texts, score_financial_page, PER_DOC_CHAR_LIMIT, top_n=18)

        if not doc_text.strip():
            print(f"[Extract] {doc_key}: no relevant pages found, skipping")
            final_data[doc_key] = {}
            continue

        prompt = build_doc_prompt(doc_key, sections, doc_text, company_type, detected_periods)

        try:
            parsed = groq_call(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a strict financial OCR parser for Indian statutory reports "
                            f"({company_type}). Extract ONLY {doc_key.upper()} data. "
                            f"Return only valid JSON."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=MAX_OUTPUT_TOKENS
            )
        except Exception as e:
            print(f"[Extract] {doc_key} extraction failed: {str(e)}")
            final_data[doc_key] = {}
            continue

        # Update global meta from first successful parse
        if company_name == "Unknown" and parsed.get("company_name"):
            company_name = parsed["company_name"]
        if parsed.get("period"):
            primary_period = parsed["period"]
        if parsed.get("all_periods"):
            for p in parsed["all_periods"]:
                if p not in all_periods:
                    all_periods.append(p)

        # Clean and store this doc's data
        raw_doc_data        = parsed.get("data", {}).get(doc_key, {})
        final_data[doc_key] = {}

        for sec_key, periods in raw_doc_data.items():
            final_data[doc_key][sec_key] = {}
            if isinstance(periods, dict):
                for period, items in periods.items():
                    if isinstance(items, dict):
                        final_data[doc_key][sec_key][period] = {
                            k: clean_val(v) for k, v in items.items()
                        }

        # Merge confidence scores
        raw_conf = parsed.get("confidence", {})
        for sec_data in final_data[doc_key].values():
            for period_data in sec_data.values():
                for item_name, item_val in period_data.items():
                    if item_name not in all_confidence:
                        all_confidence[item_name] = compute_item_confidence(
                            item_name, item_val, full_text, raw_conf
                        )

        # Respect Groq TPM between calls (2s gap)
        time.sleep(2)

    # ── Notes extraction ──────────────────────────────────────────
    notes_text   = select_pages(page_texts, score_notes_page, NOTES_TEXT_LIMIT, top_n=12)
    notes_parsed = {"notes_to_accounts": []}

    if notes_text.strip():
        notes_prompt = f"""You are extracting Notes to Accounts from an Indian statutory audit report.

Return ONLY valid JSON:
{{
  "notes_to_accounts": [
    {{
      "number": "1",
      "title": "Note Title",
      "text": "Qualitative/policy text. Empty string if none.",
      "table": [
        {{ "item": "Sub-item name", "FY2024": 1234.00, "FY2023": 1000.00 }}
      ]
    }}
  ]
}}

Rules:
- Extract EVERY numbered note and named schedule found in the document.
- "number": note number as string ("1", "2", "2a", etc.)
- "title": the heading of the note
- "text": policy or qualitative explanation — empty string if none
- "table": array of line items with per-year values. [] if the note has no table.
- Remove all commas from numbers: "1,26,44,429.00" → 12644429.00
- Use "FY20XX" format for all year keys
- NO MATH EXPRESSIONS: All values must be final floats. Do not output "100+20".
- Return ONLY the JSON. No markdown. No explanation.

Document:
{notes_text}"""

        try:
            notes_parsed = groq_call(
                messages=[
                    {"role": "system", "content": "You are a strict financial notes extractor. Return only valid JSON."},
                    {"role": "user",   "content": notes_prompt}
                ],
                max_tokens=4000
            )
        except Exception as e:
            print(f"[Extract] Notes extraction failed (non-fatal): {str(e)}")
            notes_parsed = {"notes_to_accounts": []}

    # ── Clean notes ────────────────────────────────────────────────
    raw_notes   = notes_parsed.get("notes_to_accounts", [])
    clean_notes = []
    for note in raw_notes:
        clean_table = []
        for row in note.get("table", []):
            clean_row = {"item": row.get("item", "")}
            for k, v in row.items():
                if k != "item":
                    clean_row[k] = clean_val(v)
            clean_table.append(clean_row)
        clean_notes.append({
            "number": str(note.get("number", "")),
            "title":  note.get("title", ""),
            "text":   note.get("text", ""),
            "table":  clean_table
        })

    all_periods_sorted = sorted(set(all_periods), reverse=True)

    return ExtractResponse(
        mapped={
            "companyname":       company_name,
            "period":            primary_period,
            "allperiods":        all_periods_sorted,
            "data":              final_data,
            "notes_to_accounts": clean_notes,
            "company_type":      company_type,
        },
        raw=full_text[:3000],
        confidence=all_confidence
    )


# ── HEALTH & DEBUG ────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Redwood PDF Extractor running"}

@app.get("/health")
async def health():
    return {"status": "ok", "model": "llama-3.3-70b-versatile", "mode": "per-doc-extraction"}

@app.get("/ping")
def ping():
    return {"status": "pong"}

@app.post("/debug")
async def debug_extract(file: UploadFile = File(...)):
    contents   = await file.read()
    pdf_doc    = fitz.open(stream=contents, filetype="pdf")
    page_texts = []
    page_data  = []

    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]
        text = extract_page_text_structured(page)
        if not text.strip():
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang='eng').strip()
        page_texts.append(text)
        page_data.append({
            "page":        page_num + 1,
            "pnl_score":   score_pnl_page(text),
            "bs_score":    score_bs_page(text),
            "cf_score":    score_cf_page(text),
            "notes_score": score_notes_page(text),
            "chars":       len(text),
            "preview":     text[:120]
        })

    pnl_text   = select_pages(page_texts, score_pnl_page,   PER_DOC_CHAR_LIMIT, 18)
    bs_text    = select_pages(page_texts, score_bs_page,    PER_DOC_CHAR_LIMIT, 18)
    cf_text    = select_pages(page_texts, score_cf_page,    PER_DOC_CHAR_LIMIT, 18)
    notes_text = select_pages(page_texts, score_notes_page, NOTES_TEXT_LIMIT,   12)
    pdf_doc.close()

    return {
        "total_pages":      len(page_data),
        "page_scores":      sorted(page_data, key=lambda x: x["pnl_score"] + x["bs_score"], reverse=True)[:20],
        "pnl_text_chars":   len(pnl_text),
        "bs_text_chars":    len(bs_text),
        "cf_text_chars":    len(cf_text),
        "notes_text_chars": len(notes_text),
        "pnl_preview":      pnl_text[:2000],
        "bs_preview":       bs_text[:2000],
        "cf_preview":       cf_text[:1000],
        "notes_preview":    notes_text[:1000]
    }
