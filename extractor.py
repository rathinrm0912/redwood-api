# extractor.py — Redwood Production (Groq) — Fixed: 413, .00 formatting, company name

import fitz
import io
import os
import re
import json
import pytesseract
from PIL import Image
from groq import Groq
from fastapi import FastAPI, UploadFile, File, Request, Response
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

# Groq Free Tier = 12,000 TPM hard limit
# Output reserved = 4,000 tokens (~16,000 chars)
# Input budget    = 8,000 tokens
# Overhead        = ~2,000 tokens (system + schema + instructions)
# Safe text chars = ~6,000 tokens = ~24,000 chars total
# Split: 80% financial, 20% notes, with hard cap enforced before API call
FINANCIAL_TEXT_LIMIT = 16_000
NOTES_TEXT_LIMIT     =  4_000
MAX_COMBINED_CHARS   = 19_000   # hard cap — never exceed this before sending to Groq

FINANCIAL_KEYWORDS = [
    "revenue", "income", "expenditure", "expenses", "profit", "loss",
    "balance sheet", "assets", "liabilities", "equity", "capital",
    "cash", "receivable", "payable", "depreciation", "surplus",
    "fixed assets", "current assets", "borrowings", "turnover",
    "sales", "purchases", "gross", "net", "ebitda", "reserves",
    "investments", "provisions", "creditors", "debtors", "schedule"
]

NOTES_KEYWORDS = [
    "note", "notes to", "schedule", "accounting policy", "significant",
    "depreciation method", "inventories", "taxation", "related party",
    "contingent", "commitments", "auditor", "basis of preparation",
    "revenue recognition", "lease", "segment", "earnings per share"
]

INDIAN_NUMBER_RE = re.compile(r'\d{1,2}(,\d{2})*,\d{3}(\.\d+)?|\d[\d,]+\.\d{2}')


# ── UNIFORM .00 SERIALIZER ────────────────────────────────────────
def financial_json_response(data) -> Response:
    """Forces all floats to serialize with exactly 2 decimal places."""
    def preprocess(obj):
        if isinstance(obj, float):
            return f"§{obj:.2f}§"
        if isinstance(obj, dict):
            return {k: preprocess(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [preprocess(i) for i in obj]
        return obj

    processed = preprocess(data)
    raw       = json.dumps(processed, ensure_ascii=False)
    raw       = re.sub(r'"§(-?[\d.]+)§"', r'\1', raw)
    return Response(content=raw, media_type="application/json")


# ── TEXT EXTRACTION ───────────────────────────────────────────────
def extract_page_text_structured(page) -> str:
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

        rows        = []
        current_row = [all_spans[0]]
        for i in range(1, len(all_spans)):
            if abs(all_spans[i][0] - current_row[-1][0]) <= 4:
                current_row.append(all_spans[i])
            else:
                rows.append(current_row)
                current_row = [all_spans[i]]
        rows.append(current_row)

        lines = []
        for row in rows:
            row_text = "  |  ".join(s[2] for s in sorted(row, key=lambda s: s[1]))
            lines.append(row_text)

        return "\n".join(lines)

    except Exception:
        return page.get_text().strip()


# ── PAGE SCORERS ──────────────────────────────────────────────────
def score_financial_page(text: str) -> int:
    score = 0
    lower = text.lower()
    lines = text.split('\n')

    score += len(INDIAN_NUMBER_RE.findall(text)) * 6
    score += len(re.findall(r'\d[\d,]{4,}', text)) * 3

    for kw in FINANCIAL_KEYWORDS:
        if kw in lower:
            score += 3

    # Only pipes on lines that also have numbers — prevents OCR prose inflation
    numeric_pipe_lines = sum(
        1 for l in lines if '|' in l and re.search(r'\d{2,}', l)
    )
    score += numeric_pipe_lines * 5

    # Lines with 2+ large numbers = real table data rows
    multi_num_lines = sum(
        1 for l in lines if len(re.findall(r'\d{4,}', l)) >= 2
    )
    score += multi_num_lines * 10

    year_hits = re.findall(
        r'(FY\s*\d{2,4}|20\d{2}[-\u2013]\d{2,4}|31[.\-/]\d{2}[.\-/]\d{2,4})', text
    )
    score += len(year_hits) * 5

    words      = len(text.split())
    num_count  = len(re.findall(r'\d+', text))
    pipe_count = text.count('|')

    if words > 80 and (num_count / max(words, 1)) < 0.05:
        score = int(score * 0.15)

    # OCR prose penalty: many pipes but almost no numbers
    if pipe_count > 20 and (num_count / max(pipe_count, 1)) < 0.1:
        score = int(score * 0.2)

    return score


def score_notes_page(text: str) -> int:
    score = 0
    lower = text.lower()
    lines = text.split('\n')

    for kw in NOTES_KEYWORDS:
        if kw in lower:
            score += 5

    note_hits = re.findall(
        r'\bnote\s*\d+\b|\b\d+\.\s+[A-Z]|\bschedule\s+[IVX\d]+\b',
        text, re.IGNORECASE
    )
    score += len(note_hits) * 8

    score += len(INDIAN_NUMBER_RE.findall(text)) * 2

    numeric_pipe_lines = sum(
        1 for l in lines if '|' in l and re.search(r'\d{2,}', l)
    )
    score += numeric_pipe_lines * 3

    words      = len(text.split())
    num_count  = len(re.findall(r'\d+', text))
    pipe_count = text.count('|')

    if words > 30 and (num_count / max(words, 1)) > 0.5:
        score = int(score * 0.4)

    if pipe_count > 20 and (num_count / max(pipe_count, 1)) < 0.1:
        score = int(score * 0.25)

    return score


# ── SMART TEXT SELECTOR ───────────────────────────────────────────
def select_pages(page_texts: list, scorer_fn, char_limit: int, top_n: int = 10) -> str:
    scored = []
    for page_num, text in enumerate(page_texts):
        score = scorer_fn(text)
        if score > 0:
            scored.append((score, page_num, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_n]
    top.sort(key=lambda x: x[1])

    result = ""
    for _, _, text in top:
        if len(result) + len(text) + 1 > char_limit:
            remaining = char_limit - len(result)
            if remaining > 400:
                result += text[:remaining] + "\n"
            break
        result += text + "\n"

    return result.strip()


# ── CONFIDENCE SCORER ─────────────────────────────────────────────
def compute_item_confidence(
    item_name: str, val: float, full_text: str, model_conf: dict
) -> int:
    score = 0

    model_score = model_conf.get(item_name)
    if model_score is None:
        lk = item_name.lower().strip()
        model_score = next(
            (v for k, v in model_conf.items() if k.lower().strip() == lk), None
        )
    score += min(20, int(model_score * 0.2)) if model_score is not None else 10

    if val and val != 0:         score += 20
    if val and abs(val) > 10000: score += 20

    if item_name.lower() in full_text.lower(): score += 40

    return min(score, 100)


# ── COMPANY NAME FALLBACK ─────────────────────────────────────────
def extract_company_name_fallback(full_text: str) -> str:
    candidates = re.findall(
        r'([A-Z][A-Za-z\s&,.\-]+(?:Limited|Pvt\.?\s*Ltd\.?|LLP|Inc\.?|Corporation|Enterprises|Industries|Company))',
        full_text[:2000]
    )
    if candidates:
        return max(candidates, key=len).strip()
    return "Unknown Company"


# ── MAIN EXTRACT ENDPOINT ─────────────────────────────────────────
@app.post("/extract")
async def extract_pdf(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files allowed")

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

    financial_text = select_pages(page_texts, score_financial_page, FINANCIAL_TEXT_LIMIT, top_n=10)
    notes_text     = select_pages(page_texts, score_notes_page,     NOTES_TEXT_LIMIT,     top_n=5)

    # ── Hard cap — prevents 413 on large PDFs ────────────────────
    combined = len(financial_text) + len(notes_text)
    if combined > MAX_COMBINED_CHARS:
        fin_budget   = int(MAX_COMBINED_CHARS * 0.82)
        notes_budget = MAX_COMBINED_CHARS - fin_budget
        financial_text = financial_text[:fin_budget]
        notes_text     = notes_text[:notes_budget]

    doc_keys_list = list(doc_schemas.keys())
    doc_keys_json = json.dumps(doc_keys_list)
    schema_desc   = ""
    data_hint     = ""

    for doc_key, sections in doc_schemas.items():
        desc = "\n".join([f'  "{s["key"]}": {s["title"]}' for s in sections])
        schema_desc += f'\n{doc_key.upper()} section keys:\n{desc}\n'
        data_hint   += f'    "{doc_key}": {{ "section_key": {{ "FY2024": {{ "Line Item": 1234.00 }} }} }},\n'

    prompt = f"""You are a senior financial analyst extraction engine specialising in Indian statutory audit reports (Schedule III format).

The document text has been extracted with table columns separated by " | ".
Each row: Line Item Name | Note No | VALUE_YEAR1 | VALUE_YEAR2

Return ONLY valid JSON with this exact structure:

{{
  "company_name": "string",
  "period": "FY2024",
  "all_periods": ["FY2024", "FY2023"],
  "confidence": {{ "Exact Item Name": 95 }},
  "data": {{
{data_hint}  }},
  "notes_to_accounts": [
    {{
      "number": "1",
      "title": "Note Title",
      "text": "Free-text accounting policy or observation (if any)",
      "table": [
        {{ "item": "Sub-item name", "FY2024": 1234.00, "FY2023": 1000.00 }}
      ]
    }}
  ]
}}

The "data" object MUST contain keys for: {doc_keys_json}

{schema_desc}

═══════════ FINANCIAL STATEMENTS ═══════════
{financial_text}

═══════════ NOTES TO ACCOUNTS ═══════════
{notes_text}

Extraction Rules:
1. YEAR DETECTION: First value column = most recent year, second = prior year. Normalise → "FY20XX".
2. FINANCIAL DATA:
   - Map each line item to the nearest schema section key
   - Read ONLY values physically in each cell — NEVER infer or carry over
   - "-", blank, "Nil" → 0
   - Read each year column independently — NEVER copy a value to both years
   - Remove all commas: "1,26,44,429.00" → 12644429.00. All positive floats.
3. NOTES TO ACCOUNTS:
   - Extract numbered financial notes and their tables only
   - Skip auditor reports, legal annexures, compliance paragraphs
4. CONFIDENCE: self-score 0-100 per financial item.
5. Return ONLY the JSON. No markdown."""

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict financial OCR parser for Indian statutory reports. Extract only explicitly present values. Return only valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        parsed = json.loads(response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(500, f"AI Processing Error: {str(e)}")

    # ── Clean numeric values — always 2 decimal places ────────────
    def clean_val(v):
        try:
            return round(float(str(v).replace(',', '').replace(' ', '').strip()), 2)
        except Exception:
            return 0.00

    # ── Process financial data ────────────────────────────────────
    raw_data   = parsed.get("data", {})
    final_data = {}

    for doc_key in doc_keys_list:
        final_data[doc_key] = {}
        for sec_key, periods in raw_data.get(doc_key, {}).items():
            final_data[doc_key][sec_key] = {}
            if isinstance(periods, dict):
                for period, items in periods.items():
                    if isinstance(items, dict):
                        final_data[doc_key][sec_key][period] = {
                            k: clean_val(v) for k, v in items.items()
                        }

    # ── Clean notes ───────────────────────────────────────────────
    raw_notes   = parsed.get("notes_to_accounts", [])
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

    # ── Confidence ────────────────────────────────────────────────
    raw_conf        = parsed.get("confidence", {})
    item_confidence = {}

    for doc_key, doc_data in final_data.items():
        for sec_data in doc_data.values():
            for period_data in sec_data.values():
                for item_name, item_val in period_data.items():
                    if item_name not in item_confidence:
                        item_confidence[item_name] = compute_item_confidence(
                            item_name, item_val, full_text, raw_conf
                        )

    # ── Company name — always populated ──────────────────────────
    company_name = (
        parsed.get("company_name") or
        extract_company_name_fallback(full_text)
    )

    return financial_json_response({
        "mapped": {
            "companyname":       company_name,
            "period":            parsed.get("period", "Unknown"),
            "allperiods":        parsed.get("all_periods", [parsed.get("period", "Unknown")]),
            "data":              final_data,
            "notes_to_accounts": clean_notes,
        },
        "raw":        full_text[:3000],
        "confidence": item_confidence
    })


# ── HEALTH & DEBUG ────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Redwood PDF Extractor running"}

@app.get("/health")
async def health():
    return {"status": "ok", "model": "llama-3.3-70b-versatile", "context": "131k tokens"}

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
            "page":            page_num + 1,
            "financial_score": score_financial_page(text),
            "notes_score":     score_notes_page(text),
            "chars":           len(text),
            "preview":         text[:120]
        })

    fin_text   = select_pages(page_texts, score_financial_page, FINANCIAL_TEXT_LIMIT, 10)
    notes_text = select_pages(page_texts, score_notes_page,     NOTES_TEXT_LIMIT,      5)
    pdf_doc.close()

    return {
        "total_pages":          len(page_data),
        "page_scores":          sorted(page_data, key=lambda x: x["financial_score"], reverse=True),
        "financial_text_chars": len(fin_text),
        "notes_text_chars":     len(notes_text),
        "financial_preview":    fin_text[:2000],
        "notes_preview":        notes_text[:2000]
    }
