# extractor.py — Redwood Production Final
# Fixes: regex, page scoring, large PDF support, GitHub Models (no TPM limits)

import fitz
import io
import os
import re
import json
import pytesseract
from PIL import Image
from openai import OpenAI
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

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN,
)
MODEL = "gpt-4o-mini"

FINANCIAL_TEXT_LIMIT = 24_000
NOTES_TEXT_LIMIT     = 18_000

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


class ExtractResponse(BaseModel):
    mapped: Any
    raw: str
    confidence: Any


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

    score += len(INDIAN_NUMBER_RE.findall(text)) * 4
    score += len(re.findall(r'\d[\d,]{4,}', text)) * 2

    for kw in FINANCIAL_KEYWORDS:
        if kw in lower:
            score += 3

    score += text.count("|") * 3

    year_hits = re.findall(
        r'(FY\s*\d{2,4}|20\d{2}[-\u2013]\d{2,4}|31[.\-/]\d{2}[.\-/]\d{2,4})', text
    )
    score += len(year_hits) * 5

    words     = len(text.split())
    num_count = len(re.findall(r'\d+', text))
    if words > 80 and (num_count / max(words, 1)) < 0.05:
        score = int(score * 0.3)

    return score


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
    if words > 30 and (num_count / max(words, 1)) > 0.5:
        score = int(score * 0.4)

    return score


# ── SMART PAGE SELECTOR ───────────────────────────────────────────
def select_pages(page_texts: list, scorer_fn, char_limit: int) -> str:
    total_pages = len(page_texts)
    top_n       = max(20, total_pages // 5)

    scored = []
    for page_num, text in enumerate(page_texts):
        score = scorer_fn(text)
        if score > 0:
            scored.append((score, page_num, text))

    # Fallback: if fewer than 3 pages scored, use second half positionally
    if len(scored) < 3:
        start = max(0, total_pages // 3)
        scored = [
            (1, i, page_texts[i])
            for i in range(start, total_pages)
            if page_texts[i].strip()
        ]
        if not scored:
            scored = [(1, i, page_texts[i]) for i in range(total_pages) if page_texts[i].strip()]

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_n]

    # Window padding: include preceding page for split header/data pages
    top_indices = set(p[1] for p in top)
    for _, page_num, _ in list(top):
        if page_num > 0 and (page_num - 1) not in top_indices:
            prev_text = page_texts[page_num - 1]
            if prev_text.strip():
                top.append((0, page_num - 1, prev_text))
                top_indices.add(page_num - 1)

    top.sort(key=lambda x: x[1])

    result = ""
    for _, _, text in top:
        if len(result) + len(text) + 1 <= char_limit:
            result += text + "\n"
        else:
            remaining = char_limit - len(result)
            if remaining > 200:
                result += text[:remaining] + "\n"
            continue

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


# ── MAIN EXTRACT ENDPOINT ─────────────────────────────────────────
@app.post("/extract", response_model=ExtractResponse)
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

    financial_text = select_pages(page_texts, score_financial_page, FINANCIAL_TEXT_LIMIT)
    notes_text     = select_pages(page_texts, score_notes_page,     NOTES_TEXT_LIMIT)

    if not financial_text.strip():
        financial_text = full_text[:FINANCIAL_TEXT_LIMIT]
    if not notes_text.strip():
        notes_text = full_text[:NOTES_TEXT_LIMIT]

    doc_keys_list = list(doc_schemas.keys())
    doc_keys_json = json.dumps(doc_keys_list)
    schema_desc   = ""
    data_hint     = ""

    for doc_key, sections in doc_schemas.items():
        desc = "\n".join([f'  "{s["key"]}": {s["title"]}' for s in sections])
        schema_desc += f'\n{doc_key.upper()} section keys:\n{desc}\n'
        data_hint   += f'    "{doc_key}": {{ "section_key": {{ "FY2024": {{ "Line Item": 1234.00 }} }} }},\n'

    # ── CALL 1 — Financial Data ───────────────────────────────────
    fin_prompt = f"""You are a senior financial analyst extraction engine specialising in Indian statutory audit reports (Schedule III format).

The document text has been extracted with table columns separated by " | ".
Each row: Line Item Name | Note No | VALUE_YEAR1 | VALUE_YEAR2

Return ONLY valid JSON:

{{
  "company_name": "string",
  "period": "FY2024",
  "all_periods": ["FY2024", "FY2023"],
  "confidence": {{ "Exact Item Name": 95 }},
  "data": {{
{data_hint}  }}
}}

The "data" object MUST contain keys for: {doc_keys_json}

{schema_desc}

Extraction Rules:
1. YEAR DETECTION: First value column = most recent year, second = prior year. Normalise to "FY20XX".
2. Map each line item to the nearest schema section key.
3. Read ONLY values physically in each cell — NEVER infer, carry over, or substitute.
4. "-", blank, "Nil" → 0. All values positive floats.
5. Remove all commas: "1,26,44,429.00" → 12644429.00
6. Read each year column independently — NEVER copy a value to both years.
7. Confidence: self-score 0-100 per item. 90-100=clearly readable, 60-89=inferred. Use EXACT item name string as in data.
8. Empty sections → {{}}
9. Return ONLY the JSON. No markdown. No explanation.

Document:
{financial_text}"""

    try:
        fin_response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict financial OCR parser for Indian statutory reports. Extract only explicitly present values. Return only valid JSON."
                },
                {"role": "user", "content": fin_prompt}
            ],
            model=MODEL,
            temperature=0,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        fin_parsed = json.loads(fin_response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(500, f"Financial extraction error: {str(e)}")

    # ── CALL 2 — Notes to Accounts ────────────────────────────────
    notes_parsed = {"notes_to_accounts": []}

    if notes_text.strip():
        notes_prompt = f"""You are extracting Notes to Accounts from an Indian statutory audit report.

Return ONLY valid JSON:

{{
  "notes_to_accounts": [
    {{
      "number": "1",
      "title": "Note Title",
      "text": "Any accounting policy or qualitative text. Empty string if none.",
      "table": [
        {{ "item": "Sub-item name", "FY2024": 1234.00, "FY2023": 1000.00 }}
      ]
    }}
  ]
}}

Rules:
- Extract EVERY numbered note and named schedule found in the document
- "number": note number as string ("1", "2", "2a", etc.)
- "title": the heading of the note
- "text": policy or qualitative explanation — empty string if none
- "table": array of line items with per-year values. [] if the note has no table.
- Remove all commas from numbers: "1,26,44,429.00" → 12644429.00
- Use "FY20XX" format for all year keys
- Return ONLY the JSON. No markdown. No explanation.

Document:
{notes_text}"""

        try:
            notes_response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict financial notes extractor. Return only valid JSON."
                    },
                    {"role": "user", "content": notes_prompt}
                ],
                model=MODEL,
                temperature=0,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            notes_parsed = json.loads(notes_response.choices[0].message.content)
        except Exception as e:
            print(f"Notes extraction failed (non-fatal): {str(e)}")
            notes_parsed = {"notes_to_accounts": []}

    # ── Clean numeric values ──────────────────────────────────────
    def clean_val(v):
        try:
            return float(str(v).replace(',', '').replace(' ', '').strip())
        except Exception:
            return 0.0

    raw_data   = fin_parsed.get("data", {})
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

    raw_conf        = fin_parsed.get("confidence", {})
    item_confidence = {}

    for doc_key, doc_data in final_data.items():
        for sec_data in doc_data.values():
            for period_data in sec_data.values():
                for item_name, item_val in period_data.items():
                    if item_name not in item_confidence:
                        item_confidence[item_name] = compute_item_confidence(
                            item_name, item_val, full_text, raw_conf
                        )

    return ExtractResponse(
        mapped={
            "companyname":       fin_parsed.get("company_name", "Unknown"),
            "period":            fin_parsed.get("period", "Unknown"),
            "allperiods":        fin_parsed.get("all_periods", [fin_parsed.get("period", "Unknown")]),
            "data":              final_data,
            "notes_to_accounts": clean_notes,
        },
        raw=full_text[:3000],
        confidence=item_confidence
    )


# ── HEALTH & DEBUG ────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Redwood PDF Extractor running"}

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "provider": "github-models"}

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

    fin_text   = select_pages(page_texts, score_financial_page, FINANCIAL_TEXT_LIMIT)
    notes_text = select_pages(page_texts, score_notes_page,     NOTES_TEXT_LIMIT)
    pdf_doc.close()

    return {
        "total_pages":          len(page_data),
        "page_scores":          sorted(page_data, key=lambda x: x["financial_score"], reverse=True),
        "financial_text_chars": len(fin_text),
        "notes_text_chars":     len(notes_text),
        "financial_preview":    fin_text[:2000],
        "notes_preview":        notes_text[:2000]
    }
