from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz
import io
import os
import re
import json
from typing import Any
from groq import Groq

# ── Optional OCR — only imported if Tesseract is available ────────
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

# ── Constants ─────────────────────────────────────────────────────
MAX_PAGES        = 20       # never process more than this many pages
MAX_TEXT_CHARS   = 15000    # stop collecting text after this threshold
MODEL_MAX_TOKENS = 3000     # safe for Railway memory + Groq latency
TOP_PAGES_KEPT   = 8        # only best-scored pages go to LLM
TEXT_TO_MODEL    = 12000    # chars sliced for LLM prompt

FINANCIAL_KEYWORDS = [
    "revenue", "income", "expenditure", "expenses", "profit", "loss",
    "balance sheet", "assets", "liabilities", "equity", "capital",
    "cash", "receivable", "payable", "depreciation", "surplus",
    "schedule", "fixed assets", "current assets", "borrowings",
    "turnover", "sales", "purchases", "gross", "net", "ebitda",
    "reserves", "investments", "provisions", "creditors", "debtors"
]

INDIAN_NUMBER_RE = re.compile(r'\d{1,2}(,\d{2})*,\d{3}(\.\d+)?|\d[\d,]+\.\d{2}')


# ── Response model ────────────────────────────────────────────────
class ExtractResponse(BaseModel):
    mapped: Any
    raw: str
    confidence: Any


# ── Page text extraction ──────────────────────────────────────────
def extract_page_text_structured(page) -> str:
    """Extract text from a PDF page preserving column layout via span coords."""
    try:
        raw_dict = page.get_text("dict")
        all_spans = []

        for block in raw_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text:
                        continue
                    x = span["origin"][0]
                    y = span["origin"][1]
                    all_spans.append((y, x, text))

        if not all_spans:
            return page.get_text().strip()

        all_spans.sort(key=lambda s: (s[0], s[1]))

        rows = []
        current_row = [all_spans[0]]
        for i in range(1, len(all_spans)):
            y_curr = all_spans[i][0]
            y_prev = current_row[-1][0]
            if abs(y_curr - y_prev) <= 4:
                current_row.append(all_spans[i])
            else:
                rows.append(current_row)
                current_row = [all_spans[i]]
        rows.append(current_row)

        lines = []
        for row in rows:
            row_sorted = sorted(row, key=lambda s: s[1])
            row_text = "  |  ".join(s[2] for s in row_sorted)
            lines.append(row_text)

        return "\n".join(lines)

    except Exception:
        return page.get_text().strip()


# ── Page scoring ──────────────────────────────────────────────────
def score_page(text: str) -> int:
    """Score a page by how financially relevant its content is."""
    score = 0
    lower = text.lower()
    score += len(INDIAN_NUMBER_RE.findall(text)) * 3
    score += len(re.findall(r'\d[\d,]{3,}', text))
    for kw in FINANCIAL_KEYWORDS:
        if kw in lower:
            score += 2
    score += text.count("|") * 2
    return score


# ── OCR fallback (only if no native text layer at all) ───────────
def ocr_page(page) -> str:
    """Run Tesseract OCR only if the page has zero native text."""
    if not OCR_AVAILABLE:
        return ""
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img, lang="eng").strip()
    except Exception:
        return ""


# ── Number cleaning ───────────────────────────────────────────────
def clean_number(val) -> float:
    """Safely convert any value (str/int/float/None) to a clean float."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        cleaned = val.replace(",", "").replace(" ", "").strip()
        if not cleaned or cleaned in ("-", "—", "nil", "n/a"):
            return 0.0
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    return 0.0


def clean_section(section_dict: dict) -> dict:
    """Normalize all values inside a PnL/BS section dict."""
    cleaned = {}
    for period, items in section_dict.items():
        cleaned[period] = {}
        if isinstance(items, dict):
            for item, val in items.items():
                cleaned[period][item] = clean_number(val)
    return cleaned


# ── Confidence scoring ────────────────────────────────────────────
def compute_item_confidence(item_name: str, val: float, page_text: str, model_conf: dict) -> int:
    score = 0

    # Model's own confidence
    lower_key = item_name.lower().strip()
    model_score = model_conf.get(item_name) or next(
        (v for k, v in model_conf.items() if k.lower().strip() == lower_key), None
    )
    score += min(20, int(model_score * 0.2)) if model_score is not None else 10

    if val and val != 0:
        score += 20
    if val and val > 10000:
        score += 20
    if item_name.lower() in page_text.lower():
        score += 20
    if "|" in page_text and val and val > 0:
        score += 20

    return min(score, 100)


# ── PDF text extraction pipeline ─────────────────────────────────
def extract_text_from_pdf(contents: bytes) -> tuple[str, str]:
    """
    Returns (text_for_model, raw_preview).
    Uses page scoring to send only the most relevant pages to the LLM.
    Stops processing early once enough text is collected.
    """
    doc = fitz.open(stream=contents, filetype="pdf")
    page_texts = []

    try:
        for page_num in range(min(len(doc), MAX_PAGES)):
            page = doc[page_num]

            # Try native text first
            native_text = page.get_text().strip()
            if native_text:
                structured = extract_page_text_structured(page)
                text = structured if structured.strip() else native_text
            else:
                # OCR only when truly no text layer
                text = ocr_page(page)

            if text.strip():
                page_texts.append(text)

            # Stop early — no need to parse 100-page annual reports in full
            total_so_far = sum(len(t) for t in page_texts)
            if total_so_far >= MAX_TEXT_CHARS:
                break

    finally:
        doc.close()

    if not page_texts:
        return "", ""

    # Score each page and keep only the most financially relevant ones
    scored = sorted(
        enumerate(page_texts),
        key=lambda x: score_page(x[1]),
        reverse=True
    )
    top_pages = sorted(scored[:TOP_PAGES_KEPT], key=lambda x: x[0])  # restore doc order
    selected_text = "\n\n".join(t for _, t in top_pages)

    raw_preview = selected_text[:2000]
    text_for_model = selected_text[:TEXT_TO_MODEL].strip()

    return text_for_model, raw_preview


# ── LLM prompt ────────────────────────────────────────────────────
def build_prompt(text: str, pnl_desc: str, bs_desc: str) -> str:
    return f"""You are a financial data extraction assistant for Indian financial statements.

The document text has been extracted with table columns separated by " | ".
Each row looks like:   Line Item Name  |  Note No  |  VALUE_YEAR1  |  VALUE_YEAR2

Example:
  Some Income Line Item   |  3  |  5,00,00,000.00  |  4,20,00,000.00
  Some Expense Line Item  |  7  |  1,75,50,000.00  |  1,60,25,000.00

Year headers appear above like:
  For the Year Ended 31st March, 2024  |  For the Year Ended 31st March, 2023
or:  2023-24  |  2022-23

Return ONLY valid JSON:

{{
  "company_name": "string",
  "period": "FY2024",
  "all_periods": ["FY2024", "FY2023"],
  "confidence": {{
    "Exact Item Name As In Data": 95
  }},
  "notes": ["Key auditor observation or accounting policy point"],
  "data": {{
    "pnl": {{
      "section_key": {{
        "FY2024": {{ "Line Item Name": 1234567.89 }},
        "FY2023": {{ "Line Item Name": 9876543.21 }}
      }}
    }},
    "bs": {{
      "section_key": {{
        "FY2024": {{ "Line Item Name": 5000000.00 }},
        "FY2023": {{ "Line Item Name": 4500000.00 }}
      }}
    }}
  }}
}}

P&L section keys:
{pnl_desc}

Balance Sheet section keys:
{bs_desc}

Extraction Rules:
- FIRST value column = most recent year, SECOND = prior year
- Read ONLY the value physically in each cell — NEVER infer or carry over
- If a cell shows "-", blank, or empty → output 0
- NEVER copy the same value to both years
- For sub-breakdown items → extract ONLY the final net/total line
- Normalise year labels → "FY20XX" ("31.03.2022" → "FY2022", "2023-24" → "FY2024")
- Remove ALL commas before converting: "1,26,44,429.00" → 12644429.00
- All values must be positive floats
- confidence keys MUST match exact item name strings in data output
- Score 90-100 = clearly readable; 60-80 = inferred/partially visible
- "notes" = 3-5 key auditor notes or accounting policy points. [] if none.
- Empty sections → {{}}
- Return ONLY the JSON. No markdown. No explanation.

Document:
{text}"""


# ── Main extraction endpoint ──────────────────────────────────────
@app.post("/extract", response_model=ExtractResponse)
async def extract_pdf(
    file: UploadFile = File(...),
    pnl_schema: str = Form(default="[]"),
    bs_schema: str = Form(default="[]")
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are allowed.")

    contents = await file.read()

    text_for_model, raw_preview = extract_text_from_pdf(contents)

    if not text_for_model:
        raise HTTPException(400, "No text could be extracted. Try a clearer PDF.")

    pnl_sections = json.loads(pnl_schema)
    bs_sections  = json.loads(bs_schema)
    pnl_desc = "\n".join(f'  "{s["key"]}": {s["title"]}' for s in pnl_sections)
    bs_desc  = "\n".join(f'  "{s["key"]}": {s["title"]}' for s in bs_sections)

    prompt = build_prompt(text_for_model, pnl_desc, bs_desc)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict financial data extraction assistant. "
                        "Extract ONLY values explicitly present in the document. "
                        "Never infer, guess, or fill in missing values. "
                        "Return only valid JSON."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=MODEL_MAX_TOKENS,
            response_format={"type": "json_object"}
        )
    except Exception as e:
        raise HTTPException(502, f"LLM call failed: {str(e)}")

    raw_json = chat_completion.choices[0].message.content

    try:
        parsed = json.loads(raw_json)
    except Exception:
        raise HTTPException(500, "Model returned invalid JSON. Please try again.")

    # ── Clean extracted data ──────────────────────────────────────
    data = parsed.get("data", {"pnl": {}, "bs": {}})
    clean_data = {"pnl": {}, "bs": {}}

    for key, val in data.get("pnl", {}).items():
        clean_data["pnl"][key] = clean_section(val)
    for key, val in data.get("bs", {}).items():
        clean_data["bs"][key] = clean_section(val)

    # ── Confidence map ────────────────────────────────────────────
    raw_conf = parsed.get("confidence", {})
    item_confidence = {}

    for section_data in [clean_data["pnl"], clean_data["bs"]]:
        for period_map in section_data.values():
            for period_items in period_map.values():
                for item_name, item_val in period_items.items():
                    if item_name not in item_confidence:
                        item_confidence[item_name] = compute_item_confidence(
                            item_name, item_val, text_for_model, raw_conf
                        )

    all_periods = parsed.get("all_periods", [parsed.get("period", "Unknown")])
    notes       = parsed.get("notes", [])

    return ExtractResponse(
        mapped={
            "companyname": parsed.get("company_name", ""),
            "period":      parsed.get("period", "Unknown"),
            "allperiods":  all_periods,
            "data":        clean_data,
            "notes":       notes
        },
        raw=raw_preview,
        confidence=item_confidence
    )


# ── Health check ──────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "llama-3.3-70b-versatile",
        "inference": "groq-cloud",
        "ocr_available": OCR_AVAILABLE
    }


# ── Debug endpoint ────────────────────────────────────────────────
@app.post("/debug")
async def debug_extract(file: UploadFile = File(...)):
    contents = await file.read()
    doc = fitz.open(stream=contents, filetype="pdf")
    page_data = []

    try:
        for page_num in range(min(len(doc), MAX_PAGES)):
            page = doc[page_num]
            native = page.get_text().strip()
            text = extract_page_text_structured(page) if native else ocr_page(page)
            page_data.append((score_page(text), page_num, text))
    finally:
        doc.close()

    page_data_sorted = sorted(page_data, key=lambda x: x[0], reverse=True)

    return {
        "total_pages_processed": len(page_data),
        "top_pages_by_score": [
            {"page": p, "score": s, "preview": t[:120]}
            for s, p, t in page_data_sorted[:10]
        ],
        "text_sent_to_model": "\n\n".join(
            t for _, _, t in sorted(page_data_sorted[:TOP_PAGES_KEPT], key=lambda x: x[1])
        )[:TEXT_TO_MODEL]
    }
