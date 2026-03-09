from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz
import io
import pytesseract
from PIL import Image
from groq import Groq
import os
import json
from typing import Any
from collections import defaultdict
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY", "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"))

FINANCIAL_KEYWORDS = [
    "revenue", "income", "expenditure", "expenses", "profit", "loss",
    "balance sheet", "assets", "liabilities", "equity", "capital",
    "cash", "receivable", "payable", "depreciation", "surplus",
    "schedule", "fixed assets", "current assets", "borrowings",
    "turnover", "sales", "purchases", "gross", "net", "ebitda",
    "reserves", "investments", "provisions", "creditors", "debtors"
]

INDIAN_NUMBER_RE = re.compile(r'\d{1,2}(,\d{2})*,\d{3}(\.\d+)?|\d[\d,]+\.\d{2}')

class ExtractResponse(BaseModel):
    mapped: Any
    raw: str
    confidence: Any


def extract_page_text_structured(page):
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


def score_page(text: str) -> int:
    score = 0
    lower = text.lower()
    indian_nums = INDIAN_NUMBER_RE.findall(text)
    score += len(indian_nums) * 3
    plain_nums = re.findall(r'\d[\d,]{3,}', text)
    score += len(plain_nums)
    for kw in FINANCIAL_KEYWORDS:
        if kw in lower:
            score += 2
    score += text.count("|") * 2
    return score


def compute_item_confidence(item_name: str, val: float, page_text: str, model_conf: dict) -> int:
    score = 0

    model_score = model_conf.get(item_name)
    if model_score is None:
        lower_key = item_name.lower().strip()
        model_score = next(
            (v for k, v in model_conf.items() if k.lower().strip() == lower_key),
            None
        )
    if model_score is not None:
        score += min(20, int(model_score * 0.2))
    else:
        score += 10

    if val and val != 0:
        score += 20

    if val and val > 10000:
        score += 20

    if item_name.lower() in page_text.lower():
        score += 20

    if "|" in page_text and val and val > 0:
        score += 20

    return min(score, 100)


@app.post("/extract", response_model=ExtractResponse)
async def extract_pdf(
    file: UploadFile = File(...),
    pnl_schema: str = Form(default="[]"),
    bs_schema: str = Form(default="[]")
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files allowed")

    contents = await file.read()
    doc = fitz.open(stream=contents, filetype="pdf")

    full_text = ""
    page_data = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = extract_page_text_structured(page)

        if not text.strip():
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang='eng').strip()

        full_text += text + "\n"
        page_data.append((score_page(text), text))

    doc.close()

    if not full_text.strip():
        raise HTTPException(400, "No text found. Try a clearer PDF.")

    page_data.sort(key=lambda x: x[0], reverse=True)

    text_for_model = ""
    for _, text in page_data:
        if len(text_for_model) >= 8000:
            break
        text_for_model += text + "\n"
    text_for_model = text_for_model[:8000].strip()

    if not text_for_model:
        text_for_model = full_text[:8000].strip()

    pnl_sections = json.loads(pnl_schema)
    bs_sections  = json.loads(bs_schema)

    pnl_desc = "\n".join([f'  "{s["key"]}": {s["title"]}' for s in pnl_sections])
    bs_desc  = "\n".join([f'  "{s["key"]}": {s["title"]}' for s in bs_sections])

    prompt = f"""You are a financial data extraction assistant for Indian financial statements.

The document text has been extracted with table columns separated by " | ".
Each row looks like:   Line Item Name  |  Note No  |  VALUE_YEAR1  |  VALUE_YEAR2

Example of how Indian financial tables appear:
  Some Income Line Item   |  3  |  5,00,00,000.00  |  4,20,00,000.00
  Some Expense Line Item  |  7  |  1,75,50,000.00  |  1,60,25,000.00
  Some Asset Line Item    |  4  |  8,30,00,000.00  |  7,90,00,000.00

Year column headers appear somewhere above like:
  For the Year Ended 31st March, 2024  |  For the Year Ended 31st March, 2023
or:  2023-24  |  2022-23
or:  As at 31.03.2024  |  As at 31.03.2023

Correct output for the above example:
  "Some Income Line Item"  FY_RECENT = 50000000.00   FY_PRIOR = 42000000.00
  (DIFFERENT values per year — read each column independently)

Return ONLY valid JSON:

{{
  "company_name": "string",
  "period": "FY2024",
  "all_periods": ["FY2024", "FY2023"],
  "confidence": {{
    "Exact Item Name As In Data": 95,
    "Another Item Name": 90
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
- Identify year columns from headers — FIRST value column = most recent year, SECOND = prior year
- Read ONLY the value that physically appears in each cell — NEVER infer, carry over, or substitute
- If a cell shows "-", blank, or is empty → output 0. No exceptions.
- NEVER use a value from a different row to fill a blank cell
- NEVER use totals (e.g. Total Assets, Total Liabilities) to fill line item fields
- For items that have sub-breakdown lines (e.g. Opening Stock / Purchases / Closing Stock under COGS):
    → Extract ONLY the final net/total line for that item
    → If the net total cell shows "-" or blank → output 0, even if sub-lines have values
    → Do NOT sum the sub-lines yourself
- EVERY row must produce TWO separate values, one per year column
- NEVER copy the same value to both years — read each column independently
- Normalise year labels → "FY20XX" (e.g. "31.03.2022" → "FY2022", "2023-24" → "FY2024")
- Remove ALL commas before converting: "1,26,44,429.00" → 12644429.00
- All values must be positive floats
- confidence keys MUST use the EXACT same item name string as in the data output
- Score 90-100 for items clearly readable in a table row; 60-80 for inferred or partially visible
- "notes" = 3-6 key points from auditor notes/accounting policies. [] if none.
- Empty sections → {{}}
- Return ONLY the JSON. No markdown. No explanation.

Document:
{text_for_model}"""

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a strict financial data extraction assistant. Extract ONLY values that are explicitly present in the document. Never infer, guess, or fill in missing values. Return only valid JSON."},
            {"role": "user",   "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=3000,
        response_format={"type": "json_object"}
    )

    raw_json = chat_completion.choices[0].message.content

    try:
        parsed = json.loads(raw_json)
    except Exception:
        raise HTTPException(500, "Model returned invalid JSON. Try again.")

    def clean_number(val):
        if isinstance(val, str):
            try:
                return float(val.replace(",", "").replace(" ", "").strip())
            except:
                return 0
        return val if val is not None else 0

    def clean_section(section_dict):
        cleaned = {}
        for period, items in section_dict.items():
            cleaned[period] = {}
            if isinstance(items, dict):
                for item, val in items.items():
                    cleaned[period][item] = clean_number(val)
        return cleaned

    data = parsed.get("data", {"pnl": {}, "bs": {}})

    clean_data = {"pnl": {}, "bs": {}}
    for key, val in data.get("pnl", {}).items():
        clean_data["pnl"][key] = clean_section(val)
    for key, val in data.get("bs", {}).items():
        clean_data["bs"][key] = clean_section(val)

    raw_conf = parsed.get("confidence", {})

    item_confidence = {}
    for key, val in clean_data["pnl"].items():
        for period, items in val.items():
            for item_name, item_val in items.items():
                if item_name not in item_confidence:
                    item_confidence[item_name] = compute_item_confidence(
                        item_name, item_val, text_for_model, raw_conf
                    )
    for key, val in clean_data["bs"].items():
        for period, items in val.items():
            for item_name, item_val in items.items():
                if item_name not in item_confidence:
                    item_confidence[item_name] = compute_item_confidence(
                        item_name, item_val, text_for_model, raw_conf
                    )

    all_periods = parsed.get("all_periods", [parsed.get("period", "Unknown")])
    notes = parsed.get("notes", [])

    return ExtractResponse(
    mapped={
        "companyname":  parsed.get("company_name", ""),   # ✅ fsa.js reads this
        "period":       parsed.get("period", "Unknown"),
        "allperiods":   all_periods,                       # ✅ fsa.js reads this
        "data":         clean_data,
        "notes":        notes
    },
    raw=full_text[:2000],
    confidence=item_confidence
)



@app.get("/health")
async def health():
    return {"status": "ok", "model": "llama-3.3-70b-versatile", "inference": "groq-cloud"}


@app.post("/debug")
async def debug_extract(file: UploadFile = File(...)):
    contents = await file.read()
    doc = fitz.open(stream=contents, filetype="pdf")

    full_text = ""
    page_data = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = extract_page_text_structured(page)
        if not text.strip():
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang='eng').strip()

        full_text += text + "\n"
        page_data.append((score_page(text), text))

    doc.close()

    page_data.sort(key=lambda x: x[0], reverse=True)

    text_for_model = ""
    for _, text in page_data:
        if len(text_for_model) >= 8000:
            break
        text_for_model += text + "\n"
    text_for_model = text_for_model[:8000].strip()

    return {
        "full_text_length": len(full_text),
        "text_sent_to_model_length": len(text_for_model),
        "page_scores": [(s, t[:80]) for s, t in sorted(page_data, key=lambda x: x[0], reverse=True)],
        "text_sent_to_model": text_for_model
    }
