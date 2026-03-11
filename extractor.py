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
import re


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

FINANCIAL_KEYWORDS = [
    "revenue","income","expenditure","expenses","profit","loss",
    "balance sheet","assets","liabilities","equity","capital",
    "cash","receivable","payable","depreciation","surplus",
    "schedule","fixed assets","current assets","borrowings",
    "turnover","sales","purchases","gross","net","ebitda",
    "reserves","investments","provisions","creditors","debtors"
]

INDIAN_NUMBER_RE = re.compile(r'\d{1,3}(?:,\d{2,3})*(?:\.\d+)?')


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

    if not file or not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed")

    contents = await file.read()

    try:
        doc = fitz.open(stream=contents, filetype="pdf")
    except Exception:
        raise HTTPException(400, "Invalid or corrupted PDF")

    full_text = ""

    for page_num in range(len(doc)):

        page = doc[page_num]

        text = extract_page_text_structured(page)

        if not text.strip():

            raw = page.get_text()

            if not raw.strip():
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img, lang="eng").strip()
            else:
                text = raw

        full_text += text + "\n"

    doc.close()

    if not full_text.strip():
        raise HTTPException(400, "No text found. Try a clearer PDF.")

    text_for_model = full_text[:12000].strip()

    try:
        pnl_sections = json.loads(pnl_schema) if pnl_schema else []
    except Exception:
        pnl_sections = []

    try:
        bs_sections = json.loads(bs_schema) if bs_schema else []
    except Exception:
        bs_sections = []

    pnl_desc = "\n".join([f'  "{s["key"]}": {s["title"]}' for s in pnl_sections])
    bs_desc = "\n".join([f'  "{s["key"]}": {s["title"]}' for s in bs_sections])

    prompt = f"""You are a financial data extraction assistant for Indian financial statements.

Document:
{text_for_model}
"""

    try:

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a strict financial data extraction assistant. Extract ONLY values explicitly present."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=8000,
            response_format={"type": "json_object"}
        )

    except Exception as e:
        raise HTTPException(500, f"LLM request failed: {str(e)}")

    raw_json = chat_completion.choices[0].message.content

    try:
        parsed = json.loads(raw_json)
    except Exception:
        raise HTTPException(500, "Model returned invalid JSON")

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
            "companyname": parsed.get("company_name", ""),
            "period": parsed.get("period", "Unknown"),
            "allperiods": all_periods,
            "data": clean_data,
            "notes": notes
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

            raw = page.get_text()

            if not raw.strip():
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img, lang='eng').strip()
            else:
                text = raw

        full_text += text + "\n"

        page_data.append((score_page(text), text))

    doc.close()

    return {
        "full_text_length": len(full_text),
        "text_sent_to_model": full_text[:12000],
        "page_scores": [(s, t[:80]) for s, t in sorted(page_data, key=lambda x: x[0], reverse=True)]
    }
