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

# Railway-safe limits
MAX_PAGES = 20
MAX_TEXT = 15000

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
        spans = []

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

                    spans.append((y, x, text))

        if not spans:
            return page.get_text()

        spans.sort(key=lambda s: (s[0], s[1]))

        rows = []
        current = [spans[0]]

        for i in range(1, len(spans)):
            if abs(spans[i][0] - current[-1][0]) <= 4:
                current.append(spans[i])
            else:
                rows.append(current)
                current = [spans[i]]

        rows.append(current)

        lines = []

        for row in rows:
            row_sorted = sorted(row, key=lambda s: s[1])
            line = " | ".join(s[2] for s in row_sorted)
            lines.append(line)

        return "\n".join(lines)

    except Exception:
        return page.get_text()


def score_page(text: str) -> int:

    score = 0
    lower = text.lower()

    numbers = INDIAN_NUMBER_RE.findall(text)
    score += len(numbers) * 3

    for kw in FINANCIAL_KEYWORDS:
        if kw in lower:
            score += 3

    score += text.count("|") * 2

    return score


def compute_item_confidence(item_name, val, page_text):

    score = 20

    if item_name.lower() in page_text.lower():
        score += 30

    if val and val > 0:
        score += 25

    if "|" in page_text:
        score += 25

    return min(score,100)


def run_ocr(page):

    pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
    img = Image.open(io.BytesIO(pix.tobytes("png")))

    return pytesseract.image_to_string(img, lang="eng")


@app.post("/extract", response_model=ExtractResponse)
async def extract_pdf(
    file: UploadFile = File(...),
    pnl_schema: str = Form(default="[]"),
    bs_schema: str = Form(default="[]")
):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(400,"Only PDF allowed")

    contents = await file.read()

    doc = fitz.open(stream=contents,filetype="pdf")

    page_texts = []

    for page_num in range(min(len(doc), MAX_PAGES)):

        page = doc[page_num]

        text = extract_page_text_structured(page)

        if not text.strip():
            raw = page.get_text()
            if not raw.strip():
                text = run_ocr(page)
            else:
                text = raw

        page_texts.append(text)

    doc.close()

    if not page_texts:
        raise HTTPException(400,"No readable text")

    scored_pages = sorted(
        [(score_page(t),t) for t in page_texts],
        key=lambda x: x[0],
        reverse=True
    )

    selected = []
    total_len = 0

    for score,text in scored_pages:

        if total_len > MAX_TEXT:
            break

        selected.append(text)
        total_len += len(text)

    text_for_model = "\n".join(selected)

    pnl_sections = json.loads(pnl_schema)
    bs_sections = json.loads(bs_schema)

    pnl_desc = "\n".join([f'"{s["key"]}": {s["title"]}' for s in pnl_sections])
    bs_desc = "\n".join([f'"{s["key"]}": {s["title"]}' for s in bs_sections])

    prompt = f"""
You are a financial statement extraction system for Indian financial reports.

The document text uses "|" as table column separators.

Return ONLY JSON.

Rules:

- Identify two year columns
- FIRST value column = most recent FY
- SECOND column = prior FY
- Do not infer missing values
- "-" or blank = 0
- Remove commas before converting numbers
- All numbers positive floats

Notes extraction rules:

- Extract 3–6 key auditor observations
- Prefer audit qualifications or limitations
- Ignore generic accounting policy text
- Use short clear sentences

JSON structure:

{{
"company_name":"",
"period":"",
"all_periods":[],
"confidence":{{}},
"notes":[],
"data":{{"pnl":{{}},"bs":{{}}}}
}}

P&L sections:
{pnl_desc}

Balance Sheet sections:
{bs_desc}

Document:
{text_for_model}
"""

    completion = client.chat.completions.create(
        messages=[
            {"role":"system","content":"Strict financial extraction. Never guess numbers."},
            {"role":"user","content":prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=2500,
        response_format={"type":"json_object"}
    )

    raw_json = completion.choices[0].message.content

    try:
        parsed = json.loads(raw_json)
    except:
        raise HTTPException(500,"Invalid JSON returned by model")

    def clean_number(v):
        if isinstance(v,str):
            try:
                return float(v.replace(",","").strip())
            except:
                return 0
        return v if v else 0

    clean_data = {"pnl":{},"bs":{}}

    for sec,val in parsed.get("data",{}).get("pnl",{}).items():
        clean_data["pnl"][sec] = {}
        for period,items in val.items():
            clean_data["pnl"][sec][period] = {
                k:clean_number(v) for k,v in items.items()
            }

    for sec,val in parsed.get("data",{}).get("bs",{}).items():
        clean_data["bs"][sec] = {}
        for period,items in val.items():
            clean_data["bs"][sec][period] = {
                k:clean_number(v) for k,v in items.items()
            }

    confidence = {}

    for section in clean_data["pnl"].values():
        for period,items in section.items():
            for name,val in items.items():
                if name not in confidence:
                    confidence[name] = compute_item_confidence(name,val,text_for_model)

    for section in clean_data["bs"].values():
        for period,items in section.items():
            for name,val in items.items():
                if name not in confidence:
                    confidence[name] = compute_item_confidence(name,val,text_for_model)

    return ExtractResponse(
        mapped={
            "companyname": parsed.get("company_name",""),
            "period": parsed.get("period",""),
            "allperiods": parsed.get("all_periods",[]),
            "data": clean_data,
            "notes": parsed.get("notes",[])
        },
        raw=text_for_model[:2000],
        confidence=confidence
    )


@app.get("/health")
async def health():
    return {
        "status":"ok",
        "model":"llama-3.3-70b-versatile",
        "deployment":"railway-optimized"
    }