import os
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import json
from extractor import extract_text
from ai import extract_financials
from mapper import map_to_firestore

app = FastAPI(title="Redwood PDF Extractor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Redwood PDF Extractor running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ping")
def ping():
    return {"status": "pong"}

@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    pnl_schema: str = Form(...),
    bs_schema: str = Form(...)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted")

    try:
        pnl = json.loads(pnl_schema)
        bs = json.loads(bs_schema)
    except:
        raise HTTPException(400, "Invalid schema JSON")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Run blocking functions in thread pool — prevents freezing the server
        text, method = await asyncio.to_thread(extract_text, tmp_path)
        raw_data = await asyncio.to_thread(extract_financials, text)
        mapped = map_to_firestore(raw_data["values"], pnl, bs)

        return {
            "status": "success",
            "method": method,
            "raw": raw_data["values"],
            "confidence": raw_data["confidence"],
            "mapped": mapped
        }
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)
