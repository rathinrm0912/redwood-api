import json
import os
from groq import Groq
from template import PROMPT_TEMPLATE, OUTPUT_TEMPLATE

client = Groq(api_key=os.getenv("GROQ_API_KEY", "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"))

# REPLACE the entire CONFIDENCE_INSTRUCTION with this:
CONFIDENCE_INSTRUCTION = """
Keep the exact same JSON structure as the template above. Do NOT wrap individual fields.
Additionally, add ONE extra key "_confidence" at the top level.
Use the EXACT same line item names from BOTH the P&L and Balance Sheet as keys:

"_confidence": {
    "Sales": 92,
    "Purchases": 78,
    "Balance with banks": 85,
    "Accounts receivables": 60,
    "Capital Account": 45
}

Include confidence scores for ALL line items — both P&L and Balance Sheet.

Confidence rules:
- 90-100 : exact number found directly in the text
- 60-89  : calculated or derived from nearby figures
- 30-59  : uncertain, partial match
- 0-29   : not found or guessed
"""


def extract_financials(text: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        template=json.dumps(OUTPUT_TEMPLATE, indent=2),
        text=text[:12000]
    ) + CONFIDENCE_INSTRUCTION

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    raw = response.choices[0].message.content
    parsed = json.loads(raw)
    return split_values_and_confidence(parsed)


def split_values_and_confidence(data: dict) -> dict:
    confidence = data.pop("_confidence", {})
    return {
        "values": data,       # exact same structure mapper.py expects — untouched
        "confidence": confidence
    }
