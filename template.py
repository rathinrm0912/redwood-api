OUTPUT_TEMPLATE = {
    "company_name": "",
    "period": "",
    "currency": "",
    "pnl": {
        "Revenue": {},
        "Direct Costs": {},
        "Employee Costs": {},
        "Other Indirect": {},
        "Finance Costs": {},
        "Depreciation & Amortization": {},
        "Tax": {}
    },
    "bs": {
        "Non Current Assets": {},
        "Current Assets": {},
        "Current Liabilities": {},
        "Non Current Liabilities": {},
        "Equity": {}
    }
}

PROMPT_TEMPLATE = """
You are a financial data extraction assistant.
Extract financial data from the text below and return ONLY a valid JSON object.
No explanation, no markdown, no code block — raw JSON only.

The JSON must follow this exact structure:
{template}

Rules:
- All monetary values must be plain numbers (no commas, no currency symbols)
- If a value is not found, use 0
- Keys inside pnl/bs sections are line item names → values are numbers
- period format: "FY2022-23" style
- currency: "INR" or "USD" etc

Financial Statement Text:
{text}
"""
