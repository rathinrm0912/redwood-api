def build_key_map(schema_sections):
    """Build reverse lookup: section title → key"""
    key_map = {}
    for section in schema_sections:
        if "key" in section and "title" in section:
            key = section["key"]
            title = section["title"]
            key_map[title] = key
            # Add fuzzy match
            fuzzy = title.lower().replace(" ", "").replace("&", "")
            key_map[fuzzy] = key
    return key_map

def fuzzy_match(title: str, key_map: dict) -> str | None:
    # Exact match
    if title in key_map:
        return key_map[title]
    # Fuzzy match
    fuzzy = title.lower().replace(" ", "").replace("&", "")
    return key_map.get(fuzzy)

def map_to_firestore(api_data: dict, pnl_schema: list, bs_schema: list) -> dict:
    period = api_data.get("period", "FY2024-25")
    
    pnl_map = build_key_map(pnl_schema)
    bs_map = build_key_map(bs_schema)
    
    result = {
        "period": period,
        "company_name": api_data.get("company_name", ""),
        "currency": api_data.get("currency", "INR"),
        "data": {
            "pnl": {},
            "bs": {}
        }
    }

    # Map PnL sections
    for section_title, items in api_data.get("pnl", {}).items():
        key = fuzzy_match(section_title, pnl_map)
        if key and isinstance(items, dict):
            result["data"]["pnl"][key] = {period: items}

    # Map BS sections
    for section_title, items in api_data.get("bs", {}).items():
        key = fuzzy_match(section_title, bs_map)
        if key and isinstance(items, dict):
            result["data"]["bs"][key] = {period: items}

    return result
