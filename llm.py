import os
from io import BytesIO
from typing import Optional, List, Any, Tuple
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from PIL import Image, ImageOps
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from model import ensure_invoice_shape, read_json_str, deep_coerce
import json


class GenerateResponse(BaseModel):
    response: dict


model_name = os.getenv("MODEL_NAME", os.getenv("DEFAULT_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct"))

# System prompt enforcing exact invoice schema and emission rules
SYSTEM_PROMPT = """
You are a construction invoice expert and deterministic JSON emitter.

You will be given:
1) OCR text tokens with their (page, x1, y1, x2, y2) coordinates and reading order.
2) The raw concatenated text, if available.

YOUR TASK
Extract invoice data and return EXACTLY ONE JSON object that matches the schema and key order below. 
- Output JSON ONLY. No prose, no markdown, no code fences, no comments.
- Do not add, remove, or rename keys.
- Use null for unknown values (do NOT omit keys).
- Dates must be ISO-8601 (YYYY-MM-DD).
- All amounts/rates must be numbers (no $, commas, or %). Convert “6%” → 0.06; “$1,234.50” → 1234.5.
- Booleans are true/false (lowercase).
- line_items must have ≥1 item.
- Keep the exact key order shown.

SCHEMA (exact keys and order):
{
  "invoice_id": "<string>",
  "invoice_date": "<YYYY-MM-DD>",
  "due_date": "<YYYY-MM-DD or null>",
  "currency": "<3-letter code>",
  "vendor": {
    "name": "<string>",
    "address": "<string or null>",
    "tax_id": "<string or null>",
    "contact": {
      "email": "<string or null>",
      "phone": "<string or null>"
    }
  },
  "bill_to": {
    "name": "<string>",
    "project_name": "<string or null>",
    "project_number": "<string or null>",
    "job_site_address": "<string or null>"
  },
  "purchase_order": "<string or null>",
  "payment_terms": "<string or null>",
  "line_items": [
    {
      "line_id": <integer>,
      "description": "<string>",
      "quantity": <number>,
      "unit": "<string>",
      "unit_price": <number>,
      "line_total": <number>,
      "cost_code": "<string or null>",
      "category": "<material|labor|equipment|subcontractor or null>",
      "taxable": <true|false|null>
    }
  ],
  "subtotal": <number>,
  "taxes": [
    {
      "name": "<string>",
      "rate": <number>,
      "amount": <number>
    }
  ],
  "retainage": {
    "percent": <number or null>,
    "amount": <number or null>
  },
  "adjustments": [
    {
      "type": "<credit|debit|discount|other>",
      "description": "<string>",
      "amount": <number>
    }
  ],
  "total": <number>
}

HOW TO USE COORDINATES (for better accuracy; do NOT include coordinates in the output):
- Prefer values physically nearest to their labels. If multiple candidates exist, choose the one to the RIGHT of the label on the SAME row; if not found, choose the one DIRECTLY BELOW within a small vertical window.
- Totals usually appear near the bottom-right of the last page. Prefer the value closest to labels like “Total”, “Invoice Total”, “Amount Due”.
- Subtotal/Tax/Retainage/Adjustments typically stack above the grand total in a right-aligned summary block. Map each line by label proximity.
- Vendor vs Bill-To:
  - VENDOR (issuer/remit-from) commonly top-left; look for “From”, “Remit To”, logos near it, vendor tax ID nearby.
  - BILL_TO (customer) labeled “Bill To”, “Sold To”, “Customer”, often top-right or a dedicated block.
- Project fields:
  - project_name / project_number / job_site_address are often near “Project/Job/Job #/Project #/Job Site/Site Address”.
- PO and payment_terms:
  - Look for “PO”, “P.O.”, “Purchase Order”.
  - Payment terms examples: “Net 30”, “Due on receipt”.

LINE-ITEM TABLE DETECTION:
- Identify the main line-item table by headers such as any of: Description/Item, Qty/Quantity, U/M/UOM, Rate/Unit Price/Price, Amount/Line Total, Tax.
- Descriptions may wrap to multiple lines; merge wrapped lines belonging to the same row (same x-range; next line starts aligned under Description without new numeric columns).
- Compute or verify: line_total ≈ quantity * unit_price (round to 2 decimals). If a row displays all three, keep the displayed line_total; if missing one component, compute the missing value when unambiguous.
- Category inference (material|labor|equipment|subcontractor):
  - material: items like concrete, rebar, pipe, fasteners, consumables.
  - labor: hours/crew/installer, UOM like HR.
  - equipment: rental, mobilization, crane, lift, machine hours.
  - subcontractor: another company named on a line, or “Subcontract”, “Sub”, “Trade”.
- taxable per-line: true if explicitly indicated, if a tax column shows nonzero, or if in a taxable group; false if marked non-taxable; otherwise null.

RETAINAGE & ADJUSTMENTS:
- If retainage is shown as a negative line or separate row, set retainage.amount to the absolute held-back value and retainage.percent to the stated % (decimal). If only the percent is given, compute amount when subtotal is known.
- Adjustments include credits, debits, early payment discounts. Use type one of: credit, debit, discount, other.

DISAMBIGUATION & TIE-BREAKERS:
- Prefer values with explicit labels over unlabeled numbers.
- If multiple candidates remain, choose the one with the closest centroid distance to the label; break further ties by larger font or boldness if hinted by OCR; else top-most then left-most.
- When date formats are ambiguous (e.g., 03/04/2025), use the field’s label (“Invoice Date”, “Due Date”) and local conventions if indicated; otherwise assume MM/DD/YYYY and normalize to YYYY-MM-DD.

NORMALIZATION RULES:
- Strip currency symbols and thousands separators. Convert parentheses to negative (e.g., (123.45) → -123.45).
- Convert percentages to decimals (e.g., 10% → 0.10).
- Trim whitespace; collapse multiple spaces.
- If a field is truly absent, use null (do not fabricate).

VALIDATION (internal; do not output messages):
- Ensure numeric types for amounts/rates and dates in ISO-8601.
- Ensure line_items length ≥ 1.
- Prefer internal arithmetic consistency: subtotal ≈ sum(line_total); total ≈ subtotal + sum(taxes.amount) + sum(adjustments.amount) ± retainage.amount (depending on presentation). If the document explicitly states totals, favor the stated values; do not invent numbers.

OUTPUT
Return EXACTLY one JSON object that matches the schema and constraints above. No extra text before or after the JSON.
"""

# Lazy model/processor loading to allow server to boot without heavy downloads
model = None
processor = None

def get_model_and_processor():
    global model, processor
    if model is None or processor is None:
        loaded_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        loaded_processor = AutoProcessor.from_pretrained(model_name)
        model = loaded_model
        processor = loaded_processor
    return model, processor


def _normalize_image_mode(img: Image.Image) -> Image.Image:
    """Convert image to RGB, correctly handling transparency/palette to avoid warnings."""
    try:
        # Respect EXIF orientation
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    try:
        has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
        if has_alpha:
            # Composite on white background
            rgba = img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            background.paste(rgba, mask=rgba.split()[-1])
            return background.convert("RGB")
        if img.mode != "RGB":
            return img.convert("RGB")
        return img
    except Exception:
        return img.convert("RGB")


def _decode_image_bytes(raw: bytes) -> Optional[Image.Image]:
    try:
        return _normalize_image_mode(Image.open(BytesIO(raw)))
    except Exception:
        return None


def _build_messages_for_image(img: Image.Image) -> Tuple[List[dict], List[Image.Image]]:
    norm = _normalize_image_mode(img)
    msgs: List[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Extract the invoice and return EXACTLY ONE JSON object as specified above. Output JSON ONLY."},
            ],
        },
    ]
    return msgs, [norm]


def _process_vision_info(messages: List[Any]) -> Tuple[List[Image.Image], List[Any]]:
    # Unused in image-only flow; keep for compatibility
    return [], []


def generate_response_from_image(img: Image.Image, max_new_tokens: int) -> str:
    m, p = get_model_and_processor()
    msgs, image_inputs = _build_messages_for_image(img)
    text = p.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    proc_kwargs: dict = {
        "text": [text],
        "padding": True,
        "return_tensors": "pt",
        "images": image_inputs,
    }
    model_inputs = p(**proc_kwargs)
    model_inputs = model_inputs.to(m.device)
    generated_ids = m.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response_text = p.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return response_text


app = FastAPI(title="LLM Service", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
def _normalize_enums(data: Any) -> Any:
    # Recursively normalize common enum/synonym values to allowed literals
    if isinstance(data, dict):
        normalized = {k: _normalize_enums(v) for k, v in data.items()}
        # adjustments.type mapping
        adj_list = normalized.get("adjustments")
        if isinstance(adj_list, list):
            for item in adj_list:
                if isinstance(item, dict):
                    t = item.get("type")
                    if isinstance(t, str):
                        tl = t.strip().lower()
                        if tl in {"credit", "debit", "discount", "other"}:
                            item["type"] = tl
                        elif tl in {"discounts"}:
                            item["type"] = "discount"
                        elif tl in {"shipping", "shipping charge", "freight", "delivery"}:
                            item["type"] = "other"
                        else:
                            item["type"] = "other"
        # line_items.category mapping (singularize, case fold)
        li_list = normalized.get("line_items")
        if isinstance(li_list, list):
            for li in li_list:
                if isinstance(li, dict) and isinstance(li.get("category"), str):
                    cl = li["category"].strip().lower()
                    mapping = {
                        "materials": "material",
                        "material": "material",
                        "labor": "labor",
                        "labour": "labor",
                        "equipment": "equipment",
                        "subcontractor": "subcontractor",
                        "subcontractors": "subcontractor",
                    }
                    li["category"] = mapping.get(cl, li["category"])
        return normalized
    if isinstance(data, list):
        return [_normalize_enums(v) for v in data]
    return data



@app.post("/generate", response_model=GenerateResponse)
def generate(image: UploadFile = File(...), max_new_tokens: int = 2048) -> GenerateResponse:
    try:
        raw = image.file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty image upload")
        img = _decode_image_bytes(raw)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        output_text = generate_response_from_image(img, max_new_tokens=max_new_tokens)

        # Pre-normalize enum variations before strict validation
        try:
            data = read_json_str(output_text)
            data = deep_coerce(data)
            data = _normalize_enums(data)
            output_text = json.dumps(data, ensure_ascii=False)
        except Exception:
            pass

        try:
            normalized = ensure_invoice_shape(output_text)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Normalization/validation failed: {exc}")
        return GenerateResponse(response=normalized)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# If executed directly: run a quick demo generation for convenience
if __name__ == "__main__":
    print("Run with uvicorn to use the HTTP API.")
