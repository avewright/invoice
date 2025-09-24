#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate & normalize an LLM invoice JSON to your exact structure.

Usage:
  python ensure_invoice_shape.py <<'JSON'
  { ... LLM JSON ... }
  JSON
or
  python ensure_invoice_shape.py -f path/to/invoice.json

What it does:
- Strips ```json fences and fixes trailing commas
- Converts "$1,234.50" -> 1234.5, "6%" -> 0.06, "true"/"false" -> booleans
- Validates STRICTLY (no extra keys) against the exact structure you provided
- Prints a canonical JSON object to stdout
- Emits non-fatal consistency warnings to stderr (subtotal, taxes, totals)
"""

import argparse
import json
import re
import sys
from typing import Optional, List, Literal

# ---- Pydantic v2 preferred, v1 fallback ----
try:
    from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict
    V2 = True
except Exception:
    from pydantic import BaseModel, Field, validator as field_validator, ValidationError
    V2 = False

# --------------------------
# Models (STRICT: extra=forbid)
# --------------------------
if V2:
    class StrictBaseModel(BaseModel):
        model_config = ConfigDict(extra="forbid")
else:
    class StrictBaseModel(BaseModel):
        class Config:
            extra = "forbid"

class Contact(StrictBaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None

class Vendor(StrictBaseModel):
    name: str
    address: Optional[str] = None
    tax_id: Optional[str] = None
    contact: Optional[Contact] = None

class BillTo(StrictBaseModel):
    name: str
    project_name: Optional[str] = None
    project_number: Optional[str] = None
    job_site_address: Optional[str] = None

class Tax(StrictBaseModel):
    name: str
    rate: float  # e.g., 0.06
    amount: float

class Adjustment(StrictBaseModel):
    type: Literal["credit", "debit", "discount", "other"] = "other"
    description: str
    amount: float

class LineItem(StrictBaseModel):
    line_id: int
    description: str
    quantity: float
    unit: str
    unit_price: float
    line_total: float
    cost_code: Optional[str] = None
    category: Optional[Literal["material", "labor", "equipment", "subcontractor"]] = None
    taxable: Optional[bool] = None

class Retainage(StrictBaseModel):
    percent: Optional[float] = None  # e.g., 0.10
    amount: Optional[float] = None

class Invoice(StrictBaseModel):
    invoice_id: str
    invoice_date: str
    due_date: Optional[str] = None
    currency: str
    vendor: Vendor
    bill_to: BillTo
    purchase_order: Optional[str] = None
    payment_terms: Optional[str] = None
    line_items: List[LineItem]
    subtotal: float
    taxes: List[Tax] = Field(default_factory=list)
    retainage: Optional[Retainage] = None
    adjustments: List[Adjustment] = Field(default_factory=list)
    total: float

    # Optional: forbid empty line_items
    @field_validator("line_items")
    def _non_empty_line_items(cls, v):
        if not v:
            raise ValueError("line_items must contain at least one item")
        return v

# --------------------------
# Cleaning & coercion helpers
# --------------------------
FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)
TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
MONEY_NUM_RE = re.compile(r"^\s*[-+]?\$?\s*[\d,]+(?:\.\d+)?\s*%?\s*$", re.IGNORECASE)

def strip_code_fences(s: str) -> str:
    return FENCE_RE.sub("", s).strip()

def remove_trailing_commas(s: str) -> str:
    # Safely remove trailing commas before } or ]
    # Repeat until stable in case of multiple occurrences
    prev = None
    while s != prev:
        prev = s
        s = TRAILING_COMMA_RE.sub(r"\1", s)
    return s

def coerce_scalar(x):
    # Convert common string-y numbers and booleans
    if isinstance(x, str):
        xs = x.strip()
        if xs.lower() in ("true", "false"):
            return xs.lower() == "true"
        if MONEY_NUM_RE.match(xs):
            # Remove $, commas, and whitespace
            is_percent = xs.endswith("%")
            xs = xs.replace("$", "").replace(",", "").replace("%", "").strip()
            try:
                num = float(xs)
                if is_percent:
                    return num / 100.0
                return num
            except Exception:
                return x
        return x
    return x

def deep_coerce(obj):
    if isinstance(obj, dict):
        return {k: deep_coerce(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [deep_coerce(v) for v in obj]
    return coerce_scalar(obj)

def read_json_str(src: str) -> dict:
    cleaned = remove_trailing_commas(strip_code_fences(src))
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Try a softer fix: quote keys if bare (rare), else re-raise
        raise e

# --------------------------
# Consistency checks (non-fatal)
# --------------------------
def warn(*args):
    print("WARNING:", *args, file=sys.stderr)

def consistency_warnings(inv: Invoice):
    # Line totals vs subtotal
    line_sum = sum(li.line_total for li in inv.line_items)
    if abs(line_sum - inv.subtotal) > 0.02:
        warn(f"Subtotal mismatch: sum(line_total)={line_sum:.2f} vs subtotal={inv.subtotal:.2f}")

    # Tax aggregation
    tax_sum = sum(t.amount for t in inv.taxes)
    if inv.taxes and tax_sum < 0:
        warn(f"Taxes total negative? sum(taxes)={tax_sum:.2f}")

    # Retainage sanity
    if inv.retainage:
        if inv.retainage.percent is not None and not (0 <= inv.retainage.percent <= 1.0):
            warn(f"Retainage percent out of range [0,1]: {inv.retainage.percent}")

    # Total plausibility (don’t enforce sign for retainage; varies by workflow)
    adj_sum = sum(a.amount for a in inv.adjustments)
    # Two common patterns:
    expected_minus_retainage = inv.subtotal + tax_sum + adj_sum - (inv.retainage.amount or 0.0 if inv.retainage else 0.0)
    expected_plus_retainage  = inv.subtotal + tax_sum + adj_sum + (inv.retainage.amount or 0.0 if inv.retainage else 0.0)

    if min(abs(inv.total - expected_minus_retainage), abs(inv.total - expected_plus_retainage)) > 0.02:
        warn(
            "Total looks inconsistent with typical formulas.\n"
            f"  total={inv.total:.2f}\n"
            f"  subtotal+tax+adj - retainage={expected_minus_retainage:.2f}\n"
            f"  subtotal+tax+adj + retainage={expected_plus_retainage:.2f}"
        )

# --------------------------
# Public API
# --------------------------
def ensure_invoice_shape(raw_text: str) -> dict:
    """
    Returns a normalized dict that exactly fits the required structure.
    Raises ValidationError on structural/type issues.
    """
    data = read_json_str(raw_text)
    data = deep_coerce(data)

    # Validate strictly
    try:
        inv = Invoice.model_validate(data) if V2 else Invoice.parse_obj(data)
    except ValidationError as e:
        # Pretty-print errors and exit with non-zero for pipelines
        print("VALIDATION ERROR: The JSON does not match the required invoice structure.\n", file=sys.stderr)
        for err in (e.errors() if V2 else e.errors()):
            print(err, file=sys.stderr)
        raise

    # Non-fatal sanity warnings
    consistency_warnings(inv)

    # Dump back in canonical order (the field order defined in the model)
    if V2:
        return inv.model_dump(mode="python")
    else:
        return json.loads(inv.json())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--file", help="Path to JSON file (otherwise reads stdin)")
    args = ap.parse_args()

    raw = None
    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            raw = fh.read()
    else:
        raw = sys.stdin.read()

    try:
        normalized = ensure_invoice_shape(raw)
    except ValidationError:
        sys.exit(1)

    # Print canonical JSON
    print(json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=False))

if __name__ == "__main__":
    main()
