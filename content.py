"""
Claude API text generation for synthetic document content.

Uses the Anthropic Python SDK to generate realistic structured data for
invoices, forms, contracts, and receipts. Each generator returns a dict
that matches the corresponding template's expected schema.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional

import anthropic

DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_RETRIES = 3


def _get_client(client: Optional[anthropic.Anthropic] = None) -> anthropic.Anthropic:
    """Return the provided client or create one from ANTHROPIC_API_KEY."""
    if client is not None:
        return client
    return anthropic.Anthropic()


def _parse_json(text: str) -> dict:
    """Parse JSON from an LLM response, stripping markdown code fences and prose."""
    text = text.strip()
    # Extract content between code fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()
    # Fallback: find the outermost { ... } block
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
    return json.loads(text)


def _generate(
    client: anthropic.Anthropic,
    prompt: str,
    model: str = DEFAULT_MODEL,
    retries: int = MAX_RETRIES,
) -> dict:
    """Call Claude and parse the JSON response with retry logic."""
    last_error = None
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                temperature=0.9,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            return _parse_json(text)
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(1)
                continue
        except (
            anthropic.RateLimitError,
            anthropic.InternalServerError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
        ) as e:
            last_error = e
            wait = 2 ** attempt
            print(f"    API error ({type(e).__name__}), retrying in {wait}s...")
            if attempt < retries - 1:
                time.sleep(wait)
                continue
            raise ValueError(
                f"Failed to parse JSON after {retries} attempts: {last_error}"
            ) from e


# ---------------------------------------------------------------------------
# Invoice content
# ---------------------------------------------------------------------------

def generate_invoice_content(
    client: Optional[anthropic.Anthropic] = None,
    industry: str = "technology",
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Generate realistic invoice data for a given industry."""
    client = _get_client(client)
    prompt = f"""Generate a realistic invoice for a {industry} company.
Return ONLY valid JSON with this exact structure:
{{
  "company_name": "string",
  "company_address": "string (full address on one line)",
  "invoice_number": "string (e.g. INV-2025-0042)",
  "date": "string (YYYY-MM-DD)",
  "bill_to": {{
    "name": "string (person or company name)",
    "address": "string (full address on one line)"
  }},
  "line_items": [
    {{
      "description": "string",
      "quantity": number,
      "unit_price": number
    }}
  ],
  "tax_rate": number (e.g. 0.0875 for 8.75%),
  "notes": "string (payment terms or notes)"
}}

Include 3-6 line items with realistic descriptions and prices for a {industry} company.
Use realistic company and person names. Vary the invoice amounts."""
    return _generate(client, prompt, model)


# ---------------------------------------------------------------------------
# Form content
# ---------------------------------------------------------------------------

def generate_form_content(
    client: Optional[anthropic.Anthropic] = None,
    form_type: str = "employment",
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Generate realistic form data for a given form type."""
    client = _get_client(client)
    prompt = f"""Generate a realistic {form_type} form with filled-in data.
Return ONLY valid JSON with this exact structure:
{{
  "title": "string (form title)",
  "sections": [
    {{
      "section_name": "string",
      "fields": [
        {{
          "label": "string",
          "value": "string or boolean",
          "field_type": "text" | "checkbox" | "date"
        }}
      ]
    }}
  ]
}}

Include 3-4 sections with 3-5 fields each. Use realistic data for a {form_type} form.
Mix field types: mostly text, some checkboxes and dates.
For checkboxes, value should be "true" or "false"."""
    return _generate(client, prompt, model)


# ---------------------------------------------------------------------------
# Contract content
# ---------------------------------------------------------------------------

def generate_contract_content(
    client: Optional[anthropic.Anthropic] = None,
    contract_type: str = "service agreement",
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Generate realistic contract data for a given contract type."""
    client = _get_client(client)
    prompt = f"""Generate a realistic {contract_type} contract.
Return ONLY valid JSON with this exact structure:
{{
  "title": "string (contract title in caps)",
  "parties": [
    {{
      "name": "string (company or person name)",
      "role": "string (e.g. 'Provider', 'Client', 'Landlord', 'Tenant')"
    }}
  ],
  "paragraphs": [
    "string (each paragraph is a full clause of the contract)"
  ],
  "signature_blocks": [
    {{
      "name": "string (signatory name)",
      "title": "string (job title)",
      "date": "string (YYYY-MM-DD or blank)"
    }}
  ]
}}

Include 2 parties, 5-8 paragraphs with realistic legal language for a {contract_type},
and 2 signature blocks. Keep paragraphs to 2-4 sentences each."""
    return _generate(client, prompt, model)


# ---------------------------------------------------------------------------
# Receipt content
# ---------------------------------------------------------------------------

def generate_receipt_content(
    client: Optional[anthropic.Anthropic] = None,
    store_type: str = "grocery",
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Generate realistic receipt data for a given store type."""
    client = _get_client(client)
    prompt = f"""Generate a realistic {store_type} store receipt.
Return ONLY valid JSON with this exact structure:
{{
  "store_name": "string (store name in caps)",
  "store_address": "string (address on one line)",
  "items": [
    {{
      "name": "string (item name, max 20 chars)",
      "price": number
    }}
  ],
  "subtotal": number,
  "tax": number,
  "total": number,
  "payment_method": "string (e.g. VISA ****1234, CASH, DEBIT)",
  "date": "string (YYYY-MM-DD HH:MM)",
  "transaction_id": "string (e.g. TXN-123456)"
}}

Include 5-12 items with realistic names and prices for a {store_type} store.
Make sure subtotal equals the sum of item prices, and total equals subtotal + tax."""
    return _generate(client, prompt, model)


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

GENERATORS = {
    "invoice": generate_invoice_content,
    "form": generate_form_content,
    "contract": generate_contract_content,
    "receipt": generate_receipt_content,
}

# Default variety options for each document type
VARIETY = {
    "invoice": ["technology", "healthcare", "construction", "marketing",
                 "legal", "education", "manufacturing", "consulting"],
    "form": ["employment", "insurance", "loan application", "medical intake",
             "tax", "rental application", "volunteer", "membership"],
    "contract": ["service agreement", "NDA", "lease agreement",
                 "employment contract", "consulting agreement",
                 "licensing agreement", "partnership agreement"],
    "receipt": ["grocery", "restaurant", "hardware", "pharmacy",
                "electronics", "clothing", "coffee shop", "gas station"],
}


def generate_batch(
    client: Optional[anthropic.Anthropic] = None,
    doc_type: str = "invoice",
    count: int = 5,
    model: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    """Generate *count* documents of the given type with variety.

    Cycles through industry/form_type/contract_type variants to produce
    diverse content.
    """
    client = _get_client(client)
    generator = GENERATORS.get(doc_type)
    if generator is None:
        raise ValueError(
            f"Unknown doc_type '{doc_type}'. Available: {list(GENERATORS.keys())}"
        )

    variants = VARIETY.get(doc_type, ["default"])
    results = []
    for i in range(count):
        variant = variants[i % len(variants)]
        # The second positional arg is the variant (industry/form_type/etc.)
        data = generator(client, variant, model)
        results.append(data)
        print(f"  Generated {doc_type} {i+1}/{count} ({variant})")
    return results


if __name__ == "__main__":
    # Quick test: generate one invoice
    c = anthropic.Anthropic()
    data = generate_invoice_content(c, "technology")
    print(json.dumps(data, indent=2))
