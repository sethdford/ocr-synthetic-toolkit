"""
PDF rendering for synthetic business documents + image degradation pipeline.

Four document templates (invoice, form, contract, receipt) generate PDFs with
ReportLab and return ``(pdf_bytes, ground_truth_text)``. The degradation
pipeline converts PDFs to images and applies configurable scan-like artifacts.
"""

from __future__ import annotations

import io
import math
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageFilter
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

# Try to import pdf2image; degrade gracefully if poppler isn't installed
try:
    from pdf2image import convert_from_bytes
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

PAGE_W, PAGE_H = letter  # 612 x 792 points
MARGIN = 0.75 * inch


def _new_canvas(buf: io.BytesIO) -> canvas.Canvas:
    return canvas.Canvas(buf, pagesize=letter)


# ---------------------------------------------------------------------------
# Template 1: Invoice
# ---------------------------------------------------------------------------

def render_invoice(data: dict[str, Any]) -> tuple[bytes, str]:
    """Render an invoice PDF from structured *data*.

    Expected keys: company_name, company_address, bill_to (dict with name,
    address), invoice_number, date, line_items (list of dicts with
    description, quantity, unit_price), tax_rate, notes.
    """
    buf = io.BytesIO()
    c = _new_canvas(buf)
    gt_lines: list[str] = []  # ground truth accumulator
    y = PAGE_H - MARGIN

    # --- Company header ---
    c.setFont("Helvetica-Bold", 18)
    company = data.get("company_name", "Acme Corporation")
    c.drawString(MARGIN, y, company)
    gt_lines.append(company)
    y -= 20

    c.setFont("Helvetica", 9)
    addr = data.get("company_address", "123 Business Ave, Suite 100")
    c.drawString(MARGIN, y, addr)
    gt_lines.append(addr)
    y -= 30

    # --- INVOICE title + meta ---
    c.setFont("Helvetica-Bold", 14)
    c.drawString(MARGIN, y, "INVOICE")
    gt_lines.append("INVOICE")
    y -= 18

    c.setFont("Helvetica", 10)
    inv_num = f"Invoice #: {data.get('invoice_number', '0001')}"
    c.drawString(MARGIN, y, inv_num)

    date_str = f"Date: {data.get('date', '2025-01-15')}"
    c.drawString(PAGE_W / 2, y, date_str)
    gt_lines.append(f"{inv_num}  {date_str}")
    y -= 28

    # --- Bill To ---
    c.setFont("Helvetica-Bold", 10)
    c.drawString(MARGIN, y, "Bill To:")
    gt_lines.append("Bill To:")
    y -= 14

    bill_to = data.get("bill_to", {})
    c.setFont("Helvetica", 10)
    bt_name = bill_to.get("name", "Jane Doe")
    c.drawString(MARGIN, y, bt_name)
    gt_lines.append(bt_name)
    y -= 14
    bt_addr = bill_to.get("address", "456 Client Rd, Town, ST 00000")
    c.drawString(MARGIN, y, bt_addr)
    gt_lines.append(bt_addr)
    y -= 28

    # --- Line items table ---
    col_desc = MARGIN
    col_qty = MARGIN + 300
    col_price = MARGIN + 380
    col_total = MARGIN + 450

    c.setFont("Helvetica-Bold", 10)
    headers = ["Description", "Qty", "Unit Price", "Total"]
    c.drawString(col_desc, y, headers[0])
    c.drawString(col_qty, y, headers[1])
    c.drawString(col_price, y, headers[2])
    c.drawString(col_total, y, headers[3])
    gt_lines.append("  ".join(headers))
    y -= 4
    c.line(MARGIN, y, PAGE_W - MARGIN, y)
    y -= 14

    c.setFont("Helvetica", 10)
    subtotal = 0.0
    max_desc_width = col_qty - col_desc - 10
    for item in data.get("line_items", []):
        if y < MARGIN + 40:
            c.showPage()
            y = PAGE_H - MARGIN
            c.setFont("Helvetica", 10)

        desc = str(item.get("description", "Item"))
        # Truncate description to fit before Qty column
        if c.stringWidth(desc, "Helvetica", 10) > max_desc_width:
            while len(desc) > 1 and c.stringWidth(desc + "...", "Helvetica", 10) > max_desc_width:
                desc = desc[:-1]
            desc = desc.rstrip() + "..."

        qty = item.get("quantity", 1)
        price = item.get("unit_price", 0.0)
        line_total = qty * price
        subtotal += line_total

        c.drawString(col_desc, y, desc)
        c.drawString(col_qty, y, str(qty))
        c.drawString(col_price, y, f"${price:,.2f}")
        c.drawString(col_total, y, f"${line_total:,.2f}")
        gt_lines.append(f"{desc}  {qty}  ${price:,.2f}  ${line_total:,.2f}")
        y -= 16

    y -= 8
    c.line(MARGIN, y, PAGE_W - MARGIN, y)
    y -= 18

    # --- Totals ---
    c.setFont("Helvetica", 10)
    sub_line = f"Subtotal: ${subtotal:,.2f}"
    c.drawString(col_price, y, sub_line)
    gt_lines.append(sub_line)
    y -= 16

    tax_rate = data.get("tax_rate", 0.08)
    tax_amt = subtotal * tax_rate
    tax_line = f"Tax ({tax_rate*100:.1f}%): ${tax_amt:,.2f}"
    c.drawString(col_price, y, tax_line)
    gt_lines.append(tax_line)
    y -= 16

    c.setFont("Helvetica-Bold", 11)
    total = subtotal + tax_amt
    total_line = f"Total: ${total:,.2f}"
    c.drawString(col_price, y, total_line)
    gt_lines.append(total_line)
    y -= 30

    # --- Notes ---
    notes = data.get("notes", "")
    if notes:
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(MARGIN, y, f"Notes: {notes}")
        gt_lines.append(f"Notes: {notes}")

    c.showPage()
    c.save()
    return buf.getvalue(), "\n".join(gt_lines)


# ---------------------------------------------------------------------------
# Template 2: Form
# ---------------------------------------------------------------------------

def render_form(data: dict[str, Any]) -> tuple[bytes, str]:
    """Render a form PDF (employment, insurance, application, etc.).

    Expected keys: title, sections (list of dicts with section_name, fields).
    Each field: {label, value, field_type} where field_type is
    "text", "checkbox", or "date".
    """
    buf = io.BytesIO()
    c = _new_canvas(buf)
    gt_lines: list[str] = []
    y = PAGE_H - MARGIN

    # Title
    title = data.get("title", "Application Form")
    c.setFont("Helvetica-Bold", 16)
    c.drawString(MARGIN, y, title)
    gt_lines.append(title)
    y -= 30

    for section in data.get("sections", []):
        # Section header
        sec_name = section.get("section_name", "Section")
        c.setFont("Helvetica-Bold", 12)
        c.drawString(MARGIN, y, sec_name)
        gt_lines.append(sec_name)
        y -= 6
        c.setStrokeColor(colors.grey)
        c.line(MARGIN, y, PAGE_W - MARGIN, y)
        c.setStrokeColor(colors.black)
        y -= 16

        for field in section.get("fields", []):
            label = field.get("label", "Field")
            value = str(field.get("value", ""))
            ftype = field.get("field_type", "text")

            c.setFont("Helvetica-Bold", 10)
            c.drawString(MARGIN, y, f"{label}:")
            c.setFont("Helvetica", 10)

            if ftype == "checkbox":
                # Draw a checkbox
                box_x = MARGIN + 200
                box_size = 10
                c.rect(box_x, y - 2, box_size, box_size, stroke=1, fill=0)
                checked = str(value).lower() in ("true", "yes", "1", "x")
                if checked:
                    c.drawString(box_x + 2, y, "X")
                display = f"{label}: [{'X' if checked else ' '}]"
            elif ftype == "date":
                c.drawString(MARGIN + 200, y, value)
                # Underline
                c.line(MARGIN + 200, y - 2, MARGIN + 350, y - 2)
                display = f"{label}: {value}"
            else:
                c.drawString(MARGIN + 200, y, value)
                c.line(MARGIN + 200, y - 2, PAGE_W - MARGIN, y - 2)
                display = f"{label}: {value}"

            gt_lines.append(display)
            y -= 22

            if y < MARGIN + 40:
                c.showPage()
                y = PAGE_H - MARGIN

        y -= 10  # gap between sections

    c.showPage()
    c.save()
    return buf.getvalue(), "\n".join(gt_lines)


# ---------------------------------------------------------------------------
# Template 3: Contract
# ---------------------------------------------------------------------------

def render_contract(data: dict[str, Any]) -> tuple[bytes, str]:
    """Render a contract PDF.

    Expected keys: title, parties (list of {name, role}), paragraphs
    (list of strings), signature_blocks (list of {name, title, date}).
    """
    buf = io.BytesIO()
    c = _new_canvas(buf)
    gt_lines: list[str] = []
    y = PAGE_H - MARGIN
    usable_w = PAGE_W - 2 * MARGIN

    # Title
    title = data.get("title", "Service Agreement")
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(PAGE_W / 2, y, title)
    gt_lines.append(title)
    y -= 30

    # Parties
    c.setFont("Helvetica", 10)
    for party in data.get("parties", []):
        pname = party.get("name", "Party")
        prole = party.get("role", "")
        line = f"{prole}: {pname}" if prole else pname
        c.drawString(MARGIN, y, line)
        gt_lines.append(line)
        y -= 14

    y -= 16

    # Numbered paragraphs
    c.setFont("Helvetica", 10)
    for i, para in enumerate(data.get("paragraphs", []), 1):
        prefix = f"{i}. "
        # Wrap text manually
        words = para.split()
        current_line = prefix
        for word in words:
            test = current_line + word + " "
            if c.stringWidth(test, "Helvetica", 10) > usable_w:
                c.drawString(MARGIN, y, current_line.rstrip())
                gt_lines.append(current_line.rstrip())
                y -= 14
                current_line = "   " + word + " "
                if y < MARGIN + 80:
                    c.showPage()
                    y = PAGE_H - MARGIN
            else:
                current_line = test

        if current_line.strip():
            c.drawString(MARGIN, y, current_line.rstrip())
            gt_lines.append(current_line.rstrip())
            y -= 14

        y -= 8  # paragraph gap

        if y < MARGIN + 80:
            c.showPage()
            y = PAGE_H - MARGIN

    y -= 20

    # Signature blocks
    for sig in data.get("signature_blocks", []):
        if y < MARGIN + 80:
            c.showPage()
            y = PAGE_H - MARGIN

        c.line(MARGIN, y, MARGIN + 200, y)
        y -= 14
        sig_name = sig.get("name", "Name")
        c.drawString(MARGIN, y, sig_name)
        gt_lines.append(sig_name)
        y -= 14
        sig_title = sig.get("title", "")
        if sig_title:
            c.drawString(MARGIN, y, sig_title)
            gt_lines.append(sig_title)
            y -= 14
        sig_date = f"Date: {sig.get('date', '___________')}"
        c.drawString(MARGIN, y, sig_date)
        gt_lines.append(sig_date)
        y -= 30

    c.showPage()
    c.save()
    return buf.getvalue(), "\n".join(gt_lines)


# ---------------------------------------------------------------------------
# Template 4: Receipt
# ---------------------------------------------------------------------------

def render_receipt(data: dict[str, Any]) -> tuple[bytes, str]:
    """Render a narrow receipt PDF (thermal receipt style).

    Expected keys: store_name, store_address, items (list of {name, price}),
    subtotal, tax, total, payment_method, date, transaction_id.
    """
    # Narrow page: 3 inches wide, 11 inches tall
    receipt_w = 3 * inch
    receipt_h = 11 * inch

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(receipt_w, receipt_h))
    gt_lines: list[str] = []
    margin = 0.25 * inch
    y = receipt_h - margin
    content_w = receipt_w - 2 * margin

    # Store header
    c.setFont("Courier-Bold", 10)
    store = data.get("store_name", "GENERAL STORE")
    c.drawCentredString(receipt_w / 2, y, store)
    gt_lines.append(store)
    y -= 12

    c.setFont("Courier", 8)
    addr = data.get("store_address", "123 Main St")
    c.drawCentredString(receipt_w / 2, y, addr)
    gt_lines.append(addr)
    y -= 16

    # Separator
    sep = "-" * 32
    c.drawString(margin, y, sep)
    gt_lines.append(sep)
    y -= 12

    # Items
    c.setFont("Courier", 8)
    for item in data.get("items", []):
        if y < margin + 30:
            c.showPage()
            y = receipt_h - margin
            c.setFont("Courier", 8)

        name = str(item.get("name", "Item"))[:20]
        price = item.get("price", 0.0)
        price_str = f"${price:.2f}"
        # Right-align price
        c.drawString(margin, y, name)
        c.drawRightString(receipt_w - margin, y, price_str)
        gt_lines.append(f"{name:<20} {price_str:>10}")
        y -= 11

    y -= 4
    c.drawString(margin, y, sep)
    gt_lines.append(sep)
    y -= 12

    # Totals
    c.setFont("Courier", 8)
    for label, key in [("SUBTOTAL", "subtotal"), ("TAX", "tax")]:
        val = data.get(key, 0.0)
        line_text = f"{label}:"
        val_text = f"${val:.2f}"
        c.drawString(margin, y, line_text)
        c.drawRightString(receipt_w - margin, y, val_text)
        gt_lines.append(f"{line_text:<20} {val_text:>10}")
        y -= 11

    y -= 2
    c.setFont("Courier-Bold", 9)
    total_val = data.get("total", 0.0)
    c.drawString(margin, y, "TOTAL:")
    c.drawRightString(receipt_w - margin, y, f"${total_val:.2f}")
    gt_lines.append(f"{'TOTAL:':<20} ${total_val:.2f}")
    y -= 16

    # Payment info
    c.setFont("Courier", 8)
    pay = data.get("payment_method", "CASH")
    c.drawString(margin, y, f"PAYMENT: {pay}")
    gt_lines.append(f"PAYMENT: {pay}")
    y -= 11

    date_str = data.get("date", "2025-01-15 14:30")
    c.drawString(margin, y, date_str)
    gt_lines.append(date_str)
    y -= 11

    txn = data.get("transaction_id", "TXN-000001")
    c.drawString(margin, y, txn)
    gt_lines.append(txn)
    y -= 16

    c.setFont("Courier", 8)
    thanks = "Thank you for your purchase!"
    c.drawCentredString(receipt_w / 2, y, thanks)
    gt_lines.append(thanks)

    c.showPage()
    c.save()
    return buf.getvalue(), "\n".join(gt_lines)


# ---------------------------------------------------------------------------
# Image degradation pipeline
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> list[Image.Image]:
    """Convert PDF bytes to a list of PIL Images (one per page).

    Requires poppler installed: ``brew install poppler`` (macOS).
    """
    if not HAS_PDF2IMAGE:
        raise ImportError(
            "pdf2image is required. Install it with: pip install pdf2image\n"
            "Also install poppler: brew install poppler (macOS)"
        )
    return convert_from_bytes(pdf_bytes, dpi=dpi)


def apply_blur(img: Image.Image, radius: float = 1.0) -> Image.Image:
    """Apply Gaussian blur to simulate slightly out-of-focus scan."""
    if radius <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_noise(
    img: Image.Image, sigma: float = 10.0, rng: Optional[np.random.Generator] = None,
) -> Image.Image:
    """Add Gaussian noise to simulate scanner sensor noise."""
    if sigma <= 0:
        return img
    if rng is None:
        rng = np.random.default_rng()
    arr = np.array(img, dtype=np.float32)
    noise = rng.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def apply_rotation(img: Image.Image, angle: float = 0.5) -> Image.Image:
    """Rotate image slightly to simulate skewed scan. Fills with white."""
    if abs(angle) < 0.01:
        return img
    return img.rotate(angle, resample=Image.BICUBIC, expand=False,
                      fillcolor=(255, 255, 255))


def apply_jpeg_artifacts(img: Image.Image, quality: int = 75) -> Image.Image:
    """Re-compress as JPEG to introduce compression artifacts."""
    if quality >= 100:
        return img
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def apply_ink_bleed(img: Image.Image, amount: float = 1.0) -> Image.Image:
    """Simulate ink bleed / spread using morphological dilation on dark pixels.

    Requires OpenCV. Falls back to a Pillow min-filter if unavailable.
    """
    if amount <= 0:
        return img

    # Minimum useful kernel is 3x3; map amount to odd kernel sizes
    kernel_size = 2 * max(1, int(math.ceil(amount))) + 1

    if HAS_CV2:
        arr = np.array(img)
        # Invert so ink is white (dilation expands white regions)
        inv = 255 - arr
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        dilated = cv2.dilate(inv, kernel, iterations=1)
        result = 255 - dilated
        return Image.fromarray(result)
    else:
        # Fallback: MinFilter makes dark areas spread
        return img.filter(ImageFilter.MinFilter(size=kernel_size))


def degrade_image(
    img: Image.Image,
    blur: float = 0.8,
    noise: float = 8.0,
    rotation: float = 0.3,
    jpeg_quality: int = 80,
    ink_bleed: float = 0.0,
    seed: Optional[int] = None,
) -> Image.Image:
    """Apply a composable degradation pipeline to simulate a scanned document.

    Args:
        img: Input PIL Image (typically from pdf_to_images).
        blur: Gaussian blur radius in pixels.
        noise: Gaussian noise standard deviation.
        rotation: Rotation angle in degrees.
        jpeg_quality: JPEG re-compression quality (1-100).
        ink_bleed: Ink bleed / morphological dilation amount.
        seed: Random seed for reproducible noise.

    Returns:
        Degraded PIL Image.
    """
    rng = np.random.default_rng(seed)
    img = img.convert("RGB")
    img = apply_blur(img, blur)
    img = apply_noise(img, noise, rng=rng)
    img = apply_rotation(img, rotation)
    img = apply_jpeg_artifacts(img, jpeg_quality)
    img = apply_ink_bleed(img, ink_bleed)
    return img


# ---------------------------------------------------------------------------
# Convenience: render any template by name
# ---------------------------------------------------------------------------

RENDERERS = {
    "invoice": render_invoice,
    "form": render_form,
    "contract": render_contract,
    "receipt": render_receipt,
}


def render_document(doc_type: str, data: dict[str, Any]) -> tuple[bytes, str]:
    """Dispatch to the appropriate renderer by document type name."""
    renderer = RENDERERS.get(doc_type)
    if renderer is None:
        raise ValueError(
            f"Unknown doc_type '{doc_type}'. "
            f"Available: {list(RENDERERS.keys())}"
        )
    return renderer(data)


if __name__ == "__main__":
    # Quick smoke test: render a sample invoice
    sample_data = {
        "company_name": "TechFlow Solutions",
        "company_address": "789 Innovation Blvd, San Francisco, CA 94102",
        "invoice_number": "INV-2025-0042",
        "date": "2025-01-15",
        "bill_to": {"name": "Alice Johnson", "address": "321 Client Lane, Portland, OR 97201"},
        "line_items": [
            {"description": "Web Development", "quantity": 40, "unit_price": 150.00},
            {"description": "UI/UX Design", "quantity": 20, "unit_price": 125.00},
        ],
        "tax_rate": 0.0875,
        "notes": "Payment due within 30 days.",
    }
    pdf_bytes, gt = render_invoice(sample_data)
    with open("sample_invoice.pdf", "wb") as f:
        f.write(pdf_bytes)
    print(f"Wrote sample_invoice.pdf ({len(pdf_bytes)} bytes)")
    print(f"Ground truth:\n{gt}")
