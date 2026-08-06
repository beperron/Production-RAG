"""Shared page rasterization (PyMuPDF → PIL) for the OCR/VLM lanes.

Both Tesseract and the VLM consume page images; centralizing the render keeps a
single DPI/colorspace policy. 200 DPI is the sweet spot for OCR accuracy vs.
token/throughput cost; bump for very small print.
"""

from __future__ import annotations

import io

DEFAULT_DPI = 200


def rasterize_page(page, dpi: int = DEFAULT_DPI):
    """Render a ``fitz.Page`` to a ``PIL.Image`` (RGB) at the given DPI."""
    from PIL import Image

    zoom = dpi / 72.0  # PDF user space is 72 dpi
    import fitz

    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def image_to_data_url(image, fmt: str = "PNG") -> str:
    """Encode a PIL image as a base64 data URL for the OpenAI-compatible API."""
    import base64

    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"
