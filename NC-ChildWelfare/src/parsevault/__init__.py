"""parsevault — local-first document intelligence.

A self-contained pipeline that converts documents to clean Markdown (native text
layer, Tesseract, or a local vision-LLM), builds structured metadata, classifies
document type, and serves hybrid (lexical + optional dense) retrieval — all on
your own machine, with no data egress.
"""

__version__ = "0.1.0"
