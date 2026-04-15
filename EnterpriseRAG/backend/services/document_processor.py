"""
Document processing: PDF/DOCX extraction and intelligent chunking.

Production-only dependencies (pymupdf, python-docx, langchain) are imported
lazily inside the methods that need them so the module loads cleanly in mock
mode without those packages installed.
"""

import io
import re
from typing import List, Dict, Any
from ..utils.config import settings


class DocumentProcessor:
    """Handle document extraction and chunking"""

    def __init__(self):
        # Splitter is created lazily to avoid importing langchain at startup
        self._text_splitter = None

    def _get_splitter(self):
        if self._text_splitter is None:
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
            except ImportError:
                raise ImportError(
                    "langchain is required for production mode. "
                    "Run: pip install -r requirements.prod.txt"
                )
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
            )
        return self._text_splitter

    async def extract_text(self, filename: str, file_content: bytes) -> str:
        if filename.lower().endswith(".pdf"):
            return await self._extract_pdf(file_content)
        elif filename.lower().endswith(".docx"):
            return await self._extract_docx(file_content)
        else:
            raise ValueError(f"Unsupported file type: {filename}")

    async def _extract_pdf(self, file_content: bytes) -> str:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "pymupdf is required for PDF processing. "
                "Run: pip install -r requirements.prod.txt"
            )
        doc = fitz.open(stream=file_content, filetype="pdf")
        text_parts = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_parts.append(f"\n[Page {page_num + 1}]\n{text}")
        doc.close()
        return "\n".join(text_parts)

    async def _extract_docx(self, file_content: bytes) -> str:
        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise ImportError(
                "python-docx is required for DOCX processing. "
                "Run: pip install -r requirements.prod.txt"
            )
        doc = DocxDocument(io.BytesIO(file_content))
        text_parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(text_parts)

    async def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        splitter = self._get_splitter()
        chunks = splitter.split_text(text)
        return [
            {
                "text": chunk.strip(),
                "chunk_index": i,
                "page": self._extract_page_number(chunk),
                "section": self._extract_section(chunk),
            }
            for i, chunk in enumerate(chunks)
        ]

    def _extract_page_number(self, chunk: str) -> str:
        match = re.search(r"\[Page (\d+)\]", chunk)
        return match.group(1) if match else "unknown"

    def _extract_section(self, chunk: str) -> str:
        for line in chunk.split("\n")[:3]:
            line = line.strip()
            if line.isupper() and 3 < len(line) < 80:
                return line
            if line and line[0].isdigit() and "." in line[:10]:
                return line
        return ""


# Global instance
document_processor = DocumentProcessor()
