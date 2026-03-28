"""
Document processing: PDF/DOCX extraction and intelligent chunking
"""

import io
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ..utils.config import settings


class DocumentProcessor:
    """Handle document extraction and chunking"""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    async def extract_text(self, filename: str, file_content: bytes) -> str:
        """
        Extract text from PDF or DOCX
        """
        if filename.lower().endswith(".pdf"):
            return await self._extract_pdf(file_content)
        elif filename.lower().endswith(".docx"):
            return await self._extract_docx(file_content)
        else:
            raise ValueError(f"Unsupported file type: {filename}")

    async def _extract_pdf(self, file_content: bytes) -> str:
        """
        Extract text from PDF using PyMuPDF
        """
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
        """
        Extract text from DOCX
        """
        doc = DocxDocument(io.BytesIO(file_content))
        text_parts = []

        for i, para in enumerate(doc.paragraphs):
            text = para.text.strip()
            if text:
                text_parts.append(text)

        return "\n\n".join(text_parts)

    async def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Chunk text semantically with metadata
        """
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)

        # Add metadata
        chunked_docs = []
        for i, chunk in enumerate(chunks):
            # Extract page number from chunk if available
            page = self._extract_page_number(chunk)

            # Extract section heading if available
            section = self._extract_section(chunk)

            chunked_docs.append({
                "text": chunk.strip(),
                "chunk_index": i,
                "page": page,
                "section": section,
            })

        return chunked_docs

    def _extract_page_number(self, chunk: str) -> str:
        """
        Extract page number from chunk text
        """
        import re

        match = re.search(r"\[Page (\d+)\]", chunk)
        if match:
            return match.group(1)
        return "unknown"

    def _extract_section(self, chunk: str) -> str:
        """
        Extract section heading from chunk
        """
        lines = chunk.split("\n")
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            # Look for section headers (all caps, or numbered)
            if line.isupper() and len(line) > 3 and len(line) < 80:
                return line
            if line and line[0].isdigit() and "." in line[:10]:
                return line

        return ""


# Global instance
document_processor = DocumentProcessor()
