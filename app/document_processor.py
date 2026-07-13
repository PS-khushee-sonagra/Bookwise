"""
Document Processor for BookWise

Responsibilities:
- Open a PDF
- Extract text page by page
- Return structured page data
"""

from pathlib import Path
from typing import List, Dict
import logging

# pyrefly: ignore [missing-import]
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles PDF document processing."""

    def extract_text(self, pdf_path: str | Path) -> List[Dict]:
        """
        Extract text from a PDF page by page.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of dictionaries containing page number and text.

        Raises:
            FileNotFoundError: If the PDF does not exist.
            RuntimeError: If the PDF cannot be opened.
        """

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages = []

        try:
            with fitz.open(pdf_path) as document:
                logger.info(
                    "Opened '%s' (%d pages)",
                    pdf_path.name,
                    len(document)
                )

                for page in document:
                    raw_text = page.get_text()
                    text = raw_text.strip() if isinstance(raw_text, str) else ""

                    page_number_val = page.number
                    page_num = page_number_val if page_number_val is not None else 0
                    pages.append(
                        {
                            "page_number": page_num + 1,
                            "text": text,
                        }
                    )

            logger.info(
                "Successfully extracted %d pages.",
                len(pages)
            )

            return pages

        except Exception as e:
            logger.exception("Failed to process PDF.")
            raise RuntimeError(f"Could not process PDF: {e}") from e