from pathlib import Path
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


class DocumentLoader:
    """Load and process documentation files (MD, TXT, PDF)."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )

    def load_markdown_files(self, directory: str) -> List[dict]:
        """Load all .md/.txt/.pdf files from a directory."""
        docs: List[dict] = []
        data_path = Path(directory)

        # Support multiple file types
        for file_path in list(data_path.glob("*.md")) + list(data_path.glob("*.txt")) + list(data_path.glob("*.pdf")):
            try:
                content = self._extract_text(str(file_path))
                chunks = self.splitter.split_text(content)

                for i, chunk in enumerate(chunks):
                    docs.append({
                        "content": chunk,
                        "source": file_path.name,
                        "chunk_index": i
                    })
            except Exception as e:
                print(f"⚠️ Error processing {file_path.name}: {e}")
                continue

        return docs

    def load_single_file(self, file_path: str) -> List[dict]:
        """Load a single documentation file (md/txt/pdf)."""
        try:
            content = self._extract_text(file_path)
            chunks = self.splitter.split_text(content)

            return [
                {
                    "content": chunk,
                    "source": Path(file_path).name,
                    "chunk_index": i
                }
                for i, chunk in enumerate(chunks)
            ]
        except Exception as e:
            raise ValueError(f"Failed to load {file_path}: {str(e)}")

    def _extract_text(self, file_path: str) -> str:
        """Extract text from different file types."""
        file_path = str(file_path)
        file_ext = Path(file_path).suffix.lower()

        if file_ext == ".pdf":
            return self._extract_pdf(file_path)
        elif file_ext in [".md", ".txt"]:
            return self._extract_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    @staticmethod
    def _extract_text_file(file_path: str) -> str:
        """Extract text from .md or .txt files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Fallback for different encodings
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()

    @staticmethod
    def _extract_pdf(file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf not installed. Run: pip install pypdf")

        try:
            reader = PdfReader(file_path)
            text = ""

            for page_num, page in enumerate(reader.pages):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()

            if not text.strip():
                raise ValueError("No text extracted from PDF (possibly scanned image)")

            return text

        except Exception as e:
            raise ValueError(f"PDF extraction failed: {str(e)}")