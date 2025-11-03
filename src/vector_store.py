from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv
import numpy as np
import json

load_dotenv()

# Support Streamlit Cloud secrets
def get_api_key():
    """Get API key from environment or Streamlit secrets."""
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
            return st.secrets['GOOGLE_API_KEY']
    except:
        pass
    return os.getenv("GOOGLE_API_KEY")

class VectorStoreManager:
    """Minimal numpy vector store (no C++ build tools required).

    Files:
      - index.npz: embeddings
      - meta.json: list of metadatas
      - texts.json: list of original texts
    """

    def __init__(self, collection_name: str = "codewhisperer",
                 persist_directory: str = "./chroma_data"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

        # Try Gemini embeddings, fallback to HF if quota fails or no key
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            api_key = get_api_key()
            if not api_key:
                raise RuntimeError("Missing GOOGLE_API_KEY")

            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key
            )
        except Exception:
            from langchain.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            print("⚠️ Using HuggingFace embeddings (Gemini unavailable)")

        self._emb_path = os.path.join(self.persist_directory, "index.npz")
        self._meta_path = os.path.join(self.persist_directory, "meta.json")
        self._texts_path = os.path.join(self.persist_directory, "texts.json")

        self._emb = self._load_embeddings()
        self._meta = self._load_json(self._meta_path, default=[])
        self._texts = self._load_json(self._texts_path, default=[])

    def _load_embeddings(self) -> np.ndarray:
        if os.path.exists(self._emb_path):
            data = np.load(self._emb_path)
            return data["arr_0"]
        return np.empty((0, 768), dtype=np.float32)

    @staticmethod
    def _load_json(path: str, default):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return default

    @staticmethod
    def _write_json(path: str, data) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _save(self) -> None:
        np.savez_compressed(self._emb_path, self._emb)
        self._write_json(self._meta_path, self._meta)
        self._write_json(self._texts_path, self._texts)

    def _clean_docs(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        cleaned = []
        for d in documents:
            content = (d.get("content") or "").strip()
            if content:
                cleaned.append({
                    "content": content,
                    "source": d.get("source", "unknown"),
                    "chunk_index": d.get("chunk_index", 0),
                })
        return cleaned

    def add_documents(self, documents: List[Dict[str, str]]) -> None:
        docs = self._clean_docs(documents)
        if not docs:
            return

        texts = [d["content"] for d in docs]
        metadatas = [
            {"source": d["source"], "chunk_index": str(d["chunk_index"])}
            for d in docs
        ]

        batch_size = 16
        all_vecs = []
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                vecs = self.embeddings.embed_documents(batch)
                all_vecs.extend(vecs)
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {type(e).__name__}: {e}")

        vecs_np = np.array(all_vecs, dtype=np.float32)

        if self._emb.size == 0:
            self._emb = vecs_np
        else:
            self._emb = np.vstack([self._emb, vecs_np])

        self._meta.extend(metadatas)
        self._texts.extend(texts)
        self._save()

    def search(self, query: str, k: int = 3) -> List[Tuple[str, Dict, float]]:
        if self._emb.size == 0:
            return []

        q = np.array(self.embeddings.embed_query(query), dtype=np.float32)
        norms = np.linalg.norm(self._emb, axis=1) * (np.linalg.norm(q) + 1e-8)
        scores = (self._emb @ q) / (norms + 1e-8)
        top = np.argsort(-scores)[:k]
        return [
            (self._texts[int(i)], self._meta[int(i)], float(scores[int(i)]))
            for i in top
        ]

    def clear_store(self) -> None:
        self._emb = np.empty((0, 768), dtype=np.float32)
        self._meta = []
        self._texts = []
        self._save()
