from typing import List, Tuple
from src.vector_store import VectorStoreManager


class RAGRetriever:
    """Handles retrieval-augmented generation pipeline."""

    def __init__(self, vector_store_manager: VectorStoreManager, k: int = 3):
        self.vector_store = vector_store_manager
        self.k = k

    def retrieve_context(self, query: str) -> Tuple[str, List[dict]]:
        results = self.vector_store.search(query, k=self.k)

        context_chunks: List[str] = []
        sources: List[dict] = []

        for i, (content, metadata, score) in enumerate(results, 1):
            context_chunks.append(
                f"--- Snippet {i} (from {metadata['source']}) ---\n{content}"
            )
            sources.append({
                "source": metadata["source"],
                "chunk": metadata["chunk_index"],
                "relevance_score": round(float(score), 4)
            })

        formatted_context = "\n\n".join(context_chunks)
        return formatted_context, sources

    def build_prompt(self, context: str, question: str) -> str:
        system_prompt = (
            "You are an expert code analyst and documentation specialist. "
            "Answer ONLY using the provided documentation snippets.\n\n"
            "CRITICAL RULES:\n"
            "1. Only use information from the provided snippets\n"
            "2. If the answer is not in the snippets, say: "
            "\"This information is not available in the provided documentation.\"\n"
            "3. Always cite which snippet you used\n"
            "4. Provide runnable code when applicable\n"
            "5. Be precise and avoid speculation"
        )

        user_prompt = (
            f"CONTEXT (Documentation snippets):\n{context}\n\n---\n\n"
            f"QUESTION:\n{question}\n\n---\n\n"
            "ANSWER (cite sources and be specific):"
        )

        return f"{system_prompt}\n\n{user_prompt}"

