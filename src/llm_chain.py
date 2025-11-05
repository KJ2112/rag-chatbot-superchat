import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict


load_dotenv()


# Support Streamlit Cloud secrets
def get_api_key():
    """Get API key from environment or Streamlit secrets."""
    try:
        import streamlit as st

        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except:
        pass
    return os.getenv("GOOGLE_API_KEY")


class CodeWhispererChain:
    """LangChain integration for CodeWhisperer using FREE Gemini."""

    def __init__(self, model: str = "gemini-2.0-flash", temperature: float = 0.3):
        self.llm = ChatGoogleGenerativeAI(
            model=model, temperature=temperature, api_key=get_api_key(), max_tokens=1500
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an AI assistant that answers questions using the provided documents.

POLICY:
1. Base answers only on the given context; do not fabricate details.
2. Use a neutral, professional tone. Do NOT assume any person's identity.
3. Refer to people in the third person (use their name or "they/them").
4. If the context is insufficient, say so and suggest what is missing.
5. Keep responses concise and directly address the user's question.
6. Do not add citations inline; the app will show sources separately.""",
                ),
                (
                    "user",
                    """CONTEXT:
{context}

---

QUESTION:
{question}

---

RESPONSE (neutral, third-person; avoid using "I" or "me"):
""",
                ),
            ]
        )

    def invoke(self, context: str, question: str) -> Dict:
        chain = self.prompt_template | self.llm | StrOutputParser()

        response = chain.invoke({"context": context, "question": question})

        return {"answer": response, "model": getattr(self.llm, "model_name", "gemini")}

    def invoke_with_retriever(self, retriever, question: str) -> Dict:
        context, sources = retriever.retrieve_context(question)
        result = self.invoke(context, question)
        result["sources"] = sources
        return result
