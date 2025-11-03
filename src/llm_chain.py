import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict


load_dotenv()


class CodeWhispererChain:
    """LangChain integration for CodeWhisperer using FREE Gemini."""

    def __init__(self, model: str = "gemini-2.0-flash", temperature: float = 0.3):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("GOOGLE_API_KEY"),
            max_tokens=1500
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a personal assistant that helps answer questions using personal notes and memories.

IMPORTANT: When answering questions about people mentioned in the documents, respond as if you ARE the person who wrote those notes - use first person ("I", "me", "my", "we", "us") and speak directly to the person being asked about.

RULES:
1. Base ALL answers on the provided snippets
2. Use FIRST PERSON perspective - speak as if you wrote the notes yourself
3. Be warm, personal, and conversational - like you're talking directly to a loved one
4. If information is missing, say so naturally
5. NO formal citations - just speak naturally from your heart
6. Make it feel like a personal conversation, not a documentation search"""),
            (
                "user",
                """PERSONAL NOTES/MEMORIES:
{context}

---

QUESTION:
{question}

---

RESPONSE (speak from your heart, in first person):"""
            )
        ])

    def invoke(self, context: str, question: str) -> Dict:
        chain = (
            self.prompt_template
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke({
            "context": context,
            "question": question
        })

        return {
            "answer": response,
            "model": getattr(self.llm, "model_name", "gemini")
        }

    def invoke_with_retriever(self, retriever, question: str) -> Dict:
        context, sources = retriever.retrieve_context(question)
        result = self.invoke(context, question)
        result["sources"] = sources
        return result

