import streamlit as st
from src.document_loader import DocumentLoader
from src.vector_store import VectorStoreManager
from src.retriever import RAGRetriever
from src.llm_chain import CodeWhispererChain
import os
from pathlib import Path
import time


# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="SmartReader AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>
    .main { padding-top: 2rem; }
    .stChatMessage { border-radius: 12px; padding: 1rem; }
    .block-container { padding-top: 1rem; }
    h1 { color: #667eea; }
</style>
""",
    unsafe_allow_html=True,
)


# ==================== SIDEBAR ====================
with st.sidebar:
    st.title("⚙️ Configuration")

    # Model selection
    st.subheader("🤖 AI Model")
    st.info("Using **Gemini 2.0 Flash** (FREE tier)")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower = more factual"
    )

    k_chunks = st.slider(
        "Context Chunks",
        min_value=1,
        max_value=10,
        value=3,
    )

    st.divider()
    st.subheader("📚 Knowledge Base")

    if st.button("🗑️ Clear All Documents", type="secondary", use_container_width=True):
        st.session_state.vector_store.clear_store()
        st.success("✅ Cleared!")
        st.rerun()

    st.write("**Formats:** PDF, MD, TXT")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("📤 Add to Knowledge Base", type="primary", use_container_width=True):
            loader = DocumentLoader()
            total_chunks = 0
            successful = 0

            for file in uploaded_files:
                try:
                    file_ext = Path(file.name).suffix.lower()
                    file_path = f"./temp_{file.name}"
                    
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())

                    docs = loader.load_single_file(file_path)
                    st.session_state.vector_store.add_documents(docs)
                    os.remove(file_path)
                    
                    total_chunks += len(docs)
                    successful += 1
                    st.success(f"✅ {file.name} - {len(docs)} sections")

                except Exception as e:
                    st.error(f"❌ {file.name}: {str(e)}")

            st.divider()
            st.success(f"✅ Added {total_chunks} chunks from {successful} file(s)")

    # Show stats
    if st.session_state.vector_store._emb.size > 0:
        st.divider()
        st.success(f"📊 {len(st.session_state.vector_store._meta)} sections loaded")


# ==================== HEADER ====================
st.title("📚 SmartReader AI")
st.markdown("**Semantic search and Q&A over uploaded documents** • Upload PDF, MD, or TXT files and ask questions")


# ==================== INITIALIZE ====================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStoreManager()

st.session_state.chain = CodeWhispererChain(
    model="gemini-2.0-flash",
    temperature=temperature
)
st.session_state.retriever = RAGRetriever(
    st.session_state.vector_store, k=k_chunks
)


# ==================== CHAT DISPLAY ====================
st.subheader("💬 Chat")

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])
        
        if "sources" in message and message["sources"]:
            with st.expander("📌 Sources"):
                for source in message["sources"]:
                    file_ext = Path(source['source']).suffix.lower()
                    icon = {'pdf': '📄', 'md': '📝', 'txt': '📋'}.get(file_ext, '📄')
                    st.caption(f"{icon} {source['source']} (Relevance: {source['relevance_score']})")


# ==================== CHAT INPUT ====================
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    # Add to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Get response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Processing your question..."):
            try:
                if st.session_state.vector_store._emb.size == 0:
                    st.error("❌ No documents uploaded")
                    st.info("📌 Upload documents using the sidebar first")
                else:
                    # Measure response time
                    start_time = time.time()
                    
                    result = st.session_state.chain.invoke_with_retriever(
                        st.session_state.retriever, user_input
                    )

                    end_time = time.time()
                    response_time = end_time - start_time

                    answer = result["answer"]
                    sources = result.get("sources", [])

                    st.markdown(answer)
                    
                    # Show sources
                    if sources:
                        with st.expander("📌 Sources"):
                            for source in sources:
                                file_ext = Path(source['source']).suffix.lower()
                                icon = {'pdf': '📄', 'md': '📝', 'txt': '📋'}.get(file_ext, '📄')
                                st.caption(f"{icon} {source['source']} (Relevance: {source['relevance_score']})")
                    
                    # Show performance metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"⏱️ Response time: {response_time:.2f}s")
                    with col2:
                        st.caption(f"📊 Sources used: {len(sources)}")

                    # Add to history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
            
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ Error: {error_msg}")
                
                if "API_KEY" in error_msg or "API" in error_msg:
                    st.info("💡 **Fix:** Ensure GOOGLE_API_KEY is set in .env file")
                elif "vector" in error_msg.lower() or "empty" in error_msg.lower():
                    st.info("💡 **Fix:** Upload documents using the sidebar")
                else:
                    st.info("💡 **Fix:** Try again or check your internet connection")


# ==================== FOOTER ====================
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    messages_count = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.metric("Questions", messages_count)

with col2:
    st.metric("Model", "Gemini Flash", delta="Free ✨")

with col3:
    if st.session_state.vector_store._emb.size > 0:
        st.metric("Docs Loaded", len(st.session_state.vector_store._meta))

with col4:
    st.metric("Temperature", f"{temperature:.1f}")

st.caption(
    "SmartReader AI • Powered by LangChain + Gemini API • "
    "Neutral, professional Q&A system"
)
