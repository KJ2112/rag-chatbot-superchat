import streamlit as st
from src.document_loader import DocumentLoader
from src.vector_store import VectorStoreManager
from src.retriever import RAGRetriever
from src.llm_chain import CodeWhispererChain
import os
from pathlib import Path
import json


# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================
st.set_page_config(
    page_title="SuperChat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown(
    """
<style>
    /* Main container */
    .main {
        padding-top: 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Chat messages */
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(0,0,0,0.05);
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #1a1a2e;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
        border: none;
    }
    
    /* Input */
    .stChatInput {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
    }
    
    .stChatInput:focus {
        border-color: #4f46e5;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        border-radius: 8px;
        background-color: #f0f4f8;
    }
    
    /* Status messages */
    .stStatus {
        border-radius: 8px;
    }
    
    /* Info boxes */
    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e0e0e0, transparent);
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
with st.sidebar:
    # Header
    st.markdown(
        """
        <div style='text-align: center; margin-bottom: 1rem;'>
            <h2 style='color: #4f46e5; margin: 0;'>⚙️ Settings</h2>
            <p style='color: #666; font-size: 0.9rem; margin: 0.5rem 0 0 0;'>Configure your experience</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.divider()
    
    # Model Configuration (Simplified - only Flash available)
    st.subheader("🤖 Model Configuration")
    
    model_choice = st.selectbox(
        "Select LLM",
        ["gemini-2.0-flash"],
        help="Using Gemini Flash (free tier: 5 requests/minute, 25 requests/day)",
        disabled=True  # Only one option available
    )
    
    st.info(
        "📊 **Free Tier Limits:**\n\n"
        "• 5 requests/minute\n"
        "• 25 requests/day\n"
        "• 1M token context\n"
        "• Forever free ✨"
    )
    
    temperature = st.slider(
        "Temperature (Creativity)",
        0.0, 1.0, 0.3,
        help="Lower = more factual, Higher = more creative"
    )
    
    st.divider()
    
    # Retrieval Settings
    st.subheader("🔍 Retrieval Settings")
    
    k_chunks = st.slider(
        "Results to retrieve",
        1, 10, 3,
        help="More results = more context but slower"
    )
    
    st.divider()
    
    # Knowledge Base Management
    st.subheader("📚 Knowledge Base")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset KB", use_container_width=True):
            st.session_state.vector_store.clear_store()
            st.success("✅ Knowledge base cleared!")
            st.rerun()
    
    with col2:
        if st.button("ℹ️ Stats", use_container_width=True):
            if st.session_state.vector_store._emb.size > 0:
                num_docs = len(st.session_state.vector_store._meta)
                st.info(f"📊 {num_docs} chunks indexed")
            else:
                st.warning("No documents uploaded")
    
    st.divider()
    
    # File Upload
    st.subheader("📥 Upload Documents")
    
    st.write("**Supported formats:**")
    st.caption("📄 PDF  •  📝 Markdown  •  📋 Text")
    
    uploaded_files = st.file_uploader(
        "Choose files to analyze",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        if st.button("➕ Add to Knowledge Base", type="primary", use_container_width=True):
            loader = DocumentLoader()
            total_chunks = 0
            successful_files = 0
            failed_files = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    file_ext = Path(uploaded_file.name).suffix.lower()
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Save temporary file
                    file_path = f"./temp_{uploaded_file.name}"
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Load and process
                    docs = loader.load_single_file(file_path)
                    st.session_state.vector_store.add_documents(docs)
                    
                    # Cleanup
                    os.remove(file_path)
                    
                    total_chunks += len(docs)
                    successful_files += 1
                    
                    # Update progress
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    
                    st.success(f"✅ {uploaded_file.name} ({file_ext}) → {len(docs)} chunks")

                except Exception as e:
                    failed_files.append((uploaded_file.name, str(e)))
                    st.error(f"❌ {uploaded_file.name}: {str(e)}")

            progress_bar.empty()
            status_text.empty()
            
            st.divider()
            
            if successful_files > 0:
                st.success(f"🎉 Added {total_chunks} chunks from {successful_files} file(s)!")
            
            if failed_files:
                st.warning(f"⚠️ {len(failed_files)} file(s) had issues")
    
    st.divider()
    
    # Footer
    st.markdown(
        """
        <div style='text-align: center; color: #999; font-size: 0.85rem; margin-top: 2rem;'>
            <p>💡 Powered by Gemini API</p>
            <p style='font-size: 0.8rem;'>v2.0 - Universal Document Reader</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

# Header
col1, col2 = st.columns([0.85, 0.15])

with col1:
    st.title("📄 SuperChat")
    st.markdown(
        """
        <p style='color: #666; font-size: 1.1rem; margin: -0.5rem 0 1rem 0;'>
        Universal document reader • Chat with any file • Powered by AI
        </p>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.write("")
    st.write("")
    st.metric("Status", "🟢 Active", delta="Ready", delta_color="off")

st.divider()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStoreManager()

# Auto-load documents from data folder on first run
if "data_folder_loaded" not in st.session_state:
    data_folder = Path("data")
    if data_folder.exists() and (any(data_folder.glob("*.pdf")) or any(data_folder.glob("*.md")) or any(data_folder.glob("*.txt"))):
        loader = DocumentLoader()
        try:
            docs = loader.load_markdown_files(str(data_folder))
            if docs:
                st.session_state.vector_store.add_documents(docs)
                st.session_state.data_folder_loaded = True
                # Show info message (only once)
                st.info(f"📚 Automatically loaded {len(set(doc['source'] for doc in docs))} document(s) from `data/` folder")
        except Exception as e:
            st.warning(f"⚠️ Could not auto-load documents from data folder: {e}")
            st.session_state.data_folder_loaded = True
    else:
        st.session_state.data_folder_loaded = True

# Initialize Chain and Retriever
st.session_state.chain = CodeWhispererChain(
    model=model_choice, temperature=temperature
)
st.session_state.retriever = RAGRetriever(
    st.session_state.vector_store, k=k_chunks
)

# ============================================================================
# CHAT INTERFACE
# ============================================================================

st.subheader("💬 Conversation", divider=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])
        
        if "sources" in message and message["sources"]:
            with st.expander("📌 Sources used"):
                cols = st.columns(len(message["sources"]) if len(message["sources"]) <= 3 else 3)
                
                for idx, source in enumerate(message["sources"]):
                    file_ext = Path(source['source']).suffix.lower()
                    file_type_icon = {
                        '.pdf': '📄',
                        '.md': '📝',
                        '.txt': '📋'
                    }.get(file_ext, '📄')
                    
                    col = cols[idx % 3]
                    with col:
                        st.metric(
                            file_type_icon + " File",
                            source['source'].split('/')[-1][:20],
                            f"Score: {source['relevance_score']}"
                        )

# Chat input
user_input = st.chat_input(
    "Ask me anything about your documents...",
    key="user_input"
)

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)

    # Get response from chain
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Analyzing documents..."):
            try:
                # Get retrieval results
                result = st.session_state.chain.invoke_with_retriever(
                    st.session_state.retriever, user_input
                )

                answer = result["answer"]
                sources = result.get("sources", [])

                # Display answer
                st.markdown(answer)
                
                # Display sources
                if sources:
                    with st.expander("📌 Sources used"):
                        cols = st.columns(min(len(sources), 3))
                        
                        for idx, source in enumerate(sources):
                            file_ext = Path(source['source']).suffix.lower()
                            file_type_icon = {
                                '.pdf': '📄',
                                '.md': '📝',
                                '.txt': '📋'
                            }.get(file_ext, '📄')
                            
                            col = cols[idx % 3]
                            with col:
                                st.metric(
                                    file_type_icon + " File",
                                    source['source'].split('/')[-1][:20],
                                    f"Relevance: {source['relevance_score']}"
                                )

                # Add assistant response to history
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ Error: {error_msg}")
                
                # Smart error messages
                if "GOOGLE_API_KEY" in error_msg or "API" in error_msg:
                    st.info(
                        "💡 **Fix:** Add `GOOGLE_API_KEY` to your `.env` file\n\n"
                        "Get a free key at: https://aistudio.google.com"
                    )
                elif "vector" in error_msg.lower():
                    st.info(
                        "💡 **Fix:** Upload some documents first using the sidebar\n\n"
                        "Supported formats: PDF, Markdown, Text files"
                    )
                elif "rate" in error_msg.lower():
                    st.warning(
                        "⏱️ **Rate limit hit!** (5 requests/minute)\n\n"
                        "Please wait ~1 minute before trying again."
                    )

# ============================================================================
# FOOTER & STATS
# ============================================================================

st.divider()

# Stats row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🤖 Model", "Gemini 2.0 Flash", "Free Tier")

with col2:
    st.metric("🌡️ Temperature", temperature, "Creative" if temperature > 0.5 else "Factual")

with col3:
    st.metric("📚 Context", f"{k_chunks} chunks", "Retrieved")

with col4:
    st.metric("💬 Messages", len(st.session_state.messages), "Total")

st.caption(
    "📄 **SuperChat** • Universal Document Reader • "
    "Chat with PDFs, Markdown, and Text files • Powered by Gemini API • v1.0"
)