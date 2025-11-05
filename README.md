# SuperChat - RAG-Powered Document Chatbot

A production-ready Retrieval-Augmented Generation (RAG) chatbot that enables natural language queries over your documents. Built with LangChain, vector embeddings, and Google Gemini API.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)

🔗 **Live Demo:** [Try the Streamlit App](https://rag-chatbot-superchatkj.streamlit.app/)

## 🚀 Features

- **Multi-Format Support**: Process PDF, Markdown, and Text documents
- **Semantic Search**: Advanced vector-based retrieval for contextually relevant answers
- **Source Citations**: Every response includes source references with relevance scores
- **Persistent Storage**: Vector embeddings stored locally with automatic persistence
- **Interactive UI**: Clean, modern Streamlit interface with real-time chat
- **Configurable Retrieval**: Adjustable chunk size, overlap, and retrieval count
- **Cost-Effective**: Uses free-tier Gemini API with local embeddings

## 🛠️ Tech Stack

- **LangChain** - RAG orchestration and LLM integration
- **Google Gemini 2.0 Flash** - Large Language Model (FREE tier)
- **NumPy-based Vector Store** - Lightweight, local vector database
- **HuggingFace Embeddings** - Text embeddings (all-MiniLM-L6-v2)
- **Streamlit** - Interactive web interface
- **pypdf** - PDF document parsing

## 📋 Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get free key here](https://aistudio.google.com))

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/rag-chatbot-superchat.git
cd rag-chatbot-superchat
```

> **Note**: Replace `yourusername` with your GitHub username.

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 5. Run the Application
```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

## 📁 Project Structure

```
rag-chatbot-superchat/
├── src/
│   ├── document_loader.py    # Document loading and chunking
│   ├── vector_store.py       # Vector database management
│   ├── retriever.py          # RAG retrieval logic
│   └── llm_chain.py          # LLM chain orchestration
├── data/                     # Default documents folder
├── chroma_data/              # Vector store persistence (auto-created)
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
└── README.md
```

## 🏗️ Architecture

### RAG Pipeline

1. **Document Processing**: Files are loaded and chunked using recursive text splitting
2. **Embedding Generation**: Text chunks are converted to vector embeddings
3. **Vector Storage**: Embeddings stored in local NumPy-based vector store
4. **Query Processing**: User queries are embedded and matched against stored vectors
5. **Context Retrieval**: Top-k most relevant chunks retrieved based on cosine similarity
6. **LLM Generation**: Gemini generates answers using retrieved context
7. **Response Formatting**: Answers include source citations and relevance scores

### Key Components

- **DocumentLoader**: Handles multi-format document parsing (PDF, MD, TXT)
- **VectorStoreManager**: Manages embeddings, metadata, and similarity search
- **RAGRetriever**: Implements retrieval logic with configurable top-k
- **CodeWhispererChain**: LangChain integration for prompt management and LLM calls

## 🔧 Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Your Google Gemini API key (required)

### Application Settings
- **Temperature**: Controls response creativity (0.0-1.0, default: 0.3)
- **Retrieval Count (k)**: Number of chunks to retrieve (1-10, default: 3)
- **Chunk Size**: Text splitting size (default: 800 tokens)
- **Chunk Overlap**: Overlap between chunks (default: 100 tokens)

## 📖 Usage

1. **Upload Documents**: Use the sidebar to upload PDF, Markdown, or Text files
2. **Ask Questions**: Type your question in the chat interface
3. **View Sources**: Expand the "Sources used" section to see document references
4. **Manage Knowledge Base**: Reset or view statistics using sidebar controls

## 🎯 Use Cases

- Technical documentation Q&A
- Internal knowledge base queries
- Research paper analysis
- Code documentation search
- Educational content exploration

## 🔒 Privacy & Security

- All embeddings stored locally
- Documents processed on your machine
- No data sent to third-party services (except Gemini API for LLM)
- Vector store persists between sessions

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Built with ❤️ using LangChain and Google Gemini**
