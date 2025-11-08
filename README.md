# RAG System - Streamlit UI Boilerplate

A modern, UV-ready boilerplate project for building RAG (Retrieval-Augmented Generation) systems with a Streamlit-based user interface.

## Features

- 🚀 **UV Package Manager**: Fast, modern Python package management
- 🎨 **Streamlit UI**: Beautiful, interactive web interface
- 🤖 **RAG System**: Complete RAG implementation using LangChain
- 📄 **PDF Support**: Load and query PDF documents
- 🔍 **Vector Search**: ChromaDB for efficient similarity search
- 💬 **Chat Interface**: Conversational UI for document Q&A
- 🔐 **Environment Config**: Secure API key management

## Prerequisites

- Python 3.12 or higher
- OpenAI API key

## Quick Start

### 1. Install UV (if not already installed)

```bash
pip install uv
```

### 2. Clone the repository

```bash
git clone https://github.com/roycjit/rag-2025.git
cd rag-2025
```

### 3. Install dependencies

```bash
uv sync
```

This will create a virtual environment and install all required dependencies.

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run the application

```bash
uv run streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

## Usage

1. **Configure API Key**: Enter your OpenAI API key in the sidebar
2. **Upload Documents**: Upload one or more PDF files using the file uploader
3. **Process Documents**: Click "Process Documents" to index your files
4. **Ask Questions**: Use the chat interface to ask questions about your documents

## Project Structure

```
rag-2025/
├── app.py                      # Main Streamlit application
├── pyproject.toml              # UV project configuration
├── .env.example                # Environment variables template
├── .python-version             # Python version specification
├── src/
│   └── rag_system/
│       ├── __init__.py
│       ├── document_loader.py  # Document loading and processing
│       ├── vectorstore.py      # Vector store management
│       └── qa_chain.py         # QA chain implementation
├── data/                       # Data directory (created at runtime)
└── docs/                       # Documentation

```

## Development

### Install development dependencies

```bash
uv sync --all-extras
```

### Run linter

```bash
uv run ruff check .
```

### Format code

```bash
uv run ruff format .
```

## Dependencies

Main dependencies:
- **streamlit**: Web UI framework
- **langchain**: LLM framework
- **langchain-openai**: OpenAI integration
- **chromadb**: Vector database
- **pypdf**: PDF processing
- **python-dotenv**: Environment variable management

## Configuration

You can customize the RAG system behavior by modifying:

- **Model**: Change `model_name` in `src/rag_system/qa_chain.py`
- **Embeddings**: Change embedding model in `src/rag_system/vectorstore.py`
- **Chunk Size**: Modify `chunk_size` and `chunk_overlap` in `src/rag_system/document_loader.py`
- **Retrieval**: Adjust `k` parameter in `src/rag_system/qa_chain.py`

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.