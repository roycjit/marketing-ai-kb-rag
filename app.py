"""
Streamlit-based RAG System UI

This is the main entry point for the Streamlit application that provides
a user interface for the RAG (Retrieval-Augmented Generation) system.
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.rag_system.document_loader import load_documents
from src.rag_system.qa_chain import create_qa_chain
from src.rag_system.vectorstore import create_vectorstore

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False


def sidebar():
    """Render the sidebar for document upload and configuration."""
    st.sidebar.title("⚙️ Configuration")

    # API Key input
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Enter your OpenAI API key",
    )

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.sidebar.markdown("---")
    st.sidebar.title("📄 Document Upload")

    # File uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to query",
    )

    # Process uploaded documents
    if uploaded_files and st.sidebar.button("Process Documents"):
        if not api_key:
            st.sidebar.error("Please enter your OpenAI API key first!")
            return

        with st.sidebar.status("Processing documents...", expanded=True) as status:
            try:
                # Save uploaded files temporarily
                data_dir = Path("data")
                data_dir.mkdir(exist_ok=True)

                temp_files = []
                for uploaded_file in uploaded_files:
                    temp_path = data_dir / uploaded_file.name
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    temp_files.append(str(temp_path))

                st.write("📖 Loading documents...")
                documents = load_documents(temp_files)
                st.write(f"✅ Loaded {len(documents)} document chunks")

                st.write("🔍 Creating vector store...")
                vectorstore = create_vectorstore(documents)
                st.session_state.vectorstore = vectorstore
                st.write("✅ Vector store created")

                st.write("🔗 Setting up QA chain...")
                qa_chain = create_qa_chain(vectorstore)
                st.session_state.qa_chain = qa_chain
                st.write("✅ QA chain ready")

                st.session_state.documents_loaded = True
                status.update(label="✅ Documents processed successfully!", state="complete")

            except Exception as e:
                st.sidebar.error(f"Error processing documents: {str(e)}")
                status.update(label="❌ Processing failed", state="error")

    # Display status
    if st.session_state.documents_loaded:
        st.sidebar.success("✅ System ready!")
    else:
        st.sidebar.info("📤 Upload documents to get started")


def main():
    """Main application function."""
    initialize_session_state()

    # Render sidebar
    sidebar()

    # Main content
    st.title("🤖 RAG System")
    st.markdown(
        "Ask questions about your documents using AI-powered Retrieval-Augmented Generation"
    )

    # Check if system is ready
    if not st.session_state.documents_loaded:
        st.info(
            "👈 Please upload documents and configure your API key in the sidebar to get started."
        )
        return

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.qa_chain.invoke({"query": prompt})
                    answer = response.get("result", "I couldn't generate an answer.")

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    # Show source documents in expander
                    if "source_documents" in response:
                        with st.expander("📚 View source documents"):
                            for i, doc in enumerate(response["source_documents"], 1):
                                st.markdown(f"**Source {i}:**")
                                st.markdown(doc.page_content)
                                st.markdown("---")

                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
