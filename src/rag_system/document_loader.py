"""
Document Loader Module

This module handles loading and processing documents for the RAG system.
"""

from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_documents(file_paths: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Load and split documents from PDF files.

    Args:
        file_paths: List of file paths to load
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks

    Returns:
        List of document chunks
    """
    documents = []

    for file_path in file_paths:
        if not Path(file_path).exists():
            print(f"Warning: File {file_path} does not exist, skipping...")
            continue

        # Load PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        documents.extend(docs)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents and split into {len(chunks)} chunks")

    return chunks
