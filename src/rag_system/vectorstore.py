"""
Vector Store Module

This module handles creating and managing the vector store for document embeddings.
"""

from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def create_vectorstore(documents: List[Document], persist_directory: str = "./data/chroma_db") -> Chroma:
    """
    Create a vector store from documents using OpenAI embeddings.

    Args:
        documents: List of documents to embed
        persist_directory: Directory to persist the vector store

    Returns:
        Chroma vector store instance
    """
    # Create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    print(f"Created vector store with {len(documents)} documents")

    return vectorstore


def load_vectorstore(persist_directory: str = "./data/chroma_db") -> Chroma:
    """
    Load an existing vector store.

    Args:
        persist_directory: Directory where vector store is persisted

    Returns:
        Chroma vector store instance
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    return vectorstore
