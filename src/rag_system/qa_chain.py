"""
QA Chain Module

This module handles creating and managing the question-answering chain.
"""

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI


def create_qa_chain(
    vectorstore: Chroma,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    k: int = 4,
) -> RetrievalQA:
    """
    Create a question-answering chain using the vector store.

    Args:
        vectorstore: Chroma vector store to use for retrieval
        model_name: Name of the OpenAI model to use
        temperature: Temperature for response generation
        k: Number of documents to retrieve

    Returns:
        RetrievalQA chain instance
    """
    # Create LLM
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=False,
    )

    return qa_chain
