"""
QA Chain Module

This module handles creating and managing the question-answering chain.
"""

from operator import itemgetter

from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI


def format_docs(docs):
    """Format documents for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_qa_chain(
    vectorstore: Chroma,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    k: int = 4,
):
    """
    Create a question-answering chain using the vector store.

    Args:
        vectorstore: Chroma vector store to use for retrieval
        model_name: Name of the OpenAI model to use
        temperature: Temperature for response generation
        k: Number of documents to retrieve

    Returns:
        RAG chain instance that returns dict with 'answer' and 'context'
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

    # Create prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create RAG chain
    rag_chain = {
        "context": itemgetter("input") | retriever,
        "input": itemgetter("input"),
    } | RunnablePassthrough.assign(
        answer=(
            {
                "context": lambda x: format_docs(x["context"]),
                "input": itemgetter("input"),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
    )

    return rag_chain
