from src.document_loader import load_documents, split_documents
from src.vector_store import create_vector_store, load_vector_store
from src.rag_chain import create_rag_chain
import os

def initialize_rag_system(data_dir="data", vectorstore_dir="vectorstore"):
    """Initialize RAG system safely for Streamlit"""

    # Check if vector store exists safely
    if os.path.exists(vectorstore_dir) and os.path.isdir(vectorstore_dir):
        if len(os.listdir(vectorstore_dir)) > 0:
            vectorstore = load_vector_store(vectorstore_dir)
        else:
            documents = load_documents(data_dir)
            chunks = split_documents(documents)
            vectorstore = create_vector_store(chunks, vectorstore_dir)
    else:
        documents = load_documents(data_dir)
        chunks = split_documents(documents)
        vectorstore = create_vector_store(chunks, vectorstore_dir)

    rag_chain = create_rag_chain(vectorstore)
    return rag_chain


def query_rag(rag_chain, question: str):
    """Query RAG system safely"""
    try:
        return rag_chain.invoke(question)
    except Exception as e:
        return f"❌ Error: {str(e)}"