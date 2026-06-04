from src.document_loader import load_documents, split_documents
from src.vector_store import create_vector_store, load_vector_store
from src.rag_chain import create_rag_chain
import os

def initialize_rag_system(data_dir="data", vectorstore_dir="vectorstore"):
    """Initialize the complete RAG system"""
    
    # Check if vector store already exists
    if os.path.exists(vectorstore_dir) and len(os.listdir(vectorstore_dir)) > 0:
        print("🔄 Loading existing vector store...")
        vectorstore = load_vector_store(vectorstore_dir)
    else:
        print("🔄 Loading documents and creating new vector store...")
        # Load and split documents
        documents = load_documents(data_dir)
        chunks = split_documents(documents)
        
        # Create vector store
        vectorstore = create_vector_store(chunks, vectorstore_dir)
    
    # Create RAG chain
    rag_chain = create_rag_chain(vectorstore)
    
    return rag_chain


def query_rag(rag_chain, question: str):
    """Ask a question to the RAG system"""
    try:
        response = rag_chain.invoke(question)
        return response
    except Exception as e:
        return f"❌ Error: {str(e)}"