from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
import os

def create_vector_store(chunks, persist_directory="vectorstore"):
    """Create and save FAISS vector store"""
    
    # Using a strong open-source embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print("🔄 Creating vector store... (this may take a minute)")
    
    # Create FAISS vector store
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
        distance_strategy=DistanceStrategy.COSINE
    )
    
    # Save to disk
    vectorstore.save_local(persist_directory)
    print(f"✅ Vector store saved to '{persist_directory}' folder")
    
    return vectorstore


def load_vector_store(persist_directory="vectorstore"):
    """Load existing FAISS vector store"""
    
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = FAISS.load_local(
        persist_directory, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    print(f"✅ Vector store loaded from '{persist_directory}'")
    return vectorstore