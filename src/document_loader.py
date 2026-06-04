from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter   # ← Fixed Import
from pathlib import Path
import os

def load_documents(data_dir="data"):
    """Load all documents from the data directory"""
    documents = []
    
    # Load PDFs
    pdf_loader = PyPDFDirectoryLoader(data_dir)
    pdf_docs = pdf_loader.load()
    documents.extend(pdf_docs)
    
    # Load text files if any
    txt_files = list(Path(data_dir).glob("*.txt")) + list(Path(data_dir).glob("*.md"))
    for txt_file in txt_files:
        loader = TextLoader(str(txt_file), encoding="utf-8")
        txt_docs = loader.load()
        documents.extend(txt_docs)
    
    print(f"✅ Loaded {len(documents)} documents from {data_dir}")
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks