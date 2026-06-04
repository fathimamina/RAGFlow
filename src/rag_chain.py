from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

def create_rag_chain(vectorstore):
    """Create the RAG (Retrieval + Generation) chain"""
    
    # Updated to a currently supported model
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",   # Best current replacement
        temperature=0.3,
        max_tokens=1024,
    )
    
    # Good prompt template
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful and accurate assistant. Answer the question based only on the following context.
If you don't know the answer, just say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""
    )
    
    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # RAG Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("✅ RAG Chain created successfully with llama-3.3-70b-versatile!")
    return rag_chain