from src.utils import initialize_rag_system, query_rag

def main():
    print("🚀 Welcome to RAGFlow!")
    print("=" * 50)
    
    # Initialize the RAG system
    rag_chain = initialize_rag_system()
    
    print("\n💡 Type 'exit' or 'quit' to stop.\n")
    
    while True:
        question = input("❓ Ask a question: ")
        
        if question.lower() in ['exit', 'quit', 'q']:
            print("👋 Goodbye!")
            break
        
        if question.strip() == "":
            continue
            
        print("🤔 Thinking...")
        answer = query_rag(rag_chain, question)
        print(f"\n📝 Answer:\n{answer}\n")
        print("-" * 50)

if __name__ == "__main__":
    main()