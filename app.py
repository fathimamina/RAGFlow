import streamlit as st
from src.utils import initialize_rag_system, query_rag
import os

# Page Configuration
st.set_page_config(
    page_title="RAGFlow - Chat with your Documents",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAGFlow")
st.markdown("### Intelligent Document Assistant")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    if st.button("🔄 Rebuild Vector Store"):
        if os.path.exists("vectorstore"):
            import shutil
            shutil.rmtree("vectorstore")
        st.success("Vector store cleared! Refresh the app.")
    
    st.info("Put your PDF files in the `data/` folder")

# Initialize RAG System
@st.cache_resource
def get_rag_chain():
    return initialize_rag_system()

try:
    rag_chain = get_rag_chain()
    st.success("✅ RAG System Ready!")
except Exception as e:
    st.error(f"Failed to initialize: {e}")
    st.stop()

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_rag(rag_chain, prompt)
            st.markdown(response)
    
    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})