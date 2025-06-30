import os
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_REPO_ID = "google/flan-t5-small"
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def load_llm():
    """Load and cache HuggingFace LLM."""
    hf_token = os.environ.get("HF_TOKEN")
    
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        huggingfacehub_api_token=hf_token,
        model_kwargs={"max_new_tokens": 200}
    )

@st.cache_resource
def load_vector_store():
    """Load and cache FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    return FAISS.load_local(
        DB_FAISS_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

@st.cache_resource
def create_qa_chain():
    """Create and cache QA chain."""
    template = """Use the context to answer the question. If you don't know, say you don't know.

Context: {context}
Question: {question}

Answer:"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    llm = load_llm()
    db = load_vector_store()
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
    st.title("🤖 RAG Chatbot")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Add user message
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                qa_chain = create_qa_chain()
                response = qa_chain.invoke({'query': prompt})
                
                answer = response['result']
                sources = response['source_documents']
                
                st.markdown(answer)
                
                # Show sources
                with st.expander("Sources"):
                    for i, doc in enumerate(sources, 1):
                        st.write(f"**Source {i}:**")
                        st.write(doc.page_content[:300] + "...")
                
                # Add to chat history
                st.session_state.messages.append({'role': 'assistant', 'content': answer})
    
    # Sidebar
    with st.sidebar:
        st.header("Status")
        
        if os.path.exists(DB_FAISS_PATH):
            st.success("✅ Vector store loaded")
        else:
            st.error("❌ Vector store not found")
        
        if os.environ.get("HF_TOKEN"):
            st.success("✅ HuggingFace token found")
        else:
            st.error("❌ HuggingFace token missing")
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()