import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_REPO_ID = "google/flan-t5-small"
DB_FAISS_PATH = "vectorstore/db_faiss"

def load_llm():
    """Load HuggingFace LLM."""
    hf_token = os.environ.get("HF_TOKEN")
    
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        huggingfacehub_api_token=hf_token,
        model_kwargs={"max_new_tokens": 200}
    )

def load_vector_store():
    """Load FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    return FAISS.load_local(
        DB_FAISS_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

def create_qa_chain():
    """Create QA chain."""
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
    """Main chatbot loop."""
    qa_chain = create_qa_chain()
    print("Chatbot ready! Type 'quit' to exit.")
    
    while True:
        query = input("\nEnter your question: ")
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        if query.strip():
            response = qa_chain.invoke({'query': query})
            print(f"\nAnswer: {response['result']}")

if __name__ == "__main__":
    main()