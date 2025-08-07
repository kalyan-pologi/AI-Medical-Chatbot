# 🤖 Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot powered by Streamlit that provides intelligent medical information based on "The Gale Encyclopedia of Medicine" content.

## 🏥 Features

- **RAG-Powered Q&A**: Intelligent answers based on medical encyclopedia content
- **Source Citations**: View the source documents for each answer
- **Clean Chat Interface**: Modern UI with message history
- **Vector Search**: Fast document retrieval using FAISS
- **Real-time Responses**: Powered by HuggingFace language models
- **Medical Knowledge**: Comprehensive medical information from trusted sources

## 📋 Prerequisites

- Python 3.8 or higher
- HuggingFace account and API token (optional but recommended)
- Git

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd medical-chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Set Up Environment Variables (Optional)

Create a `.env` file in the project root:
```env
HF_TOKEN=your_huggingface_token_here
```

**Note**: The application will work without the HuggingFace token, but responses may be limited.

## 🎯 Usage

### Running the Application

1. **Start the Streamlit app:**
```bash
python -m streamlit run app.py
```

2. **Open your browser and navigate to:**
```
http://localhost:8501
```

3. **Start asking medical questions!**

### Example Questions

- "What are the symptoms of diabetes?"
- "How is hypertension treated?"
- "What causes heart disease?"
- "What are the side effects of aspirin?"


## 🔧 Configuration

### HuggingFace Model
The application uses `google/flan-t5-small` by default. You can modify the model in `app.py`:

```python
HUGGINGFACE_REPO_ID = "google/flan-t5-small"
```

### Vector Store
The FAISS vector store is located at `vectorstore/db_faiss/`. The application automatically loads this pre-processed database.

## 📚 Dependencies

- **streamlit**: Web application framework
- **langchain**: LLM orchestration
- **langchain-huggingface**: HuggingFace integration
- **langchain-community**: Community integrations
- **faiss-cpu**: Vector similarity search
- **sentence-transformers**: Text embeddings
- **transformers**: HuggingFace transformers
- **python-dotenv**: Environment variable management
- **pypdf**: PDF processing
