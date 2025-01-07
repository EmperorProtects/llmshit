## Ollama RAG app

A Streamlit-based chatbot that implements Retrieval-Augmented Generation (RAG) using Ollama and ChromaDB.

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install streamlit requests chromadb ollama uuid
```

3. Ensure Ollama is running locally on port 11434

## Usage

1. Start the application:

```bash
streamlit run app.py
```

2. Access the web interface (typically at <http://localhost:8501>)
3. Upload text documents using the file uploader
4. Start chatting with the bot

## How it Works

1. **Document Processing**

   - Users can upload text documents
   - Documents are embedded and stored in ChromaDB

2. **Chat Interaction**

   - User queries are processed using semantic search
   - Relevant context is retrieved from the document collection
   - Ollama generates responses based on the context

3. **Storage**
   - Conversations are stored in the session
   - Document embeddings persist in ChromaDB

## Examples
