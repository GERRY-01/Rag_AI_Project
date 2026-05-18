# RAG PDF Chatbot using Flask + Groq + LangChain

A Retrieval-Augmented Generation (RAG) chatbot that allows users to query a PDF document and ask questions about its content.  
The system retrieves relevant context from the PDF using FAISS vector search and generates intelligent answers using a Large Language Model (LLM) via Groq API.

---

## Features

- Load and process PDF documents
- Split text into semantic chunks
- Generate embeddings using Sentence Transformers
- Fast similarity search using FAISS
- AI-powered answers using Groq LLM (LLaMA 3)
- Simple Flask web interface
- Real-time question answering from documents

---

## Project Architecture

PDF Document  
↓  
Text Splitting (LangChain)  
↓  
Embeddings (SentenceTransformers)  
↓  
Vector Store (FAISS)  
↓  
User Query  
↓  
Retriever (Similarity Search)  
↓  
Context + Prompt  
↓  
Groq LLM (LLaMA 3)  
↓  
Generated Answer  

---

## Tech Stack

- Python
- Flask
- LangChain
- FAISS (Vector Database)
- SentenceTransformers (Embeddings)
- Groq API (LLM)
- HTML/CSS (Frontend)

---

## Project Structure

pdf_rag_chatbot/
│
├── app.py
├── .env
├── .gitignore
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── data/
│   └── report.pdf
│
└── README.md

---

## Installation & Setup

### 1. Clone the repository

git clone https://github.com/GERRY-01/Rag_AI_Project
cd PDF_RAG_CHATBOT

---

### 2. Create a virtual environment

python -m venv .venv

Activate it:

Linux / Mac:
source .venv/bin/activate

Windows:
.venv\Scripts\activate

---

### 3. Install dependencies

pip install -r requirements.txt

---

### 4. Setup environment variables

Open the `.env` file in the root directory and replace the placeholder value with your Groq API key:

GROQ_API_KEY=your_groq_api_key_here

---

### 5. Run the application

python app.py

Open in browser:

http://127.0.0.1:5000

---

## How It Works

1. PDF is loaded using LangChain PDF loader  
2. Text is split into chunks  
3. Each chunk is converted into embeddings  
4. Embeddings are stored in FAISS vector database  
5. User enters a question  
6. Retriever finds relevant chunks  
7. Context is sent to Groq LLM  
8. LLM generates a final answer  

---

## Example Use Cases

- Ask questions from research papers  
- Summarize documents  
- Extract key insights from reports  
- Query technical PDFs  

---

## Security Notes

- API keys are stored in `.env`
- `.env` file is excluded using `.gitignore`
- Never upload secrets to GitHub

---

## Future Improvements

- Add PDF upload feature in UI
- Add chat history memory
- Add streaming responses (real-time typing)
- Support multiple PDFs
- Deploy to cloud (Render / Railway / AWS)
- Add authentication system

---
