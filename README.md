# RAG Assistant - Document Q&A System

A Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask questions based only on the uploaded content.

## 🎯 Overview

This application implements a RAG pipeline that:
- Accepts document uploads (TXT, PDF, CSV, DOCX)
- Chunks and embeds documents using sentence-transformers
- Stores embeddings in FAISS vector database
- Retrieves relevant context for user queries
- Generates answers using Groq's LLM (Llama 3.3 70B)
- Returns "I don't have enough information" when answer isn't found in documents

## 🛠️ Tech Stack

- **Python 3.10+**
- **FastAPI** - Web framework
- **SentenceTransformers** - Text embeddings (all-mpnet-base-v2)
- **FAISS** - Vector similarity search
- **Groq** - LLM for answer generation (Llama 3.3 70B)
- **PyPDF2, python-docx, pandas** - Document processing

