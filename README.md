# RAG_DEMO

A minimal Retrieval-Augmented Generation (RAG) demo built with LangChain.

It takes a short text document, splits it into chunks, embeds the chunks into a
vector store, retrieves the most relevant chunks for a query, and then uses an
LLM (via Groq) to answer questions grounded in that retrieved context.

## How it works

1. **Load document** – a sample text about AI, ML, NLP, Computer Vision, etc.
2. **Split** – `RecursiveCharacterTextSplitter` breaks it into chunks.
3. **Embed** – `sentence-transformers/all-MiniLM-L6-v2` converts chunks to vectors.
4. **Store** – chunks are persisted in a local Chroma vector store (`./chroma_db`).
5. **Retrieve** – similarity search fetches the top-k relevant chunks.
6. **Generate** – a Groq-hosted LLM answers questions using the retrieved context.

## Requirements

- Python 3.9+
- A [Groq](https://console.groq.com/) API key

Install dependencies:

```bash
pip install langchain langchain-text-splitters langchain-huggingface langchain-community langchain-groq langchain-classic chromadb sentence-transformers python-dotenv
```

## Setup

Create a `.env` file in the project root (it is git-ignored):

```env
API_KEY=your_groq_api_key_here
```

## Usage

```bash
python RAGDEMO.py
```

The script prints each pipeline step, runs a live similarity search, then asks a
few sample questions and prints the LLM answers.

## Project structure

```
.
├── RAGDEMO.py      # main RAG pipeline
├── .env            # API key (not committed)
├── .gitignore
└── README.md
```

## Notes

- Embeddings are computed locally with HuggingFace models; generation calls the Groq API.
- The vector store is written to `./chroma_db` and is excluded from version control.
