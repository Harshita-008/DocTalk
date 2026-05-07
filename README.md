# DocTalk

DocTalk is a RAG-powered document chat application built for Assignment 03. A user can upload a PDF, TXT, or CSV file, the app indexes the document into a vector database, and the user can ask natural-language questions that are answered only from the uploaded document with page citations.

Deployed link: https://doctalk-ysjx.onrender.com

## Features

- Upload PDF, TXT, or CSV documents.
- Extract text from uploaded files.
- Split documents into structure-aware chunks.
- Generate embeddings for chunks.
- Store chunk embeddings and metadata in ChromaDB.
- Retrieve relevant context using vector search, keyword scoring, and reranking.
- Generate grounded answers using retrieved context.
- Refuse questions that cannot be answered from the document.
- Show page-level citations for answers.
- Use a React frontend with a FastAPI backend.

## RAG Pipeline

```text
Upload document
  -> Load and extract text
  -> Clean and chunk text
  -> Generate embeddings
  -> Store chunks in ChromaDB
  -> Retrieve relevant chunks for a question
  -> Filter noisy or unrelated context
  -> Generate a grounded answer
  -> Return answer with citations
```

## Chunking Strategy

DocTalk uses a structure-aware chunking strategy implemented in `app/ingestion/chunker.py`.

The chunker:

- normalizes extracted text,
- preserves useful structure such as headings, paragraphs, numbered lists, and labelled points,
- removes low-value lines such as review questions or boilerplate,
- creates overlapping chunks so nearby context is not lost,
- stores metadata such as page number, chunk index, section title, and context window.

This keeps retrieved context more coherent than fixed-size splitting alone.

## Tech Stack

- Backend: Python, FastAPI, Uvicorn
- Frontend: React, Vite
- Document loading: PyMuPDF, pypdf, LangChain document objects
- Embeddings: Sentence Transformers with fallback lexical embeddings
- Vector database: ChromaDB
- Generation: OpenAI API when configured, with local fallback behavior

## Project Structure

```text
DocTalk/
|-- app/
|   |-- agent/
|   |   |-- generator.py
|   |   |-- guardrails.py
|   |   `-- prompt.py
|   |-- ingestion/
|   |   |-- chunker.py
|   |   |-- document_loader.py
|   |   |-- embedder.py
|   |   `-- pdf_loader.py
|   |-- retrieval/
|   |   |-- retriever.py
|   |   `-- vector_store.py
|   |-- config.py
|   `-- main.py
|-- frontend/
|   |-- src/
|   |-- index.html
|   |-- package.json
|   `-- vite.config.js
|-- sample/
|-- requirements.txt
|-- runtime.txt
|-- .gitignore
`-- README.md
```

## Local Setup

Clone the repository:

```bash
git clone https://github.com/Harshita-008/DocTalk.git
cd DocTalk
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=all-MiniLM-L6-v2
ENABLE_SENTENCE_TRANSFORMERS=true
```

Start the backend:

```bash
uvicorn app.main:app --reload
```

Install and start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## Usage

1. Upload a PDF, TXT, or CSV document.
2. Wait for indexing to complete.
3. Ask questions about the document.
4. Review the grounded answer and citations.

If the document does not contain the answer, the app returns:

```text
I cannot answer this question from the provided document.
```

## Sample Questions

Use `sample/sample.pdf` to test the application.

Valid questions:

1. What is entrepreneurship?
   Expected: A definition of entrepreneurship with citation.

2. What are types of entrepreneurship?
   Expected: A list of entrepreneurship classifications with citations.

3. What are the problems faced by entrepreneurs in India?
   Expected: Bullet points grounded in the document.

4. Why is entrepreneurship important for economic development?
   Expected: Bullet points explaining its economic importance.

5. What was the main issue in the Satyam case study?
   Expected: A document-grounded answer identifying accounting fraud as the main issue.

Invalid questions:

1. What is machine learning?
   Expected: Refusal.

2. Who is Elon Musk?
   Expected: Refusal.

3. Write a Python program.
   Expected: Refusal.

Expected refusal:

```text
I cannot answer this question from the provided document.
```

## API Endpoints

- `GET /` - health check
- `POST /upload` - upload and index a document
- `POST /chat?query=your_question` - ask a question
- `GET /debug/retrieve?query=your_question` - inspect retrieved chunks

## Limitations

- Only one active uploaded document is indexed at a time.
- Scanned image-only PDFs are not supported because OCR is not included.
- Citation granularity is page-level.
- Very large or complex PDFs may require more memory on the deployed server.
