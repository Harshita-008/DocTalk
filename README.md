# DocTalk

DocTalk is a RAG-powered document question-answering backend. A user uploads a PDF, TXT, or CSV file, the backend extracts and chunks the document, stores embeddings in ChromaDB Cloud, and answers questions using only retrieved document context with page-level citations.

Frontend deployed link: https://doc-talk-frontend-nine.vercel.app/

Frontend repository: https://github.com/Harshita-008/DocTalk-Frontend

## Features

- Upload PDF, TXT, or CSV documents.
- Extract readable text from uploaded files.
- Split documents into structure-aware chunks with overlap and page metadata.
- Generate document and query embeddings with Gemini embeddings.
- Store chunks, embeddings, page numbers, source metadata, and context windows in ChromaDB Cloud.
- Retrieve context using vector search, keyword ranking, reranking, and guardrail filtering.
- Generate answers with an OpenRouter/OpenAI-compatible chat model.
- Use extractive grounding and support checks to reduce hallucinations.
- Refuse questions that cannot be answered from the uploaded document.
- Return page-level citations for supported answers.
- Support a separate React/Vite frontend through CORS configuration.

## RAG Pipeline

```text
Upload document
  -> Validate file type and size
  -> Save upload temporarily
  -> Extract text from PDF/TXT/CSV
  -> Clean text and repair common PDF spacing artifacts
  -> Split text into structured overlapping chunks
  -> Generate Gemini embeddings for chunks
  -> Upsert chunks, metadata, and embeddings into ChromaDB Cloud
  -> Embed the user question
  -> Retrieve vector candidates
  -> Add keyword-ranked candidates
  -> Rerank and filter noisy or weak evidence
  -> Build grounded context from selected chunks
  -> Generate or extract an answer using only context
  -> Return answer with page citations
```

## Chunking Strategy

DocTalk uses a structure-aware chunking strategy implemented in `app/ingestion/chunker.py`.

The chunker:

- normalizes extracted text,
- restores useful boundaries around headings, numbered lists, bullets, and labelled items,
- removes low-value lines such as review questions, contents headings, and boilerplate,
- splits long text into sentence/word-based chunks,
- keeps configurable word overlap between chunks,
- stores metadata such as page number, chunk index, section title, source name, source type, and context window.

This makes retrieval more reliable than plain fixed-size splitting while keeping the deployment lightweight.

## Tech Stack

- Backend: Python, FastAPI, Uvicorn
- Deployment entrypoint: Vercel serverless function through `api/index.py`
- Frontend: separate React/Vite repository
- Document loading: `pypdf` for PDFs, standard Python readers for TXT/CSV
- Embeddings: Google Gemini embeddings via `google-genai`
- Vector database: ChromaDB Cloud
- Generation: OpenRouter or OpenAI-compatible chat completions through the `openai` SDK
- Configuration: environment variables loaded with `python-dotenv`

## Project Structure

```text
DocTalk/
|-- api/
|   `-- index.py
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
|-- requirements.txt
|-- runtime.txt
|-- vercel.json
|-- .vercelignore
|-- .gitignore
`-- README.md
```

The frontend is maintained separately at:

```text
https://github.com/Harshita-008/DocTalk-Frontend
```

## Local Setup

Clone the backend repository:

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

Create a `.env` file in the project root.

If you store your OpenRouter key in `OPENAI_API_KEY`, use:

```env
OPENAI_API_KEY=your_openrouter_api_key
OPENAI_LLM_MODEL=openai/gpt-4o-mini
LLM_PROVIDER=openrouter
CHROMADB_API_KEY=your_chromadb_api_key
GEMINI_API_KEY=your_gemini_api_key
FRONTEND_ORIGINS=http://localhost:5173,https://doc-talk-frontend-nine.vercel.app
```

Alternatively, you can use the explicit OpenRouter variables:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_LLM_MODEL=openai/gpt-4o-mini
CHROMADB_API_KEY=your_chromadb_api_key
GEMINI_API_KEY=your_gemini_api_key
FRONTEND_ORIGINS=http://localhost:5173,https://doc-talk-frontend-nine.vercel.app
```

Optional configuration:

```env
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
MAX_UPLOAD_MB=10
CHUNK_SIZE=350
CHUNK_OVERLAP=70
TOP_K=12
MAX_CONTEXT_CHUNKS=3
RERANK_TOP_N=3
CONTEXT_WINDOW_SIZE=1
SIMILARITY_THRESHOLD=0.20
```

Start the backend locally:

```bash
uvicorn app.main:app --reload
```

The backend runs at:

```text
http://127.0.0.1:8000
```

For frontend development, use the separate frontend repository:

```bash
git clone https://github.com/Harshita-008/DocTalk-Frontend.git
cd DocTalk-Frontend
npm install
npm run dev
```

Open the frontend locally:

```text
http://localhost:5173
```

## Usage

1. Open the deployed frontend or the local frontend.
2. Upload a PDF, TXT, or CSV document.
3. Wait for the backend to extract, chunk, embed, and index the file.
4. Ask questions about the uploaded document.
5. Review the grounded answer and page citations.

If the document does not contain enough information to answer, DocTalk returns:

```text
I cannot answer this question from the provided document.
```

## Sample Questions

Use any readable PDF, TXT, or CSV document and ask questions that can be answered from that document.

Valid examples:

1. What is the main topic of the document?
   Expected: A document-grounded summary with citation.

2. What is [a term defined in the document]?
   Expected: The definition found in the document with citation.

3. What are the types, steps, components, or categories discussed?
   Expected: A bullet list when the document contains a list or section.

4. Why is [a concept from the document] important?
   Expected: A concise explanation using only document evidence.

5. Describe [a framework, process, law, or policy mentioned in the document].
   Expected: A focused answer grounded in the retrieved section.

Invalid examples:

1. Who is Elon Musk?
   Expected: Refusal unless the uploaded document discusses him.

2. What is blockchain-based cyber security?
   Expected: Refusal unless the uploaded document contains that information.

3. Write a Python program.
   Expected: Refusal because it is not a document-grounded question.

Expected refusal:

```text
I cannot answer this question from the provided document.
```

## API Endpoints

- `GET /` - health check.
- `POST /upload` - upload and index one document.
- `POST /chat?query=your_question` - ask a question about the indexed document.
- `GET /debug/retrieve?query=your_question` - inspect retrieved chunks for debugging.

## Limitations

- Only one active uploaded document is indexed at a time.
- Uploads are stored temporarily; Vercel/serverless environments use `/tmp`.
- ChromaDB collection reset on upload means a new upload replaces the previous indexed document.
- Scanned image-only PDFs are not supported because OCR is not included.
- Citation granularity is page-level, not sentence-level.
- Very large, image-heavy, or complex PDFs may exceed memory, time, or upload limits.
- The answer generator is restricted to retrieved document context and may refuse if retrieval does not find enough evidence.
