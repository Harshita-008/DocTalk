import os
import re
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.agent.generator import REFUSAL, generate_answer
from app.agent.guardrails import filter_relevant_chunks
from app.config import MAX_CONTEXT_CHUNKS, MAX_UPLOAD_BYTES, MAX_UPLOAD_MB
from app.ingestion.chunker import chunk_text
from app.ingestion.document_loader import SUPPORTED_EXTENSIONS, load_document
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore


DB = None
retriever = None

FRONTEND_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
]

app = FastAPI(title="DocTalk")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        *FRONTEND_ORIGINS,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB = VectorStore()
retriever = Retriever()


@app.get("/")
async def health_check():
    return {"status": "ok", "service": "DocTalk"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global DB, retriever

    # Vercel serverless functions only allow writes to /tmp.
    # Use /tmp when available (serverless), otherwise fall back to local data/.
    upload_dir = "/tmp" if os.path.isdir("/tmp") else "data"
    os.makedirs(upload_dir, exist_ok=True)
    filename = os.path.basename(file.filename or "")
    extension = _detect_upload_extension(filename, file.content_type)
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        detail = (
            f"Please upload a supported file: {supported}. "
            f"Received filename '{filename or 'unknown'}' with content type '{file.content_type or 'unknown'}'."
        )
        raise HTTPException(status_code=400, detail=detail)

    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)
    file_path = os.path.join(upload_dir, f"{uuid.uuid4().hex}_{safe_name}")

    total_size = 0
    try:
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File is too large. Maximum upload size is {MAX_UPLOAD_MB} MB.",
                    )
                buffer.write(chunk)

        docs = load_document(file_path, original_filename=filename, source_extension=extension)
        chunks = chunk_text(docs)
        if not chunks:
            detail = (
                f"No readable text could be extracted from this file. "
                f"Received {len(docs)} document sections from {filename or 'the upload'}."
            )
            raise HTTPException(status_code=400, detail=detail)

        DB = VectorStore(reset=True)
        DB.add_documents(chunks)
        retriever = Retriever()
    except HTTPException:
        raise
    except MemoryError:
        raise HTTPException(
            status_code=507,
            detail="The file needs more memory than this backend instance has. Try a smaller file or a larger Render plan.",
        )
    except Exception as exc:
        print("UPLOAD ERROR:", repr(exc))
        raise HTTPException(status_code=500, detail="File upload failed during processing.")
    finally:
        try:
            await file.close()
        except Exception:
            pass

    return {
        "message": "File uploaded and processed successfully",
        "pages": len(docs),
        "chunks": len(chunks),
    }


def _detect_upload_extension(filename, content_type):
    extension = os.path.splitext(filename or "")[1].lower()
    if extension:
        return extension

    content_type = (content_type or "").lower()
    if "csv" in content_type:
        return ".csv"
    if "plain" in content_type or "text" in content_type:
        return ".txt"
    if "pdf" in content_type:
        return ".pdf"
    return extension


@app.post("/chat")
async def chat(query: str):
    try:
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        if retriever is None:
            raise HTTPException(status_code=400, detail="Please upload a document first.")

        results = retriever.retrieve(query)
        filtered = filter_relevant_chunks(results, query=query, max_chunks=MAX_CONTEXT_CHUNKS + 6)
        filtered = _dominant_source_chunks(filtered)

        if not filtered:
            return {"answer": REFUSAL, "citations": []}

        context = "\n\n".join([
            f"Page {chunk['page']}:\n{chunk.get('window_text') or chunk['text']}"
            for chunk in filtered
        ])
        if not context.strip():
            return {"answer": REFUSAL, "citations": []}

        answer = generate_answer(context, query)
        if not answer or len(answer.split()) < 3 or answer == REFUSAL:
            return {"answer": REFUSAL, "citations": []}

        return {
            "answer": answer,
            "citations": _select_citations(answer, query, filtered),
        }

    except HTTPException:
        raise
    except Exception as exc:
        print("ERROR:", str(exc))
        return {"answer": "An error occurred", "citations": []}


@app.get("/debug/retrieve")
async def debug_retrieve(query: str):
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if retriever is None:
        raise HTTPException(status_code=400, detail="Please upload a document first.")

    results = retriever.retrieve(query)
    filtered = filter_relevant_chunks(results, query=query, max_chunks=MAX_CONTEXT_CHUNKS + 6)
    filtered = _dominant_source_chunks(filtered)
    return {
        "query": query,
        "chunks": [
            {
                "page": chunk.get("page"),
                "chunk_index": chunk.get("chunk_index"),
                "score": chunk.get("score"),
                "guardrail_score": chunk.get("guardrail_score"),
                "text": (chunk.get("text") or "")[:500],
                "window_text": (chunk.get("window_text") or "")[:800],
            }
            for chunk in filtered
        ],
    }


def _dominant_source_chunks(chunks):
    if not chunks:
        return chunks

    source_counts = {}
    for chunk in chunks:
        source = chunk.get("source_name") or ""
        if not source:
            continue
        source_counts[source] = source_counts.get(source, 0) + 1

    if len(source_counts) <= 1:
        return chunks

    dominant = max(source_counts, key=source_counts.get)
    return [chunk for chunk in chunks if (chunk.get("source_name") or "") == dominant]


STOPWORDS = {
    "a", "an", "and", "are", "as", "by", "for", "from", "in", "is", "it",
    "of", "on", "or", "that", "the", "this", "to", "was", "were", "with",
    "what", "which", "who", "why", "how", "does", "did", "do", "page",
}

GENERIC_CITATION_TERMS = {
    "document", "paper", "papers", "research", "scientific", "science",
    "section", "sections", "writing", "written", "should", "main",
    "question", "answer",
}


def _select_citations(answer, query, chunks):
    answer_terms = set(_content_terms(answer))
    distinctive_answer_terms = answer_terms - GENERIC_CITATION_TERMS
    query_terms = set(_content_terms(query))
    page_scores = {}
    page_order = {}

    has_distinctive_overlap = any(
        distinctive_answer_terms.intersection(set(_content_terms(chunk.get("text", ""))))
        for chunk in chunks
    )

    for order, chunk in enumerate(chunks):
        page = int(chunk.get("page", 0) or 0)
        if page <= 0:
            continue

        evidence_text = f"{chunk.get('text', '')}\n{chunk.get('window_text', '')}"
        chunk_terms = set(_content_terms(evidence_text))
        answer_overlap_terms = (
            distinctive_answer_terms.intersection(chunk_terms)
            if has_distinctive_overlap
            else answer_terms.intersection(chunk_terms)
        )
        answer_overlap = len(answer_overlap_terms)
        query_overlap = len(query_terms.intersection(chunk_terms))
        if has_distinctive_overlap and answer_overlap <= 0:
            continue

        score = answer_overlap * 2 + query_overlap + float(chunk.get("guardrail_score", 0) or 0)

        if score <= 0:
            continue
        page_scores[page] = max(page_scores.get(page, 0), score)
        page_order.setdefault(page, order)

    if not page_scores and chunks:
        first_page = int(chunks[0].get("page", 0) or 0)
        return [f"Page {first_page}"] if first_page > 0 else []

    ordered_pages = sorted(
        page_scores,
        key=lambda page: (-page_scores[page], page_order.get(page, 9999), page),
    )
    max_pages = 3 if answer.count("- ") >= 3 else 2
    return [f"Page {page}" for page in sorted(ordered_pages[:max_pages])]


def _anchored_citation_pages(query, chunks):
    """Find pages whose text shares the most content terms with the query.

    Returns at most 2 candidate pages with high query-term overlap, or an
    empty list when no chunk clears the threshold (citation falls back to the
    general scoring path in _select_citations).
    """
    query_terms = set(_content_terms(query))
    if not query_terms or not chunks:
        return []

    page_hits = {}
    for chunk in chunks:
        text = f"{chunk.get('text', '')}\n{chunk.get('window_text', '')}".lower()
        chunk_terms = set(_content_terms(text))
        overlap = len(query_terms & chunk_terms)
        page = int(chunk.get("page", 0) or 0)
        if page > 0 and overlap > 0:
            page_hits[page] = max(page_hits.get(page, 0), overlap)

    if not page_hits:
        return []

    threshold = max(page_hits.values()) * 0.6
    strong = [p for p, hits in page_hits.items() if hits >= threshold]
    return sorted(strong)[:2]


def _content_terms(text):
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", (text or "").lower())
        if token not in STOPWORDS
    ]
