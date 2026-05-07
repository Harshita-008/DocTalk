import chromadb
from app.config import CHROMADB_API_KEY

from app.config import EMBEDDING_BATCH_SIZE
from app.ingestion.embedder import get_embeddings


COLLECTION_NAME = "pdf_docs"
DB_PATH = "./data/db"


class VectorStore:
    def __init__(self, reset=False):
        self.client = chromadb.CloudClient(
            api_key=CHROMADB_API_KEY,
            tenant='b1a76c24-662d-4e22-b9e5-59bce8ad652b',
            database='DocTalk'
        )

        if reset:
            self._clear_collection()

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _clear_collection(self):
        try:
            collection = self.client.get_collection(COLLECTION_NAME)
            existing = collection.get(include=[])
            ids = existing.get("ids", [])
            batch_size = 500
            for start in range(0, len(ids), batch_size):
                collection.delete(ids=ids[start:start + batch_size])
        except Exception:
            pass

        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    def add_documents(self, chunks):
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        metadatas = [
            {
                "page": int(c["page"]),
                "chunk_index": int(c.get("chunk_index", i)),
                "section_title": str(c.get("section_title") or ""),
                "window_text": str(c.get("window_text") or c.get("text") or ""),
                "source_name": str(c.get("source_name") or ""),
                "source_type": str(c.get("source_type") or "document"),
            }
            for i, c in enumerate(chunks)
        ]
        ids = [
            f"page-{c['page']}-chunk-{c.get('chunk_index', i)}"
            for i, c in enumerate(chunks)
        ]

        batch_size = max(1, EMBEDDING_BATCH_SIZE)
        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            batch_texts = texts[start:end]
            embeddings = get_embeddings(batch_texts)

            self.collection.upsert(
                documents=batch_texts,
                metadatas=metadatas[start:end],
                ids=ids[start:end],
                embeddings=embeddings
            )

    def query(self, query_text, top_k=5):
        query_embedding = get_embeddings([query_text])[0]
        count = self.collection.count()

        if count == 0:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"]
        )

        return results

    def get_all(self):
        return self.collection.get(include=["documents", "metadatas"])

    def count(self):
        return self.collection.count()
