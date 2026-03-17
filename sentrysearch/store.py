"""ChromaDB vector store."""

import hashlib

import chromadb


_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=".sentrysearch_db")
    return _client


def get_collection(name: str = "clips") -> chromadb.Collection:
    """Get or create the ChromaDB collection."""
    client = _get_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def store_embeddings(
    collection: chromadb.Collection,
    chunks: list[dict],
) -> None:
    """Store chunk embeddings in ChromaDB.

    Args:
        collection: ChromaDB collection.
        chunks: List of chunk dicts with 'embedding', 'description', and metadata.
    """
    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for chunk in chunks:
        # Deterministic ID from source file + start time
        raw_id = f"{chunk['source_file']}:{chunk['start_time']}"
        doc_id = hashlib.sha256(raw_id.encode()).hexdigest()[:16]

        ids.append(doc_id)
        embeddings.append(chunk["embedding"])
        documents.append(chunk["description"])
        metadatas.append(
            {
                "source_file": chunk["source_file"],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "chunk_path": chunk["chunk_path"],
            }
        )

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def query_collection(
    collection: chromadb.Collection,
    query_embedding: list[float],
    n_results: int = 5,
) -> list[dict]:
    """Query the collection and return ranked results.

    Returns:
        List of dicts with keys: source_file, start_time, end_time, chunk_path, score.
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    hits = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        hits.append(
            {
                "source_file": meta["source_file"],
                "start_time": meta["start_time"],
                "end_time": meta["end_time"],
                "chunk_path": meta["chunk_path"],
                "score": 1.0 - distance,  # cosine similarity
            }
        )
    return hits
