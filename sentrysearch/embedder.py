"""Gemini embedding client."""

import os

import google.generativeai as genai


def _get_client():
    """Configure and return the Gemini client."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    genai.configure(api_key=api_key)
    return genai


def embed_text(text: str) -> list[float]:
    """Embed a single text string using Gemini."""
    client = _get_client()
    result = client.embed_content(
        model="models/text-embedding-004",
        content=text,
    )
    return result["embedding"]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Generate embeddings for video chunks.

    Each chunk dict is augmented with a 'description' field (placeholder for
    future frame-analysis) and an 'embedding' field.

    Args:
        chunks: List of chunk dicts from chunker.chunk_video.

    Returns:
        List of chunk dicts with added 'embedding' key.
    """
    results = []
    for chunk in chunks:
        # Build a text description from metadata.
        # In a full implementation this would use Gemini vision to describe frames.
        description = (
            f"Video clip from {os.path.basename(chunk['source_file'])} "
            f"starting at {chunk['start_time']:.1f}s to {chunk['end_time']:.1f}s"
        )
        embedding = embed_text(description)
        results.append({**chunk, "description": description, "embedding": embedding})
    return results
