"""Standalone CLI for embedding extracted PDF pages in external ChromaDB."""

import argparse
import os
import re

import chromadb
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings


PAGE_PATTERN = re.compile(r"--- Page (\d+) ---")


def chunk_by_page(text: str) -> list[dict]:
    """Split extracted PDF text into page-numbered chunks."""
    blocks = re.split(PAGE_PATTERN, text)
    return [
        {"page": int(blocks[index]), "text": blocks[index + 1].strip()}
        for index in range(1, len(blocks), 2)
        if blocks[index + 1].strip()
    ]


def embed_and_store(chunks, collection, model) -> None:
    """Embed page chunks and send them to a ChromaDB collection."""
    texts = [chunk["text"] for chunk in chunks]
    if not texts:
        print("No text content found to embed.")
        return

    embeddings = model.embed_documents(texts)
    collection.upsert(
        ids=[str(chunk["page"]) for chunk in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"page": chunk["page"]} for chunk in chunks],
    )
    print(f"Stored {len(texts)} page chunks in external ChromaDB.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index extracted PDF text in external ChromaDB")
    parser.add_argument("input_txt", help="Path to extracted text")
    parser.add_argument("--collection", default="pdf_pages")
    args = parser.parse_args()

    with open(args.input_txt, encoding="utf-8") as text_file:
        chunks = chunk_by_page(text_file.read())

    client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "localhost"),
        port=int(os.getenv("CHROMA_PORT", "8001")),
        ssl=os.getenv("CHROMA_SSL", "false").lower() in {"1", "true", "yes", "on"},
    )
    collection = client.get_or_create_collection(name=args.collection)
    embed_and_store(
        chunks,
        collection,
        NVIDIAEmbeddings(model=os.getenv("NVIDIA_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")),
    )


if __name__ == "__main__":
    main()
