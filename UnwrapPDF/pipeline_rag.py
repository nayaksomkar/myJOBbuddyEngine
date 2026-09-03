"""Standalone CLI for embedding page-level extracted PDF text."""

import re       # regex – for splitting text on page markers
import argparse # for parsing CLI arguments

from sentence_transformers import SentenceTransformer  # generates embeddings
import chromadb                                          # vector database
from chromadb.config import Settings                     # disables telemetry


# regex that matches markers like "--- Page 3 ---"
PAGE_PATTERN = re.compile(r"--- Page (\d+) ---")


def chunk_by_page(text: str):
    """
    Splits extracted text on '--- Page N ---' markers.
    Returns a list of dicts: {'page': int, 'text': str}
    """
    # split keeps the matched numbers in the result list
    blocks = re.split(PAGE_PATTERN, text)
    chunks = []
    # blocks layout: [before, page_num, content, page_num, content, ...]
    for i in range(1, len(blocks), 2):
        page_num = blocks[i]                # the page number as a string
        content = blocks[i + 1].strip()     # the page content
        if content:                         # skip empty pages
            chunks.append({"page": int(page_num), "text": content})
    return chunks


def embed_and_store(chunks, collection, model):
    """
    Embeds each chunk using sentence-transformers,
    then upserts into a ChromaDB collection with page metadata.
    """
    texts = [c["text"] for c in chunks]  # collect all page texts
    if not texts:
        print("No text content found to embed.")
        return

    # convert texts into vector embeddings (one per page)
    embeddings = model.encode(texts, show_progress_bar=True)
    # use page number as a unique ID string
    ids = [str(c["page"]) for c in chunks]
    # store which page each embedding came from
    metadatas = [{"page": c["page"]} for c in chunks]

    # insert or update in ChromaDB
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    print(f"Stored {len(ids)} page chunks in ChromaDB.")


def main():
    """
    CLI entry point.
    Reads a .txt file from pdf_extract.py, chunks by page,
    embeds with sentence-transformers, and stores in ChromaDB.
    """
    parser = argparse.ArgumentParser(
        description="Index extracted PDF text into ChromaDB"
    )
    # positional: the .txt file produced by pdf_extract.py
    parser.add_argument(
        "input_txt",
        help="Path to extracted .txt file from pdf_extract.py"
    )
    # optional: name of the collection inside ChromaDB
    parser.add_argument(
        "--collection",
        default="pdf_pages",
        help="ChromaDB collection name"
    )
    # optional: where to persist the ChromaDB data on disk
    parser.add_argument(
        "--db",
        default="./chroma_db",
        help="ChromaDB persistent directory"
    )

    args = parser.parse_args()

    # read the entire extracted text file into memory
    with open(args.input_txt, encoding="utf-8") as f:
        text = f.read()

    # split into page-level chunks
    chunks = chunk_by_page(text)
    print(f"Found {len(chunks)} page chunks.")

    # load a small, fast embedding model (384-dim vectors)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # connect to a persistent ChromaDB (saves to disk, not in-memory)
    client = chromadb.PersistentClient(
        path=args.db,
        settings=Settings(anonymized_telemetry=False)
    )
    # create the collection if it doesn't exist yet
    collection = client.get_or_create_collection(name=args.collection)

    # embed everything and write to ChromaDB
    embed_and_store(chunks, collection, model)


if __name__ == "__main__":
    main()
