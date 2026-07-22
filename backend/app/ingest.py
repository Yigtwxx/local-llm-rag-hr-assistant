"""Index the knowledge base: chunk, embed, and persist to the vector store.

Run with:  uv run python -m app.ingest
"""

import argparse
import asyncio
import time

from app.chunking import load_knowledge_base
from app.config import get_settings
from app.llm import OllamaClient
from app.retrieval import VectorStore

EMBED_BATCH_SIZE = 16


async def ingest(*, rebuild: bool = True) -> int:
    """Chunk every document, embed it, and store it. Returns the chunk count."""
    settings = get_settings()
    kb_dir = settings.resolve_kb_dir()

    print(f"Knowledge base : {kb_dir}")
    chunks = load_knowledge_base(kb_dir, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        print("No Markdown documents found — nothing to index.")
        return 0

    files = sorted({chunk.source_file for chunk in chunks})
    tokens = sum(chunk.token_estimate for chunk in chunks)
    print(f"Documents      : {len(files)} ({', '.join(files)})")
    avg = tokens // len(chunks)
    print(f"Chunks         : {len(chunks)} (~{tokens} tokens, avg {avg})")

    client = OllamaClient(settings)
    store = VectorStore(settings)

    try:
        if rebuild:
            store.reset()

        print(f"Embedding with : {settings.embedding_model}")
        started = time.perf_counter()

        for offset in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[offset : offset + EMBED_BATCH_SIZE]
            embeddings = await client.embed([chunk.text for chunk in batch])
            store.add(batch, embeddings)
            done = min(offset + EMBED_BATCH_SIZE, len(chunks))
            print(f"  indexed {done}/{len(chunks)}", end="\r", flush=True)

        elapsed = time.perf_counter() - started
        probe = await client.embed(["boyut kontrolu"])
        vector_size = len(probe[0]) if probe else 0

        print(f"\nStored         : {store.count()} chunks in {elapsed:.1f}s")
        print(f"Vector size    : {vector_size} dimensions")
        print(f"Storage path   : {settings.resolve_storage_dir()}")
        return store.count()
    finally:
        await client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Index the HR knowledge base.")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Add to the existing collection instead of rebuilding it.",
    )
    args = parser.parse_args()
    asyncio.run(ingest(rebuild=not args.append))


if __name__ == "__main__":
    main()
