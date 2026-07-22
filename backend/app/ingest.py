"""Index the knowledge base: chunk, embed, and persist to the vector store.

Run with:  uv run python -m app.ingest
"""

import argparse
import asyncio
import time

from app import suggestions
from app.chunking import load_knowledge_base
from app.config import get_settings
from app.llm import OllamaClient, OllamaError
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

    # Follow-up questions come from the reviewed file, never from a model at
    # ingest time, so building the index twice yields the same chips twice.
    reviewed = suggestions.load(settings.resolve_suggestions_file())
    if reviewed:
        chunks = [
            chunk.model_copy(
                update={
                    "suggested_questions": reviewed.get(
                        suggestions.PassageKey(chunk.source_file, chunk.section), []
                    )
                }
            )
            for chunk in chunks
        ]
        attached = sum(len(chunk.suggested_questions) for chunk in chunks)
        print(f"Suggestions    : {attached} questions from {len(reviewed)} passages")
    else:
        print(
            "Suggestions    : none "
            "(run `python -m app.gen_suggestions`, review, then re-ingest)"
        )

    files = sorted({chunk.source_file for chunk in chunks})
    tokens = sum(chunk.token_estimate for chunk in chunks)
    print(f"Documents      : {len(files)} ({', '.join(files)})")
    avg = tokens // len(chunks)
    print(f"Chunks         : {len(chunks)} (~{tokens} tokens, avg {avg})")

    client = OllamaClient(settings)

    try:
        # Embed everything *before* touching the collection. The rebuild used to
        # drop the index first and write batch by batch, so anything that failed
        # part-way — Ollama not running is the ordinary case — left the machine
        # with no index at all and a traceback: measured, 39 chunks became 0 and
        # the assistant refused every question until someone re-ran this.
        #
        # The whole corpus is held in memory for the length of one call. At this
        # scale that is a few hundred kilobytes; revisit if the knowledge base
        # grows by orders of magnitude.
        print(f"Embedding with : {settings.embedding_model}")
        started = time.perf_counter()

        embeddings: list[list[float]] = []
        for offset in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[offset : offset + EMBED_BATCH_SIZE]
            embeddings.extend(await client.embed([chunk.text for chunk in batch]))
            done = min(offset + EMBED_BATCH_SIZE, len(chunks))
            print(f"  embedded {done}/{len(chunks)}", end="\r", flush=True)

        elapsed = time.perf_counter() - started
        vector_size = len(embeddings[0]) if embeddings else 0

        # Opened only now, because opening it creates `chroma.sqlite3`. The
        # Docker entrypoint treats that file as "an index already exists" and
        # skips ingest, so a first run that died before writing anything left
        # the container starting up with an empty index on every subsequent
        # boot — the exact outcome the entrypoint's `set -e` exists to prevent.
        store = VectorStore(settings)
        if rebuild:
            store.reset()
        store.add(chunks, embeddings)

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
    try:
        asyncio.run(ingest(rebuild=not args.append))
    except OllamaError as exc:
        # Forgetting to start Ollama is the ordinary way this fails, and a
        # traceback buries the one line that says so. The index is untouched.
        raise SystemExit(
            f"\nerror: {exc}\nThe existing index was left as it was."
        ) from None


if __name__ == "__main__":
    main()
