"""Draft the follow-up questions offered after an answer.

Runs the local chat model once per passage and asks it for questions that
passage answers. The output is *not* indexed directly: it is written to
`data/suggested-questions.yaml` for a human to read, edit and commit, and
`app.ingest` picks up the committed file. See `app/suggestions.py` for why.

Deterministic by construction — temperature 0 and a fixed seed, the same
settings the benchmark uses — so re-running it on an unchanged knowledge base
produces an unchanged file and the review diff stays empty.

Run with:
    uv run python -m app.gen_suggestions
    uv run python -m app.gen_suggestions --only 01-izin-politikasi.md
"""

import argparse
import asyncio

from app.chunking import load_knowledge_base
from app.config import get_settings
from app.llm import OllamaClient, OllamaError
from app.rag import load_prompt
from app.suggestions import PassageKey, clean_question, dump, load

# Two short questions need far fewer tokens than an answer. Capped explicitly so
# a model that starts explaining itself cannot run on.
MAX_TOKENS = 160


async def generate(only: str | None) -> int:
    """Write suggestions for every passage. Returns the question count."""
    settings = get_settings()
    kb_dir = settings.resolve_kb_dir()
    chunks = load_knowledge_base(kb_dir, settings.chunk_size, settings.chunk_overlap)
    if only:
        chunks = [chunk for chunk in chunks if chunk.source_file == only]
    if not chunks:
        raise SystemExit(f"No passages to process in {kb_dir}.")

    client = OllamaClient(settings)
    model = settings.chat_model_primary
    template = load_prompt("suggest_tr.txt")
    output = settings.resolve_suggestions_file()

    # `--only` regenerates one document, so it has to start from what is already
    # committed. Writing just the generated passages replaced the whole file:
    # measured, regenerating one of four documents took the file from 37
    # reviewed passages to 9 and deleted the other three documents' questions —
    # which are hand-edited work, the entire reason this detours through a file.
    #
    # Entries for the document being regenerated are dropped rather than merged,
    # so a renamed heading leaves no orphan behind pointing at a section that no
    # longer exists.
    questions: dict[PassageKey, list[str]] = {}
    if only:
        questions = {
            key: value for key, value in load(output).items() if key.file != only
        }
        print(f"Keeping   : {len(questions)} passages from other documents")
    total = 0

    print(f"Model     : {model}")
    print(f"Passages  : {len(chunks)}\n")

    try:
        for index, chunk in enumerate(chunks, start=1):
            prompt = template.format(
                count=settings.suggestions_per_chunk,
                title=chunk.doc_title,
                section=chunk.section,
                text=chunk.text,
            )
            pieces: list[str] = []
            async for piece in client.chat_stream(
                model,
                [{"role": "user", "content": prompt}],
                think=False,
                max_tokens=MAX_TOKENS,
            ):
                if piece.content:
                    pieces.append(piece.content)

            drafted = [
                question
                for line in "".join(pieces).splitlines()
                if (question := clean_question(line))
            ][: settings.suggestions_per_chunk]

            questions[PassageKey(chunk.source_file, chunk.section)] = drafted
            total += len(drafted)
            print(f"  [{index}/{len(chunks)}] {chunk.section} → {len(drafted)}")
            for question in drafted:
                print(f"        {question}")
    except OllamaError as exc:
        raise SystemExit(f"Generation failed: {exc}") from exc
    finally:
        await client.aclose()

    output.parent.mkdir(parents=True, exist_ok=True)
    dump(output, questions)

    print(f"\nWrote {total} questions to {output}")
    print("Review and edit that file, then run: uv run python -m app.ingest")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draft follow-up questions for review."
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Limit to one knowledge-base file, e.g. 01-izin-politikasi.md",
    )
    args = parser.parse_args()
    asyncio.run(generate(args.only))


if __name__ == "__main__":
    main()
