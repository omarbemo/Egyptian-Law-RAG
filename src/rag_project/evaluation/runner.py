from dataclasses import dataclass, asdict
from typing import Any

from rag_project.Hybird_Search import ask


@dataclass
class EvalSample:
    """
    Raw output from one RAG evaluation example.
    """

    id: str
    language: str
    question: str
    reference: str
    expected_article: int | None

    answer: str
    retrieved_contexts: list[str]
    retrieved_ids: list[Any]

    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def run_one(
    item: dict,
    language: str,
) -> EvalSample:
    """
    Run the RAG pipeline for one golden-dataset item.
    """

    question = item[f"q_{language}"]
    reference = item[f"a_{language}"]

    try:
        answer, contexts, ids = ask(
            question,
            lang=language,
        )

        return EvalSample(
            id=item["id"],
            language=language,
            question=question,
            reference=reference,
            expected_article=item.get("article"),
            answer=answer,
            retrieved_contexts=list(contexts or []),
            retrieved_ids=list(ids or []),
        )

    except Exception as exc:

        return EvalSample(
            id=item["id"],
            language=language,
            question=question,
            reference=reference,
            expected_article=item.get("article"),
            answer="",
            retrieved_contexts=[],
            retrieved_ids=[],
            error=f"{type(exc).__name__}: {exc}",
        )


def run_evaluation(
    golden: list[dict],
    language: str,
) -> list[EvalSample]:
    """
    Run the RAG pipeline over the complete golden dataset.

    The RAG is called exactly once per question.
    """

    if language not in {"en", "ar"}:
        raise ValueError(
            "language must be 'en' or 'ar'"
        )

    results = []

    total = len(golden)

    print(
        f"\nRunning RAG evaluation: "
        f"{language.upper()} | {total} questions\n"
    )

    for index, item in enumerate(golden, start=1):

        print(
            f"[{index:>3}/{total}] "
            f"{item['id']}"
        )

        sample = run_one(
            item=item,
            language=language,
        )

        results.append(sample)

        if sample.error:
            print(
                f"    ERROR: {sample.error}"
            )
        else:
            print(
                f"    retrieved: "
                f"{len(sample.retrieved_contexts)} chunks"
            )

    successful = sum(
        1 for r in results
        if r.error is None
    )

    failed = total - successful

    print("\n" + "-" * 50)
    print(f"Completed: {successful}/{total}")

    if failed:
        print(f"Failed:    {failed}")

    print("-" * 50)

    return results