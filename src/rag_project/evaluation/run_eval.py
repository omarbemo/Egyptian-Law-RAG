from pathlib import Path

from rag_project.evaluation.config import (
    GOLDEN_DATASET,
    RESULTS_DIR,
    LANGUAGES,
    MAX_WORKERS,
)

from rag_project.evaluation.dataset import load_golden

from rag_project.evaluation.runner import run_evaluation

from rag_project.evaluation.metrics import (
    evaluate_samples,
    calculate_retrieval_stats,
)

from rag_project.evaluation.report import create_excel_report


# ============================================================
# Display helpers
# ============================================================

def print_header():
    print()
    print("=" * 60)
    print("              RAG EVALUATION")
    print("=" * 60)


def print_language_summary(
    language: str,
    samples,
    ragas_result,
    retrieval_stats,
):
    """
    Print a concise summary to the terminal.
    """

    ragas_df = ragas_result.to_pandas()

    print()
    print("-" * 60)
    print(f"{language.upper()}")
    print("-" * 60)

    total = len(samples)

    successful = sum(
        sample.error is None
        for sample in samples
    )

    failed = total - successful

    print(f"Questions:          {total}")
    print(f"RAG successful:     {successful}")
    print(f"RAG failed:         {failed}")

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    hit_rate = retrieval_stats.get(
        "article_hit_rate"
    )

    if hit_rate is not None:

        print(
            f"Article Hit@K:      "
            f"{hit_rate:.2%}"
        )

    # --------------------------------------------------------
    # RAGAS
    # --------------------------------------------------------

    metric_columns = [
        (
            "Faithfulness",
            "faithfulness",
        ),
        (
            "Context Recall",
            "llm_context_recall",
        ),
        (
            "Context Precision",
            "llm_context_precision_with_reference",
        ),
        (
            "Factual Correctness",
            "factual_correctness",
        ),
    ]

    for label, column in metric_columns:

        if column not in ragas_df.columns:
            continue

        values = ragas_df[column].dropna()

        if values.empty:
            continue

        print(
            f"{label:<20}"
            f"{values.mean():.2f}"
        )


# ============================================================
# Run one language
# ============================================================

def evaluate_language(
    golden,
    language: str,
):
    """
    Run the complete evaluation for one language.
    """

    print()
    print(f"Starting {language.upper()} evaluation...")

    # --------------------------------------------------------
    # 1. Run the RAG
    # --------------------------------------------------------

    samples = run_evaluation(
        golden=golden,
        language=language,
    )

    # --------------------------------------------------------
    # 2. Retrieval metrics
    # --------------------------------------------------------

    retrieval_stats = calculate_retrieval_stats(
        samples
    )

    # --------------------------------------------------------
    # 3. RAGAS
    # --------------------------------------------------------

    print(
        f"\nRunning RAGAS for "
        f"{language.upper()}..."
    )

    ragas_result, valid_samples = evaluate_samples(
        samples=samples,
        max_workers=MAX_WORKERS,
    )

    # --------------------------------------------------------
    # 4. Report
    # --------------------------------------------------------

    output_dir = (
        RESULTS_DIR
        / (
            "english"
            if language == "en"
            else "arabic"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "detailed.xlsx"
    )

    create_excel_report(
        samples=samples,
        ragas_result=ragas_result,
        output_path=output_path,
    )

    # --------------------------------------------------------
    # 5. Terminal summary
    # --------------------------------------------------------

    print_language_summary(
        language=language,
        samples=samples,
        ragas_result=ragas_result,
        retrieval_stats=retrieval_stats,
    )

    return {
        "language": language,
        "samples": samples,
        "ragas_result": ragas_result,
        "retrieval_stats": retrieval_stats,
        "report": output_path,
    }


# ============================================================
# Main
# ============================================================

def main():

    print_header()

    # --------------------------------------------------------
    # Load golden dataset
    # --------------------------------------------------------

    print(
        f"\nLoading golden dataset:\n"
        f"{GOLDEN_DATASET}"
    )

    golden = load_golden(
        GOLDEN_DATASET
    )

    print(
        f"Loaded {len(golden)} "
        f"evaluation questions."
    )

    # --------------------------------------------------------
    # Run languages
    # --------------------------------------------------------

    results = []

    for language in LANGUAGES:

        result = evaluate_language(
            golden=golden,
            language=language,
        )

        results.append(result)

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("              EVALUATION COMPLETE")
    print("=" * 60)

    print("\nReports:")

    for result in results:

        print(
            f"  {result['language'].upper():<5} "
            f"{result['report']}"
        )

    print()


if __name__ == "__main__":
    main()