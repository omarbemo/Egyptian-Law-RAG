# evaluation/metrics.py
import sys
import types
import os


# ============================================================
# RAGAS 0.4.3 compatibility patch
# MUST happen before importing RAGAS
# ============================================================

def _patch_vertexai_import():

    module_name = (
        "langchain_community.chat_models.vertexai"
    )

    if module_name in sys.modules:
        return

    stub = types.ModuleType(module_name)

    class ChatVertexAI:
        pass

    stub.ChatVertexAI = ChatVertexAI

    sys.modules[module_name] = stub


_patch_vertexai_import()


# ============================================================
# Now import RAGAS
# ============================================================

from datasets import Dataset

from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper

from ragas.metrics import (
    Faithfulness,
    LLMContextRecall,
    LLMContextPrecisionWithReference,
    FactualCorrectness,
)

from langchain_groq import ChatGroq

from rag_project.evaluation.runner import EvalSample




# ============================================================
# Judge LLM
# ============================================================

def create_judge():
    """
    Create the LLM used by RAGAS to judge the RAG outputs.
    """

    api_key = os.environ.get("GROQ_API_KEY2")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set."
        )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0,
    )

    return LangchainLLMWrapper(llm)


# ============================================================
# RAGAS metrics
# ============================================================

def create_metrics():
    """
    Return the RAGAS metrics used by this project.
    """

    return [
        Faithfulness(),
        LLMContextRecall(),
        LLMContextPrecisionWithReference(),
        FactualCorrectness(),
    ]


# ============================================================
# Convert our samples → RAGAS Dataset
# ============================================================

def samples_to_dataset(
    samples: list[EvalSample],
) -> tuple[Dataset, list[EvalSample]]:

    rows = {
        "user_input": [],
        "retrieved_contexts": [],
        "response": [],
        "reference": [],
    }

    valid_samples = []

    for sample in samples:

        if sample.error:
            continue

        rows["user_input"].append(
            sample.question
        )

        rows["retrieved_contexts"].append(
            sample.retrieved_contexts
        )

        rows["response"].append(
            sample.answer
        )

        rows["reference"].append(
            sample.reference
        )

        valid_samples.append(sample)

    dataset = Dataset.from_dict(rows)

    return dataset, valid_samples


# ============================================================
# Run RAGAS
# ============================================================

def evaluate_samples(
    samples: list[EvalSample],
    max_workers: int = 2,
):
    """
    Run RAGAS over successful RAG samples.

    Returns:
        result, valid_samples
    """

    dataset, valid_samples = samples_to_dataset(
        samples
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "No successful samples available for RAGAS evaluation."
        )

    judge = create_judge()

    result = evaluate(
        dataset=dataset,
        metrics=create_metrics(),
        llm=judge,
        run_config=RunConfig(
            max_workers=max_workers
        ),
    )

    return result, valid_samples

def article_hit(
    sample: EvalSample,
) -> bool:

    if sample.expected_article is None:
        return False

    return sample.expected_article in sample.retrieved_ids

def calculate_retrieval_stats(
    samples: list[EvalSample],
) -> dict:

    valid = [
        sample
        for sample in samples
        if sample.error is None
    ]

    if not valid:
        return {
            "article_hit_rate": 0.0,
            "successful_questions": 0,
        }

    hits = sum(
        article_hit(sample)
        for sample in valid
    )

    return {
        "article_hit_rate": hits / len(valid),
        "successful_questions": len(valid),
    }