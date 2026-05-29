"""Evaluation glue.

Wraps RAGAS and DeepEval into a single `evaluate()` call so every notebook
ends the same way: pipeline in, metrics dict out. Recipes that need finer
control (recipe 37 for RAGAS, recipe 38 for DeepEval-in-CI) call the
underlying libraries directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


@dataclass
class EvalSample:
    """One row in an evaluation set."""

    question: str
    expected_answer: str
    contexts: list[str]
    actual_answer: str


def evaluate(
    samples: Sequence[EvalSample],
    *,
    use_ragas: bool = True,
    use_deepeval: bool = False,
) -> dict[str, float]:
    """Run the requested metric suites and merge their results."""
    out: dict[str, float] = {}
    if use_ragas:
        out.update(_run_ragas(samples))
    if use_deepeval:
        out.update(_run_deepeval(samples))
    return out


# ---------------------------------------------------------------------------
# RAGAS
# ---------------------------------------------------------------------------
def _build_nebius_langchain_llm():
    """Build a LangChain ChatOpenAI pointed at whatever provider the cookbook uses."""
    import os
    from langchain_openai import ChatOpenAI
    from .providers import PROVIDERS

    provider = os.environ.get("PROVIDER", "nebius")
    spec = PROVIDERS[provider]
    return ChatOpenAI(
        model=spec.default_chat_model,
        api_key=os.environ.get(spec.api_key_env),
        base_url=os.environ.get(spec.base_url_env) if spec.base_url_env else None,
        temperature=0.0,
    )


def _build_nebius_langchain_embeddings():
    """Build a LangChain embeddings wrapper pointed at the cookbook provider."""
    import os
    from langchain_openai import OpenAIEmbeddings
    from .providers import PROVIDERS

    provider = os.environ.get("PROVIDER", "nebius")
    spec = PROVIDERS[provider]
    return OpenAIEmbeddings(
        model=spec.default_embed_model,
        api_key=os.environ.get(spec.api_key_env),
        base_url=os.environ.get(spec.base_url_env) if spec.base_url_env else None,
        check_embedding_ctx_length=False,
    )


def _run_ragas(samples: Sequence[EvalSample]) -> dict[str, float]:
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    judge_llm = LangchainLLMWrapper(_build_nebius_langchain_llm())
    judge_embeddings = LangchainEmbeddingsWrapper(_build_nebius_langchain_embeddings())

    ds = Dataset.from_list(
        [
            {
                "question": s.question,
                "answer": s.actual_answer,
                "contexts": s.contexts,
                "ground_truth": s.expected_answer,
            }
            for s in samples
        ]
    )
    result = ragas_evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    df = result.to_pandas()
    out: dict[str, float] = {}
    for col in df.columns:
        try:
            out[f"ragas_{col}"] = float(df[col].mean())
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# DeepEval
# ---------------------------------------------------------------------------
def _build_deepeval_model():
    """Build a DeepEval-compatible judge model pointed at the cookbook provider."""
    from deepeval.models.base_model import DeepEvalBaseLLM
    from .providers import LLMClient

    class _CookbookJudge(DeepEvalBaseLLM):
        def __init__(self):
            self._client = LLMClient()

        def load_model(self):
            return self._client

        def generate(self, prompt: str) -> str:
            return self._client.chat(prompt)

        async def a_generate(self, prompt: str) -> str:
            return self.generate(prompt)

        def get_model_name(self) -> str:
            return f"cookbook-{self._client.provider}-{self._client.chat_model}"

    return _CookbookJudge()


def _run_deepeval(samples: Sequence[EvalSample]) -> dict[str, float]:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    judge = _build_deepeval_model()
    metrics = [
        AnswerRelevancyMetric(threshold=0.7, model=judge),
        FaithfulnessMetric(threshold=0.7, model=judge),
        ContextualPrecisionMetric(threshold=0.7, model=judge),
    ]
    aggregated: dict[str, list[float]] = {m.__class__.__name__: [] for m in metrics}
    for s in samples:
        tc = LLMTestCase(
            input=s.question,
            actual_output=s.actual_answer,
            expected_output=s.expected_answer,
            retrieval_context=s.contexts,
        )
        for m in metrics:
            m.measure(tc)
            aggregated[m.__class__.__name__].append(float(m.score or 0.0))
    return {
        f"deepeval_{name.lower()}": sum(scores) / max(1, len(scores))
        for name, scores in aggregated.items()
    }


# ---------------------------------------------------------------------------
# Convenience for the in-notebook smoke checks
# ---------------------------------------------------------------------------
def run_qa_against(
    pipeline: Callable[[str], tuple[str, list[str]]],
    questions: Iterable[dict],
) -> list[EvalSample]:
    """Run a `pipeline(question) -> (answer, contexts)` over an eval set."""
    out: list[EvalSample] = []
    for row in questions:
        answer, contexts = pipeline(row["question"])
        out.append(
            EvalSample(
                question=row["question"],
                expected_answer=row.get("answer", ""),
                contexts=contexts,
                actual_answer=answer,
            )
        )
    return out


__all__ = ["EvalSample", "evaluate", "run_qa_against"]
