"""
Financial relevance evaluation tests using DeepEval.
"""

import os
import pytest
from deepeval import assert_test

from src.evaluation.metrics import (
    create_financial_relevance_metric,
    create_calculation_coherence_metric,
)


def _should_run_evaluation() -> bool:
    """Check if evaluation tests should run based on environment variables."""
    return os.environ.get("RUN_EVALUATION", "").lower() == "true"


def _has_valid_openai_key() -> bool:
    """Check if we have a valid OpenAI API key for evaluation tests."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    return api_key and api_key != "test-key" and len(api_key) > 10


@pytest.mark.evaluation
@pytest.mark.slow
class TestFinancialRelevance:
    """Test suite for financial relevance evaluation."""

    def test_financial_relevance_basic(self, financial_test_case):
        """Test basic financial relevance metric."""
        if not _should_run_evaluation():
            pytest.skip("Set RUN_EVALUATION=true to run evaluation tests")

        if not _has_valid_openai_key():
            pytest.skip("Valid OpenAI API key required for evaluation tests")
        metric = create_financial_relevance_metric(threshold=0.6)

        assert_test(financial_test_case, [metric])

    def test_calculation_coherence(self, financial_test_case):
        """Test calculation coherence metric."""
        if not _should_run_evaluation():
            pytest.skip("Set RUN_EVALUATION=true to run evaluation tests")

        if not _has_valid_openai_key():
            pytest.skip("Valid OpenAI API key required for evaluation tests")
        metric = create_calculation_coherence_metric(threshold=0.7)

        assert_test(financial_test_case, [metric])

    def test_combined_metrics(self, financial_test_case):
        """Test multiple metrics together."""
        if not _should_run_evaluation():
            pytest.skip("Set RUN_EVALUATION=true to run evaluation tests")

        if not _has_valid_openai_key():
            pytest.skip("Valid OpenAI API key required for evaluation tests")
        metrics = [
            create_financial_relevance_metric(threshold=0.6),
            create_calculation_coherence_metric(threshold=0.6),
        ]

        assert_test(financial_test_case, metrics)
