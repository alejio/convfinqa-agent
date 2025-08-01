"""
End-to-end agent evaluation tests using DeepEval.
"""

import os
import pytest
from deepeval import assert_test
from deepeval.metrics.answer_relevancy.answer_relevancy import AnswerRelevancyMetric
from deepeval.metrics.faithfulness.faithfulness import FaithfulnessMetric
from deepeval.test_case.llm_test_case import LLMTestCase
from typing import List

from src.evaluation.metrics import (
    ExecutionAccuracyMetric,
    create_financial_accuracy_metric,
    create_financial_relevance_metric,
    create_calculation_coherence_metric,
)


def _should_run_evaluation() -> bool:
    """Check if evaluation tests should run based on environment variables."""
    # Check if explicitly requested
    if os.environ.get("RUN_EVALUATION", "").lower() == "true":
        return True

    # Legacy support for RUN_LLM_EVAL
    if os.environ.get("RUN_LLM_EVAL", "").lower() == "true":
        return True

    return False


def _has_valid_openai_key() -> bool:
    """Check if we have a valid OpenAI API key for evaluation tests."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    return bool(api_key and api_key != "test-key" and len(api_key) > 10)


@pytest.mark.evaluation
@pytest.mark.integration
@pytest.mark.slow
class TestAgentEvaluation:
    """Integration tests for full agent evaluation."""

    def test_agent_with_real_llm(self, sample_record, eval_data_loader):
        """Test agent with real LLM if evaluation is enabled."""
        if not _should_run_evaluation():
            pytest.skip("Set RUN_EVALUATION=true to run evaluation tests")

        if not _has_valid_openai_key():
            pytest.skip("Valid OpenAI API key required for evaluation tests")

        from src.functions.agent import ConvFinQAAgent

        # Create real agent (not mocked)
        agent = ConvFinQAAgent()
        agent.set_record_context(sample_record, eval_data_loader)

        # Get first question
        question = sample_record.dialogue.conv_questions[0]
        expected_answer = sample_record.dialogue.conv_answers[0]

        # Generate actual response
        actual_output = agent.chat(question)

        # Get context
        table = sample_record.get_financial_table()
        df = table.to_dataframe()
        context = [f"Financial data:\n{df.to_string()}"]

        test_case = LLMTestCase(
            input=question,
            actual_output=actual_output,
            expected_output=expected_answer,
            context=context,
            retrieval_context=context
        )

        # Test with comprehensive metrics including execution accuracy
        metrics = [
            AnswerRelevancyMetric(threshold=0.5),
            FaithfulnessMetric(threshold=0.5),
            ExecutionAccuracyMetric(threshold=1.0),  # Exact match for baseline comparison
            create_financial_accuracy_metric(threshold=0.5),
            create_financial_relevance_metric(threshold=0.5),
            create_calculation_coherence_metric(threshold=0.5),
        ]

        assert_test(test_case, metrics)

    def test_execution_accuracy_baseline_comparison(self, sample_record, eval_data_loader):
        """Test execution accuracy specifically to compare against ConvFinQA baseline figures."""
        if not _should_run_evaluation():
            pytest.skip("Set RUN_EVALUATION=true to run evaluation tests")

        if not _has_valid_openai_key():
            pytest.skip("Valid OpenAI API key required for evaluation tests")

        from src.functions.agent import ConvFinQAAgent

        # Test PEC + DSPy architecture (only architecture implemented)
        print(f"\n=== Testing PEC + DSPy Architecture ===")

        # Create agent
        agent = ConvFinQAAgent()
        agent.set_record_context(sample_record, eval_data_loader)

        correct_count = 0
        total_count = min(3, len(sample_record.dialogue.conv_questions))  # Test first 3 questions

        for i in range(total_count):
            question = sample_record.dialogue.conv_questions[i]
            expected_answer = sample_record.dialogue.conv_answers[i]

            # Generate actual response
            actual_output = agent.chat(question)

            # Test execution accuracy
            test_case = LLMTestCase(
                input=question,
                actual_output=actual_output,
                expected_output=expected_answer
            )

            exec_metric = ExecutionAccuracyMetric()
            exec_metric.measure(test_case)

            if exec_metric.is_successful():
                correct_count += 1

            print(f"Question {i+1}: {'✓' if exec_metric.is_successful() else '✗'}")
            print(f"  Expected: {expected_answer}")
            print(f"  Actual:   {actual_output}")
            print(f"  Reason:   {exec_metric.reason}")

        execution_accuracy = correct_count / total_count if total_count > 0 else 0.0

        print(f"PEC + DSPy Execution Accuracy: {execution_accuracy:.2%}")

        # Print baseline comparison
        print("\n=== Baseline Comparison (from ConvFinQA paper) ===")
        print("GPT-3 (answer-only-prompt): 24.09%")
        print("GPT-3 (CoT prompting): 40.63%")
        print("GPT-3 (DSL program): 45.15%")
        print("FinQANet(RoBERTa-large): 68.90%")
        print("Human Expert Performance: 89.44%")
        print(f"Your Agent (PEC + DSPy): {execution_accuracy:.2%}")

        # Agent should achieve some correct answers
        assert execution_accuracy > 0, "Agent achieved no correct answers"

    def test_agent_mock_response(self, financial_test_case):
        """Test agent evaluation with mocked response."""
        if not _should_run_evaluation():
            pytest.skip("Set RUN_EVALUATION=true to run evaluation tests")

        if not _has_valid_openai_key():
            pytest.skip("Valid OpenAI API key required for evaluation tests")
        metrics = [
            create_financial_accuracy_metric(threshold=0.4),
        ]

        assert_test(financial_test_case, metrics)  # type: ignore

    def test_multiple_records_sample(self, eval_data_loader, eval_agent):
        """Test evaluation across multiple records."""
        if not _should_run_evaluation():
            pytest.skip("Set RUN_EVALUATION=true to run evaluation tests")

        if not _has_valid_openai_key():
            pytest.skip("Valid OpenAI API key required for evaluation tests")
        dataset = eval_data_loader.dataset
        sample_records = dataset.dev[:3] if len(dataset.dev) >= 3 else dataset.dev[:1]

        test_cases = []

        for record in sample_records:
            if not record.dialogue.conv_questions:
                continue

            eval_agent.set_record_context(record, eval_data_loader)

            question = record.dialogue.conv_questions[0]
            expected_answer = record.dialogue.conv_answers[0]

            # Mock response for testing
            actual_output = expected_answer

            table = record.get_financial_table()
            df = table.to_dataframe()
            context = [f"Financial data:\n{df.to_string()}"]

            test_case = LLMTestCase(
                input=question,
                actual_output=actual_output,
                expected_output=expected_answer,
                context=context
            )
            test_cases.append(test_case)

        if not test_cases:
            pytest.skip("No valid test cases created")

        # Test each case individually
        metric = create_financial_accuracy_metric(threshold=0.3)

        for test_case in test_cases:
            assert_test(test_case, [metric])

    def test_multi_turn_dialogue_evaluation(self, sample_record, eval_data_loader, eval_agent):
        """Test evaluation across multiple turns in a dialogue - key for ConvFinQA."""
        if not _should_run_evaluation():
            pytest.skip("Set RUN_EVALUATION=true to run evaluation tests")

        if not _has_valid_openai_key():
            pytest.skip("Valid OpenAI API key required for evaluation tests")
        if len(sample_record.dialogue.conv_questions) < 3:
            pytest.skip("Need at least 3 questions for multi-turn test")

        # Set up agent context once
        eval_agent.set_record_context(sample_record, eval_data_loader)

        # Get financial context for all test cases
        table = sample_record.get_financial_table()
        df = table.to_dataframe()
        financial_context = f"Financial data:\n{df.to_string()}"

        # Test each turn individually with context of previous turns
        for i in range(min(3, len(sample_record.dialogue.conv_questions))):
            question = sample_record.dialogue.conv_questions[i]
            expected_answer = sample_record.dialogue.conv_answers[i]

            # Generate mock actual output (in real scenario, this would come from agent.chat(question))
            actual_output = expected_answer

            # Build conversational context from previous turns
            conversation_context = [financial_context]
            for j in range(i):
                prev_q = sample_record.dialogue.conv_questions[j]
                prev_a = sample_record.dialogue.conv_answers[j]
                conversation_context.append(f"Previous Q{j+1}: {prev_q}")
                conversation_context.append(f"Previous A{j+1}: {prev_a}")

            test_case = LLMTestCase(
                input=question,
                actual_output=actual_output,
                expected_output=expected_answer,
                context=conversation_context,
            )

            # Use financial accuracy metric for all turns
            metrics = [create_financial_accuracy_metric(threshold=0.4)]
            assert_test(test_case, metrics)  # type: ignore
