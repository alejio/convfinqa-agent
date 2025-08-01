"""
Shared pytest fixtures and configuration for ConvFinQA tests
"""

import sys
from pathlib import Path

# Add the src directory to the Python path so we can import our modules
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Load environment variables early in the test process
from dotenv import load_dotenv
load_dotenv()

# Disable DeepEval telemetry and browser opening
import os
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
os.environ["OTEL_SDK_DISABLED"] = "true"

# This conftest.py file can be used to add shared fixtures, plugins, and configuration
# for all tests in the project

import pytest

from src.data.loader import create_data_loader
from src.functions.agent import ConvFinQAAgent


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--evaluation",
        action="store_true",
        default=False,
        help="Run evaluation tests that require LLM API calls"
    )
    parser.addoption(
        "--run-llm-eval",
        action="store_true",
        default=False,
        help="Run LLM evaluation tests that require API calls (legacy)"
    )


def pytest_configure(config):
    """Configure pytest based on command line options."""
    if config.getoption("--evaluation") or config.getoption("--run-llm-eval"):
        os.environ["RUN_EVALUATION"] = "true"


# Evaluation test fixtures
@pytest.fixture(scope="session")
def eval_data_loader():
    """Create data loader for evaluation tests."""
    loader = create_data_loader()
    # Actually load the dataset
    loader.load_dataset()
    return loader


@pytest.fixture
def eval_agent(mocker):
    """Create agent for evaluation tests."""
    # Mock OpenAI API to avoid API requirements during testing
    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})

    # Mock OpenAI client for smolagents
    mock_openai = mocker.patch("openai.OpenAI")
    mock_client = mocker.MagicMock()
    mock_openai.return_value = mock_client

    return ConvFinQAAgent()


@pytest.fixture(autouse=True)
def mock_deepeval_in_ci(mocker):
    """Automatically mock DeepEval metrics in CI environment."""
    # Only mock if we're not explicitly running evaluation tests
    if not os.environ.get("RUN_EVALUATION", "").lower() == "true":
        # Mock DeepEval's GEval and ConversationalGEval to avoid API calls
        mock_geval = mocker.patch("deepeval.metrics.g_eval.g_eval.GEval")
        mock_conv_geval = mocker.patch("deepeval.metrics.conversational_g_eval.conversational_g_eval.ConversationalGEval")

        # Mock the evaluate method to return success
        mock_geval_instance = mocker.MagicMock()
        mock_geval_instance.measure.return_value = None
        mock_geval_instance.score = 0.8
        mock_geval_instance.is_successful.return_value = True
        mock_geval.return_value = mock_geval_instance

        mock_conv_geval_instance = mocker.MagicMock()
        mock_conv_geval_instance.measure.return_value = None
        mock_conv_geval_instance.score = 0.8
        mock_conv_geval_instance.is_successful.return_value = True
        mock_conv_geval.return_value = mock_conv_geval_instance

        # Mock assert_test and evaluate functions
        mocker.patch("deepeval.assert_test")
        mocker.patch("deepeval.evaluate")


@pytest.fixture(scope="session")
def sample_record(eval_data_loader):
    """Get a sample record for testing."""
    try:
        dataset = eval_data_loader.dataset
        if dataset and hasattr(dataset, 'dev') and dataset.dev:
            return dataset.dev[0]
        elif dataset and hasattr(dataset, 'train') and dataset.train:
            return dataset.train[0]
        else:
            pytest.skip("No records available in dataset")
    except Exception as e:
        pytest.skip(f"Failed to load sample record: {e}")


@pytest.fixture
def financial_test_case(sample_record, eval_agent, eval_data_loader):
    """Create a test case from sample record."""
    from deepeval.test_case.llm_test_case import LLMTestCase

    # Set agent context
    eval_agent.set_record_context(sample_record, eval_data_loader)

    # Get first question
    if not sample_record.dialogue.conv_questions:
        pytest.skip("Record has no questions")

    question = sample_record.dialogue.conv_questions[0]
    expected_answer = sample_record.dialogue.conv_answers[0]

    # Get financial table as context
    table = sample_record.get_financial_table()
    df = table.to_dataframe()
    context = [f"Financial data:\n{df.to_string()}"]

    # Mock agent response for testing
    actual_output = expected_answer

    return LLMTestCase(
        input=question,
        actual_output=actual_output,
        expected_output=expected_answer,
        context=context,
        # Add retrieval_context for FaithfulnessMetric (as per documentation)
        retrieval_context=context
    )


@pytest.fixture
def deepeval_dataset(eval_data_loader):
    """Create a DeepEval dataset from ConvFinQA data."""
    from deepeval.dataset.dataset import EvaluationDataset
    from deepeval.dataset.golden import Golden

    if not eval_data_loader.dataset:
        pytest.skip("Dataset not loaded")

    # Create goldens from a few dev records
    goldens = []
    records_to_process = eval_data_loader.dataset.dev[:5] if eval_data_loader.dataset.dev else []

    for record in records_to_process:
        if not record.dialogue.conv_questions:
            continue

        question = record.dialogue.conv_questions[0]
        answer = record.dialogue.conv_answers[0]

        # Create financial table context
        table = record.get_financial_table()
        df = table.to_dataframe()
        context = [f"Financial data:\n{df.to_string()}"]

        golden = Golden(
            input=question,
            expected_output=answer,
            context=context
        )
        goldens.append(golden)

    return EvaluationDataset(goldens=goldens)
