"""
Custom evaluation metrics for ConvFinQA using DeepEval's proper G-Eval pattern.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepeval.metrics.base_metric import BaseMetric
from deepeval.metrics.conversational_g_eval.conversational_g_eval import (
    ConversationalGEval,
)
from deepeval.metrics.g_eval.g_eval import GEval
from deepeval.test_case.conversational_test_case import TurnParams
from deepeval.test_case.llm_test_case import LLMTestCase, LLMTestCaseParams
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ..core.logger import get_logger

logger = get_logger(__name__)
console = Console()

if TYPE_CHECKING:
    from ..data.loader import DataLoader
    from ..functions.agent import ConvFinQAAgent


class ExecutionAccuracyMetric(BaseMetric):
    """
    Execution Accuracy metric for ConvFinQA using exact numeric matching.

    This metric implements the exact evaluation methodology used in the ConvFinQA paper
    to replicate baseline figures. It performs exact match on numeric outputs without
    using LLM-as-a-judge.
    """

    def __init__(self, threshold: float = 1.0):
        """Initialize execution accuracy metric.

        Args:
            threshold: Threshold for success (1.0 for exact match)
        """
        self.threshold: float = threshold
        self.score: float = 0.0
        self.reason: str = ""
        self.success: bool = False

    @property
    def __name__(self) -> str:
        return "Execution Accuracy"

    def measure(self, test_case: LLMTestCase) -> float:
        """Measure execution accuracy by exact numeric match.

        Args:
            test_case: The test case containing actual_output and expected_output

        Returns:
            Score of 1.0 for exact match, 0.0 otherwise
        """
        if not test_case.actual_output or not test_case.expected_output:
            self.reason = "Missing actual_output or expected_output"
            self.score = 0.0
            self.success = False
            return self.score

        # Extract numeric values from both outputs
        actual_num = self._extract_numeric_value(test_case.actual_output)
        expected_num = self._extract_numeric_value(test_case.expected_output)

        # Check for exact match
        if self._numeric_match(actual_num, expected_num):
            self.score = 1.0
            self.reason = f"Exact match: {actual_num} == {expected_num}"
            self.success = True
        else:
            self.score = 0.0
            self.reason = (
                f"No match: {actual_num} != {expected_num} (actual vs expected)"
            )
            self.success = False

        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async version of measure (same implementation since no LLM calls)."""
        return self.measure(test_case)

    def is_successful(self) -> bool:
        """Check if the metric was successful."""
        return self.success

    def _extract_numeric_value(self, text: str) -> str | None:
        """Extract the primary numeric value from text using shared utility."""
        from ..core.numeric_utils import NumericExtractor

        return NumericExtractor.extract_primary_numeric_value(text)

    def _numeric_match(self, actual_num: str | None, expected_num: str | None) -> bool:
        """Check if two numeric values match exactly using shared utility."""
        from ..core.numeric_utils import NumericExtractor

        return NumericExtractor.numeric_match(actual_num, expected_num)


def _evaluate_single_record(
    record: Any,
    data_loader: "DataLoader",
    model: str,
    max_questions_per_record: int | None = None,
    add_delay: bool = True,
    token_optimized: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate a single record with its own agent instance.

    This function is designed to be thread-safe by creating a separate agent
    instance for each record evaluation.

    Args:
        record: The record to evaluate
        data_loader: DataLoader instance
        model: The model name to use
        max_questions_per_record: Maximum questions per record (None = all questions)
        add_delay: Whether to add small delays to reduce rate limiting
        token_optimized: Whether to enable token optimization for reduced usage

    Returns:
        List of result dictionaries for this record
    """
    # Import here to avoid circular imports
    from ..functions.agent import ConvFinQAAgent

    # Add small random delay at start to stagger parallel requests (only for parallel mode)
    if add_delay:
        import random
        import time

        initial_delay = random.uniform(0.1, 0.5)
        time.sleep(initial_delay)

    # Create a separate agent instance for this record (thread-safe)
    agent = ConvFinQAAgent(model=model, token_optimized=token_optimized)

    # Set record context for agent and start fresh conversation
    agent.set_record_context(record, data_loader)

    # Clear any previous conversation state to prevent context bleed
    agent.clear_history()

    # Determine if this is a hybrid conversation
    is_hybrid = getattr(record.features, "has_type2_question", False)

    # Process questions in this record
    questions_to_process = record.dialogue.conv_questions
    if max_questions_per_record:
        questions_to_process = questions_to_process[:max_questions_per_record]

    record_results = []

    for turn, (question, expected_answer) in enumerate(
        zip(
            questions_to_process,
            record.dialogue.conv_answers[: len(questions_to_process)],
            strict=False,
        ),
        1,
    ):
        try:
            # Add small delay between questions to reduce rate limiting pressure (only for parallel mode)
            if add_delay and turn > 1:
                import random
                import time

                inter_question_delay = random.uniform(0.2, 0.8)
                time.sleep(inter_question_delay)

            # Get agent response
            actual_answer = agent.chat(question)

            # Check execution accuracy using existing metric (with question context for DSPy)
            is_correct = calculate_execution_accuracy(
                actual_answer, expected_answer, question
            )

            # Record result
            record_results.append(
                {
                    "record_id": record.id,
                    "turn": turn,
                    "question": question,
                    "expected": expected_answer,
                    "actual": actual_answer,
                    "is_correct": is_correct,
                    "is_hybrid_conversation": is_hybrid,
                }
            )

        except Exception as e:
            # Record as incorrect on error
            record_results.append(
                {
                    "record_id": record.id,
                    "turn": turn,
                    "question": question,
                    "expected": expected_answer,
                    "actual": f"ERROR: {str(e)}",
                    "is_correct": False,
                    "is_hybrid_conversation": is_hybrid,
                }
            )

    return record_results


def evaluate_agent_on_dataset(
    data_loader: "DataLoader",
    agent: "ConvFinQAAgent",
    max_records: int | None = None,
    max_questions_per_record: int | None = None,
    parallel: bool = False,
    max_workers: int | None = None,
    checkpoint_file: str | None = None,
    resume: bool = False,
    token_optimized: bool = False,
) -> "ConvFinQAEvaluationResults":
    """Evaluate agent on dataset with ConvFinQA baseline comparison format.

    Args:
        data_loader: DataLoader instance
        agent: ConvFinQAAgent instance (only used for model name if parallel=True)
        max_records: Maximum number of records to evaluate (None = all dev records)
        max_questions_per_record: Maximum questions per record (None = all questions)
        parallel: Whether to process records in parallel (default: False)
        max_workers: Maximum number of worker threads for parallel processing (default: 4, max 8 for OpenAI rate limits)
        checkpoint_file: Path to checkpoint file for saving/resuming progress
        resume: Whether to resume from checkpoint if available
        token_optimized: Whether to enable token optimization for reduced usage

    Returns:
        ConvFinQAEvaluationResults with comprehensive metrics
    """
    results = ConvFinQAEvaluationResults()

    # Setup checkpoint handling
    checkpoint_path = None
    if checkpoint_file:
        checkpoint_path = Path(checkpoint_file)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to resume from checkpoint
    processed_record_ids = set()
    if resume and checkpoint_path and checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                checkpoint_data = json.load(f)

            # Restore previous results
            for result_data in checkpoint_data.get("results", []):
                results.add_result(**result_data)

            processed_record_ids = set(checkpoint_data.get("processed_record_ids", []))

            console.print(
                f"[green]Resumed from checkpoint: {len(processed_record_ids)} records already processed[/green]"
            )
        except Exception as e:
            console.print(
                f"[yellow]Failed to load checkpoint: {e}. Starting fresh.[/yellow]"
            )
            processed_record_ids = set()

    # Use dev split for evaluation (matches paper methodology)
    if data_loader.dataset is None:
        raise ValueError("Dataset not loaded")
    all_records = data_loader.dataset.dev
    if max_records:
        all_records = all_records[:max_records]

    # Filter out already processed records if resuming
    records_to_evaluate = [
        record for record in all_records if record.id not in processed_record_ids
    ]

    if not records_to_evaluate:
        console.print(
            "[green]All records already processed! Loading final results...[/green]"
        )
        return results

    def save_checkpoint() -> None:
        """Save current progress to checkpoint file."""
        if checkpoint_path:
            try:
                checkpoint_data = {
                    "processed_record_ids": list(processed_record_ids),
                    "results": [
                        {
                            "record_id": r["record_id"],
                            "turn": r["turn"],
                            "question": r["question"],
                            "expected": r["expected"],
                            "actual": r["actual"],
                            "is_correct": r["is_correct"],
                            "is_hybrid_conversation": r["is_hybrid"],
                        }
                        for r in results.detailed_results
                    ],
                    "total_records": len(all_records),
                    "max_records": max_records,
                    "max_questions_per_record": max_questions_per_record,
                }

                with open(checkpoint_path, "w") as f:
                    json.dump(checkpoint_data, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")

    total_to_process = len(records_to_evaluate)
    console.print(
        f"[dim]Processing {total_to_process} records ({len(processed_record_ids)} already completed)...[/dim]"
    )

    if parallel:
        # Parallel evaluation using ThreadPoolExecutor with OpenAI rate limit considerations
        import concurrent.futures
        from concurrent.futures import ThreadPoolExecutor

        # Limit to 4 workers by default, max 8 to avoid rate limiting
        if max_workers is None:
            max_workers = 4
        elif max_workers > 8:
            logger.warning(
                f"max_workers={max_workers} may cause rate limiting, consider ≤8"
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[cyan]({task.fields[current_acc]:.1f}% acc)[/cyan]"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Evaluating records...", total=total_to_process, current_acc=0.0
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all records for parallel processing
                future_to_record = {
                    executor.submit(
                        _evaluate_single_record,
                        record,
                        data_loader,
                        agent.model,  # Use agent's model name
                        max_questions_per_record,
                        True,  # add_delay=True for parallel processing
                        token_optimized,  # pass token_optimized flag
                    ): record
                    for record in records_to_evaluate
                }

                # Collect results as they complete
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_record):
                    try:
                        record = future_to_record[future]
                        record_results = future.result()

                        # Add all results from this record
                        for result in record_results:
                            results.add_result(
                                record_id=result["record_id"],
                                turn=result["turn"],
                                question=result["question"],
                                expected=result["expected"],
                                actual=result["actual"],
                                is_correct=result["is_correct"],
                                is_hybrid_conversation=result["is_hybrid_conversation"],
                            )

                        # Mark record as processed and save checkpoint
                        processed_record_ids.add(record.id)
                        save_checkpoint()

                        completed_count += 1
                        current_acc = (
                            results.execution_accuracy
                            if results.total_questions > 0
                            else 0.0
                        )
                        progress.update(
                            task,
                            advance=1,
                            current_acc=current_acc,
                            description=f"Evaluating records... (Record {record.id[:8]}...)",
                        )

                    except Exception as e:
                        # Handle any unexpected errors
                        record = future_to_record[future]
                        logger.error(f"Record {record.id} generated an exception: {e}")
                        processed_record_ids.add(
                            record.id
                        )  # Mark as processed even if failed
                        save_checkpoint()
                        completed_count += 1
                        progress.update(task, advance=1)
    else:
        # Sequential evaluation with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[cyan]({task.fields[current_acc]:.1f}% acc)[/cyan]"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Evaluating records...", total=total_to_process, current_acc=0.0
            )

            for record in records_to_evaluate:
                try:
                    record_results = _evaluate_single_record(
                        record,
                        data_loader,
                        agent.model,
                        max_questions_per_record,
                        False,  # add_delay=False for sequential
                        token_optimized,  # pass token_optimized flag
                    )

                    # Add all results from this record
                    for result in record_results:
                        results.add_result(
                            record_id=result["record_id"],
                            turn=result["turn"],
                            question=result["question"],
                            expected=result["expected"],
                            actual=result["actual"],
                            is_correct=result["is_correct"],
                            is_hybrid_conversation=result["is_hybrid_conversation"],
                        )

                    # Mark record as processed and save checkpoint
                    processed_record_ids.add(record.id)
                    save_checkpoint()

                    current_acc = (
                        results.execution_accuracy
                        if results.total_questions > 0
                        else 0.0
                    )
                    progress.update(
                        task,
                        advance=1,
                        current_acc=current_acc,
                        description=f"Evaluating records... (Record {record.id[:8]}...)",
                    )

                except KeyboardInterrupt:
                    console.print(
                        "\n[yellow]Evaluation interrupted. Progress saved to checkpoint.[/yellow]"
                    )
                    save_checkpoint()
                    raise
                except Exception as e:
                    # Handle any unexpected errors at record level
                    logger.error(f"Record {record.id} generated an exception: {e}")
                    processed_record_ids.add(
                        record.id
                    )  # Mark as processed even if failed
                    save_checkpoint()
                    progress.update(task, advance=1)

    # Clean up checkpoint file if all records completed successfully
    if checkpoint_path and len(processed_record_ids) == len(all_records):
        try:
            checkpoint_path.unlink()
            console.print("[dim]Checkpoint file removed (evaluation complete)[/dim]")
        except Exception:
            pass  # Don't worry if cleanup fails

    return results


def calculate_execution_accuracy(
    actual_answer: str, expected_answer: str, question: str = ""
) -> bool:
    """Calculate execution accuracy by exact match on numeric outputs.

    This matches the evaluation methodology used in the ConvFinQA paper
    for comparing against baseline figures. Uses enhanced DSPy extraction for better accuracy.

    Args:
        actual_answer: The agent's response
        expected_answer: The ground truth answer
        question: Original question for context (helps DSPy extraction)

    Returns:
        True if the answers match exactly (for execution accuracy), False otherwise
    """
    # Try enhanced DSPy extraction first if available
    try:
        import os

        import dspy

        from .dspy_extraction import DSPyExecutionAccuracyMetric

        # Enhanced DSPy auto-configuration for better performance
        if not hasattr(dspy.settings, "lm") or dspy.settings.lm is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    # Use GPT-4o-mini for cost-effective intelligent extraction
                    lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key)
                    dspy.configure(lm=lm)
                    logger.info(
                        "Enhanced DSPy configured with OpenAI GPT-4o-mini for intelligent extraction"
                    )
                except Exception as e:
                    logger.debug(f"Failed to auto-configure enhanced DSPy: {e}")

        # Check if DSPy is now configured for enhanced processing
        if hasattr(dspy.settings, "lm") and dspy.settings.lm is not None:
            metric = DSPyExecutionAccuracyMetric()
            from deepeval.test_case.llm_test_case import LLMTestCase

            test_case = LLMTestCase(
                input=question,
                actual_output=actual_answer,
                expected_output=expected_answer,
            )

            # Use enhanced DSPy extraction with better context
            result = metric.measure(test_case, question=question)
            logger.debug(
                f"Enhanced DSPy extraction - Score: {result}, Reason: {metric.reason}"
            )
            return metric.is_successful()

    except Exception as e:
        logger.debug(
            f"Enhanced DSPy extraction failed, falling back to basic matching: {e}"
        )

    # Fallback to basic numeric extraction if DSPy is not available
    try:
        # Clean both answers for comparison
        actual_clean = _extract_numeric_value_basic(actual_answer)
        expected_clean = _extract_numeric_value_basic(expected_answer)

        # Check for exact numeric match
        if actual_clean and expected_clean:
            try:
                actual_float = float(actual_clean)
                expected_float = float(expected_clean)

                # Use small tolerance for floating point comparison
                tolerance = 1e-6
                match = abs(actual_float - expected_float) < tolerance

                logger.debug(
                    f"Basic numeric comparison: {actual_float} vs {expected_float}, match: {match}"
                )
                return match
            except ValueError:
                # If numeric conversion fails, fall back to string comparison
                match = actual_clean.strip() == expected_clean.strip()
                logger.debug(
                    f"String comparison fallback: '{actual_clean}' vs '{expected_clean}', match: {match}"
                )
                return match

        return False

    except Exception as e:
        logger.error(f"All extraction methods failed: {e}")
        return False


def _extract_numeric_value_basic(text: str) -> str | None:
    """Basic numeric extraction fallback when DSPy is not available."""
    from ..core.numeric_utils import NumericExtractor

    return NumericExtractor.extract_basic_numeric_value(text)


def create_financial_accuracy_metric(threshold: float = 0.7) -> GEval:
    """Create a G-Eval metric for financial calculation accuracy."""
    return GEval(
        name="Financial Accuracy",
        criteria=(
            "Evaluate whether the actual output contains accurate financial calculations, "
            "correct numerical values, and factually correct financial information. "
            "Consider mathematical precision, proper use of financial formulas, "
            "and alignment with provided financial data."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.CONTEXT,
        ],
        threshold=threshold,
    )


def create_financial_relevance_metric(threshold: float = 0.7) -> GEval:
    """Create a G-Eval metric for financial question relevance."""
    return GEval(
        name="Financial Relevance",
        criteria=(
            "Determine if the actual output directly addresses the financial question asked. "
            "The response should be relevant to financial analysis, use appropriate "
            "financial terminology, and focus on the specific financial aspects mentioned "
            "in the input question."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=threshold,
    )


def create_calculation_coherence_metric(threshold: float = 0.7) -> GEval:
    """Create a G-Eval metric for financial calculation coherence."""
    return GEval(
        name="Calculation Coherence",
        criteria=(
            "Assess whether the financial calculations follow logical steps, "
            "use consistent methodologies, and show clear reasoning from input data "
            "to final results. Check for proper sequence of calculations and "
            "reasonable intermediate steps."
        ),
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.CONTEXT,
        ],
        threshold=threshold,
    )


def create_conversational_financial_metric(
    threshold: float = 0.7,
) -> ConversationalGEval:
    """Create a ConversationalGEval metric for multi-turn financial conversations."""
    return ConversationalGEval(
        name="Conversational Financial Analysis",
        criteria=(
            "Evaluate the quality of conversational financial question answering. "
            "Assess whether the assistant maintains context across turns, provides "
            "consistent financial analysis, builds upon previous questions and answers, "
            "and demonstrates understanding of the ongoing financial discussion. "
            "Consider accuracy, relevance, and conversational flow."
        ),
        evaluation_params=[
            TurnParams.CONTENT
        ],  # Fixed: Use TurnParams instead of LLMTestCaseParams
        threshold=threshold,
    )


# Convenience function to get all financial metrics
def get_financial_metrics(threshold: float = 0.7) -> list[GEval]:
    """Get a list of all financial evaluation metrics."""
    return [
        create_financial_accuracy_metric(threshold),
        create_financial_relevance_metric(threshold),
        create_calculation_coherence_metric(threshold),
    ]


class ConvFinQAEvaluationResults:
    """Container for ConvFinQA evaluation results matching baseline format."""

    def __init__(self) -> None:
        self.total_questions = 0
        self.correct_answers = 0
        self.results_by_turn: dict[
            int, dict[str, int]
        ] = {}  # turn_number -> {correct: int, total: int}
        self.results_by_type = {
            "simple": {"correct": 0, "total": 0},
            "hybrid": {"correct": 0, "total": 0},
        }
        self.detailed_results: list[
            dict[str, Any]
        ] = []  # List of individual question results

    def add_result(
        self,
        record_id: str,
        turn: int,
        question: str,
        expected: str,
        actual: str,
        is_correct: bool,
        is_hybrid_conversation: bool = False,
    ) -> None:
        """Add a single question result."""
        self.total_questions += 1
        if is_correct:
            self.correct_answers += 1

        # Track by turn
        if turn not in self.results_by_turn:
            self.results_by_turn[turn] = {"correct": 0, "total": 0}
        self.results_by_turn[turn]["total"] += 1
        if is_correct:
            self.results_by_turn[turn]["correct"] += 1

        # Track by conversation type
        conv_type = "hybrid" if is_hybrid_conversation else "simple"
        self.results_by_type[conv_type]["total"] += 1
        if is_correct:
            self.results_by_type[conv_type]["correct"] += 1

        # Store detailed result
        self.detailed_results.append(
            {
                "record_id": record_id,
                "turn": turn,
                "question": question[:100] + "..." if len(question) > 100 else question,
                "expected": expected,
                "actual": actual,
                "is_correct": is_correct,
                "is_hybrid": is_hybrid_conversation,
            }
        )

    @property
    def execution_accuracy(self) -> float:
        """Calculate execution accuracy (Exe Acc) matching ConvFinQA baseline format."""
        return (
            (self.correct_answers / self.total_questions * 100)
            if self.total_questions > 0
            else 0.0
        )

    def get_turn_accuracy(self, turn: int) -> float:
        """Get accuracy for a specific turn number."""
        if turn not in self.results_by_turn or self.results_by_turn[turn]["total"] == 0:
            return 0.0
        return (
            float(
                self.results_by_turn[turn]["correct"]
                / self.results_by_turn[turn]["total"]
            )
            * 100
        )

    def get_conversation_type_accuracy(self, conv_type: str) -> float:
        """Get accuracy for simple or hybrid conversations."""
        if self.results_by_type[conv_type]["total"] == 0:
            return 0.0
        return (
            float(
                self.results_by_type[conv_type]["correct"]
                / self.results_by_type[conv_type]["total"]
            )
            * 100
        )

    def print_baseline_comparison(self) -> None:
        """Print results in format matching ConvFinQA baseline table."""
        print("\n" + "=" * 60)  # noqa: T201
        print("CONVFINQA EVALUATION RESULTS")  # noqa: T201
        print("=" * 60)  # noqa: T201

        print("\nModel Performance:")  # noqa: T201
        print(f"Your Agent (Smol):        {self.execution_accuracy:.2f}%")  # noqa: T201

        print("\nBaseline Comparison (from ConvFinQA paper):")  # noqa: T201
        print("-" * 50)  # noqa: T201
        print(f"{'Model':<30} {'Method':<25} {'Exe Acc':<8}")  # noqa: T201
        print("-" * 50)  # noqa: T201
        print(f"{'GPT-3':<30} {'answer-only-prompt':<25} {'24.09':<8}")  # noqa: T201
        print(f"{'GPT-3':<30} {'CoT prompting':<25} {'40.63':<8}")  # noqa: T201
        print(f"{'GPT-3':<30} {'DSL program':<25} {'45.15':<8}")  # noqa: T201
        print(f"{'FinQANet(RoBERTa-large)':<30} {'DSL program':<25} {'68.90':<8}")  # noqa: T201
        print("-" * 50)  # noqa: T201
        print(f"{'Human Expert':<30} {'':<25} {'89.44':<8}")  # noqa: T201
        print(  # noqa: T201
            f"{'Your Agent (Smol)':<30} {'Function-calling':<25} {f'{self.execution_accuracy:.2f}':<8}"
        )
        print("-" * 50)  # noqa: T201

        # Performance by conversation type
        print("\nPerformance by Conversation Type:")  # noqa: T201
        simple_acc = self.get_conversation_type_accuracy("simple")
        hybrid_acc = self.get_conversation_type_accuracy("hybrid")
        print(  # noqa: T201
            f"Simple Conversations:  {simple_acc:.2f}% ({self.results_by_type['simple']['correct']}/{self.results_by_type['simple']['total']})"
        )
        print(  # noqa: T201
            f"Hybrid Conversations:  {hybrid_acc:.2f}% ({self.results_by_type['hybrid']['correct']}/{self.results_by_type['hybrid']['total']})"
        )

        # Performance by turn (showing difficulty progression)
        print("\nPerformance by Turn (Multi-turn Analysis):")  # noqa: T201
        for turn in sorted(self.results_by_turn.keys()):
            turn_acc = self.get_turn_accuracy(turn)
            correct = self.results_by_turn[turn]["correct"]
            total = self.results_by_turn[turn]["total"]
            print(f"Turn {turn}:  {turn_acc:.2f}% ({correct}/{total})")  # noqa: T201

        print("\nOverall Statistics:")  # noqa: T201
        print(f"Total Questions Evaluated: {self.total_questions}")  # noqa: T201
        print(f"Correct Answers: {self.correct_answers}")  # noqa: T201
        print(f"Execution Accuracy: {self.execution_accuracy:.2f}%")  # noqa: T201
        print("=" * 60)  # noqa: T201
