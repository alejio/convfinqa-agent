"""
Main typer app for ConvFinQA
"""

from typing import Any

import typer
from rich import print as rich_print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.logger import get_logger
from ..data.loader import create_data_loader
from ..functions.agent import ConvFinQAAgent
from .error_handler import handle_cli_errors, handle_table_display_errors

logger = get_logger(__name__)
console = Console()

app = typer.Typer(
    name="main",
    help="ConvFinQA: Conversational Financial Question Answering with Smol",
    add_completion=True,
    no_args_is_help=True,
)


@app.command()
@handle_cli_errors
def chat(
    record_id: str = typer.Argument(..., help="ID of the record to chat about"),
    session_id: str = typer.Option(
        None, "--session", help="Resume a previous conversation session"
    ),
    show_sessions: bool = typer.Option(
        False, "--list-sessions", help="List available sessions for this record"
    ),
    model: str = typer.Option("gpt-4o-mini", "--model", help="LLM model to use"),
) -> None:
    """Ask questions about a specific record with multi-turn conversation support using Smol"""
    # Initialize data loader
    data_loader = create_data_loader()

    # Load the specific record
    record = data_loader.get_record(record_id)

    # Initialize LLM agent with Smol
    agent = ConvFinQAAgent(model=model)

    # Handle session listing
    if show_sessions:
        sessions = agent.conversation_manager.list_sessions()
        record_sessions = [s for s in sessions if s[2] == record_id]

        if not record_sessions:
            console.print(
                f"[yellow]No previous sessions found for record {record_id}[/yellow]"
            )
            return

        console.print(
            Panel(f"[bold]Available Sessions for {record_id}[/bold]", style="blue")
        )
        session_table = Table()
        session_table.add_column("Session ID", style="cyan")
        session_table.add_column("Created", style="green")
        session_table.add_column("Turns", style="magenta")

        for s_id, created, _ in record_sessions:
            # Count turns by loading the session briefly
            temp_state = agent.conversation_manager._load_session(s_id)
            turn_count = len(temp_state.turns)
            session_table.add_row(
                s_id[:8] + "...", created.strftime("%Y-%m-%d %H:%M"), str(turn_count)
            )

        console.print(session_table)
        console.print(
            "\n[dim]Use --session <session_id> to resume a conversation[/dim]"
        )
        return

    # Show agent configuration
    console.print(
        Panel(
            f"[bold blue]ConvFinQA Agent (Smol)[/bold blue]\n"
            f"Model: {model}\n"
            f"Architecture: Smol CodeAgent",
            title="🤖 ConvFinQA Agent",
        )
    )

    # Set record context
    agent.set_record_context(record, data_loader)

    # Show record information
    table_count = 1 if record.doc.table else 0
    console.print(
        Panel(
            f"[bold green]Record: {record.id}[/bold green]\n"
            f"Tables: {table_count}\n"
            f"Dialogue turns: {len(record.dialogue.conv_questions)}",
            title="📊 Financial Document",
        )
    )

    current_session = agent.get_current_session_id()

    # Get the financial table
    table = record.get_financial_table()
    df = table.to_dataframe()

    # Display record information
    session_display = current_session[:8] + "..." if current_session else "None"
    console.print(
        Panel(
            f"[bold]Record: {record_id}[/bold] | Session: {session_display} | Smol",
            style="blue",
        )
    )

    # Show conversation context if resuming
    if (
        session_id
        and agent.conversation_manager.current_state
        and agent.conversation_manager.current_state.turns
    ):
        console.print(
            Panel("[bold]Resuming Previous Conversation[/bold]", style="green")
        )
        # Show last few turns
        recent_turns = agent.conversation_manager.current_state.turns[-3:]
        for turn in recent_turns:
            rich_print(
                f"[dim]User: {turn.user_message[:80]}{'...' if len(turn.user_message) > 80 else ''}[/dim]"
            )
            rich_print(
                f"[dim]Assistant: {turn.assistant_response[:80]}{'...' if len(turn.assistant_response) > 80 else ''}[/dim]"
            )
        console.print()
    else:
        # Display initial context for new conversations
        console.print(f"[dim]Pre-text: {record.doc.pre_text[:100]}...[/dim]")
        console.print(f"[dim]Post-text: {record.doc.post_text[:100]}...[/dim]")

        # Display table schema
        schema_table = Table(title="Table Schema")
        schema_table.add_column("Column", style="cyan")
        schema_table.add_column("Type", style="magenta")
        schema_table.add_column("Nullable", style="green")

        for col in table.table_schema.columns:
            schema_table.add_row(col.name, col.column_type, str(col.nullable))

        console.print(schema_table)

        # Display financial data
        console.print(Panel("[bold]Financial Data[/bold]", style="green"))
        rich_print(df.to_string())

    # Interactive chat loop with enhanced commands
    console.print(
        Panel(
            "[bold]Multi-Turn Chat Mode (Smol)[/bold] - Commands: 'exit', 'clear', 'context', 'session'",
            style="yellow",
        )
    )

    while True:
        message = input(">>> ")
        command = message.strip().lower()

        if command in {"exit", "quit"}:
            break
        elif command == "clear":
            agent.clear_history()
            rich_print("[green]Conversation history cleared[/green]")
            continue
        elif command == "context":
            context = agent.conversation_manager.get_conversation_context()
            rich_print(f"[cyan][bold]Current Context:[/bold]\n{context}[/cyan]")
            continue
        elif command == "session":
            current_session_id = agent.get_current_session_id()
            turn_count = (
                len(agent.conversation_manager.current_state.turns)
                if agent.conversation_manager.current_state
                else 0
            )
            rich_print(
                f"[cyan][bold]Current Session:[/bold] {current_session_id}\n[bold]Turns:[/bold] {turn_count}[/cyan]"
            )
            continue

        try:
            # Get response from LLM agent with multi-turn context
            response = agent.chat(message)
            rich_print(f"[blue][bold]assistant:[/bold] {response}[/blue]")
        except Exception as e:
            logger.error(f"Chat error: {e}")
            rich_print(
                f"[red][bold]Error:[/bold] Sorry, I encountered an error: {e}[/red]"
            )


@app.command()
@handle_cli_errors
def show_record(
    record_id: str = typer.Argument(..., help="ID of the record to display"),
) -> None:
    """Display detailed information about a specific record"""
    data_loader = create_data_loader()
    record = data_loader.get_record(record_id)

    # Display record details
    console.print(Panel(f"[bold]Record Details: {record_id}[/bold]", style="blue"))

    # Show dialogue
    dialogue_table = Table(title="Dialogue")
    dialogue_table.add_column("Turn", style="cyan")
    dialogue_table.add_column("Question", style="magenta")
    dialogue_table.add_column("Answer", style="white")
    dialogue_table.add_column("Program", style="green")

    for i, (question, answer, program) in enumerate(
        zip(
            record.dialogue.conv_questions,
            record.dialogue.conv_answers,
            record.dialogue.turn_program,
            strict=False,
        )
    ):
        dialogue_table.add_row(
            str(i + 1),
            question[:80] + "..." if len(question) > 80 else question,
            answer[:30] + "..." if len(answer) > 30 else answer,
            program[:50] + "..." if len(program) > 50 else program,
        )

    console.print(dialogue_table)

    # Show financial table
    _display_financial_table(record)


@app.command()
@handle_cli_errors
def list_records(
    limit: int = typer.Option(10, "--limit", help="Number of records to display"),
    dev_only: bool = typer.Option(
        False, "--dev-only", help="Show only dev set records"
    ),
    train_only: bool = typer.Option(
        False, "--train-only", help="Show only train set records"
    ),
) -> None:
    """List available records in the dataset"""
    data_loader = create_data_loader()

    # Determine which split to show
    split = None
    if dev_only and train_only:
        rich_print("[red]Error: Cannot specify both --dev-only and --train-only[/red]")
        raise typer.Exit(1)
    elif dev_only:
        split = "dev"
    elif train_only:
        split = "train"

    tables = data_loader.list_available_tables(split=split)

    # Update the panel title based on split
    if split == "dev":
        title = f"[bold]Dev Set Records (showing first {limit})[/bold]"
    elif split == "train":
        title = f"[bold]Train Set Records (showing first {limit})[/bold]"
    else:
        title = f"[bold]Available Records (showing first {limit})[/bold]"

    console.print(Panel(title, style="blue"))

    for i, table_id in enumerate(tables[:limit]):
        rich_print(f"{i + 1:3d}. {table_id}")

    if len(tables) > limit:
        rich_print(f"... and {len(tables) - limit} more records")

    split_info = f" ({split} set)" if split else ""
    rich_print(f"\n[bold]Total records{split_info}: {len(tables)}[/bold]")


@app.command()
@handle_cli_errors
def dataset_stats() -> None:
    """Display dataset statistics"""
    data_loader = create_data_loader()
    stats = data_loader.get_dataset_statistics()

    console.print(Panel("[bold]Dataset Statistics[/bold]", style="blue"))

    # Basic counts
    stats_table = Table(title="Record Counts")
    stats_table.add_column("Split", style="cyan")
    stats_table.add_column("Count", style="magenta")

    stats_table.add_row("Train", str(stats["train_records"]))
    stats_table.add_row("Dev", str(stats["dev_records"]))
    stats_table.add_row("Total", str(stats["total_records"]))

    console.print(stats_table)

    # Column types
    if stats["column_type_distribution"]:
        col_table = Table(title="Column Type Distribution")
        col_table.add_column("Type", style="cyan")
        col_table.add_column("Count", style="magenta")

        for col_type, count in stats["column_type_distribution"].items():
            col_table.add_row(col_type, str(count))

        console.print(col_table)

    # Sample table shapes
    if stats["sample_table_shapes"]:
        rich_print("\n[bold]Sample table shapes:[/bold]")
        for i, shape in enumerate(stats["sample_table_shapes"]):
            rich_print(f"  {i + 1}. {shape[0]} rows × {shape[1]} columns")


@handle_table_display_errors
def _display_financial_table(record: Any) -> None:
    """Helper function to display financial table with error handling."""
    table = record.get_financial_table()
    df = table.to_dataframe()

    console.print(Panel("[bold]Financial Table[/bold]", style="green"))
    rich_print(df.to_string())


@app.command()
@handle_cli_errors
def evaluate(
    max_records: int = typer.Option(
        5,
        "--max-records",
        help="Maximum number of dev records to evaluate (default: 5)",
    ),
    max_questions: int = typer.Option(
        3, "--max-questions", help="Maximum questions per record (default: 3)"
    ),
    model: str = typer.Option("gpt-4o-mini", "--model", help="LLM model to use"),
    pytest_mode: bool = typer.Option(
        False, "--pytest", help="Run pytest evaluation tests instead"
    ),
    parallel: bool = typer.Option(
        False, "--parallel", help="Process records in parallel for faster evaluation"
    ),
    max_workers: int = typer.Option(
        None,
        "--max-workers",
        help="Maximum number of worker threads for parallel processing (default: 4, max recommended: 8)",
    ),
    checkpoint_file: str = typer.Option(
        ".checkpoints/evaluation_checkpoint.json",
        "--checkpoint",
        help="Path to checkpoint file for saving/resuming progress",
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Resume from checkpoint if available"
    ),
    no_checkpoint: bool = typer.Option(
        False, "--no-checkpoint", help="Disable checkpoint saving/loading"
    ),
) -> None:
    """Run comprehensive evaluation with ConvFinQA baseline comparison and progress tracking"""
    if pytest_mode:
        # Original pytest-based evaluation
        console.print(
            Panel(
                f"[bold]ConvFinQA Evaluation (Pytest Mode)[/bold]\n"
                f"Model: {model}\n"
                f"Running existing DeepEval test infrastructure",
                style="blue",
            )
        )

        import os
        import subprocess

        cmd = ["uv", "run", "pytest", "tests/evaluation/", "-m", "evaluation", "-v"]
        env = os.environ.copy()
        env["RUN_EVALUATION"] = "true"

        try:
            result = subprocess.run(cmd, env=env, cwd=".")
            if result.returncode == 0:
                console.print(
                    Panel(
                        "[bold green]Pytest evaluation completed![/bold green]",
                        style="green",
                    )
                )
            else:
                console.print(
                    Panel(
                        f"[bold red]Pytest evaluation failed with exit code {result.returncode}[/bold red]",
                        style="red",
                    )
                )
        except Exception as e:
            console.print(f"[red]Failed to run pytest evaluation: {e}[/red]")
        return

    # New comprehensive evaluation mode
    execution_mode = "Parallel" if parallel else "Sequential"
    worker_info = f" (max workers: {max_workers})" if parallel and max_workers else ""

    console.print(
        Panel(
            f"[bold]ConvFinQA Baseline Comparison Evaluation[/bold]\n"
            f"Model: {model}\n"
            f"Token Optimized: ✅ Always Enabled\n"
            f"Max Records: {max_records} (from dev split)\n"
            f"Max Questions/Record: {max_questions}\n"
            f"Evaluation Method: Execution Accuracy (Exe Acc)\n"
            f"Processing: {execution_mode}{worker_info}",
            style="blue",
        )
    )

    # Check for OpenAI API key
    import os

    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[yellow]Warning: OPENAI_API_KEY not found. Set your API key to run real evaluations.[/yellow]"
        )
        return

    try:
        # Initialize components
        from ..evaluation.metrics import evaluate_agent_on_dataset

        console.print("[dim]Initializing data loader and agent...[/dim]")
        data_loader = create_data_loader()
        data_loader.load_dataset()  # Load the dataset explicitly
        agent = ConvFinQAAgent(model=model)

        # Run evaluation with parallelism options
        console.print(
            f"[dim]Starting {'parallel' if parallel else 'sequential'} evaluation on up to {max_records} dev records...[/dim]"
        )
        # Setup checkpoint parameters
        checkpoint_path = None if no_checkpoint else checkpoint_file

        results = evaluate_agent_on_dataset(
            data_loader=data_loader,
            agent=agent,
            max_records=max_records,
            max_questions_per_record=max_questions,
            parallel=parallel,
            max_workers=max_workers,
            checkpoint_file=checkpoint_path,
            resume=resume,
        )

        # Print results in baseline comparison format
        results.print_baseline_comparison()

        # Additional analysis if requested
        if max_records <= 10:  # Only show details for small evaluations
            console.print("\n" + "=" * 60)
            console.print("DETAILED RESULTS")
            console.print("=" * 60)

            correct_examples = [r for r in results.detailed_results if r["is_correct"]][
                :3
            ]
            incorrect_examples = [
                r for r in results.detailed_results if not r["is_correct"]
            ][:3]

            if correct_examples:
                console.print("\n[green]✓ Sample Correct Answers:[/green]")
                for ex in correct_examples:
                    console.print(
                        f"  Record: {ex['record_id'][:12]}... Turn: {ex['turn']}"
                    )
                    console.print(f"  Q: {ex['question'][:80]}...")
                    console.print(f"  Expected: {ex['expected']}")
                    console.print(f"  Got: {ex['actual']}")
                    console.print()

            if incorrect_examples:
                console.print("\n[red]✗ Sample Incorrect Answers:[/red]")
                for ex in incorrect_examples:
                    console.print(
                        f"  Record: {ex['record_id'][:12]}... Turn: {ex['turn']}"
                    )
                    console.print(f"  Q: {ex['question'][:80]}...")
                    console.print(f"  Expected: {ex['expected']}")
                    console.print(f"  Got: {ex['actual']}")
                    console.print()

        execution_info = f"{'Parallel' if parallel else 'Sequential'} Evaluation"
        console.print(
            Panel(
                f"[bold green]{execution_info} Complete![/bold green]\n"
                f"Your Agent: {results.execution_accuracy:.2f}% Execution Accuracy\n"
                f"Questions Evaluated: {results.total_questions}\n"
                f"Compare with ConvFinQA baselines above.",
                style="green",
            )
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted by user[/yellow]")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        console.print(f"[red]Evaluation failed: {e}[/red]")
        console.print(
            "[dim]Try with --pytest flag to run original evaluation tests[/dim]"
        )


@app.command()
@handle_cli_errors
def agent_info(
    model: str = typer.Option("gpt-4o-mini", "--model", help="Model to use"),
) -> None:
    """Show information about the Smol agent"""
    console.print(
        Panel(
            f"[bold blue]ConvFinQA Agent (Smol)[/bold blue]\n\n"
            f"🔧 Model: [cyan]{model}[/cyan]\n"
            f"🏗️  Architecture: [yellow]Smol CodeAgent[/yellow]\n"
            f"⚡ Framework: [green]HuggingFace smolagents[/green]",
            title="🤖 Agent Configuration",
        )
    )

    console.print(
        Panel(
            "[bold blue]Smol Architecture[/bold blue]\n\n"
            "🎯 Agent: [cyan]CodeAgent with native tool orchestration[/cyan]\n"
            "⚙️  Model: [cyan]LiteLLMModel for OpenAI integration[/cyan]\n"
            "🔍 Tools: [cyan]Custom financial analysis functions[/cyan]\n"
            "🔁 Conversation: [yellow]Built-in memory management[/yellow]\n",
            title="🏗️ Smol Components",
        )
    )

    console.print(
        Panel(
            "[bold blue]Key Features[/bold blue]\n\n"
            "• Direct LLM function-calling via Smol\n"
            "• Simplified architecture with native tool integration\n"
            "• Automatic conversation memory management\n"
            "• Native OpenAI API integration via LiteLLM\n"
            "• Streamlined financial analysis workflow\n"
            "• Optimized token-efficient prompts for cost reduction\n"
            "• Built-in error handling and retry logic\n"
            "• Compatible with existing evaluation framework\n",
            title="✨ Smol System",
        )
    )


if __name__ == "__main__":
    app()
