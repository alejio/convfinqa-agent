# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ConvFinQA (Conversational Financial Question Answering) assignment project that implements an LLM-driven prototype for answering conversational dialogue questions about financial documents. The project uses a cleaned version of the ConvFinQA Dataset and follows a milestone-based development approach.

## Development Environment

- **Python Version**: 3.12
- **Environment Manager**: UV (Astral)
- **CLI Framework**: Typer
- **Package Management**: pyproject.toml
- **LLM Framework**: smolagents + OpenAI API
- **Agent Architecture**: Function-calling with tool integration

## Common Commands

### Environment Setup
```bash
# Install UV environment manager
brew install uv

# Setup environment and install dependencies
uv sync

# Add a package to the environment
uv add <package_name>

# Set up OpenAI API key (required for LLM functionality)
export OPENAI_API_KEY="your_api_key_here"
# or add to .env file
```

### Running the Application
```bash
# Run the main CLI application
uv run main

# Chat with a specific record
uv run main chat <record_id>

# Show detailed record information
uv run main show_record <record_id>

# List available records
uv run main list_records --limit 10

# Show dataset statistics
uv run main dataset_stats

# Run agent evaluation on dev set
uv run main evaluate --max-records 14 --parallel
```

### Development Commands
```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test markers
uv run pytest -m "not slow"
uv run pytest -m integration

# Lint code
uv run ruff check src/
uv run ruff format src/

# Type checking
uv run mypy src/

# Pre-commit hooks
uv run pre-commit run --all-files
```

## Key Dependencies

### Core LLM Integration
- **openai>=1.0.0**: OpenAI API integration for GPT models
- **smolagents>=1.20.0**: Agent framework providing function-calling capabilities (ACTIVELY USED)
- **litellm>=1.0.0**: Multi-provider LLM integration for smolagents OpenAI compatibility
- **deepeval>=3.3.0**: LLM evaluation framework for assessment and metrics

### Evaluation & Testing
- **deepeval>=3.3.0**: LLM evaluation framework for assessment and metrics
- **pytest-mock>=3.14.1**: Advanced mocking capabilities for LLM and agent testing

### Core Dependencies
- **typer**: CLI framework with rich formatting
- **pydantic**: Data validation and serialization
- **pandas**: Financial data processing
- **rich**: Enhanced console output

## Architecture Overview

### Core Components

1. **Data Models** (`src/core/models.py`):
   - `ConvFinQADataset`: Main dataset container with train/dev splits
   - `Record`: Individual conversation record with document and dialogue
   - `Document`: Contains pre/post text and financial table data
   - `FinancialTable`: Structured representation of financial data with schema
   - `TableSchema`: Metadata about table structure and column types
   - `Dialogue`: Conversation structure with questions, answers, and programs

2. **Data Loader** (`src/data/loader.py`):
   - `DataLoader`: Central class for loading and managing ConvFinQA data
   - Handles JSON dataset loading with Pydantic validation
   - Caches records for efficient lookup
   - Converts raw data to pandas DataFrames

3. **LLM Agent System** (`src/functions/`):
   - `ConvFinQAAgent` (`agent.py`): **SIMPLIFIED** smolagents CodeAgent wrapper (147 lines vs 527 lines)
   - Function-calling tools (`tools.py`): `list_tables()`, `show-table()`, `query-table()`, `compute()`
   - Mathematical functions (`math.py`): Financial computation library with ConvFinQA constants
   - Context management for record state and conversation history
   - **DEPRECATED**: Complex PEC components (`planner.py`, `executor.py`, `critic.py`, `dspy_modules.py`)

4. **CLI Interface** (`src/cli/`):
   - Enhanced Typer-based command-line interface (`main.py`)
   - Comprehensive error handling system (`error_handler.py`)
   - Rich formatting for console output
   - Commands: `chat`, `show_record`, `list_records`, `dataset_stats`, `evaluate`
   - **Fully functional chat with LLM integration**
   - **Complete evaluation pipeline** with parallel processing and baseline comparison

5. **Evaluation System** (`src/evaluation/`):
   - **ExecutionAccuracyMetric**: Primary evaluation metric with exact numeric matching
   - **G-Eval Metrics**: Financial Accuracy, Financial Relevance, Calculation Coherence
   - **DSPy Integration** (`dspy_extraction.py`): Intelligent answer extraction and normalization
   - **Parallel Evaluation**: Thread-safe concurrent processing for performance
   - **Baseline Comparison**: Comprehensive comparison with ConvFinQA paper results
   - **Performance Breakdown**: Analysis by conversation type and turn number

6. **Core Infrastructure** (`src/core/`):
   - Logging (`logger.py`): Configurable logging with environment variables
   - Exception handling (`exceptions.py`): Custom exception types
   - **Conversation Management** (`conversation.py`): Multi-turn dialogue state tracking with entity extraction
   - Supports LOG_LEVEL and LOG_FORMAT configuration

### Data Flow

1. **Loading**: JSON dataset → Pydantic models → Cached records
2. **Processing**: Raw table data → FinancialTable with schema → pandas DataFrame
3. **Context Management**: Record context → Function-calling tools → LLM agent
4. **LLM Integration**: User queries → Agent function calls → Financial computations → Structured responses
5. **Evaluation Pipeline**: Agent responses → Multiple metrics → Baseline comparison → Detailed analysis
6. **Caching**: In-memory record cache for efficient lookup
7. **Interface**: CLI commands access data through DataLoader and Agent system

### Key Design Patterns

- **Factory Pattern**: `create_data_loader()` for DataLoader instantiation
- **Agent Pattern**: **SIMPLIFIED** smolagents CodeAgent with native function-calling
- **Tool Pattern**: Modular function definitions for specific financial operations
- **Context Management**: Global state management for record and data loader access
- **Conversation State Pattern**: Session persistence with entity tracking for multi-turn dialogue continuity
- **Schema Validation**: Pydantic models ensure data integrity
- **Caching**: Record-level caching for performance
- **Type Safety**: Full type hints and mypy compliance
- **Structured Logging**: Environment-configurable logging with mathematical operation tracking
- **Error Handling**: Centralized error management with rich user feedback

## Milestone Roadmap

This project follows a 5-milestone development approach:

1. **Milestone 1**: Data Ingestion & Schema (✅ COMPLETED)
2. **Milestone 2**: One-Turn Function-Calling QA (✅ COMPLETED)
3. **Milestone 3**: Multi-Turn Dialogue & State (✅ COMPLETED)
4. **Milestone 4**: ~~Planner–Executor–Critic Agent~~ **SIMPLIFIED to Smol** (✅ COMPLETED)
5. **Milestone 5**: Evaluation, Metrics & CI (✅ COMPLETED)

## Testing & Evaluation

- **Framework**: pytest with coverage reporting + DeepEval for LLM evaluation
- **Test Structure**: Unit tests, integration tests, and end-to-end agent evaluation
- **Markers**: `slow`, `integration` for test categorization
- **Coverage**: HTML and terminal coverage reports
- **Fixtures**: Shared fixtures in `tests/conftest.py`

### Test Categories

- Data loading and validation
- Table extraction and schema detection
- CLI command functionality
- **LLM agent functionality and function-calling integration**
- **Mathematical function testing and computation validation**
- **Tool function testing with context management**
- **Conversation state management and entity extraction testing**
- **Agent evaluation testing with baseline comparison**
- Integration tests for milestone features

### Evaluation System

- **Primary Metric**: Execution Accuracy (exact numeric matching)
- **Enhanced Metrics**: Financial Accuracy, Financial Relevance, Calculation Coherence (G-Eval)
- **DSPy Integration**: Intelligent answer extraction and normalization
- **Parallel Evaluation**: Thread-safe concurrent processing for faster evaluation
- **Baseline Comparison**: Comprehensive comparison with ConvFinQA paper results
- **Performance Breakdown**: Analysis by conversation type (simple/hybrid) and turn number

## Code Quality

- **Linting**: Ruff with comprehensive rule set
- **Type Checking**: mypy with strict configuration
- **Formatting**: Ruff formatter
- **Pre-commit**: Automated code quality checks
- **Documentation**: Docstrings for all modules and functions

## Configuration

- **Environment Variables**: LOG_LEVEL, LOG_FORMAT, OPENAI_API_KEY via .env
- **Package Config**: pyproject.toml with tool configurations including LLM dependencies
- **Test Config**: pytest.ini_options with mock integration for agent testing
- **Linting Config**: Ruff configuration with selected rules
- **LLM Config**: OpenAI GPT-4.1 model with native smolagents function-calling

## LLM Integration & Performance (COMPLETE)

The project features **complete LLM integration and evaluation** with:
- **Native Smol CodeAgent**: Direct function-calling orchestration via smolagents framework
- **Active tool definitions**: `list_tables()`, `show_table()`, `query_table()`, `compute()`
- **OpenAI GPT-4.1 model**: Latest OpenAI model with 1M token context window and enhanced coding capabilities
- **LiteLLM integration**: Seamless OpenAI API integration through smolagents
- **Conversation history management** with context preservation and entity extraction for dialogue continuity
- **Mathematical computation library** with ConvFinQA constants and operations
- **Context-aware financial analysis** with automatic table data access
- **~75% code reduction**: Simplified from 527-line PEC architecture to 147-line smolagents wrapper

### Performance Results
- **Execution Accuracy**: 45.26% on ConvFinQA dev set (62/137 correct)
- **Outperforms baselines**: Beats GPT-3 answer-only (24.09%) and CoT prompting (40.63%)
- **Matches DSL program baseline**: Equals GPT-3 DSL program performance (45.15%)
- **Multi-turn capability**: Handles complex conversational context across turns
- **Performance by turn**: Turn 1: 52.00%, Turn 2: 38.00%, Turn 3: 45.95%
- **Performance by type**: Simple conversations: 45.26% (62/137), Hybrid conversations: 0.00% (0/0)

## Important Notes

### Architecture Simplification (Major Update)
- **SIMPLIFIED from PEC to Smol**: Replaced complex 527-line Planner-Executor-Critic architecture with 147-line smolagents CodeAgent
- **Native function-calling**: Direct tool orchestration via smolagents framework instead of custom PEC pipeline
- **Maintained all functionality**: Same CLI commands, conversation management, and tool capabilities
- **Improved maintainability**: Single agent class vs. 4 complex PEC components

### Project Completion Status ✅ FINALIZED
- **ALL MILESTONES COMPLETED** - Full ConvFinQA prototype implemented and evaluated
- **PROJECT FINALIZED** - Development work complete, ready for submission/deployment
- **Current Branch**: `task/token-optimised-refactor` - Token optimization improvements and DSPy enhancements
- **Evaluation Results**: Achieved **45.26% execution accuracy** on dev set, matching GPT-3 DSL program baseline
- **Comprehensive evaluation framework** with DeepEval integration and baseline comparison
- **Chat functionality is fully operational** with real OpenAI GPT-4.1 integration
- **Enhanced CLI structure** with comprehensive error handling in `src/cli/`
- **Function-calling ecosystem** complete with tools, math functions, and agent wrapper
- **Production-ready**: Finalized for deployment and research use

### Technical Details
- Uses in-memory caching for fast data access
- Project follows strict typing and validation patterns
- All numeric processing handles comma separators and string conversion
- **Context management system** maintains record state across function calls
- **OpenAI GPT-4.1 default model**: Latest model with 1M token context and enhanced performance
- **LiteLLM integration**: Proper model_id configuration to avoid Claude model defaults
- **Production-ready evaluation**: Comprehensive metrics framework with DeepEval and DSPy integration
