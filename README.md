# ConvFinQA Agent

A conversational financial question answering system built with LLM function-calling and smolagents framework.

## Overview

This project implements an LLM-driven prototype for answering multi-turn conversational questions about financial documents using the ConvFinQA dataset. The agent uses OpenAI GPT-4.1 with function-calling capabilities to analyze financial tables and provide accurate numerical answers.

## Key Features

- **Multi-turn Dialogue**: Handles conversational context across question turns with entity tracking
- **Financial Table Analysis**: Direct querying and computation on financial data with schema detection
- **Function-calling Tools**: `list_tables()`, `show_table()`, `query_table()`, `compute()` for structured data access
- **Evaluation Framework**: Comprehensive metrics using DeepEval and DSPy with baseline comparison
- **Interactive CLI**: Chat interface with rich formatting and conversation management

## Quick Start

```bash
# Setup environment
brew install uv
uv sync
export OPENAI_API_KEY="your_api_key_here"

# Run the CLI
uv run main

# Chat with a specific record
uv run main chat <record_id>

# Evaluate on dev set
uv run main evaluate --max-records 14 --parallel
```

## Performance

- **Execution Accuracy**: 45.26% on ConvFinQA dev set (62/137 correct)
- **Outperforms**: GPT-3 answer-only (24.09%) and CoT prompting (40.63%)
- **Matches**: GPT-3 DSL program baseline (45.15%)

## Architecture

Built with Python 3.12, UV package manager, smolagents framework, and OpenAI API integration. Uses Pydantic for data validation, Typer for CLI, and pandas for financial data processing.

See `CLAUDE.md` for detailed development instructions and architecture overview.
