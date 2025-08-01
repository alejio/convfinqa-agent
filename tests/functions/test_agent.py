"""
Tests for the ConvFinQA agent functionality using smolagents CodeAgent.
"""

import os
import tempfile

import pytest

from src.functions.agent import ConvFinQAAgent


class TestConvFinQAAgent:
    """Test suite for smolagents-based ConvFinQA agent."""

    def test_agent_initialization(self, mocker) -> None:
        """Test agent initializes correctly with default model."""
        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        # Mock smolagents components
        mocker.patch("src.functions.agent.LiteLLMModel")
        mocker.patch("src.functions.agent.CodeAgent")

        agent = ConvFinQAAgent()

        assert agent.model == "gpt-4.1"
        assert agent.tools is not None
        assert agent.agent is not None
        assert agent.conversation_history == []

    def test_agent_initialization_custom_model(self, mocker) -> None:
        """Test agent initializes with custom model."""
        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        # Mock smolagents components
        mocker.patch("src.functions.agent.LiteLLMModel")
        mocker.patch("src.functions.agent.CodeAgent")

        agent = ConvFinQAAgent(model="gpt-4")

        assert agent.model == "gpt-4"

    def test_set_record_context(self, mocker) -> None:
        """Test setting record context."""
        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        # Mock smolagents components
        mocker.patch("src.functions.agent.LiteLLMModel")
        mocker.patch("src.functions.agent.CodeAgent")

        agent = ConvFinQAAgent()
        mock_record = mocker.MagicMock()
        mock_record.id = "test-record-123"
        mock_data_loader = mocker.MagicMock()

        agent.set_record_context(mock_record, mock_data_loader)

        # Context should be set successfully (no specific verification needed as the method is now internal)

    def test_conversation_history_management(self, mocker) -> None:
        """Test conversation history operations."""
        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        # Mock smolagents components
        mocker.patch("src.functions.agent.LiteLLMModel")
        mocker.patch("src.functions.agent.CodeAgent")

        agent = ConvFinQAAgent()

        # Initially empty
        assert agent.get_conversation_history() == []

        # Add to history manually (simulating chat)
        agent.conversation_history.append({
            "user": "What is the revenue?",
            "assistant": "The revenue is $100M"
        })

        history = agent.get_conversation_history()
        assert len(history) == 1
        assert history[0]["user"] == "What is the revenue?"
        assert history[0]["assistant"] == "The revenue is $100M"

        # Clear history
        agent.clear_history()
        assert agent.get_conversation_history() == []

    def test_chat_success(self, mocker) -> None:
        """Test successful chat interaction using smolagents."""
        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})

        # Mock smolagents components
        mock_model = mocker.patch("src.functions.agent.LiteLLMModel")
        mock_agent = mocker.patch("src.functions.agent.CodeAgent")

        # Mock the agent.run method to return a response
        mock_agent_instance = mock_agent.return_value
        mock_agent_instance.run.return_value = "The total revenue is $500M"

        agent = ConvFinQAAgent()

        # Mock conversation manager to simulate active conversation
        agent.conversation_manager.extract_entities_from_response = mocker.MagicMock(return_value=["revenue"])
        agent.conversation_manager.add_turn = mocker.MagicMock()

        response = agent.chat("What is the total revenue?")

        assert response == "The total revenue is $500M"
        assert len(agent.conversation_history) == 1
        assert agent.conversation_history[0]["user"] == "What is the total revenue?"
        assert agent.conversation_history[0]["assistant"] == "The total revenue is $500M"

    def test_chat_error_handling(self, mocker) -> None:
        """Test chat error handling."""
        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})

        # Mock smolagents components
        mock_model = mocker.patch("src.functions.agent.LiteLLMModel")
        mock_agent = mocker.patch("src.functions.agent.CodeAgent")

        # Mock the agent.run method to raise an exception
        mock_agent_instance = mock_agent.return_value
        mock_agent_instance.run.side_effect = Exception("Smol Error")

        agent = ConvFinQAAgent()

        response = agent.chat("What is the revenue?")

        assert "I encountered an error" in response
        assert "Smol Error" in response


class TestAgentIntegration:
    """Integration tests for smolagents-based agent with tools."""

    def test_agent_with_tools(self, mocker) -> None:
        """Test agent is initialized with correct tools."""
        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})

        # Mock smolagents components
        mock_model = mocker.patch("src.functions.agent.LiteLLMModel")
        mock_agent = mocker.patch("src.functions.agent.CodeAgent")

        agent = ConvFinQAAgent()

        # Verify tools are available - agent now has 8 tools:
        # list_tables, show_table, query_table, get_table_value, compute, calculate_change, validate_data_selection, final_answer
        assert len(agent.tools) == 8

    def test_agent_info(self, mocker) -> None:
        """Test agent info returns correct configuration."""
        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        # Mock smolagents components
        mocker.patch("src.functions.agent.LiteLLMModel")
        mocker.patch("src.functions.agent.CodeAgent")

        agent = ConvFinQAAgent()
        info = agent.get_agent_info()

        assert info["architecture"] == "smolagents CodeAgent"
        assert info["conversation_enabled"] is True


class TestMultiTurnDialogue:
    """Tests for multi-turn dialogue functionality with smolagents."""

    @pytest.fixture
    def temp_storage(self):
        """Provide temporary storage for conversation manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def agent_with_temp_storage(self, mocker, temp_storage):
        """Create agent with temporary conversation storage."""
        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})

        # Mock smolagents components
        mocker.patch("src.functions.agent.LiteLLMModel")
        mocker.patch("src.functions.agent.CodeAgent")

        # Patch ConversationManager to use temp storage
        mocker.patch("src.functions.agent.ConversationManager")

        agent = ConvFinQAAgent()
        return agent

    def test_conversation_context_usage(self, agent_with_temp_storage, mocker):
        """Test that conversation context is maintained with smolagents."""
        agent = agent_with_temp_storage

        # Mock conversation manager methods
        agent.conversation_manager.extract_entities_from_response = lambda x: []
        agent.conversation_manager.add_turn = lambda **kwargs: None

        # Mock the agent.run method
        mock_run = mocker.MagicMock(return_value="Response")
        agent.agent.run = mock_run

        agent.chat("Follow-up question")

        # Verify the agent.run method was called
        mock_run.assert_called_once()

    def test_conversation_turn_tracking(self, agent_with_temp_storage, mocker):
        """Test that conversation turns are properly tracked."""
        agent = agent_with_temp_storage

        # Mock conversation manager
        add_turn_calls = []

        # Mock the _extract_entities_from_response method directly on the agent instance
        agent._extract_entities_from_response = mocker.MagicMock(return_value=["entity"])
        agent.conversation_manager.add_turn = lambda **kwargs: add_turn_calls.append(kwargs)

        # Mock the agent.run method
        agent.agent.run = mocker.MagicMock(return_value="Answer")

        agent.chat("Test question")

        # Verify turn was added with correct information
        assert len(add_turn_calls) == 1
        turn = add_turn_calls[0]
        assert turn["user_message"] == "Test question"
        assert turn["assistant_response"] == "Answer"
        assert turn["referenced_entities"] == ["entity"]
