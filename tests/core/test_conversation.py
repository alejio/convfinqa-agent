"""
Tests for conversation state management and multi-turn dialogue.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.core.conversation import ConversationManager
from src.core.models import ConversationState, ConversationTurn


class TestConversationManager:
    """Tests for ConversationManager class."""

    @pytest.fixture
    def temp_storage(self):
        """Provide a temporary storage directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def conversation_manager(self, temp_storage):
        """Provide a ConversationManager with temporary storage."""
        return ConversationManager(storage_dir=temp_storage)

    def test_start_new_conversation(self, conversation_manager):
        """Test starting a new conversation."""
        record_id = "test_record_123"
        state = conversation_manager.start_conversation(record_id)

        assert state.record_id == record_id
        assert len(state.turns) == 0
        assert state.session_id is not None
        assert conversation_manager.current_state == state

    def test_start_conversation_with_session_id(self, conversation_manager):
        """Test starting a conversation with specific session ID."""
        record_id = "test_record_123"
        session_id = "custom_session_123"

        state = conversation_manager.start_conversation(record_id, session_id)

        assert state.record_id == record_id
        assert state.session_id == session_id

    def test_add_conversation_turn(self, conversation_manager):
        """Test adding a turn to the conversation."""
        # Start conversation
        conversation_manager.start_conversation("test_record")

        # Add a turn
        turn = conversation_manager.add_turn(
            user_message="What is the revenue?",
            assistant_response="The revenue is $100M.",
            tool_calls=[{"tool_name": "query_table", "call_expression": "query_table('revenue', 'table1')"}],
            referenced_entities=["revenue"]
        )

        assert turn.user_message == "What is the revenue?"
        assert turn.assistant_response == "The revenue is $100M."
        assert len(turn.tool_calls) == 1
        assert "revenue" in turn.referenced_entities
        assert len(conversation_manager.current_state.turns) == 1

    def test_conversation_context_generation(self, conversation_manager):
        """Test conversation context generation."""
        # Start conversation
        conversation_manager.start_conversation("test_record")

        # Add multiple turns
        conversation_manager.add_turn(
            "What is the revenue?",
            "Revenue is $100M.",
            referenced_entities=["revenue"]
        )
        conversation_manager.add_turn(
            "What about profit?",
            "Profit is $20M.",
            referenced_entities=["profit"]
        )

        context = conversation_manager.get_conversation_context()
        assert "Recent conversation history:" in context
        assert "What is the revenue?" in context
        assert "What about profit?" in context

    def test_entity_extraction(self, conversation_manager, mocker):
        """Test entity extraction from response."""
        # Mock the DSPy entity extraction to return expected entities
        mock_instance = mocker.patch("src.core.financial_terms.get_financial_terms_instance")
        mock_extractor = mocker.MagicMock()
        mock_extractor.extract_entities.return_value = ["revenue", "profit", "Q1", "2023"]
        mock_instance.return_value = mock_extractor

        entities = conversation_manager.extract_entities_from_response(
            "The revenue in Q1 2023 was $100M, showing strong profit margins."
        )

        expected_entities = {"revenue", "profit", "Q1", "2023"}
        assert expected_entities.issubset(set(entities))

    def test_session_persistence(self, conversation_manager, temp_storage):
        """Test session saving and loading."""
        # Start conversation and add turns
        conversation_manager.start_conversation("test_record")
        session_id = conversation_manager.current_state.session_id

        conversation_manager.add_turn(
            "Test question",
            "Test response",
            referenced_entities=["test"]
        )

        # Verify session file exists
        session_file = Path(temp_storage) / f"{session_id}.json"
        assert session_file.exists()

        # Load session and verify content
        with open(session_file) as f:
            data = json.load(f)

        assert data["session_id"] == session_id
        assert data["record_id"] == "test_record"
        assert len(data["turns"]) == 1

    def test_resume_conversation(self, conversation_manager, temp_storage):
        """Test resuming a previous conversation."""
        # Create and save a conversation
        original_session_id = "test_session_123"
        conversation_manager.start_conversation("test_record", original_session_id)
        conversation_manager.add_turn("Original question", "Original response")

        # Create new manager and resume
        new_manager = ConversationManager(storage_dir=temp_storage)
        resumed_state = new_manager.start_conversation("test_record", original_session_id)

        assert resumed_state.session_id == original_session_id
        assert len(resumed_state.turns) == 1
        assert resumed_state.turns[0].user_message == "Original question"

    def test_list_sessions(self, conversation_manager):
        """Test listing available sessions."""
        # Create multiple sessions
        conversation_manager.start_conversation("record1", "session1")
        conversation_manager.add_turn("Q1", "A1")

        conversation_manager.start_conversation("record2", "session2")
        conversation_manager.add_turn("Q2", "A2")

        sessions = conversation_manager.list_sessions()
        assert len(sessions) == 2

        session_ids = [s[0] for s in sessions]
        assert "session1" in session_ids
        assert "session2" in session_ids

    def test_delete_session(self, conversation_manager):
        """Test deleting a session."""
        # Create session
        conversation_manager.start_conversation("test_record", "deletable_session")
        conversation_manager.add_turn("Test", "Response")

        # Delete session
        result = conversation_manager.delete_session("deletable_session")
        assert result is True

        # Verify session no longer exists
        sessions = conversation_manager.list_sessions()
        session_ids = [s[0] for s in sessions]
        assert "deletable_session" not in session_ids


class TestConversationModels:
    """Tests for conversation data models."""

    def test_conversation_turn_creation(self):
        """Test creating a ConversationTurn."""
        turn = ConversationTurn(
            turn_id="turn_1",
            user_message="What is the revenue?",
            assistant_response="Revenue is $100M",
            timestamp=datetime.now(),
            tool_calls=[{"tool_name": "query_table"}],
            referenced_entities=["revenue"],
            computation_results=[{"result": "100000000"}]
        )

        assert turn.turn_id == "turn_1"
        assert turn.user_message == "What is the revenue?"
        assert len(turn.tool_calls) == 1
        assert "revenue" in turn.referenced_entities

    def test_conversation_state_context_tracking(self):
        """Test conversation state context tracking."""
        state = ConversationState(
            session_id="test_session",
            record_id="test_record",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        # Add turn with entities
        turn = ConversationTurn(
            turn_id="turn_1",
            user_message="Show revenue",
            assistant_response="Revenue is $100M",
            timestamp=datetime.now(),
            referenced_entities=["revenue", "financial_table"]
        )

        state.add_turn(turn)

        # Check entity references are tracked
        assert "revenue" in state.entity_references
        assert state.entity_references["revenue"] == ["turn_1"]
        assert state.find_entity_references("revenue") == ["turn_1"]

    def test_conversation_context_generation(self):
        """Test conversation context string generation."""
        state = ConversationState(
            session_id="test_session",
            record_id="test_record",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        # Add multiple turns
        for i in range(3):
            turn = ConversationTurn(
                turn_id=f"turn_{i+1}",
                user_message=f"Question {i+1}",
                assistant_response=f"Answer {i+1}",
                timestamp=datetime.now()
            )
            state.add_turn(turn)

        context = state.get_conversation_context(max_turns=2)

        # Should only include last 2 turns
        assert "Question 2" in context
        assert "Question 3" in context
        assert "Question 1" not in context
        assert "Recent conversation history:" in context

    def test_empty_conversation_context(self):
        """Test context generation for empty conversation."""
        state = ConversationState(
            session_id="test_session",
            record_id="test_record",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        context = state.get_conversation_context()
        assert context == "This is the start of a new conversation."
