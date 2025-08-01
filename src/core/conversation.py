"""
Conversation state management for multi-turn dialogue.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.logger import get_logger
from .models import ConversationState, ConversationTurn

logger = get_logger(__name__)


class ConversationManager:
    """Manages conversation state and persistence for multi-turn dialogue."""

    def __init__(self, storage_dir: str = ".convfinqa_sessions"):
        """Initialize conversation manager with storage directory.

        Args:
            storage_dir: Directory to store conversation session files.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.current_state: ConversationState | None = None

    def start_conversation(
        self, record_id: str, session_id: str | None = None
    ) -> ConversationState:
        """Start a new conversation or resume existing one.

        Args:
            record_id: ID of the record being discussed.
            session_id: Optional session ID to resume. If None, creates new session.

        Returns:
            The conversation state.
        """
        if session_id and self._session_exists(session_id):
            self.current_state = self._load_session(session_id)
            logger.info(f"Resumed conversation session: {session_id}")
        else:
            new_session_id = session_id or str(uuid.uuid4())
            self.current_state = ConversationState(
                session_id=new_session_id,
                record_id=record_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            logger.info(f"Started new conversation session: {new_session_id}")

        return self.current_state

    def add_turn(
        self,
        user_message: str,
        assistant_response: str,
        tool_calls: list[dict[str, Any]] | None = None,
        referenced_entities: list[str] | None = None,
        computation_results: list[dict[str, Any]] | None = None,
    ) -> ConversationTurn:
        """Add a new turn to the current conversation.

        Args:
            user_message: The user's message.
            assistant_response: The assistant's response.
            tool_calls: List of tool calls made during this turn.
            referenced_entities: Entities referenced in this turn.
            computation_results: Results from computations in this turn.

        Returns:
            The created conversation turn.
        """
        if not self.current_state:
            raise ValueError("No active conversation. Start a conversation first.")

        turn_id = f"turn_{len(self.current_state.turns) + 1}"
        turn = ConversationTurn(
            turn_id=turn_id,
            user_message=user_message,
            assistant_response=assistant_response,
            timestamp=datetime.now(),
            tool_calls=tool_calls or [],
            referenced_entities=referenced_entities or [],
            computation_results=computation_results or [],
        )

        self.current_state.add_turn(turn)
        self._save_session()

        logger.info(f"Added conversation turn: {turn_id}")
        return turn

    def get_conversation_context(self, max_turns: int = 5) -> str:
        """Get conversation context for the LLM.

        Args:
            max_turns: Maximum number of recent turns to include.

        Returns:
            Formatted conversation context string.
        """
        if not self.current_state:
            return "This is the start of a new conversation."

        return self.current_state.get_conversation_context(max_turns)

    def extract_entities_from_response(self, response: str) -> list[str]:
        """Extract referenced entities from an assistant response.

        This is a simple implementation that can be enhanced with NLP.

        Args:
            response: The assistant's response text.

        Returns:
            List of extracted entity references.
        """
        entities = []

        # Simple keyword-based entity extraction
        financial_keywords = [
            "revenue",
            "profit",
            "loss",
            "assets",
            "liabilities",
            "equity",
            "cash flow",
            "expenses",
            "income",
            "margin",
            "quarter",
            "year",
            "Q1",
            "Q2",
            "Q3",
            "Q4",
            "2019",
            "2020",
            "2021",
            "2022",
            "2023",
        ]

        response_lower = response.lower()
        for keyword in financial_keywords:
            if keyword.lower() in response_lower:
                entities.append(keyword)

        # Look for table references
        if "table" in response_lower:
            entities.append("financial_table")

        return entities

    def _session_exists(self, session_id: str) -> bool:
        """Check if a session file exists."""
        session_file = self.storage_dir / f"{session_id}.json"
        return session_file.exists()

    def _load_session(self, session_id: str) -> ConversationState:
        """Load conversation state from file."""
        session_file = self.storage_dir / f"{session_id}.json"
        try:
            with open(session_file) as f:
                data = json.load(f)
            return ConversationState.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            raise

    def _save_session(self) -> None:
        """Save current conversation state to file."""
        if not self.current_state:
            return

        session_file = self.storage_dir / f"{self.current_state.session_id}.json"
        try:
            with open(session_file, "w") as f:
                json.dump(
                    self.current_state.model_dump(mode="json"),
                    f,
                    indent=2,
                    default=str,  # Handle datetime serialization
                )
            logger.debug(f"Saved session: {self.current_state.session_id}")
        except Exception as e:
            logger.error(f"Failed to save session {self.current_state.session_id}: {e}")

    def list_sessions(self) -> list[tuple[str, datetime, str]]:
        """List all available sessions.

        Returns:
            List of tuples: (session_id, created_at, record_id)
        """
        sessions = []
        for session_file in self.storage_dir.glob("*.json"):
            try:
                with open(session_file) as f:
                    data = json.load(f)
                sessions.append(
                    (
                        data["session_id"],
                        datetime.fromisoformat(data["created_at"]),
                        data["record_id"],
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to read session file {session_file}: {e}")

        return sorted(sessions, key=lambda x: x[1], reverse=True)

    def delete_session(self, session_id: str) -> bool:
        """Delete a conversation session.

        Args:
            session_id: ID of the session to delete.

        Returns:
            True if deleted successfully, False otherwise.
        """
        session_file = self.storage_dir / f"{session_id}.json"
        try:
            session_file.unlink()
            logger.info(f"Deleted session: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
