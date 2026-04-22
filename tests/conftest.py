import pytest
import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, Mock

from utils.config import Settings
from core.conversation_manager import ConversationManager
from core.task_state import TaskStateManager


@pytest.fixture
def test_settings() -> Settings:
    """Fixture for test settings."""
    return Settings(
        openai_api_key="test-key",
        log_level="DEBUG",
        mcp_servers="",
        rag_enabled=False,
        database_url="sqlite:///:memory:",
    )


@pytest.fixture
def mock_agent(test_settings: Settings):
    """Fixture for a mock agent."""
    return Mock()


@pytest.fixture
async def conversation_manager() -> AsyncGenerator[ConversationManager, None]:
    """Fixture for conversation manager."""
    manager = ConversationManager()
    yield manager
    await manager.close()


@pytest.fixture
async def task_state() -> AsyncGenerator[TaskStateManager, None]:
    """Fixture for task state."""
    state = TaskStateManager()
    yield state


@pytest.fixture
def mcp_orchestrator(test_settings: Settings):
    """Fixture for MCP orchestrator."""
    return Mock()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_client():
    """Fixture for async HTTP client."""
    import httpx
    async with httpx.AsyncClient() as client:
        yield client