import pytest
import json
from datetime import datetime
from core.conversation_manager import ConversationManager, Message, Conversation


class TestMessage:
    """Test Message model."""
    
    def test_message_creation(self):
        """Test basic message creation."""
        message = Message(
            role="user",
            content="Hello, world!",
            timestamp=datetime.now()
        )
        
        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert isinstance(message.timestamp, datetime)
        assert message.function_call is None
        
    def test_message_to_dict(self):
        """Test message serialization to dictionary."""
        timestamp = datetime.now()
        message = Message(
            role="assistant",
            content="How can I help you?",
            timestamp=timestamp
        )
        
        message_dict = message.model_dump()
        
        assert message_dict["role"] == "assistant"
        assert message_dict["content"] == "How can I help you?"
        assert message_dict["timestamp"] == timestamp.isoformat()
        
    def test_message_with_function_call(self):
        """Test message with function call."""
        message = Message(
            role="assistant",
            content="",
            function_call={
                "name": "search_database",
                "arguments": '{"query": "test"}'
            }
        )
        
        assert message.function_call is not None
        assert message.function_call["name"] == "search_database"
        assert '{"query": "test"}' in message.function_call["arguments"]


class TestConversation:
    """Test Conversation model."""
    
    def test_conversation_creation(self):
        """Test basic conversation creation."""
        conversation = Conversation(
            conversation_id="test-conv-123",
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!")
            ]
        )
        
        assert conversation.conversation_id == "test-conv-123"
        assert len(conversation.messages) == 2
        assert conversation.messages[0].role == "user"
        assert conversation.messages[1].role == "assistant"
        
    def test_add_message(self):
        """Test adding messages to conversation."""
        conversation = Conversation(conversation_id="test")
        
        # Add first message
        conversation.add_message("user", "First message")
        assert len(conversation.messages) == 1
        assert conversation.messages[0].content == "First message"
        
        # Add second message
        conversation.add_message("assistant", "Second message")
        assert len(conversation.messages) == 2
        assert conversation.messages[1].content == "Second message"
        
    def test_get_messages_for_api(self):
        """Test converting messages for API format."""
        conversation = Conversation(
            conversation_id="test",
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi"),
                Message(role="user", content="How are you?")
            ]
        )
        
        api_messages = conversation.get_messages_for_api()
        
        assert len(api_messages) == 3
        assert api_messages[0]["role"] == "user"
        assert api_messages[0]["content"] == "Hello"
        assert api_messages[1]["role"] == "assistant"
        assert api_messages[1]["content"] == "Hi"


@pytest.mark.asyncio
class TestConversationManager:
    """Test ConversationManager functionality."""
    
    async def test_create_conversation(self, conversation_manager: ConversationManager):
        """Test creating a new conversation."""
        conv_id = await conversation_manager.create_conversation()
        
        assert conv_id is not None
        conversation = await conversation_manager.get_conversation(conv_id)
        assert conversation is not None
        assert len(conversation.messages) == 0
        
    async def test_add_and_get_message(self, conversation_manager: ConversationManager):
        """Test adding and retrieving messages."""
        conv_id = await conversation_manager.create_conversation()
        
        # Add a message
        message_id = await conversation_manager.add_message(
            conv_id, "user", "Test message"
        )
        
        assert message_id is not None
        
        # Get the conversation and verify message
        conversation = await conversation_manager.get_conversation(conv_id)
        assert len(conversation.messages) == 1
        assert conversation.messages[0].content == "Test message"
        assert conversation.messages[0].role == "user"
        
    async def test_multiple_messages(self, conversation_manager: ConversationManager):
        """Test adding multiple messages to conversation."""
        conv_id = await conversation_manager.create_conversation()
        
        messages = [
            ("user", "Hello"),
            ("assistant", "Hi there!"),
            ("user", "How can you help me?"),
        ]
        
        for role, content in messages:
            await conversation_manager.add_message(conv_id, role, content)
            
        conversation = await conversation_manager.get_conversation(conv_id)
        assert len(conversation.messages) == 3
        
        # Verify messages are in correct order
        assert conversation.messages[0].content == "Hello"
        assert conversation.messages[1].content == "Hi there!"
        assert conversation.messages[2].content == "How can you help me?"
        
    async def test_get_nonexistent_conversation(self, conversation_manager: ConversationManager):
        """Test getting a non-existent conversation."""
        conversation = await conversation_manager.get_conversation("non-existent-id")
        assert conversation is None
        
    async def test_delete_conversation(self, conversation_manager: ConversationManager):
        """Test deleting a conversation."""
        conv_id = await conversation_manager.create_conversation()
        
        # Add some messages
        await conversation_manager.add_message(conv_id, "user", "Test")
        
        # Delete the conversation
        result = await conversation_manager.delete_conversation(conv_id)
        assert result is True
        
        # Verify it's deleted
        conversation = await conversation_manager.get_conversation(conv_id)
        assert conversation is None
        
    async def test_delete_nonexistent_conversation(self, conversation_manager: ConversationManager):
        """Test deleting a non-existent conversation."""
        result = await conversation_manager.delete_conversation("non-existent-id")
        assert result is False
        
    async def test_conversation_history(self, conversation_manager: ConversationManager):
        """Test retrieving conversation history."""
        # Create multiple conversations
        conv_ids = []
        for i in range(3):
            conv_id = await conversation_manager.create_conversation()
            conv_ids.append(conv_id)
            await conversation_manager.add_message(conv_id, "user", f"Message {i}")
            
        # Get history
        history = await conversation_manager.get_conversation_history()
        
        assert len(history) >= 3  # Might include default conversation
        
        # Verify our conversations are in history
        history_ids = [conv.conversation_id for conv in history]
        for conv_id in conv_ids:
            assert conv_id in history_ids