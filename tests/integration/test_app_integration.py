import pytest
import json
import asyncio
from httpx import AsyncClient
from unittest.mock import AsyncMock, Mock, patch

from app.main import app
from utils.config import Settings


@pytest.mark.asyncio
class TestAppIntegration:
    """Integration tests for the main application."""
    
    async def test_health_endpoint(self, async_client: AsyncClient):
        """Test health check endpoint."""
        response = await async_client.get("http://test/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        
    async def test_conversation_endpoints(self, async_client: AsyncClient):
        """Test conversation-related endpoints."""
        # Create a new conversation
        response = await async_client.post("/api/conversations/")
        assert response.status_code == 200
        data = response.json()
        assert "conversation_id" in data
        
        conversation_id = data["conversation_id"]
        
        # Get the conversation
        response = await async_client.get(f"/api/conversations/{conversation_id}")
        assert response.status_code == 200
        conv_data = response.json()
        assert conv_data["conversation_id"] == conversation_id
        assert conv_data["messages"] == []
        
        # Add a message
        message_data = {
            "role": "user",
            "content": "Hello, world!"
        }
        response = await async_client.post(
            f"/api/conversations/{conversation_id}/messages",
            json=message_data
        )
        assert response.status_code == 200
        msg_data = response.json()
        assert "message_id" in msg_data
        
        # Get conversation again to verify message was added
        response = await async_client.get(f"/api/conversations/{conversation_id}")
        assert response.status_code == 200
        conv_data = response.json()
        assert len(conv_data["messages"]) == 1
        assert conv_data["messages"][0]["role"] == "user"
        assert conv_data["messages"][0]["content"] == "Hello, world!"
        
        # Get conversation history
        response = await async_client.get("/api/conversations/")
        assert response.status_code == 200
        history_data = response.json()
        assert isinstance(history_data, list)
        
        # Delete the conversation
        response = await async_client.delete(f"/api/conversations/{conversation_id}")
        assert response.status_code == 200
        delete_data = response.json()
        assert delete_data["success"] is True
        
    async def test_task_endpoints(self, async_client: AsyncClient):
        """Test task-related endpoints."""
        # Create a new task
        task_data = {
            "description": "Test integration task",
            "context": {"test": "data"}
        }
        response = await async_client.post("/api/tasks/", json=task_data)
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        
        task_id = data["task_id"]
        
        # Get the task
        response = await async_client.get(f"/api/tasks/{task_id}")
        assert response.status_code == 200
        task_data = response.json()
        assert task_data["task_id"] == task_id
        assert task_data["description"] == "Test integration task"
        assert task_data["status"] == "pending"
        
        # Update task status
        update_data = {
            "status": "in_progress"
        }
        response = await async_client.put(
            f"/api/tasks/{task_id}/status",
            json=update_data
        )
        assert response.status_code == 200
        update_resp = response.json()
        assert update_resp["success"] is True
        
        # Get task to verify status update
        response = await async_client.get(f"/api/tasks/{task_id}")
        task_data = response.json()
        assert task_data["status"] == "in_progress"
        
        # Add task result
        result_data = {
            "key": "search_results",
            "result": {"items": ["item1", "item2"]}
        }
        response = await async_client.post(
            f"/api/tasks/{task_id}/results",
            json=result_data
        )
        assert response.status_code == 200
        
        # Get active tasks
        response = await async_client.get("/api/tasks/active")
        assert response.status_code == 200
        active_tasks = response.json()
        assert isinstance(active_tasks, list)
        
        # Mark task as completed
        complete_data = {
            "status": "completed",
            "completion_message": "Task completed successfully"
        }
        response = await async_client.put(
            f"/api/tasks/{task_id}/status",
            json=complete_data
        )
        assert response.status_code == 200
        
        # Get task history
        response = await async_client.get("/api/tasks/history")
        assert response.status_code == 200
        history = response.json()
        assert isinstance(history, list)
        
    @patch("openai.OpenAI")
    async def test_process_message(self, mock_openai, async_client: AsyncClient):
        """Test processing a message through the agent."""
        # Mock OpenAI response
        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=Mock(
                choices=[
                    Mock(
                        message=Mock(
                            content="Test AI response",
                            function_call=None
                        )
                    )
                ]
            )
        )
        mock_openai.return_value = mock_client
        
        # First create a conversation
        response = await async_client.post("/api/conversations/")
        conversation_id = response.json()["conversation_id"]
        
        # Process a message
        process_data = {
            "message": "What can you help me with?"
        }
        response = await async_client.post(
            f"/api/conversations/{conversation_id}/process",
            json=process_data
        )
        
        assert response.status_code == 200
        process_response = response.json()
        
        assert "response" in process_response
        assert process_response["response"] == "Test AI response"
        assert "conversation_id" in process_response
        assert process_response["conversation_id"] == conversation_id
        
    async def test_mcp_endpoints(self, async_client: AsyncClient):
        """Test MCP-related endpoints."""
        # Get MCP servers status
        response = await async_client.get("/api/mcp/servers")
        assert response.status_code == 200
        servers_data = response.json()
        assert isinstance(servers_data, list)
        
        # Get tools from MCP servers
        response = await async_client.get("/api/mcp/tools")
        assert response.status_code == 200
        tools_data = response.json()
        assert isinstance(tools_data, list)
        
    async def test_invalid_endpoints(self, async_client: AsyncClient):
        """Test handling of invalid requests."""
        # Non-existent endpoint
        response = await async_client.get("/api/nonexistent")
        assert response.status_code == 404
        
        # Invalid conversation ID
        response = await async_client.get("/api/conversations/invalid-id")
        assert response.status_code == 404
        
        # Invalid task ID
        response = await async_client.get("/api/tasks/invalid-id")
        assert response.status_code == 404
        
        # Invalid JSON in request
        response = await async_client.post(
            "/api/conversations/",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        
    async def test_cors_headers(self, async_client: AsyncClient):
        """Test CORS headers are present."""
        response = await async_client.options("http://test/health")
        
        # Check for CORS headers
        headers = response.headers
        assert "access-control-allow-origin" in headers
        assert "access-control-allow-methods" in headers
        assert "access-control-allow-headers" in headers
        
    async def test_error_handling(self, async_client: AsyncClient):
        """Test error handling in the application."""
        # Try to process message without conversation
        process_data = {
            "message": "Test message"
        }
        response = await async_client.post(
            "/api/conversations/invalid-id/process",
            json=process_data
        )
        
        # Should return appropriate error
        assert response.status_code in [404, 500]
        error_data = response.json()
        assert "error" in error_data or "detail" in error_data