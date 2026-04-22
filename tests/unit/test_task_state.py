import pytest
import json
from datetime import datetime
from core.task_state import TaskStateManager, Task, TaskStatus


class TestTask:
    """Test Task model."""
    
    def test_task_creation(self):
        """Test basic task creation."""
        task = Task(
            task_id="task-123",
            description="Test task"
        )
        
        assert task.task_id == "task-123"
        assert task.description == "Test task"
        assert task.status == TaskStatus.PENDING
        assert isinstance(task.created_at, datetime)
        assert task.last_updated is not None
        assert task.context == {}
        assert task.results == {}
        
    def test_task_with_context(self):
        """Test task with context."""
        task = Task(
            task_id="task-456",
            description="Task with context",
            context={"user_id": "123", "priority": "high"}
        )
        
        assert task.context["user_id"] == "123"
        assert task.context["priority"] == "high"
        
    def test_task_status_updates(self):
        """Test task status updates."""
        task = Task(
            task_id="task-789",
            description="Task for status testing"
        )
        
        # Test initial status
        assert task.status == TaskStatus.PENDING
        
        # Update status
        task.update_status(TaskStatus.IN_PROGRESS)
        assert task.status == TaskStatus.IN_PROGRESS
        
        # Add result
        task.add_result("search", {"items": ["item1", "item2"]})
        assert "search" in task.results
        assert task.results["search"]["items"] == ["item1", "item2"]
        
        # Complete task
        task.update_status(TaskStatus.COMPLETED, "Task completed successfully")
        assert task.status == TaskStatus.COMPLETED
        assert task.completion_message == "Task completed successfully"
        
    def test_task_serialization(self):
        """Test task serialization to dictionary."""
        task = Task(
            task_id="task-serial",
            description="Serialization test",
            context={"test": "data"}
        )
        
        task_dict = task.model_dump()
        
        assert task_dict["task_id"] == "task-serial"
        assert task_dict["description"] == "Serialization test"
        assert task_dict["status"] == "pending"
        assert task_dict["context"] == {"test": "data"}
        
    def test_task_from_dict(self):
        """Test creating task from dictionary."""
        task_data = {
            "task_id": "task-from-dict",
            "description": "Task from dict",
            "status": "in_progress",
            "context": {"key": "value"},
            "results": {"step1": {"result": "data"}}
        }
        
        task = Task(**task_data)
        
        assert task.task_id == "task-from-dict"
        assert task.description == "Task from dict"
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.context["key"] == "value"
        assert task.results["step1"]["result"] == "data"


@pytest.mark.asyncio
class TestTaskState:
    """Test TaskState functionality."""
    
    async def test_create_task(self, task_state: TaskStateManager):
        """Test creating a new task."""
        task_id = await task_state.create_task("Test task description")
        
        assert task_id is not None
        
        task = await task_state.get_task(task_id)
        assert task is not None
        assert task.description == "Test task description"
        assert task.status == TaskStatus.PENDING
        
    async def test_create_task_with_context(self, task_state: TaskState):
        """Test creating a task with context."""
        context = {
            "user_id": "user123",
            "session_id": "session456",
            "priority": "high"
        }
        
        task_id = await task_state.create_task(
            description="Task with context",
            context=context
        )
        
        task = await task_state.get_task(task_id)
        assert task.context["user_id"] == "user123"
        assert task.context["session_id"] == "session456"
        assert task.context["priority"] == "high"
        
    async def test_update_task_status(self, task_state: TaskState):
        """Test updating task status."""
        task_id = await task_state.create_task("Status update test")
        
        # Update to in progress
        success = await task_state.update_task_status(
            task_id, TaskStatus.IN_PROGRESS
        )
        assert success is True
        
        task = await task_state.get_task(task_id)
        assert task.status == TaskStatus.IN_PROGRESS
        
        # Update to completed with message
        success = await task_state.update_task_status(
            task_id,
            TaskStatus.COMPLETED,
            "Task completed successfully"
        )
        assert success is True
        
        task = await task_state.get_task(task_id)
        assert task.status == TaskStatus.COMPLETED
        assert task.completion_message == "Task completed successfully"
        
    async def test_update_nonexistent_task(self, task_state: TaskState):
        """Test updating status of non-existent task."""
        success = await task_state.update_task_status(
            "non-existent-id",
            TaskStatus.IN_PROGRESS
        )
        assert success is False
        
    async def test_add_task_result(self, task_state: TaskState):
        """Test adding results to a task."""
        task_id = await task_state.create_task("Task with results")
        
        # Add initial result
        await task_state.add_task_result(
            task_id,
            "search",
            {"query": "test", "results": ["item1", "item2"]}
        )
        
        task = await task_state.get_task(task_id)
        assert "search" in task.results
        assert task.results["search"]["query"] == "test"
        assert len(task.results["search"]["results"]) == 2
        
        # Add another result
        await task_state.add_task_result(
            task_id,
            "analysis",
            {"summary": "Analysis complete", "score": 0.95}
        )
        
        task = await task_state.get_task(task_id)
        assert "analysis" in task.results
        assert task.results["analysis"]["score"] == 0.95
        
    async def test_get_active_tasks(self, task_state: TaskState):
        """Test retrieving active tasks."""
        # Create tasks with different statuses
        pending_id = await task_state.create_task("Pending task")
        in_progress_id = await task_state.create_task("In progress task")
        completed_id = await task_state.create_task("Completed task")
        
        # Update statuses
        await task_state.update_task_status(in_progress_id, TaskStatus.IN_PROGRESS)
        await task_state.update_task_status(completed_id, TaskStatus.COMPLETED)
        
        # Get active tasks (pending + in_progress)
        active_tasks = await task_state.get_active_tasks()
        
        active_ids = [task.task_id for task in active_tasks]
        assert pending_id in active_ids
        assert in_progress_id in active_ids
        assert completed_id not in active_ids
        
    async def test_get_task_history(self, task_state: TaskState):
        """Test retrieving task history."""
        # Create multiple tasks
        task_ids = []
        for i in range(5):
            task_id = await task_state.create_task(f"Task {i}")
            task_ids.append(task_id)
            
        # Update some tasks to completed
        await task_state.update_task_status(task_ids[0], TaskStatus.COMPLETED)
        await task_state.update_task_status(task_ids[2], TaskStatus.COMPLETED)
        
        # Get all tasks (history)
        all_tasks = await task_state.get_task_history()
        
        # Should have at least our 5 tasks
        assert len(all_tasks) >= 5
        
        # Check that our tasks are in the history
        history_ids = [task.task_id for task in all_tasks]
        for task_id in task_ids:
            assert task_id in history_ids
            
    async def test_cleanup_completed_tasks(self, task_state: TaskState):
        """Test cleaning up old completed tasks."""
        # Create and complete some tasks
        old_task_ids = []
        for i in range(3):
            task_id = await task_state.create_task(f"Old task {i}")
            await task_state.update_task_status(task_id, TaskStatus.COMPLETED)
            old_task_ids.append(task_id)
            
        # Create some active tasks
        active_task_ids = []
        for i in range(2):
            task_id = await task_state.create_task(f"Active task {i}")
            if i == 0:
                await task_state.update_task_status(task_id, TaskStatus.IN_PROGRESS)
            active_task_ids.append(task_id)
            
        # Run cleanup (in real scenario this would check timestamps)
        # For testing, we just verify the method exists and runs
        await task_state.cleanup_completed_tasks(max_age_days=1)
        
        # Verify active tasks still exist
        for task_id in active_task_ids:
            task = await task_state.get_task(task_id)
            assert task is not None