import pytest
import sys
import os
from pathlib import Path

# Add the original path for backward compatibility testing
original_path = Path(__file__).parent.parent.parent / "route_planner"
sys.path.insert(0, str(original_path))


@pytest.mark.skipif(
    not (original_path / "enhanced_app.py").exists(),
    reason="Original enhanced_app.py not found"
)
class TestBackwardCompatibility:
    """Tests for backward compatibility with original enhanced_app.py."""
    
    def test_import_original_modules(self):
        """Test that original modules can still be imported."""
        try:
            # Try to import key modules from original structure
            from route_planner.enhanced_app import main as original_main
            from route_planner.enhanced_agent import EnhancedAgent
            
            # If we get here, imports succeeded
            assert original_main is not None
            assert EnhancedAgent is not None
            
        except ImportError as e:
            pytest.fail(f"Failed to import original modules: {e}")
            
    def test_compare_functionality(self):
        """Compare key functionality between old and new implementations."""
        # This test verifies that the same core functions exist
        # in both old and new implementations
        
        # Import original
        from route_planner.enhanced_agent import EnhancedAgent as OriginalAgent
        
        # Import new
        from core.agent import Agent as NewAgent
        
        # Check that both have essential methods
        original_methods = dir(OriginalAgent)
        new_methods = dir(NewAgent)
        
        # Essential methods that should exist in both
        essential_methods = [
            'process_message',
            'initialize',
            'close',
        ]
        
        for method in essential_methods:
            assert method in original_methods or f"_{method}" in original_methods, \
                f"Original Agent missing method: {method}"
            assert method in new_methods or f"_{method}" in new_methods, \
                f"New Agent missing method: {method}"
                
    def test_api_compatibility(self):
        """Test that the new API is compatible with the old one."""
        # Import FastAPI app from both versions
        from fastapi import FastAPI
        from unittest.mock import patch
        
        # Mock settings for new app
        with patch('utils.config.Settings') as mock_settings:
            mock_settings.return_value.openai_api_key = "test-key"
            
            # Import new app
            from app.main import app as new_app
            
            # Check new app has expected routes
            new_routes = []
            for route in new_app.routes:
                if hasattr(route, 'path'):
                    new_routes.append(route.path)
                    
            # Expected routes that should exist
            expected_routes = [
                '/health',
                '/api/conversations',
                '/api/conversations/{conversation_id}',
                '/api/conversations/{conversation_id}/messages',
                '/api/conversations/{conversation_id}/process',
                '/api/tasks',
                '/api/tasks/{task_id}',
                '/api/tasks/active',
                '/api/tasks/history',
                '/api/mcp/servers',
                '/api/mcp/tools',
            ]
            
            # Check all expected routes exist
            for expected_route in expected_routes:
                assert any(expected_route in route for route in new_routes), \
                    f"Missing route in new app: {expected_route}"
                    
    def test_data_structure_compatibility(self):
        """Test that data structures are compatible."""
        # Check that Conversation structure is similar
        from core.conversation_manager import Conversation as NewConversation
        
        # New conversation should have essential attributes
        new_conv = NewConversation(conversation_id="test")
        
        assert hasattr(new_conv, 'conversation_id')
        assert hasattr(new_conv, 'messages')
        assert hasattr(new_conv, 'add_message')
        assert hasattr(new_conv, 'get_messages_for_api')
        
        # Check Message structure
        from core.conversation_manager import Message as NewMessage
        
        new_msg = NewMessage(role="user", content="test")
        
        assert hasattr(new_msg, 'role')
        assert hasattr(new_msg, 'content')
        assert hasattr(new_msg, 'timestamp')
        
    def test_settings_compatibility(self):
        """Test that settings structure is compatible."""
        from utils.config import Settings as NewSettings
        
        # New settings should have essential configuration options
        new_settings = NewSettings()
        
        # Essential settings that should exist
        assert hasattr(new_settings, 'openai_api_key')
        assert hasattr(new_settings, 'log_level')
        assert hasattr(new_settings, 'mcp_servers')
        assert hasattr(new_settings, 'rag_enabled')
        assert hasattr(new_settings, 'database_url')
        assert hasattr(new_settings, 'port')
        
        # Check default values
        assert new_settings.openai_api_key == ""
        assert new_settings.log_level == "INFO"
        assert new_settings.rag_enabled is True
        assert new_settings.port == 7860
        
    def test_database_compatibility(self):
        """Test database compatibility."""
        from utils.database import DatabaseManager
        
        # Database manager should have essential methods
        db_manager = DatabaseManager("sqlite:///:memory:")
        
        assert hasattr(db_manager, 'initialize')
        assert hasattr(db_manager, 'close')
        assert hasattr(db_manager, 'execute_query')
        assert hasattr(db_manager, 'fetch_all')
        
    def test_async_helpers_compatibility(self):
        """Test async helpers compatibility."""
        from utils.async_helpers import AsyncHelper
        
        # Async helper should have essential methods
        async_helper = AsyncHelper()
        
        assert hasattr(async_helper, 'run_async')
        assert hasattr(async_helper, 'create_task')
        assert hasattr(async_helper, 'wait_all')
        
    @pytest.mark.asyncio
    async def test_async_operations(self):
        """Test that async operations work correctly."""
        # Test async initialization
        from core.agent import Agent
        from utils.config import Settings
        
        settings = Settings(openai_api_key="test-key")
        
        # Should be able to create agent async
        agent = Agent(settings=settings)
        
        # Should be able to initialize (even if mocked)
        await agent.initialize()
        
        # Should be able to close
        await agent.close()
        
    def test_dependency_structure(self):
        """Test that dependency structure is similar."""
        # Check that we have the same core dependencies
        import pkg_resources
        
        # Core dependencies that should be present
        core_dependencies = [
            'pydantic',
            'fastapi',
            'uvicorn',
            'openai',
            'aiofiles',
            'rich',
            'tenacity',
        ]
        
        installed_packages = {pkg.key for pkg in pkg_resources.working_set}
        
        # Check core dependencies are installed
        for dep in core_dependencies:
            assert dep in installed_packages, f"Missing core dependency: {dep}"
            
    def test_project_structure(self):
        """Test that project structure is correct."""
        # Check essential directories exist
        project_root = Path(__file__).parent.parent.parent
        
        essential_dirs = [
            'app',
            'core',
            'services',
            'utils',
            'docs',
            'tests',
        ]
        
        for dir_name in essential_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Missing directory: {dir_name}"
            assert dir_path.is_dir(), f"Not a directory: {dir_name}"
            
    def test_documentation_structure(self):
        """Test that documentation structure is correct."""
        project_root = Path(__file__).parent.parent.parent
        docs_dir = project_root / 'docs'
        
        essential_docs = [
            'USER_GUIDE.md',
            'ARCHITECTURE.md',
            'DEPLOYMENT.md',
        ]
        
        for doc_file in essential_docs:
            doc_path = docs_dir / doc_file
            assert doc_path.exists(), f"Missing documentation: {doc_file}"
            assert doc_path.is_file(), f"Not a file: {doc_file}"