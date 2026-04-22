import pytest
import os
from unittest.mock import patch, mock_open
from utils.config import Settings


class TestSettings:
    """Test Settings configuration."""
    
    def test_default_settings(self):
        """Test that default settings are properly set."""
        settings = Settings()
        
        assert settings.openai_api_key == ""
        assert settings.log_level == "INFO"
        assert settings.mcp_servers == ""
        assert settings.rag_enabled is True
        assert settings.database_url == "sqlite:///logsit.db"
        assert settings.port == 7860
        
    def test_settings_from_env(self):
        """Test settings loaded from environment variables."""
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-api-key",
            "LOG_LEVEL": "DEBUG",
            "MCP_SERVERS": "server1,server2",
            "RAG_ENABLED": "false",
            "DATABASE_URL": "sqlite:///test.db",
            "PORT": "8000",
        }):
            settings = Settings()
            
            assert settings.openai_api_key == "test-api-key"
            assert settings.log_level == "DEBUG"
            assert settings.mcp_servers == "server1,server2"
            assert settings.rag_enabled is False
            assert settings.database_url == "sqlite:///test.db"
            assert settings.port == 8000
            
    def test_settings_validation(self):
        """Test settings validation."""
        # Test invalid port
        with pytest.raises(ValueError):
            Settings(port=-1)
        
        # Test invalid log level
        with pytest.raises(ValueError):
            Settings(log_level="INVALID")
            
    def test_settings_to_dict(self):
        """Test serialization to dictionary."""
        settings = Settings(
            openai_api_key="test-key",
            log_level="DEBUG",
            mcp_servers="server1",
            rag_enabled=True,
            database_url="sqlite:///test.db",
            port=7860,
        )
        
        settings_dict = settings.model_dump()
        
        assert settings_dict["openai_api_key"] == "test-key"
        assert settings_dict["log_level"] == "DEBUG"
        assert settings_dict["mcp_servers"] == "server1"
        assert settings_dict["rag_enabled"] is True
        assert settings_dict["database_url"] == "sqlite:///test.db"
        assert settings_dict["port"] == 7860
        
    def test_settings_model_dump_json(self):
        """Test JSON serialization."""
        settings = Settings(
            openai_api_key="test-key",
            log_level="DEBUG",
        )
        
        json_str = settings.model_dump_json()
        
        assert '"openai_api_key": "test-key"' in json_str
        assert '"log_level": "DEBUG"' in json_str
        assert '"rag_enabled": true' in json_str