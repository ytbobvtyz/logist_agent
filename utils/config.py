"""
Configuration management for Logsit Agent using Pydantic Settings.
"""

import json
import os
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openrouter_api_key: str = Field(default="", description="OpenRouter API key (alternative)")
    yandex_maps_api_key: str = Field(default="", description="Yandex Maps API key")
    
    # Database configuration
    database_url: str = Field(
        default="sqlite:///logsit.db",
        description="Database connection URL"
    )
    
    # MCP servers configuration
    mcp_servers: str = Field(
        default="",
        description="Comma-separated list of MCP server configurations in format 'name1:command1,name2:command2'"
    )
    
    # RAG configuration
    rag_enabled: bool = Field(
        default=True,
        description="Enable RAG (Retrieval Augmented Generation)"
    )
    rag_index_path: str = Field(
        default="data/faiss_index",
        description="Path to FAISS index for RAG"
    )
    rag_metadata_path: str = Field(
        default="data/metadata.db",
        description="Path to metadata database for RAG"
    )
    
    # LLM configuration
    default_model: str = Field(
        default="gpt-3.5-turbo",
        description="Default LLM model to use"
    )
    summarization_model: str = Field(
        default="gpt-3.5-turbo",
        description="LLM model for summarization"
    )
    
    # Application configuration
    app_host: str = Field(
        default="0.0.0.0",
        description="Host to bind the application to"
    )
    app_port: int = Field(
        default=7860,
        description="Port to run the application on"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    # Conversation settings
    max_conversations: int = Field(
        default=100,
        description="Maximum number of conversations to keep in memory"
    )
    summarization_trigger: int = Field(
        default=10,
        description="Number of messages after which to trigger summarization"
    )
    
    # Agent configuration
    agent_timeout: int = Field(
        default=30,
        description="Timeout for agent operations in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed operations"
    )
    
    # Security configuration
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    @validator('app_port')
    def validate_port(cls, v):
        """Validate that port is within valid range."""
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of {valid_levels}')
        return v.upper()
    
    def get_mcp_servers_dict(self) -> Dict[str, Dict[str, str]]:
        """Parse MCP servers string into a dictionary."""
        if not self.mcp_servers:
            return {}
        
        servers = {}
        for server_config in self.mcp_servers.split(','):
            if ':' in server_config:
                name, command = server_config.split(':', 1)
                servers[name.strip()] = {"command": command.strip()}
            else:
                # If no colon, use the whole string as name with default python command
                name = server_config.strip()
                servers[name] = {"command": f"python {name}.py"}
        
        return servers
    
    @property
    def is_openai_available(self) -> bool:
        """Check if OpenAI API key is available."""
        return bool(self.openai_api_key)
    
    @property
    def is_openrouter_available(self) -> bool:
        """Check if OpenRouter API key is available."""
        return bool(self.openrouter_api_key)
    
    @property
    def is_yandex_maps_available(self) -> bool:
        """Check if Yandex Maps API key is available."""
        return bool(self.yandex_maps_api_key)
    
    def model_dump_safe(self, exclude_sensitive: bool = True) -> Dict[str, Any]:
        """
        Dump settings to dictionary, optionally excluding sensitive information.
        """
        data = self.model_dump()
        
        if exclude_sensitive:
            # Mask API keys
            for key in ['openai_api_key', 'openrouter_api_key', 'yandex_maps_api_key']:
                if key in data and data[key]:
                    data[key] = '***' + data[key][-4:] if len(data[key]) > 4 else '***'
        
        return data


# Create global settings instance
settings = Settings()


if __name__ == "__main__":
    # Print current settings (with sensitive data masked)
    print("Current Logsit Agent Settings:")
    print("=" * 50)
    for key, value in settings.model_dump_safe().items():
        print(f"{key}: {value}")
    print("=" * 50)