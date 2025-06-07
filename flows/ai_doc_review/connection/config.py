from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Define the settings with default values
    debug: bool = False
    serve_static: bool = True
    
    # Database settings
    cosmos_url: str = ""
    cosmos_key: str = ""
    database_name: str = "state"
    issues_container: str = "issues"
    feedback_container: str = "feedback"
    
    # Storage settings
    storage_account_url: str = ""
    storage_container_name: str = "documents"
    
    # Hugging Face Inference API settings
    hf_model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    hf_api_token: str = ""  # Hugging Face API token (required for Inference API)
    hf_api_url: str = "https://api-inference.huggingface.co/models"  # Inference API base URL
    hf_max_length: int = 2048
    hf_temperature: float = 0.7
    hf_batch_size: int = 10
    hf_wait_for_model: bool = True  # Whether to wait if model is loading
    
    # Logging settings
    appinsights_instrumentation_key: str = ""
    log_level: str = "INFO"
    
    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
