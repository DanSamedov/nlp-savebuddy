from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "nlp-savebuddy"
    APP_ENV: str = "local"
    LOG_LEVEL: str = "INFO"
    
    # NLP Models
    QA_MODEL_NAME: str
    EMBEDDING_MODEL_NAME: str
    
    # Logic Thresholds
    MATCH_THRESHOLD: float

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore"
    )

settings = Settings()
