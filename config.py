from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "Kanoon Wala"
    ENV: str = "development"

    DATABASE_URL: str
    SYNC_DATABASE_URL: str

    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # SMS_PROVIDER: "dev" (logs OTP, no real SMS), "twilio", or "msg91"
    SMS_PROVIDER: str = "dev"
    OTP_EXPIRY_SECONDS: int = 300

    # Twilio
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_FROM_NUMBER: str = ""

    # MSG91 (India-focused alternative)
    MSG91_AUTH_KEY: str = ""
    MSG91_SENDER_ID: str = "KANOON"
    MSG91_TEMPLATE_ID: str = ""

    ANTHROPIC_API_KEY: str = ""
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    LLM_MODEL: str = "claude-sonnet-4-6"

    ADMIN_BOOTSTRAP_PHONE: str = ""

    # Storage
    DOCUMENT_STORAGE_DIR: str = "/data/user_documents"
    AUDIO_STORAGE_DIR: str = "/data/audio"
    MEDIA_BASE_URL: str = "http://localhost:8000/media"

    # Voice
    WHISPER_MODEL_SIZE: str = "small"  # tiny/base/small/medium — small handles Hindi+English well
    TTS_VOICE_HI: str = "hi-IN-SwaraNeural"
    TTS_VOICE_EN: str = "en-IN-NeerjaNeural"


settings = Settings()
