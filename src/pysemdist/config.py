from pydantic import BaseSettings, Field
from typing import Optional

class Settings(BaseSettings):
    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")
    db_name: str = Field(default="", alias="DB_NAME")
    db_user: str = Field(default="", alias="DB_USER")
    db_password: str = Field(default="", alias="DB_PASSWORD")
    db_pool_min: int = Field(default=1, alias="DB_POOL_MIN")
    db_pool_max: int = Field(default=10, alias="DB_POOL_MAX")
    db_connect_timeout: int = Field(default=10, alias="DB_CONNECT_TIMEOUT")
    db_statement_timeout_ms: int = Field(default=60000, alias="DB_STATEMENT_TIMEOUT_MS")

    default_country_code: str = Field(default="US")
    output_dir: str = Field(default="data/zbronze")

    s3_bucket: Optional[str] = Field(default=None, alias="S3_BUCKET")
    s3_prefix: str = Field(default="pysemdist/exports", alias="S3_PREFIX")
    s3_upload_enabled: bool = Field(default=False, alias="S3_UPLOAD")  # "1" / "true" to enable

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
