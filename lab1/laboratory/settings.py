from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    ENVIRONMENT: str
    APP_NAME: str

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        if value not in ("dev", "test", "prod"):
            raise ValueError(
                f"The `value` argument should be one of (`dev`, `test`, `prod`), given {value}"
            )
        return value
