from src.settings import Settings
import os


def test_settings():
    os.environ["API_KEY"] = "dummy_key"
    settings = Settings()

    assert settings.ENVIRONMENT == "test"
    assert settings.APP_NAME == "app_test"
