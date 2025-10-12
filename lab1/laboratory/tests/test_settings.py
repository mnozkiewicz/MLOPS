from src.settings import Settings
from src.main import export_envs


def test_settings_loads_env_test():
    export_envs("test")

    settings = Settings()

    print(settings)
    assert settings.ENVIRONMENT == "test"
    assert settings.APP_NAME == "app_test"
