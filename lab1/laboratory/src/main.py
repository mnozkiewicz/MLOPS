import argparse
from dotenv import load_dotenv
from settings import Settings
import os
import yaml


def export_secrets():
    try:
        with open("secrets.yaml", "r") as file:
            secrets = yaml.safe_load(file)

        for key, value in secrets.items():
            os.environ[key] = str(value)

    except FileNotFoundError:
        raise ValueError("secrets.yaml not found. No secrets loaded.")


def export_envs(environment: str = "dev") -> None:
    Settings.validate_environment(environment)
    env_file = f".envs/.env.{environment}"
    load_dotenv(env_file)

    export_secrets()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load environment variables from specified.env file."
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="dev",
        help="The environment to load (dev, test, prod)",
    )
    args = parser.parse_args()

    export_envs(args.environment)

    settings = Settings()

    print("APP_NAME: ", settings.APP_NAME)
    print("ENVIRONMENT: ", settings.ENVIRONMENT)
    print("API_KEY: ", settings.API_KEY)
