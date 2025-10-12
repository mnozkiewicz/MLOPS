import argparse
from dotenv import load_dotenv
from settings import Settings


def export_envs(environment: str = "dev") -> None:
    Settings.validate_environment(environment)
    env_file = f".envs/.env.{environment}"
    load_dotenv(env_file)


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
