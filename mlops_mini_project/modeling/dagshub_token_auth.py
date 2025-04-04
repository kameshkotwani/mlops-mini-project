import os
from dotenv import load_dotenv

load_dotenv()

def get_dagshub_token() -> None:
    """Retrieve the Dagshub token from environment variables."""
    try:
        token = os.getenv('DAGSHUB_TOKEN')
        if not token:
            raise ValueError("DAGSHUB_TOKEN environment variable not set.")
        os.environ['MLFLOW_TRACKING_USERNAME'] = token
        os.environ['MLFLOW_TRACKING_PASSWORD'] = token

    except Exception as e:
        raise EnvironmentError(f"Error retrieving DAGSHUB_TOKEN: {e}")