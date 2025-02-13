# register model
from mlops_mini_project.config import MLFLOW_TRACKING_URI, REPORTS_DIR,LOGS_DIR,STAGING
import json
import mlflow
import logging
import dagshub
from pathlib import Path

dagshub.init(
    repo_owner='kameshkotwani',
    repo_name='mlops-mini-project',
    mlflow=True
)

# Set up MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler(Path(LOGS_DIR) / f"{Path(__file__).stem}.log")
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise


def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.MlflowClient()

        client.set_registered_model_alias(
                name=model_name,
                version=model_version.version,
                alias=STAGING
        )

        logger.debug(f'Model {model_name} version {model_version.version} registered with alias {STAGING}')
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise


def main():
    try:
        model_info_path = REPORTS_DIR /  'model_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "model"
        register_model(model_name, model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()