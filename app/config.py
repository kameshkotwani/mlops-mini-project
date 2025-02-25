from pathlib import Path
MLFLOW_TRACKING_URI  = "https://dagshub.com/kameshkotwani/mlops-mini-project.mlflow"
# Paths
PROJ_ROOT = Path(__file__).resolve().parents[0]
import mlflow

ARTIFACTS_DIR = PROJ_ROOT / "artifacts"

MODEL_NAME = 'model.pkl'
VECTORIZER_NAME = 'vectorizer.pkl'
import dagshub
dagshub.init(repo_owner='kameshkotwani', repo_name='mlops-mini-project', mlflow=True)
