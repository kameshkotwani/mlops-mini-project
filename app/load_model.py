from app.config import  ARTIFACTS_DIR, MODEL_NAME,VECTORIZER_NAME,MLFLOW_TRACKING_URI
import pickle
import mlflow
import os
class ModelLoader:
    def __init__(self):
        self.vectorizer_name = VECTORIZER_NAME
        self.model = None
        self.vectorizer = None
        self.client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    def load_model(self):
        """Loads the ML model and vectorizer from MLflow."""
        try:
            # create artifacts directory if it doesn't exist to store the vectorizer and model
            os.makedirs(ARTIFACTS_DIR, exist_ok=True)
            
            # Get the model by alias and load it new way (since model staging is deprecated)
            model_info = self.client.get_model_version_by_alias("model","staging")
            self.model = mlflow.pyfunc.load_model(model_info.source)
            
            # Download only the vectorizer.pkl from the model's artifacts
            self.vectorizer_uri = mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{model_info.run_id}/{self.vectorizer_name}",
                dst_path=ARTIFACTS_DIR
            )
            # # Load the vectorizer from the model's artifacts
            # self.vectorzier_uri = mlflow.artifacts.download_artifacts(
            #     run_id=f"{model_info.run_id}",
            #     dst_path=ARTIFACTS_DIR
            # )
            
           
            # read the downloaded vectorizer
            with open(ARTIFACTS_DIR / self.vectorizer_name, "rb") as f:
                self.vectorizer = pickle.load(f)
            
            return self.model, self.vectorizer
        
        except Exception as e:
            print(f"Error loading model: {e}")
            



# Example usage
if __name__ == "__main__":
    loader = ModelLoader()
    model, vectorizer = loader.load_model()
    print(type(model))
    print(type(vectorizer))
    print(vectorizer.transform(['this is a test']))


