import sys
from pathlib import Path

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, Query
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from matplotlib.pylab import vectorize
from pydantic import BaseModel, Field
import numpy as np

# relative imports
from app.data_preprocessing import normalize_text
from app.config import MODEL_NAME,VECTORIZER_NAME
from app.load_model import ModelLoader

# get the model from mlflow
# Load model and vectorizer at startup
# Load model and vectorizer once during startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    #TODO: Load the model and vectorizer here
    print("Loading model and vectorizer...")
    loader = ModelLoader()
    

    model,vectorizer = loader.load_model()
    app.state.model = model
    app.state.vectorizer = vectorizer

    if model is None or vectorizer is None:
        raise RuntimeError("Failed to load model or vectorizer")

    print("Model and vectorizer loaded")

    # Yield control back to FastAPI
    yield
    # Cleanup logic (if needed) when the app shuts down
    print("Shutting down application...")

# Creating a FastAPI instance
app = FastAPI(lifespan=lifespan)

# Set up Jinja2 for rendering templates
templates = Jinja2Templates(directory="app/templates")

# Home route (renders the form)
@app.get("/")
async def root(request: Request, prediction: str = Query(None), user_input: str = Query(None)):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction, "user_input": user_input})

# Define Input Schema using Pydantic
class TextInput(BaseModel):
    text: str = Field(..., examples=["Input some sentiment text"])

# Form Submission Route (Ensuring User Input Stays)
@app.post("/predict/")
def predict(user_input: str = Form(...)):
    try:
        # loading the user data
        df = normalize_text(user_input)

        user_input = df['content'].astype(str)[0]
        print(user_input)

        # applying bag of words using vectorizer
        transformed_user_input = app.state.vectorizer.transform([user_input])
        # getting the prediction value

        prediction = app.state.model.predict(transformed_user_input)[0]

        output_sentiment = "negative" if prediction == 0 else "positive"

        # Redirect to home page with the prediction and user_input as query parameters
        return RedirectResponse(url=f"/?prediction={output_sentiment}&user_input={user_input}", status_code=303)

    except Exception as e:
        return RedirectResponse(url="/?error=Prediction failed", status_code=303)