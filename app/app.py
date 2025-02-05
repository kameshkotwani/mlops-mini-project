from fastapi import FastAPI, Request, Form, Query
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import numpy as np

# get the model from mlflow

# Creating a FastAPI instance
app = FastAPI()

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
        # TODO: Load the model and do a true prediction from Dagshub
        prediction = np.random.randint(0, 2)
        output_sentiment = "negative" if prediction == 0 else "positive"

        # Redirect to home page with the prediction and user_input as query parameters
        return RedirectResponse(url=f"/?prediction={output_sentiment}&user_input={user_input}", status_code=303)

    except Exception as e:
        return RedirectResponse(url="/?error=Prediction failed", status_code=303)
