# Sentiment Analysis using FastAPI,Logistic Regression, and MLOps.

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>


# Sentiment Analysis MLOps Project Workflow  

This Sentiment Analysis MLOps Pipeline automates the process of data ingestion, preprocessing, model training, evaluation, versioning, and deployment. The project enables real-time sentiment predictions using Machine Learning (ML) and MLOps best practices.

 What This Project Does
✔ Predicts sentiment from user text (real-time classification).
✔ Implements MLOps best practices (MLflow, DVC, CI/CD).
✔ Optimizes performance using different ML models.
✔ Deploys an API for scalable real-time inference.
✔ Ensures reproducibility with proper version control.



## 1. Data Collection & Preprocessing  
- **Data Source:** Public sentiment datasets (e.g., IMDb, Twitter, Amazon Reviews).  
- **Preprocessing Steps:**  
  - Remove stopwords, punctuation, and special characters.  
  - Tokenization and vectorization using Bag of Words, CountVectorizer, or TF-IDF.  
  - Handle class imbalance using SMOTE or undersampling.  

## 2. Model Training & Experimentation  
- **Algorithms Used:**  
  - Logistic Regression as the baseline model.  
  - Naïve Bayes, Random Forest, and XGBoost for performance comparison.  
  - Deep Learning models such as LSTMs or Transformers (BERT) for improved accuracy.  
- **Hyperparameter tuning** with GridSearchCV to optimize model performance.  

## 3. Model Tracking & Versioning (MLOps Integration)  
- **MLflow** for:  
  - Experiment tracking (logging hyperparameters, metrics).  
  - Model versioning (storing multiple trained models).  
- **DVC (Data Version Control)** to track dataset versions and ensure reproducibility.  

## 4. Model Evaluation  
- **Metrics Used:**  
  - Accuracy, Precision, Recall, and F1-score.  
  - Confusion Matrix and ROC Curve for interpretability.  
- Compare ML models to select the best-performing one.  

## 5. Model Deployment (FastAPI on AWS)  
- **Containerization using Docker** to package the application.  
- **FastAPI API endpoint** to expose the trained model for real-time inference.  
- **Deployment on AWS (EC2/S3)** for scalability.  

## 6. Real-Time Predictions & API Usage  
- User sends text input via API.  
- The model processes and classifies sentiment as Positive, Neutral, or Negative.  
- Response is returned in JSON format with confidence scores.

- ## 7. System Design

![Blank diagram - Sentiment-Analysis-System Design](https://github.com/user-attachments/assets/447d404b-ac6c-49a2-8f1d-80458b7c9483)

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mlops_mini_project and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mlops_mini_project   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mlops_mini_project a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

