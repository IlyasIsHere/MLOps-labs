from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

app = FastAPI()


class TitanicFeatures(BaseModel):
    pclass: int
    sex: str
    age: float | None
    sibsp: int
    parch: int
    fare: float | None
    embarked: str | None


TEMPLATES = Jinja2Templates(directory="./templates")

TITANIC_MODEL_PATH = "best_model.pkl"

titanic_pipeline = joblib.load(str(TITANIC_MODEL_PATH))


@app.get('/titanic', response_class=HTMLResponse)
def titanic_form(request: Request):
    return TEMPLATES.TemplateResponse('titanic_form.html', {'request': request})


@app.post('/titanic/predict', response_class=HTMLResponse)
async def titanic_predict(request: Request):
    form = await request.form()
    try:
        features = TitanicFeatures(**form)
    except Exception as e:
        return TEMPLATES.TemplateResponse('titanic_form.html', {'request': request, 'error': f'Invalid input: {e}'})

    # Convert to DataFrame for the pipeline
    X = pd.DataFrame([{
        'Pclass': features.pclass,
        'Sex': features.sex,
        'Age': features.age,
        'SibSp': features.sibsp,
        'Parch': features.parch,
        'Fare': features.fare,
        'Embarked': features.embarked
    }])
    
    try:
        pred = titanic_pipeline.predict(X)[0]
        proba = titanic_pipeline.predict_proba(X)[0, 1]
    except Exception as e:
        return TEMPLATES.TemplateResponse('titanic_form.html', {'request': request, 'error': f'Prediction error: {e}'})

    return TEMPLATES.TemplateResponse('titanic_form.html', {'request': request, 'prediction': int(pred), 'probability': round(float(proba) * 100, 1)})


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
