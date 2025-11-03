from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pathlib import Path
import joblib
import numpy as np

app = FastAPI()


class FurnitureFeatures(BaseModel):
    category: int
    sellable_online: int
    other_colors: int
    depth: float
    height: float
    width: float

import pickle
import pandas as pd


TEMPLATES = Jinja2Templates(directory="./templates")

FURN_MODEL_PATH = "model.pkl"

furn_pipeline = joblib.load(str(FURN_MODEL_PATH))


@app.get('/furniture', response_class=HTMLResponse)
def furniture_form(request: Request):
    return TEMPLATES.TemplateResponse('furniture_form.html', {'request': request})


@app.post('/furniture/predict', response_class=HTMLResponse)
async def furniture_predict(request: Request):
    form = await request.form()
    try:
        features = FurnitureFeatures(
            **form
        )
        print(features)
    except Exception as e:
        return TEMPLATES.TemplateResponse('furniture_form.html', {'request': request, 'error': f'Invalid input: {e}'})

    X = np.array([[features.category, features.sellable_online, features.other_colors, features.depth, features.height, features.width]])
    try:
        pred = furn_pipeline.predict(X)[0]
    except Exception as e:
        return TEMPLATES.TemplateResponse('furniture_form.html', {'request': request, 'error': f'Prediction error: {e}'})

    return TEMPLATES.TemplateResponse('furniture_form.html', {'request': request, 'prediction': float(pred)})


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)


    