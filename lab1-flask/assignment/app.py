import joblib
import pandas as pd
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__)

BASE = Path(__file__).parent
MODEL_LOCAL = BASE / 'best_model.pkl'

if not MODEL_LOCAL.exists():
    raise RuntimeError(f"Required model not found: {MODEL_LOCAL}. Run the training script to create 'best_model.pkl' in this folder.")

pipeline = joblib.load(MODEL_LOCAL)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    form = request.form
    try:
        data = {
            'Pclass': [int(form.get('pclass'))],
            'Sex': [form.get('sex')],
            'Age': [float(form.get('age')) if form.get('age') not in (None, '') else None],
            'SibSp': [int(form.get('sibsp'))],
            'Parch': [int(form.get('parch'))],
            'Fare': [float(form.get('fare')) if form.get('fare') not in (None, '') else None],
            'Embarked': [form.get('embarked')]
        }
    except Exception as e:
        return render_template('index.html', prediction_text=f'Invalid input: {e}')

    X = pd.DataFrame(data)
    try:
        proba = pipeline.predict_proba(X)[0, 1]
        pred = int(pipeline.predict(X)[0])
    except Exception as e:
        return render_template('index.html', prediction_text=f'Prediction error: {e}')

    proba_pct = round(float(proba) * 100, 1)
    result_text = f'Predicted survival: {pred} (probability {proba_pct}%)'
    return render_template('index.html', prediction_text=result_text)


if __name__ == '__main__':
    app.run(debug=True)
