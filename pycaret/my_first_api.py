
import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model('my_first_api')

# Define predict function
@app.post('/predict')
def predict(sepal_length, sepal_width, petal_length, petal_width):
    data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]])
    data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    predictions = predict_model(model, data=data) 
    return {'prediction': list(predictions['Label'])}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)