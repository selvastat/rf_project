from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import pickle
import io

# Load model
with open("app/iris_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.post("/predict_csv/")
async def predict_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    # Extract features and predict
    features = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    preds = model.predict(features)

    # Store predictions
    df["prediction"] = preds

    # Return as downloadable CSV
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return StreamingResponse(output, media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=predictions.csv"
    })