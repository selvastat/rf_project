from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import pandas as pd
import mlflow.sklearn
from io import StringIO, BytesIO

app = FastAPI()

# Load model
# The line `model = joblib.load("app/iris_model.pkl")` is loading a machine learning model from a file
# named "iris_model.pkl" using the joblib library. The model is then stored in the variable `model`
# and can be used for making predictions on new data.
model = joblib.load("app/iris_model.pkl")


@app.post("/predict-csv/")
async def predict_csv(file: UploadFile = File(...)):
    # Read CSV file
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode("utf-8")))

    # Make predictions
    preds = model.predict(df)
    df["prediction"] = preds

    # Convert DataFrame to CSV
    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    # Return CSV as downloadable file
    return StreamingResponse(output, media_type="text/csv", headers={
        "Content-Disposition": "attachment; filename=predictions.csv"
    })
