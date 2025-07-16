import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import register_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import joblib
import os

# Load data iris
df = pd.read_csv("iris_data.csv")
X = df.drop("species", axis=1)
y = df["species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Start MLflow experiment
mlflow.set_experiment("Iris_RF_Classifier")
with mlflow.start_run() as run:
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
     # Log model
    mlflow.sklearn.log_model(clf, artifact_path="clf")
    # Log params and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", clf.score(X_test, y_test))
    
        # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    cm_df = pd.DataFrame(cm, columns=[f"Pred_{i}" for i in range(cm.shape[1])],
                            index=[f"True_{i}" for i in range(cm.shape[0])])
    cm_df.to_csv("confusion_matrix.csv")
    mlflow.log_artifact("confusion_matrix.csv")
    
        # Classification report
    report = classification_report(y_test, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv("classification_report.csv")
    mlflow.log_artifact("classification_report.csv")

    # Save model using joblib
    os.makedirs("app", exist_ok=True)
    joblib.dump(clf, "app/iris_model.pkl")
    print(f"✅ Model saved to app/iris_model.pkl with accuracy: {acc}")
    
    import logging

    logging.basicConfig(filename='training.log', level=logging.INFO)
    logging.info("Training started...")

    # Later log this file to MLflow
    mlflow.log_artifact('training.log')
    
    run_id = run.info.run_id

    # Register model
    model_uri = f"runs:/{run_id}/clf"
    model_details = mlflow.register_model(
        model_uri=model_uri,
        name="rf_model_registered"
    )
