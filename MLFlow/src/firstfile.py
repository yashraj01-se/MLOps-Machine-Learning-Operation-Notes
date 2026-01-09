import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = load_wine()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

#Define Rf model
max_depth = 10
n_estimators = 10

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Wine_Quality_Classification")
# Start MLflow experiment
with mlflow.start_run():
    # Intialize and train the model:
    rf=RandomForestClassifier(max_depth=max_depth,
                              n_estimators=n_estimators)
    rf.fit(X_train,
           y_train)

    # Make predictions
    y_pred=rf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test,
                            y_pred)
    cm = confusion_matrix(y_test,
                          y_pred)

    #log Confusion matrix as image
    plt.figure(figsize=(8,6))
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=data.target_names,
                yticklabels=data.target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig("confusion_matrix.png")
    plt.close()

    
   
    # Log parameters and metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)
    mlflow.set_tag("author", "Yashraj Sharma")
    mlflow.set_tag("project", "Wine Quality Classification")

    mlflow.sklearn.log_model(rf, "random_forest_model")

    print(f"Model accuracy:{accuracy}")