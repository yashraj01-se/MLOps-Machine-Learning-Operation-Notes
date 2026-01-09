from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Define model and hyperparameter grid
rf = RandomForestClassifier()
param_grid = {
    'n_estimators': [10,50, 100],
    'max_depth': [None, 10, 20, 30],
}

# Apply GridSearchCV
grid_search = GridSearchCV(estimator=rf,
                           param_grid=param_grid,
                           cv=5,
                           n_jobs=-1,
                           verbose=2)

# # Start code Execution without MLflow
# grid_search.fit(X_train, y_train)

# # Get the best hyperparameters
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_

# print(f"Best Hyperparameters: {best_params}")
# print(f"Best Cross-Validation Score: {best_score}")

# Start code Execution with MLflow:
mlflow.set_experiment("Breast_Cancer_Classification_Hyperparameter_Tuning")

mlflow.set_tracking_uri("http://localhost:5000")
with mlflow.start_run():
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Log best hyperparameters and score to MLflow
    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_score", best_score)

    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Cross-Validation Score: {best_score}")

    # Log the best model
    best_model = grid_search.best_estimator_
    mlflow.sklearn.log_model(best_model, "best_random_forest_model")

    # Log the current script
    mlflow.log_artifact(__file__)

    # Set tags
    mlflow.set_tag("author", "Yashraj Sharma")
    mlflow.set_tag("project", "Breast Cancer Classification Hyperparameter Tuning")


    # log Training and testing dataset as artifacts:
    train_df=X_train.copy()
    train_df['target']=y_train

    train_df=mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, "train_dataset.csv")

    X_test_df=X_test.copy()
    X_test_df['target']=y_test

    test_df=mlflow.data.from_pandas(X_test_df)
    mlflow.log_input(test_df, "test_dataset.csv")
    