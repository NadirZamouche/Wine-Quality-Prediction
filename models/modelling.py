# Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import joblib

# 1 Loading the precoessed data
df = pd.read_csv("../data/processed/winequality_merged.csv")
df.head()

# 2 Data Splitting
# Assuming df is your DataFrame
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=21
)
X_train.head()
y_train.head()


# 3 Model Selection
def evaluate_models(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate different classification models and compare their performance.

    Parameters:
    X_train (array-like): Training features.
    y_train (array-like): Training labels.
    X_test (array-like): Testing features.
    y_test (array-like): Testing labels.

    Returns:
    pd.DataFrame: A DataFrame containing model names, training and testing performance metrics.
    """

    # Initialize models
    models = [
        ("SVR", SVR()),
        ("Logistic Regression", LinearRegression()),
        ("Decision Tree", DecisionTreeRegressor()),
        ("Random Forest", RandomForestRegressor()),
        ("XGBRegressor", XGBRegressor()),
    ]

    # Initialize result DataFrame
    result = pd.DataFrame(
        columns=[
            "Model",
            "Train_MSE",
            "Train_MAE",
            "Train_R2",
            "Test_MSE",
            "Test_MAE",
            "Test_R2",
        ]
    )

    for model_name, model in models:
        if model_name == "XGBRegressor":
            # Train the model
            model.fit(X_train.values, y_train)

            # Cross-validation
            kfold = StratifiedKFold(n_splits=5)
            cross_val_MSE = -cross_val_score(
                model,
                X_train.values,
                y_train,
                cv=kfold,
                scoring="neg_mean_squared_error",
            ).mean()
            cross_val_MAE = -cross_val_score(
                model,
                X_train.values,
                y_train,
                cv=kfold,
                scoring="neg_mean_absolute_error",
            ).mean()
            cross_val_R2 = cross_val_score(
                model, X_train.values, y_train, cv=kfold, scoring="r2"
            ).mean()

            # Test the model
            predictions = model.predict(X_test.values)
            test_MSE = mean_squared_error(y_test, predictions)
            test_MAE = mean_absolute_error(y_test, predictions)
            test_R2 = r2_score(y_test, predictions)
        else:
            # Train the model
            model.fit(X_train, y_train)

            # Cross-validation
            kfold = StratifiedKFold(n_splits=5)
            cross_val_MSE = -cross_val_score(
                model, X_train, y_train, cv=kfold, scoring="neg_mean_squared_error"
            ).mean()
            cross_val_MAE = -cross_val_score(
                model, X_train, y_train, cv=kfold, scoring="neg_mean_absolute_error"
            ).mean()
            cross_val_R2 = cross_val_score(
                model, X_train, y_train, cv=kfold, scoring="r2"
            ).mean()

            # Test the model
            predictions = model.predict(X_test)
            test_MSE = mean_squared_error(y_test, predictions)
            test_MAE = mean_absolute_error(y_test, predictions)
            test_R2 = r2_score(y_test, predictions)

        # Store results
        result.loc[len(result)] = [
            model_name,
            cross_val_MSE,
            cross_val_MAE,
            cross_val_R2,
            test_MSE,
            test_MAE,
            test_R2,
        ]
    return result


evaluate_models(X_train, y_train, X_test, y_test)

# 4 Model Tuning (RandomForestRegressor)
# Initialize a Random Forest Classifier
rf_model = RandomForestRegressor()

# Fine-tuning parameters
param_grid = {
    "n_estimators": [100, 200, 300],  # number of trees
    "max_depth": [None, 10, 20, 30],  # tree depth
    "min_samples_split": [2, 5, 10],  # min samples to split a node
    "min_samples_leaf": [1, 2, 4],  # min samples per leaf
    "max_features": ["auto", "sqrt", "log2"],  # features considered per split
    "bootstrap": [True, False],  # whether to use bootstrap samples
    "random_state": [42],  # reproducibility
}

# Initialize Stratified K-Fold cross-validation
kfold = StratifiedKFold(n_splits=5)

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=kfold,
    scoring="neg_mean_absolute_error",
    verbose=2,
    n_jobs=-1,
)
grid_search.fit(X_train.values, y_train)

# Access the best estimator directly
best_estimator_params = grid_search.best_estimator_.get_params()
best_estimator_params

# Now you can create a new RandomForestClassifier using the best parameters
best_rf_model = RandomForestRegressor(**best_estimator_params)
best_rf_model.fit(X_train.values, y_train)

# Training set
kfold = StratifiedKFold(n_splits=5)
cross_val_MSE = -cross_val_score(
    best_rf_model, X_train.values, y_train, cv=kfold, scoring="neg_mean_squared_error"
).mean()
cross_val_MAE = -cross_val_score(
    best_rf_model, X_train.values, y_train, cv=kfold, scoring="neg_mean_absolute_error"
).mean()
cross_val_R2 = cross_val_score(
    best_rf_model, X_train.values, y_train, cv=kfold, scoring="r2"
).mean()

# Test set
predictions = best_rf_model.predict(X_test.values)
test_MSE = mean_squared_error(y_test, predictions)
test_MAE = mean_absolute_error(y_test, predictions)
test_R2 = r2_score(y_test, predictions)


print(f"Train_MSE: {cross_val_MSE:.4f}")
print(f"Train_MAE: {cross_val_MAE:.4f}")
print(f"Train_R2: {cross_val_R2:.4f}")
print(f"Test_MSE: {test_MSE:.4f}")
print(f"Test_MAE: {test_MAE:.4f}")
print(f"Test_R2: {test_R2:.4f}")

# 5 Feature Importance
# Create a subplot with desired aspect ratio
fig, ax = plt.subplots(figsize=(5, 10))  # Adjust the size here (width, height)

# Plot feature importance
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
ax.barh(range(X_train.shape[1]), importances[indices], align="center")
ax.set_yticks(range(X_train.shape[1]))
ax.set_yticklabels([X_train.columns[i] for i in indices])
ax.invert_yaxis()  # Invert y-axis to have the most important feature at the top
ax.set_xlabel("Feature Importance")
ax.set_title("Random Forest Regressor Feature Importance")

plt.show()

# Combine features and target for the test set
test_set = pd.concat([X_test, y_test], axis=1)

# 6 Save to CSV
test_set.to_csv("../data/processed/winequality_test.csv", index=False)

# 7 Save Model
ref_cols = list(X.columns)
target = "quality"
joblib.dump(value=[best_rf_model, ref_cols, target], filename="../models/model.pkl")
