# Libraries
import joblib
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from data_pipe import preprocess_data_retrain

# 1 Run preprocessing pipeline
df = preprocess_data_retrain()

# 2 Load Model + Metadata
model, ref_column, saved_target = joblib.load("../models/model.pkl")

# 3 Separate features and target using saved metadata
X = df[ref_column]
y = df[saved_target]

# 4 Retrain the model on new data
model.fit(X, y)

# 5 Evaluate performance
test = pd.read_csv("../data/processed/winequality_test.csv")
X_test = test[ref_column]  # ensures consistent features
y_test = test[saved_target]

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

print("Model Evaluation")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Average CV R²: {cv_scores.mean():.4f}")

# 6 Save updated model (overwrite or version up)
joblib.dump((model, ref_column, saved_target), "../models/model.pkl")
