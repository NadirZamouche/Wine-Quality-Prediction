# Libraries
import joblib
from data_pipe import preprocess_data_predict, label_encoder, scaler

# 1 Preprocess new incoming data
df = preprocess_data_predict()

# 2 Load trained model + metadata
model, ref_column, target = joblib.load("../models/model.pkl")

# 3 Ensure the same feature order as training
X = df[ref_column]

# 4 Make predictions
predictions = model.predict(X)

# 5 Append predictions to the dataframe
df[target] = predictions

# 6 Reverse scaling for numeric features (exclude 'wine color')
numeric_features = df.drop(columns=["wine color", target]).columns
df[numeric_features] = scaler.inverse_transform(df[numeric_features])

# 7 Reverse label encoding for 'wine color'
df["wine color"] = label_encoder.inverse_transform(df["wine color"])

# 8 Save predictions to CSV
df.to_csv("../data/processed/predictions.csv", index=False)
