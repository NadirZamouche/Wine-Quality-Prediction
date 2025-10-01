# Libraires
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Keep global encoder & scaler so training and inference share them
label_encoder = LabelEncoder()
scaler = StandardScaler()


def preprocess_data_retrain():
    """
    Preprocess wine quality datasets (training only).
    Returns preprocessed dataframe with target column.
    """
    # 1. Load datasets
    df_red = pd.read_csv("../data/raw/winequality-red.csv", sep=";")
    df_white = pd.read_csv("../data/raw/winequality-white.csv", sep=";")

    # 2. Add 'wine color'
    last_index = df_red.shape[1] - 1
    df_red.insert(last_index, "wine color", "red")
    df_white.insert(last_index, "wine color", "white")

    # 3. Merge datasets
    df = pd.concat([df_red, df_white], ignore_index=True)

    # 4. Encode 'wine color'
    df["wine color"] = label_encoder.fit_transform(df["wine color"])

    # 5. Scale features
    features_to_scale = df.drop(columns=["wine color", "quality"]).columns
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    return df


def preprocess_data_predict():
    """
    Preprocess new incoming data (inference only).
    - Encodes 'wine color'
    - Scales numerical features using trained scaler
    - Ensures column order matches training
    """
    # 1. Load datasets
    df_red = pd.read_csv("../data/raw/winequality-red.csv", sep=";")
    df_white = pd.read_csv("../data/raw/winequality-white.csv", sep=";")

    # 2. Add 'wine color'
    last_index = df_red.shape[1]
    df_red.insert(last_index, "wine color", "red")
    df_white.insert(last_index, "wine color", "white")

    # 3. Merge datasets
    df = pd.concat([df_red, df_white], ignore_index=True)

    # 4. Encode 'wine color'
    df["wine color"] = label_encoder.fit_transform(df["wine color"])

    # 5. Scale features
    features_to_scale = df.drop(columns=["wine color"]).columns
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    return df
