import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Euqlidean distance to hydrology
    df["Distance_To_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # Difference between fire-point and road distances
    df["Fire_Road_Difference"] = (
        df["Horizontal_Distance_To_Fire_Points"]
        - df["Horizontal_Distance_To_Roadways"]
    )

    # Mean hillshade across the day
    df["Mean_Hillshade"] = (
        df["Hillshade_9am"] + df["Hillshade_Noon"] + df["Hillshade_3pm"]
    ) / 3.0

    # Elevation × hydrology distance interaction
    df["Elev_Hydro_Interaction"] = df["Elevation"] * df["Distance_To_Hydrology"]

    # High-elevation binary flag
    df["High_Elevation_Flag"] = (df["Elevation"] > 3000).astype(int)

    return df


def get_X_y(df: pd.DataFrame, target_col: str = "Cover_Type"):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def preprocess(path: str, target_col: str = "Cover_Type"):
    df = load_data(path)
    df = engineer_features(df)
    return get_X_y(df, target_col)


if __name__ == "__main__":
    import os

    base = os.path.join(os.path.dirname(__file__), "..", "data")
    train_path = os.path.join(base, "train.csv")

    X, y = preprocess(train_path)
    print("Feature matrix shape :", X.shape)
    print("Target distribution   :\n", y.value_counts().sort_index())
    print("Engineered columns    :", [c for c in X.columns if c not in pd.read_csv(train_path).columns])
