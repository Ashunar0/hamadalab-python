import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set random state
RANDOM_STATE = 42


def load_data():
    if os.path.exists("train_with_features.csv") and os.path.exists(
        "test_with_features.csv"
    ):
        print("Loading existing data...")
        train_data = pd.read_csv("train_with_features.csv")
        test_data = pd.read_csv("test_with_features.csv")
        data = pd.concat([train_data, test_data], ignore_index=True)
    else:
        print("Data files not found!")
        return None
    return data


def feature_engineering(df):
    df_fe = df.copy()

    # New features
    if "release_speed" in df_fe.columns and "pfx_x" in df_fe.columns:
        df_fe["velocity_abs_pfx_x_ratio"] = df_fe["release_speed"] / (
            df_fe["pfx_x"].abs() + 0.1
        )

    if "release_speed" in df_fe.columns and "pfx_z" in df_fe.columns:
        df_fe["velocity_times_pfx_z"] = df_fe["release_speed"] * df_fe["pfx_z"]

    if "pfx_x" in df_fe.columns and "pfx_z" in df_fe.columns:
        df_fe["pfx_z_minus_abs_pfx_x"] = df_fe["pfx_z"] - df_fe["pfx_x"].abs()

    return df_fe


def get_feature_lists():
    base_features = [
        "release_speed",
        "release_spin_rate",
        "spin_axis",
        "pfx_x",
        "pfx_z",
        "release_pos_x",
        "release_pos_z",
        "p_throws",
    ]

    derived_features_11 = [
        "normalized_spin_axis",
        "movement_angle",
        "abs_horizontal_movement",
        "movement_magnitude",
        "spin_efficiency",
        "speed_spin_ratio",
        "horizontal_vertical_ratio",
        "release_position_magnitude",
        "vertical_rise",
        "sink_rate",
        "spin_axis_deviation_from_fastball",
    ]

    new_features_3 = [
        "velocity_abs_pfx_x_ratio",
        "velocity_times_pfx_z",
        "pfx_z_minus_abs_pfx_x",
    ]

    features_19 = base_features + derived_features_11
    features_22 = features_19 + new_features_3

    return features_19, features_22, new_features_3


def analyze_sl_fc_confusion(y_true, y_pred, le, model_name):
    print(f"\n--- {model_name} SL vs FC Analysis ---")

    try:
        sl_idx = list(le.classes_).index("SL")
        fc_idx = list(le.classes_).index("FC")
    except ValueError:
        print("SL or FC not found in classes")
        return

    cm = confusion_matrix(y_true, y_pred)
    sl_true_count = cm[sl_idx, sl_idx]
    fc_true_count = cm[fc_idx, fc_idx]
    sl_as_fc = cm[sl_idx, fc_idx]
    fc_as_sl = cm[fc_idx, sl_idx]

    print(f"SL Classified as SL (Correct): {sl_true_count}")
    print(f"FC Classified as FC (Correct): {fc_true_count}")
    print(f"SL Misclassified as FC: {sl_as_fc}")
    print(f"FC Misclassified as SL: {fc_as_sl}")

    return sl_as_fc


def main():
    print("Starting evaluation...")
    data = load_data()
    if data is None:
        return

    df = feature_engineering(data)
    df = df.dropna(subset=["pitch_type"])
    le = LabelEncoder()
    y = le.fit_transform(df["pitch_type"])

    features_19, features_22, new_features_3 = get_feature_lists()

    X_19 = df[features_19].values
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_19, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    print("\nTraining with 19 features...")
    model_19 = xgb.XGBClassifier(
        learning_rate=0.1,
        max_depth=7,
        n_estimators=200,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model_19.fit(X_train, y_train)
    y_pred_19 = model_19.predict(X_valid)
    sl_as_fc_19 = analyze_sl_fc_confusion(y_valid, y_pred_19, le, "19 Features")

    print("\nTraining with 22 features...")
    X_22 = df[features_22].values
    X_train_22, X_valid_22, y_train_22, y_valid_22 = train_test_split(
        X_22, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    model_22 = xgb.XGBClassifier(
        learning_rate=0.1,
        max_depth=7,
        n_estimators=200,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model_22.fit(X_train_22, y_train_22)
    y_pred_22 = model_22.predict(X_valid_22)
    sl_as_fc_22 = analyze_sl_fc_confusion(y_valid_22, y_pred_22, le, "22 Features")

    # Feature Importance
    print("\n--- Feature Importance (Top 10) ---")
    importance = model_22.feature_importances_
    feat_imp = pd.DataFrame({"feature": features_22, "importance": importance})
    feat_imp = feat_imp.sort_values("importance", ascending=False)
    print(feat_imp.head(10))

    print("\n--- New Features Importance ---")
    print(feat_imp[feat_imp["feature"].isin(new_features_3)])

    print("\n=== Improvement Report ===")
    print(f"SL -> FC Misclassifications (19 features): {sl_as_fc_19}")
    print(f"SL -> FC Misclassifications (22 features): {sl_as_fc_22}")
    reduction = sl_as_fc_19 - sl_as_fc_22
    percent = (reduction / sl_as_fc_19) * 100 if sl_as_fc_19 else 0
    print(f"Reduction: {reduction}")
    print(f"Improvement: {percent:.2f}%")


if __name__ == "__main__":
    main()
