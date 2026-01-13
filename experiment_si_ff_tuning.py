import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import os

# Configuration
RANDOM_STATE = 42
OUTPUT_FILE = "si_ff_tuning_results.txt"


def load_data():
    if os.path.exists("train_with_features.csv") and os.path.exists(
        "test_with_features.csv"
    ):
        print("Loading data...")
        train_data = pd.read_csv("train_with_features.csv")
        test_data = pd.read_csv("test_with_features.csv")
        data = pd.concat([train_data, test_data], ignore_index=True)
        return data
    else:
        print("Data not found.")
        return None


def feature_engineering(df):
    df_fe = df.copy()
    # Ensure SI/FF features (copies of logic from main notebook/analysis scripts)
    if "pfx_z" in df_fe.columns and "vertical_rise" not in df_fe.columns:
        df_fe["vertical_rise"] = df_fe["pfx_z"]

    if (
        "pfx_x" in df_fe.columns
        and "pfx_z" in df_fe.columns
        and "sink_rate" not in df_fe.columns
    ):
        df_fe["sink_rate"] = -df_fe["pfx_z"] / (np.abs(df_fe["pfx_x"]) + 0.01)

    if (
        "normalized_spin_axis" in df_fe.columns
        and "spin_axis_deviation_from_fastball" not in df_fe.columns
    ):
        df_fe["spin_axis_deviation_from_fastball"] = np.abs(
            df_fe["normalized_spin_axis"] - 180
        )

    return df_fe


def get_features():
    # 22 Features from main notebook
    return [
        "release_speed",
        "release_spin_rate",
        "spin_axis",
        "pfx_x",
        "pfx_z",
        "release_pos_x",
        "release_pos_z",
        "p_throws",
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
        "velocity_abs_pfx_x_ratio",
        "velocity_times_pfx_z",
        "pfx_z_minus_abs_pfx_x",
    ]


def custom_eval_metric(y_pred, y_true):
    # This function is for XGBoost's feval if needed, but we use GridSearchCV here.
    # We want to maximize Macro F1 or specifically SI F1.
    return "f1_macro", f1_score(y_true, y_pred, average="macro")


def main():
    data = load_data()
    if data is None:
        return

    df = feature_engineering(data)
    df = df.dropna(subset=["pitch_type"])

    # Filter valid features
    feature_list = [f for f in get_features() if f in df.columns]
    print(f"Using {len(feature_list)} features.")

    le = LabelEncoder()
    y = le.fit_transform(df["pitch_type"])
    X = df[feature_list]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Initial Model (Baseline)
    print("\nTraining Baseline Model...")
    baseline_clf = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,  # Default often 6
        min_child_weight=1,  # Default 1
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    baseline_clf.fit(X_train, y_train)
    y_pred_base = baseline_clf.predict(X_test)

    # Get SI and FF indices
    try:
        si_idx = list(le.classes_).index("SI")
        ff_idx = list(le.classes_).index("FF")
        print(f"Indices - SI: {si_idx}, FF: {ff_idx}")
    except:
        print("SI/FF not found")
        return

    # Baseline confusion portion
    cm_base = confusion_matrix(y_test, y_pred_base)
    si_ff_err_base = cm_base[si_idx, ff_idx]
    print(f"Baseline SI->FF Errors: {si_ff_err_base}")

    # Tuning Grid
    # We suspect we need deeper trees or lower min_child_weight to capture the "intermediate" cluster
    param_grid = {
        "max_depth": [6, 8, 10],
        "min_child_weight": [
            1,
            0.5,
            0.1,
        ],  # Lower weight allows leaf nodes with fewer samples (granular clusters)
        "gamma": [0, 0.1, 0.2],
    }

    print("\nStarting Grid Search...")
    grid_search = GridSearchCV(
        estimator=xgb.XGBClassifier(
            learning_rate=0.1, n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        param_grid=param_grid,
        scoring="f1_macro",  # optimizing global performance is safer, but we check SI->FF specifically
        cv=3,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    best_clf = grid_search.best_estimator_
    print(f"\nBest Params: {grid_search.best_params_}")

    # Evaluate Best Model
    y_pred_best = best_clf.predict(X_test)
    cm_best = confusion_matrix(y_test, y_pred_best)
    si_ff_err_best = cm_best[si_idx, ff_idx]

    print(f"Best SI->FF Errors: {si_ff_err_best}")
    print(f"Improvement: {si_ff_err_base - si_ff_err_best} fewer misclassifications")

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        f.write(f"Baseline SI->FF Errors: {si_ff_err_base}\n")
        f.write(f"Best Params: {grid_search.best_params_}\n")
        f.write(f"Best SI->FF Errors: {si_ff_err_best}\n")
        f.write(f"Improvement: {si_ff_err_base - si_ff_err_best}\n")


if __name__ == "__main__":
    main()
