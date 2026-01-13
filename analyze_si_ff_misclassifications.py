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
OUTPUT_DIR = "analysis_results/si_ff_dist"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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

    # New features (SI/FF specific and generic)
    # Ensure SI/FF features are present or calculate them
    if "pfx_z" in df_fe.columns:
        df_fe["vertical_rise"] = df_fe["pfx_z"]

    if "pfx_x" in df_fe.columns and "pfx_z" in df_fe.columns:
        # Avoid division by zero
        df_fe["sink_rate"] = -df_fe["pfx_z"] / (np.abs(df_fe["pfx_x"]) + 0.01)
        df_fe["horizontal_vertical_ratio"] = df_fe["pfx_x"] / (df_fe["pfx_z"] + 0.001)

    if "normalized_spin_axis" in df_fe.columns:
        df_fe["spin_axis_deviation_from_fastball"] = np.abs(
            df_fe["normalized_spin_axis"] - 180
        )

    # Legacy engineered features
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


def plot_distributions(df_analysis, feature, title):
    plt.figure(figsize=(10, 6))

    # Define groups
    groups = [
        ("Correct SI", df_analysis[df_analysis["group"] == "Correct SI"][feature]),
        ("Correct FF", df_analysis[df_analysis["group"] == "Correct FF"][feature]),
        (
            "Misclassified SI->FF",
            df_analysis[df_analysis["group"] == "Misclassified SI->FF"][feature],
        ),
    ]

    for label, data in groups:
        sns.kdeplot(data, label=label, fill=True, alpha=0.3)

    plt.title(f"Distribution of {feature} ({title})")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    filename = f"{OUTPUT_DIR}/{feature}_dist.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot: {filename}")


def main():
    print("Starting SI->FF Misclassification Analysis...")
    data = load_data()
    if data is None:
        return

    df = feature_engineering(data)
    df = df.dropna(subset=["pitch_type"])
    le = LabelEncoder()
    y = le.fit_transform(df["pitch_type"])

    _, features_22, new_features_3 = get_feature_lists()

    # Analyze important features + basic ones
    features_to_analyze = [
        "vertical_rise",  # pfx_z
        "sink_rate",
        "spin_axis_deviation_from_fastball",
        "horizontal_vertical_ratio",
        "release_speed",
        "pfx_x",
        "pfx_z",
        "spin_axis",
        "normalized_spin_axis",
        "release_pos_x",
        "release_pos_z",
        "movemnt_angle",
        "spin_efficiency",
    ]

    # Ensure all features exist
    features_22 = [f for f in features_22 if f in df.columns]

    X_22 = df[features_22].values

    # We need indices to track back to original dataframe for analysis
    indices = np.arange(len(df))
    X_train, X_valid, y_train, y_valid, idx_train, idx_valid = train_test_split(
        X_22, y, indices, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    print("Training model with features...")
    model = xgb.XGBClassifier(
        learning_rate=0.1,
        max_depth=7,
        n_estimators=200,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    # Get class indices
    try:
        si_idx = list(le.classes_).index("SI")
        ff_idx = list(le.classes_).index("FF")
    except ValueError:
        print("SI or FF not found in classes")
        return

    # Identify sample indices
    mask_correct_si = (y_valid == si_idx) & (y_pred == si_idx)
    mask_correct_ff = (y_valid == ff_idx) & (y_pred == ff_idx)
    mask_mis_si_ff = (y_valid == si_idx) & (y_pred == ff_idx)

    idx_correct_si = idx_valid[mask_correct_si]
    idx_correct_ff = idx_valid[mask_correct_ff]
    idx_mis_si_ff = idx_valid[mask_mis_si_ff]

    print(f"Count Correct SI: {len(idx_correct_si)}")
    print(f"Count Correct FF: {len(idx_correct_ff)}")
    print(f"Count Misclassified SI->FF: {len(idx_mis_si_ff)}")

    if len(idx_mis_si_ff) == 0:
        print("No misclassified SI->FF samples found in validation set. Exiting.")
        return

    # Create analysis dataframe
    df_si_correct = df.iloc[idx_correct_si].copy()
    df_si_correct["group"] = "Correct SI"

    df_ff_correct = df.iloc[idx_correct_ff].copy()
    df_ff_correct["group"] = "Correct FF"

    df_si_mis = df.iloc[idx_mis_si_ff].copy()
    df_si_mis["group"] = "Misclassified SI->FF"

    df_analysis = pd.concat([df_si_correct, df_ff_correct, df_si_mis])

    # Print statistics
    print("\n=== Feature Statistics by Group ===")
    from scipy import stats

    def calculate_cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_se

    for feature in features_to_analyze:
        if feature in df_analysis.columns:
            print(f"\nFeature: {feature}")
            stats_df = df_analysis.groupby("group")[feature].agg(
                ["mean", "std", "min", "median", "max", "count"]
            )
            print(stats_df)

            # Statistical Comparison: Misclassified SI vs Correct FF
            mis_si_data = df_si_mis[feature]
            cor_ff_data = df_ff_correct[feature]
            cor_si_data = df_si_correct[feature]

            # KS Test (similarity of distributions)
            ks_stat, ks_p = stats.ks_2samp(mis_si_data, cor_ff_data)

            # Cohen's d (effect size of difference)
            d_val = calculate_cohens_d(mis_si_data, cor_ff_data)

            print(f"--- Comparison: Misclassified SI vs Correct FF ---")
            print(
                f"Difference in Means: {np.mean(mis_si_data) - np.mean(cor_ff_data):.4f}"
            )
            print(f"Cohen's d: {d_val:.4f} (Close to 0 means very similar)")
            print(f"KS Statistic: {ks_stat:.4f} (Lower = more similar distributions)")

            # Comparison: Misclassified SI vs Correct SI (to show how different they are)
            d_val_si = calculate_cohens_d(mis_si_data, cor_si_data)
            print(f"--- Comparison: Misclassified SI vs Correct SI ---")
            print(f"Cohen's d: {d_val_si:.4f} (Higher means very different)")

            plot_distributions(df_analysis, feature, "SI vs FF Analysis")


if __name__ == "__main__":
    main()
