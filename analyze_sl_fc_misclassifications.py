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
OUTPUT_DIR = "analysis_results/sl_fc_dist"
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


def plot_distributions(df_analysis, feature, title):
    plt.figure(figsize=(10, 6))

    # Define groups
    groups = [
        ("Correct SL", df_analysis[df_analysis["group"] == "Correct SL"][feature]),
        ("Correct FC", df_analysis[df_analysis["group"] == "Correct FC"][feature]),
        (
            "Misclassified SL->FC",
            df_analysis[df_analysis["group"] == "Misclassified SL->FC"][feature],
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
    print("Starting SL->FC Misclassification Analysis...")
    data = load_data()
    if data is None:
        return

    df = feature_engineering(data)
    df = df.dropna(subset=["pitch_type"])
    le = LabelEncoder()
    y = le.fit_transform(df["pitch_type"])

    _, features_22, new_features_3 = get_feature_lists()

    # Analyze important features + basic ones
    features_to_analyze = new_features_3 + [
        "release_speed",
        "pfx_x",
        "pfx_z",
        "spin_axis",
        "normalized_spin_axis",
        "release_pos_x",
        "release_pos_z",
        "spin_efficiency",
        "movement_angle",
        "spin_axis_deviation_from_fastball",
        "speed_spin_ratio",
    ]

    X_22 = df[features_22].values

    # We need indices to track back to original dataframe for analysis
    indices = np.arange(len(df))
    X_train, X_valid, y_train, y_valid, idx_train, idx_valid = train_test_split(
        X_22, y, indices, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    print("Training model with 22 features...")
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
        sl_idx = list(le.classes_).index("SL")
        fc_idx = list(le.classes_).index("FC")
    except ValueError:
        print("SL or FC not found in classes")
        return

    # Identify sample indices
    mask_correct_sl = (y_valid == sl_idx) & (y_pred == sl_idx)
    mask_correct_fc = (y_valid == fc_idx) & (y_pred == fc_idx)
    mask_mis_sl_fc = (y_valid == sl_idx) & (y_pred == fc_idx)

    idx_correct_sl = idx_valid[mask_correct_sl]
    idx_correct_fc = idx_valid[mask_correct_fc]
    idx_mis_sl_fc = idx_valid[mask_mis_sl_fc]

    print(f"Count Correct SL: {len(idx_correct_sl)}")
    print(f"Count Correct FC: {len(idx_correct_fc)}")
    print(f"Count Misclassified SL->FC: {len(idx_mis_sl_fc)}")

    if len(idx_mis_sl_fc) == 0:
        print("No misclassified SL->FC samples found in validation set. Exiting.")
        return

    # Create analysis dataframe
    df_sl_correct = df.iloc[idx_correct_sl].copy()
    df_sl_correct["group"] = "Correct SL"

    df_fc_correct = df.iloc[idx_correct_fc].copy()
    df_fc_correct["group"] = "Correct FC"

    df_sl_mis = df.iloc[idx_mis_sl_fc].copy()
    df_sl_mis["group"] = "Misclassified SL->FC"

    df_analysis = pd.concat([df_sl_correct, df_fc_correct, df_sl_mis])

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

            # Statistical Comparison: Misclassified SL vs Correct FC
            mis_sl_data = df_sl_mis[feature]
            cor_fc_data = df_fc_correct[feature]
            cor_sl_data = df_sl_correct[feature]

            # KS Test (similarity of distributions)
            ks_stat, ks_p = stats.ks_2samp(mis_sl_data, cor_fc_data)

            # Cohen's d (effect size of difference)
            d_val = calculate_cohens_d(mis_sl_data, cor_fc_data)

            print(f"--- Comparison: Misclassified SL vs Correct FC ---")
            print(
                f"Difference in Means: {np.mean(mis_sl_data) - np.mean(cor_fc_data):.4f}"
            )
            print(f"Cohen's d: {d_val:.4f} (Close to 0 means very similar)")
            print(f"KS Statistic: {ks_stat:.4f} (Lower = more similar distributions)")

            # Comparison: Misclassified SL vs Correct SL (to show how different they are)
            d_val_sl = calculate_cohens_d(mis_sl_data, cor_sl_data)
            print(f"--- Comparison: Misclassified SL vs Correct SL ---")
            print(f"Cohen's d: {d_val_sl:.4f} (Higher means very different)")

            plot_distributions(df_analysis, feature, "SL vs FC Analysis")


if __name__ == "__main__":
    main()
