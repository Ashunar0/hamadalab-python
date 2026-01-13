import pandas as pd
import numpy as np
import os


def analyze_fs_ch():
    print("Loading data...")
    if os.path.exists("train_with_features.csv"):
        df = pd.read_csv("train_with_features.csv")
    else:
        print("Error: train_with_features.csv not found.")
        return

    # Filter for FS and CH
    target_pitches = ["FS", "CH"]
    df_subset = df[df["pitch_type"].isin(target_pitches)].copy()

    print(f"FS/CH Subset Shape: {df_subset.shape}")
    print(df_subset["pitch_type"].value_counts())

    # Feature Stats
    # Key features for FS vs CH: Speed, Vertical Drop (pfx_z), Horizontal Fade (pfx_x)
    features = ["release_speed", "pfx_x", "pfx_z", "release_spin_rate"]

    print("\nFeature Comparison (Mean +/- Std):")
    for p_type in target_pitches:
        subset = df_subset[df_subset["pitch_type"] == p_type]
        print(f"\nType: {p_type} (n={len(subset)})")
        for feat in features:
            if feat in subset.columns:
                print(
                    f"  {feat}: {subset[feat].mean():.2f} +/- {subset[feat].std():.2f}"
                )

    # Percentiles for key features
    print("\nPercentiles (Speed):")
    for p_type in target_pitches:
        subset = df_subset[df_subset["pitch_type"] == p_type]
        percs = np.percentile(subset["release_speed"].dropna(), [10, 50, 90])
        print(f"  {p_type}: 10%={percs[0]:.2f}, 50%={percs[1]:.2f}, 90%={percs[2]:.2f}")

    print("\nPercentiles (pfx_z - Vertical):")
    for p_type in target_pitches:
        subset = df_subset[df_subset["pitch_type"] == p_type]
        percs = np.percentile(subset["pfx_z"].dropna(), [10, 50, 90])
        print(f"  {p_type}: 10%={percs[0]:.2f}, 50%={percs[1]:.2f}, 90%={percs[2]:.2f}")

    print("\nPercentiles (pfx_x - Horizontal):")
    for p_type in target_pitches:
        subset = df_subset[df_subset["pitch_type"] == p_type]
        percs = np.percentile(subset["pfx_x"].dropna(), [10, 50, 90])
        print(f"  {p_type}: 10%={percs[0]:.2f}, 50%={percs[1]:.2f}, 90%={percs[2]:.2f}")


if __name__ == "__main__":
    analyze_fs_ch()
