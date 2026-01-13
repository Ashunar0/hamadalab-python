import pandas as pd
import numpy as np
import os


def analyze_sl_st():
    print("Loading data...")
    if os.path.exists("train_with_features.csv"):
        df = pd.read_csv("train_with_features.csv")
    else:
        print("Error: train_with_features.csv not found.")
        return

    # Filter for SL and ST
    target_pitches = ["SL", "ST"]
    df_subset = df[df["pitch_type"].isin(target_pitches)].copy()

    print(f"SL/ST Subset Shape: {df_subset.shape}")
    print(df_subset["pitch_type"].value_counts())

    # Feature Stats
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

    # Percentiles for pfx_x (Horizontal Break) is usually the key difference
    print("\nHorizontal Break (pfx_x) Percentiles:")
    for p_type in target_pitches:
        subset = df_subset[df_subset["pitch_type"] == p_type]
        percs = np.percentile(subset["pfx_x"].dropna(), [10, 25, 50, 75, 90])
        print(
            f"  {p_type}: 10%={percs[0]:.2f}, 25%={percs[1]:.2f}, 50%={percs[2]:.2f}, 75%={percs[3]:.2f}, 90%={percs[4]:.2f}"
        )

    # Check Overlap in 'Grey Zone'
    # Sweepers essentially have MORE horizontal break (negative pfx_x for RHP, but data might be mixed handedness so check abs?)
    # Usually pfx_x is negative for RHP sliders.
    # Let's check the sign of pfx_x.

    print("\npfx_x sign check (to infer handedness mix):")
    print(df_subset.groupby("pitch_type")["pfx_x"].apply(lambda x: (x > 0).mean()))

    # If mixed, we might need to look at abs(pfx_x) or handle L/R separately if handedness column exists.
    # Checking columns
    print("\nColumns:", df.columns.tolist())


if __name__ == "__main__":
    analyze_sl_st()
