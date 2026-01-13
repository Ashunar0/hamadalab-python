import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def analyze_fs_ch_deep():
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv("train_with_features.csv")

    # Filter for FS and CH only
    df_fs_ch = df[df["pitch_type"].isin(["FS", "CH"])].copy()

    # Define features to analyze
    # Includes standard features + any engineered ones if available
    features = [
        "release_speed",
        "release_spin_rate",
        "spin_axis",
        "pfx_x",
        "pfx_z",
        "release_pos_x",
        "release_pos_z",
        "movement_angle",
        "spin_efficiency",
    ]

    # Check which features exist
    available_features = [f for f in features if f in df.columns]

    print(f"\nAnalyzing {len(available_features)} features for FS vs CH separation.")
    print(f"FS count: {len(df_fs_ch[df_fs_ch['pitch_type'] == 'FS'])}")
    print(f"CH count: {len(df_fs_ch[df_fs_ch['pitch_type'] == 'CH'])}")

    # 1. Statistical Separation Analysis (KS Test & Cohen's d)
    results = []

    for feat in available_features:
        fs_data = df_fs_ch[df_fs_ch["pitch_type"] == "FS"][feat].dropna()
        ch_data = df_fs_ch[df_fs_ch["pitch_type"] == "CH"][feat].dropna()

        # Means and Stds
        fs_mean = fs_data.mean()
        ch_mean = ch_data.mean()
        fs_std = fs_data.std()
        ch_std = ch_data.std()

        # Cohen's d (Effect Size)
        pooled_std = np.sqrt((fs_std**2 + ch_std**2) / 2)
        cohens_d = abs(fs_mean - ch_mean) / pooled_std

        # KS Test (Distribution Difference)
        ks_stat, p_val = stats.ks_2samp(fs_data, ch_data)

        results.append(
            {
                "Feature": feat,
                "FS_Mean": fs_mean,
                "CH_Mean": ch_mean,
                "Diff": fs_mean - ch_mean,
                "Cohens_D": cohens_d,
                "KS_Stat": ks_stat,
            }
        )

    results_df = pd.DataFrame(results).sort_values("Cohens_D", ascending=False)
    print("\nFeature Separation Analysis (Sorted by Effect Size):")
    print(results_df.to_string(index=False))

    # 2. Deep Dive into Top Discriminators
    top_feature = results_df.iloc[0]["Feature"]
    print(f"\nTop Discriminator: {top_feature}")

    # 3. Analyze 2D Interaction (Top 2 Features)
    # Usually Spin vs pfx_z, or Speed vs pfx_x
    # Let's check pfx_z vs pfx_x (Movement Profile)
    print("\nAnalyzing Movement Profile (pfx_x vs pfx_z):")

    # Define zones
    # FS: Typically lower pfx_z (drops more), lower pfx_x (less fade)
    # CH: Higher pfx_z (float/fade), higher pfx_x (more fade)

    fs_mean_x = df_fs_ch[df_fs_ch["pitch_type"] == "FS"]["pfx_x"].mean()
    ch_mean_x = df_fs_ch[df_fs_ch["pitch_type"] == "CH"]["pfx_x"].mean()
    fs_mean_z = df_fs_ch[df_fs_ch["pitch_type"] == "FS"]["pfx_z"].mean()
    ch_mean_z = df_fs_ch[df_fs_ch["pitch_type"] == "CH"]["pfx_z"].mean()

    print(
        f"Centroids -> FS: ({fs_mean_x:.2f}, {fs_mean_z:.2f}), CH: ({ch_mean_x:.2f}, {ch_mean_z:.2f})"
    )

    # 4. Find 'Pure' clusters vs 'Overlap' clusters
    # We want to identify the subset of FS that looks like CH, and vice versa

    # Case A: FS looking like CH
    # High Spin (>1800), High pfx_z (> 0.5), High pfx_x (if separation exists)

    # Let's test a more aggressive combo
    # Calculate how many FS would be flipped if we used a linear boundary between centroids

    # 5. Handedness Check
    # pfx_x sign depends on handedness. Need to normalize or split.
    # Assuming 'p_throws' column exists. If not, infer from pfx_x distribution (bimodal).

    # 5. Handedness Analysis
    if "p_throws" in df.columns:
        print("\nSeparating by Handedness (0/1):")
        for hand in [0, 1]:
            subset = df_fs_ch[df_fs_ch["p_throws"] == hand]
            count = len(subset)
            print(f"Hand: {hand}, Count: {count}")

            if count > 0:
                print(
                    subset.groupby("pitch_type")[
                        ["pfx_x", "pfx_z", "release_spin_rate"]
                    ].mean()
                )

                # Deduce Handedness
                mean_pfx_x = subset["pfx_x"].mean()
                hand_str = (
                    "RHP" if mean_pfx_x < 0 else "LHP"
                )  # Assuming general population is dominated by CH/FS which fade arm-side
                print(f"  -> Likely {hand_str} (Mean pfx_x: {mean_pfx_x:.2f})")

                # 6. Simulate Round 2 Logic (Spin + pfx_x)
                # Logic:
                # FS (Splitter) should have LOWER Spin and LESS Fade (closer to 0 pfx_x)
                # CH (Changeup) should have HIGHER Spin and MORE Fade (farther from 0 pfx_x)

                # Determine "More Fade" direction
                fade_dir = -1 if mean_pfx_x < 0 else 1

                # Thresholds (Aggressive)
                # Spin Boundary: ~1600 (Midpoint between 1357 and 1795)
                # Fade Boundary: Mean of means (-1.05 for RHP)

                # Let's try combinatorial logic
                # FS -> CH candidates: High Spin (>1600) AND Large Fade (abs(pfx_x) > 1.2)
                # CH -> FS candidates: Low Spin (<1500) AND Small Fade (abs(pfx_x) < 0.8)

                candidates_fs_to_ch = subset[
                    (subset["pitch_type"] == "FS")
                    & (subset["release_spin_rate"] > 1600)
                    & (subset["pfx_x"].abs() > 1.2)
                ]

                candidates_ch_to_fs = subset[
                    (subset["pitch_type"] == "CH")
                    & (subset["release_spin_rate"] < 1500)
                    & (subset["pfx_x"].abs() < 0.8)
                ]

                print(
                    f"  [Sim] FS -> CH Candidates: {len(candidates_fs_to_ch)} ({len(candidates_fs_to_ch) / len(subset[subset['pitch_type'] == 'FS']) * 100:.1f}%)"
                )
                print(
                    f"  [Sim] CH -> FS Candidates: {len(candidates_ch_to_fs)} ({len(candidates_ch_to_fs) / len(subset[subset['pitch_type'] == 'CH']) * 100:.1f}%)"
                )


if __name__ == "__main__":
    analyze_fs_ch_deep()
