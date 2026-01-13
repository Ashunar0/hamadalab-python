import pandas as pd
import os


def check_fs_ch_impact():
    if not os.path.exists("train_with_features.csv"):
        return

    df = pd.read_csv("train_with_features.csv")

    # Filter FS/CH
    fs_mask = df["pitch_type"] == "FS"
    ch_mask = df["pitch_type"] == "CH"

    print(f"Original FS: {fs_mask.sum()}")
    print(f"Original CH: {ch_mask.sum()}")

    # Thresholds (Hypothesis)
    # FS -> CH: High Spin (> 1700) AND Less Drop (pfx_z > 0.4)
    # CH -> FS: Low Spin (< 1300) AND More Drop (pfx_z < 0.3)

    fs_to_ch = df[fs_mask & (df["release_spin_rate"] > 1800)]  # Simple spin first
    ch_to_fs = df[ch_mask & (df["release_spin_rate"] < 1300)]

    print(f"\nSimple Spin Thresholds:")
    print(
        f"FS -> CH (Spin > 1800): {len(fs_to_ch)} ({len(fs_to_ch) / fs_mask.sum() * 100:.1f}%)"
    )
    print(
        f"CH -> FS (Spin < 1300): {len(ch_to_fs)} ({len(ch_to_fs) / ch_mask.sum() * 100:.1f}%)"
    )

    # Combo Thresholds
    fs_to_ch_combo = df[
        fs_mask & (df["release_spin_rate"] > 1700) & (df["pfx_z"] > 0.4)
    ]
    ch_to_fs_combo = df[
        ch_mask & (df["release_spin_rate"] < 1400) & (df["pfx_z"] < 0.35)
    ]

    print(f"\nCombo Thresholds (Spin + Drop):")
    print(
        f"FS -> CH (Spin > 1700 & pfx_z > 0.4): {len(fs_to_ch_combo)} ({len(fs_to_ch_combo) / fs_mask.sum() * 100:.1f}%)"
    )
    print(
        f"CH -> FS (Spin < 1400 & pfx_z < 0.35): {len(ch_to_fs_combo)} ({len(ch_to_fs_combo) / ch_mask.sum() * 100:.1f}%)"
    )


if __name__ == "__main__":
    check_fs_ch_impact()
