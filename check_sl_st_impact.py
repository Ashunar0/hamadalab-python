import pandas as pd
import os


def check_relabel_impact():
    if not os.path.exists("train_with_features.csv"):
        return

    df = pd.read_csv("train_with_features.csv")

    # Filter SL/ST
    sl_mask = df["pitch_type"] == "SL"
    st_mask = df["pitch_type"] == "ST"

    print(f"Original SL: {sl_mask.sum()}")
    print(f"Original ST: {st_mask.sum()}")

    # Thresholds
    # Attempt 1: 0.85 split
    sl_to_st = df[sl_mask & (df["pfx_x"] > 0.9)]
    st_to_sl = df[st_mask & (df["pfx_x"] < 0.8)]

    print(f"\nPotential Changes:")
    print(
        f"SL -> ST (pfx_x > 0.9): {len(sl_to_st)} ({len(sl_to_st) / sl_mask.sum() * 100:.1f}%)"
    )
    print(
        f"ST -> SL (pfx_x < 0.8): {len(st_to_sl)} ({len(st_to_sl) / st_mask.sum() * 100:.1f}%)"
    )

    # Checking velocity of these candidates
    print("\nVelocity Check of Candidates:")
    print(f"SL->ST Candidates Mean Speed: {sl_to_st['release_speed'].mean():.2f}")
    print(f"ST->SL Candidates Mean Speed: {st_to_sl['release_speed'].mean():.2f}")
    print(f"Global SL Mean Speed: {df[sl_mask]['release_speed'].mean():.2f}")
    print(f"Global ST Mean Speed: {df[st_mask]['release_speed'].mean():.2f}")


if __name__ == "__main__":
    check_relabel_impact()
