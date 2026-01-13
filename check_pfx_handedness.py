import pandas as pd
import os


def check_handedness():
    if not os.path.exists("train_with_features.csv"):
        return

    df = pd.read_csv("train_with_features.csv")

    # Filter SL/ST
    df = df[df["pitch_type"].isin(["SL", "ST"])]

    print("p_throws counts:")
    print(df["p_throws"].value_counts())

    # Mean pfx_x by Handedness
    print("\nMean pfx_x by Handedness:")
    print(df.groupby(["pitch_type", "p_throws"])["pfx_x"].agg(["mean", "count", "std"]))


if __name__ == "__main__":
    check_handedness()
