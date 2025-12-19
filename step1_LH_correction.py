import pandas as pd
import os


def load_data():
    """Loads train.csv and test.csv if they exist."""
    if os.path.exists("train.csv") and os.path.exists("test.csv"):
        print("Loading existing train.csv and test.csv...")
        train_data = pd.read_csv("train.csv")
        test_data = pd.read_csv("test.csv")
        return train_data, test_data
    else:
        print("Error: train.csv or test.csv not found.")
        return None, None


def normalize_lhp(df):
    """
    Normalizes Left-Handed Pitcher (LHP) data to look like Right-Handed Pitcher (RHP) data.

    Transformations for LHP (p_throws == 1):
    - x-coordinates (pfx_x, release_pos_x, plate_x) -> multiplied by -1
    - spin_axis -> 360 - spin_axis
    """
    if "p_throws" not in df.columns:
        print("Warning: 'p_throws' column not found. Skipping normalization.")
        return df

    # Create a mask for LHP
    # Assuming standard encoding from reference notebook: R=0, L=1
    lhp_mask = df["p_throws"] == 1

    print(f"Transforming {lhp_mask.sum()} LHP rows out of {len(df)} total rows.")

    # Columns to flip (multiply by -1)
    x_cols = ["pfx_x", "release_pos_x", "plate_x"]
    for col in x_cols:
        if col in df.columns:
            print(f"Flipping {col} for LHP...")
            df.loc[lhp_mask, col] = df.loc[lhp_mask, col] * -1

    # Transform spin_axis (360 - value)
    if "spin_axis" in df.columns:
        print("Reflecting spin_axis for LHP...")
        # Make sure to handle potential NaN or existing values correctly if needed,
        # but basic arithmetic should work fine on series.
        # Note: 360 - x effectively reflects it across the vertical axis in polar coordinates
        # (or just flips the direction of spin).
        df.loc[lhp_mask, "spin_axis"] = 360 - df.loc[lhp_mask, "spin_axis"]

    return df


def main():
    train_df, test_df = load_data()

    if train_df is not None and test_df is not None:
        print("Applying LHP normalization to Train data...")
        train_df_corrected = normalize_lhp(train_df.copy())

        print("Applying LHP normalization to Test data...")
        test_df_corrected = normalize_lhp(test_df.copy())

        # Save corrections
        train_df_corrected.to_csv("train_corrected.csv", index=False)
        test_df_corrected.to_csv("test_corrected.csv", index=False)
        print("Saved corrected data to 'train_corrected.csv' and 'test_corrected.csv'.")

        # Verification snippet
        print("\nVerification (First 5 LHP rows from Train):")
        lhp_train = train_df[train_df["p_throws"] == 1].head(5)
        lhp_corr = train_df_corrected[train_df_corrected["p_throws"] == 1].head(5)

        cols_to_check = [
            c for c in ["pfx_x", "release_pos_x", "spin_axis"] if c in train_df.columns
        ]

        print("Original:")
        print(lhp_train[cols_to_check])
        print("Corrected:")
        print(lhp_corr[cols_to_check])


if __name__ == "__main__":
    main()
