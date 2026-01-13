import json
import os

NOTEBOOK_PATH = "asao_step0_1219-with_features copy.ipynb"

NEW_FEATURES = [
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


def update_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: {NOTEBOOK_PATH} not found")
        return

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    updated = False

    for cell in cells:
        if cell["cell_type"] == "code":
            source = cell.get("source", [])
            # Search for the line defining final_new_features or feature_cols_fe
            for i, line in enumerate(source):
                if "final_new_features =" in line or "feature_cols_fe =" in line:
                    # Found the target cell.
                    # We will replace the entire source of this cell to be safe and clean.
                    # But verifying context: usually this cell defines the final list.

                    # Check if it looks like the one we saw (lines 996-997)
                    # "final_new_features = ['spin_axis_sin', ...]"

                    if "final_new_features =" in line and "[" in line:
                        print("Found feature definition cell.")

                        # Construct new source
                        new_source = []
                        new_source.append("# 特徴量エンジニアリング後の特徴量リスト\n")
                        new_source.append("# 既存の8個の特徴量 + 11個の新特徴量\n")

                        # Define the list nicely
                        features_str = "final_new_features = [\n"
                        for feat in NEW_FEATURES:
                            features_str += f"    '{feat}',\n"
                        features_str += "]\n"

                        new_source.append(features_str)
                        new_source.append(
                            "feature_cols_fe = list(dict.fromkeys(feature_cols + final_new_features))\n"
                        )
                        new_source.append("\n")
                        new_source.append("# 欠損値の処理\n")
                        new_source.append(
                            "df_fe = df_fe.dropna(subset=feature_cols_fe + ['pitch_type'])\n"
                        )
                        new_source.append("\n")
                        new_source.append("# 特徴量とターゲットを抽出\n")
                        new_source.append("X_fe = df_fe[feature_cols_fe].values\n")
                        new_source.append("y_fe = df_fe['pitch_type'].values\n")
                        new_source.append("\n")
                        new_source.append("# ターゲットのエンコーディング\n")
                        new_source.append("le_fe = LabelEncoder()\n")
                        new_source.append("y_fe_encoded = le_fe.fit_transform(y_fe)\n")
                        new_source.append("\n")
                        new_source.append(
                            "# データの分割\n"
                        )  # Preserving the trailing comment if it was there or logically next

                        # Replace source
                        cell["source"] = new_source
                        updated = True
                        break
            if updated:
                break

    if updated:
        with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Successfully updated {NOTEBOOK_PATH}")
    else:
        print("Could not find the cell to update.")


if __name__ == "__main__":
    update_notebook()
