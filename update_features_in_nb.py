import json

notebook_path = "/Users/asaoyushi/Documents/02_hamadalab/01_B3/03_sem-b3-python/sem-b3-python-project/asao_step0_1219-with_features.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# The new features list
new_features_line = "new_features = ['normalized_spin_axis', 'movement_angle', 'abs_horizontal_movement', 'movement_magnitude', 'spin_efficiency', 'speed_spin_ratio', 'horizontal_vertical_ratio', 'release_position_magnitude']\n"
feature_update_line = "feature_cols_fe = feature_cols + [feat for feat in new_features if feat in df_fe.columns]\n"

found = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        # Look for the specific line to identify the correct cell
        for i, line in enumerate(source):
            if "feature_cols_fe = feature_cols  # 元の特徴量のみを使用" in line:
                # We found the target line.
                # We will clear the previous commented out lines about new_features just before it if they exist
                # But to be safe and clean, let's just replace this block.

                # Check context
                # source[i-1] might be "# feature_cols_fe = ..."
                # source[i-2] might be "# new_features = ..."

                # Let's verify context for precise replacement
                start_idx = i
                # Walk back to find the start of the block if possible, or just replace the line.
                # The user wants "All models to use 16 features" so we should definitely inject the new_features definition.

                # Based on file view:
                # 1184: "# new_features = ['spin_axis_sin', ...]"
                # 1185: "# feature_cols_fe = ..."
                # 1186: "feature_cols_fe = feature_cols ..."

                if (
                    i >= 2
                    and "new_features =" in source[i - 2]
                    and "feature_cols_fe =" in source[i - 1]
                ):
                    # Replace the 3 lines with our 2 lines
                    source[i - 2] = new_features_line
                    source[i - 1] = feature_update_line
                    source[i] = ""  # Clear the old line or just remove it.
                    # Better to assign slices
                    cell["source"] = (
                        source[: i - 2]
                        + [new_features_line, feature_update_line]
                        + source[i + 1 :]
                    )
                else:
                    # Fallback if precise lines differ: just replace the target line with the definition + update
                    cell["source"][i] = new_features_line + feature_update_line

                found = True
                break
        if found:
            break

if found:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook updated successfully.")
else:
    print("Target line not found.")
