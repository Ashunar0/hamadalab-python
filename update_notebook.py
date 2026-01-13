import json
import os

notebook_path = "/Users/asaoyushi/Documents/02_hamadalab/01_B3/03_sem-b3-python/sem-b3-python-project/asao_step0_1310-with_features_total_22.ipynb"


def update_notebook():
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    # 1. Update calculation cell
    calc_cell_found = False
    for cell in cells:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "spin_per_mph" in source and "df_fe['spin_per_mph'] =" in source:
                # Found the feature engineering cell
                print("Found feature engineering cell.")

                # Check if already added
                if "velocity_abs_pfx_x_ratio" in source:
                    print("Features already added to calculation cell.")
                    calc_cell_found = True
                    break

                # Append new code
                new_code = [
                    "\n",
                    "# 4. SL/FC識別のための追加特徴量 (2026/01/10追加)\n",
                    "# velocity_abs_pfx_x_ratio: 球速 / (|pfx_x| + 0.1)\n",
                    "# velocity_times_pfx_z: 球速 * pfx_z\n",
                    "# pfx_z_minus_abs_pfx_x: pfx_z - |pfx_x|\n",
                    "\n",
                    "if 'release_speed' in df_fe.columns and 'pfx_x' in df_fe.columns:\n",
                    "    df_fe['velocity_abs_pfx_x_ratio'] = df_fe['release_speed'] / (df_fe['pfx_x'].abs() + 0.1)\n",
                    '    print("velocity_abs_pfx_x_ratio を作成しました。")\n',
                    "\n",
                    "if 'release_speed' in df_fe.columns and 'pfx_z' in df_fe.columns:\n",
                    "    df_fe['velocity_times_pfx_z'] = df_fe['release_speed'] * df_fe['pfx_z']\n",
                    '    print("velocity_times_pfx_z を作成しました。")\n',
                    "\n",
                    "if 'pfx_x' in df_fe.columns and 'pfx_z' in df_fe.columns:\n",
                    "    df_fe['pfx_z_minus_abs_pfx_x'] = df_fe['pfx_z'] - df_fe['pfx_x'].abs()\n",
                    '    print("pfx_z_minus_abs_pfx_x を作成しました。")\n',
                ]

                # Insert before the checking part
                # Finding where to insert. The original has:
                # ...
                # else:
                #    print("警告: ...")
                #
                # # 新しい特徴量を確認
                # new_feature_list = ...

                # We append to source before the last part or just append?
                # The original source is a list of strings.

                # Let's find "new_feature_list =" and insert before it
                insert_idx = -1
                for i, line in enumerate(cell["source"]):
                    if "new_feature_list =" in line:
                        insert_idx = i
                        break

                if insert_idx != -1:
                    cell["source"] = (
                        cell["source"][:insert_idx]
                        + new_code
                        + cell["source"][insert_idx:]
                    )
                else:
                    cell["source"].extend(new_code)

                calc_cell_found = True
                break

    if not calc_cell_found:
        print("Could not find suitable cell for calculation or already added.")

    # 2. Update feature list cell
    list_cell_found = False
    for cell in cells:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "final_new_features = [" in source:
                print("Found feature list cell.")

                if "velocity_abs_pfx_x_ratio" in source:
                    print("Features already added to list.")
                    list_cell_found = True
                    break

                # Find the closing bracket ]
                insert_idx = -1
                for i, line in enumerate(cell["source"]):
                    if (
                        "]" in line and "feature_cols_fe =" not in line
                    ):  # closing the list
                        insert_idx = i
                        break

                if insert_idx != -1:
                    new_features = [
                        "    'velocity_abs_pfx_x_ratio',\n",
                        "    'velocity_times_pfx_z',\n",
                        "    'pfx_z_minus_abs_pfx_x',\n",
                    ]
                    cell["source"] = (
                        cell["source"][:insert_idx]
                        + new_features
                        + cell["source"][insert_idx:]
                    )
                    list_cell_found = True
                    break

    if not list_cell_found:
        print("Could not find feature list cell.")

    if calc_cell_found and list_cell_found:
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Updated {notebook_path}")
    else:
        print("Aborted update due to missing cells.")


if __name__ == "__main__":
    update_notebook()
