import json


def patch_notebook_sl_st():
    notebook_path = "asao_improvement_sl_st.ipynb"

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if "id" in cell:
            del cell["id"]

        if cell["cell_type"] == "code":
            source = cell["source"]
            is_target_cell = False

            # Find the same cell where we added the Others->FF fix
            for line in source:
                if (
                    "df['pitch_type'] = df['pitch_type'].replace(rare_pitches, 'Others')"
                    in line
                ):
                    is_target_cell = True
                    break

            if is_target_cell:
                print("Found target cell.")
                # We want to APPEND the SL/ST fix after the Others->FF fix
                # But first, let's identify if the Others->FF fix is already there (it should be since we copied)
                # We will check if "Others -> FF" is in the source.

                # Careful: The previous patch might have left the cell in a specific state.
                # Let's read the current source to be sure where to append.
                # It likely ends with `print(df['pitch_type'].value_counts())`

                # I will append the new logic at the very end of this cell

                new_logic = [
                    "\n",
                    "# Improvement 2: SL (Slider) vs ST (Sweeper) Cleanup\n",
                    "# 背景: STは横変化(pfx_x)が大きいSLの一種であり、境界が曖昧。\n",
                    "# 分析: SLで横変化が大きい(>0.9)ものはSTに近く（球速も遅い）、STで横変化が小さい(<0.8)ものはSLに近い（球速も速い）。\n",
                    "# 対策: 物理的特徴に基づいてラベルを修正し、モデルの混乱を防ぐ。\n",
                    "\n",
                    "# 1. SL -> ST (High Break)\n",
                    "sl_to_st_mask = (df['pitch_type'] == 'SL') & (df['pfx_x'] > 0.9)\n",
                    'print(f"\\nSLのうち、STに近い(pfx_x > 0.9)投球数: {sl_to_st_mask.sum()}")\n',
                    "if sl_to_st_mask.sum() > 0:\n",
                    "    df.loc[sl_to_st_mask, 'pitch_type'] = 'ST'\n",
                    "\n",
                    "# 2. ST -> SL (Low Break)\n",
                    "st_to_sl_mask = (df['pitch_type'] == 'ST') & (df['pfx_x'] < 0.8)\n",
                    'print(f"STのうち、SLに近い(pfx_x < 0.8)投球数: {st_to_sl_mask.sum()}")\n',
                    "if st_to_sl_mask.sum() > 0:\n",
                    "    df.loc[st_to_sl_mask, 'pitch_type'] = 'SL'\n",
                    "\n",
                    'print("\\nSL/ST ラベル修正後の球種の内訳:")\n',
                    "print(df['pitch_type'].value_counts())\n",
                ]

                source.extend(new_logic)
                cell["source"] = source
                print("Patch applied (SL/ST logic appended).")

    # Ensure nbformat minor
    if nb.get("nbformat") == 4 and nb.get("nbformat_minor", 0) > 4:
        nb["nbformat_minor"] = 4

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook patched (SL/ST) successfully.")


if __name__ == "__main__":
    patch_notebook_sl_st()
