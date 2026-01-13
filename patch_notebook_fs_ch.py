import json


def patch_notebook_fs_ch():
    notebook_path = "asao_improvement_fs_ch.ipynb"

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if "id" in cell:
            del cell["id"]

        if cell["cell_type"] == "code":
            source = cell["source"]
            is_target_cell = False

            # Find where we added the SL/ST fix in the previous step
            # It ends with printing SL/ST counts
            for line in source:
                if (
                    'print("SL/ST ラベル修正後の球種の内訳:")' in line
                    or 'print("SL/ST ラベル修正後の球種の内訳:")' in line
                ):
                    is_target_cell = True
                    break

            if is_target_cell:
                print("Found target cell.")
                # Append FS/CH logic

                new_logic = [
                    "\n",
                    "# Improvement 3: FS (Splitter) vs CH (Changeup) Cleanup\n",
                    "# 背景: FSとCHは球速・軌道が似ているが、FSはスピンが少なく(落ちる)、CHはスピンが多い(横に曲がる傾向)。\n",
                    "# 分析: FSでスピンが多く落ちないものはCHに近く、CHでスピンが少なく落ちるものはFSに近い。\n",
                    "# 対策: Spin Rate と pfx_z (縦変化) を組み合わせてラベルを修正。\n",
                    "\n",
                    "# 1. FS -> CH (High Spin & Less Drop)\n",
                    "# Criteria: Spin > 1700 rpm AND pfx_z > 0.4 (CH Mean 0.47)\n",
                    "fs_to_ch_mask = (df['pitch_type'] == 'FS') & (df['release_spin_rate'] > 1700) & (df['pfx_z'] > 0.4)\n",
                    'print(f"\\nFSのうち、CHに近い(Spin>1700, pfx_z>0.4)投球数: {fs_to_ch_mask.sum()}")\n',
                    "if fs_to_ch_mask.sum() > 0:\n",
                    "    df.loc[fs_to_ch_mask, 'pitch_type'] = 'CH'\n",
                    "\n",
                    "# 2. CH -> FS (Low Spin & More Drop)\n",
                    "# Criteria: Spin < 1400 rpm AND pfx_z < 0.35 (FS Mean 0.28)\n",
                    "ch_to_fs_mask = (df['pitch_type'] == 'CH') & (df['release_spin_rate'] < 1400) & (df['pfx_z'] < 0.35)\n",
                    'print(f"CHのうち、FSに近い(Spin<1400, pfx_z<0.35)投球数: {ch_to_fs_mask.sum()}")\n',
                    "if ch_to_fs_mask.sum() > 0:\n",
                    "    df.loc[ch_to_fs_mask, 'pitch_type'] = 'FS'\n",
                    "\n",
                    'print("\\nFS/CH ラベル修正後の球種の内訳:")\n',
                    "print(df['pitch_type'].value_counts())\n",
                ]

                source.extend(new_logic)
                cell["source"] = source
                print("Patch applied (FS/CH logic appended).")

    # Ensure nbformat minor
    if nb.get("nbformat") == 4 and nb.get("nbformat_minor", 0) > 4:
        nb["nbformat_minor"] = 4

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook patched (FS/CH) successfully.")


if __name__ == "__main__":
    patch_notebook_fs_ch()
