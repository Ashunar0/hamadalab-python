"""
ノートブックにSI/FF用の3つの新特徴量を追加するスクリプト
"""

import json

NOTEBOOK_PATH = "asao_step0_1219-with_features copy.ipynb"

# 追加する特徴量計算コード
NEW_FEATURES_CODE = """
# === SI/FF 識別用の追加特徴量 (3つ) ===
print("\\n" + "=" * 60)
print("SI/FF 識別用の追加特徴量 (3つ)")
print("=" * 60)

# 1. vertical_rise (d=1.975) - 縦方向の浮き上がり成分
train_df["vertical_rise"] = train_df["pfx_z"]
test_df["vertical_rise"] = test_df["pfx_z"]
print("✓ vertical_rise: 縦方向の浮き上がり成分 (SI=沈む, FF=浮く)")

# 2. sink_rate (d=0.544) - 沈み率
train_df["sink_rate"] = -train_df["pfx_z"] / (np.abs(train_df["pfx_x"]) + 0.01)
test_df["sink_rate"] = -test_df["pfx_z"] / (np.abs(test_df["pfx_x"]) + 0.01)
print("✓ sink_rate: 沈み率 (負の値=沈む、SI特有)")

# 3. spin_axis_deviation_from_fastball (d=0.713) - 4シームからの回転軸のずれ
train_df["spin_axis_deviation_from_fastball"] = np.abs(train_df["normalized_spin_axis"] - 180)
test_df["spin_axis_deviation_from_fastball"] = np.abs(test_df["normalized_spin_axis"] - 180)
print("✓ spin_axis_deviation_from_fastball: 4シーム回転軸(180°)からのずれ")

print(f"\\n合計特徴量数: {len(train_df.columns)}")
"""


def main():
    print(f"ノートブックを読み込み中: {NOTEBOOK_PATH}")

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    cells = notebook["cells"]

    # 特徴量計算セルを探す（normalized_spin_axisを含むセル）
    feature_cell_idx = None
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "normalized_spin_axis" in source and "train_df" in source:
                feature_cell_idx = i
                print(f"特徴量計算セル発見: セル {i}")
                break

    if feature_cell_idx is None:
        print("エラー: 特徴量計算セルが見つかりません")
        return

    # 新しいセルを作成
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": NEW_FEATURES_CODE.strip().split("\n"),
    }
    # 各行に改行を追加
    new_cell["source"] = [line + "\n" for line in new_cell["source"][:-1]] + [
        new_cell["source"][-1]
    ]

    # 特徴量計算セルの後に挿入
    cells.insert(feature_cell_idx + 1, new_cell)
    print(f"新しいセルをセル {feature_cell_idx + 1} に挿入しました")

    # feature_cols_fe を探して更新
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "feature_cols_fe" in source and "release_position_magnitude" in source:
                print(f"feature_cols_fe セル発見: セル {i}")
                # 新しい特徴量を追加
                old_source = "".join(cell["source"])
                if "vertical_rise" not in old_source:
                    new_source = old_source.replace(
                        "'release_position_magnitude'",
                        "'release_position_magnitude',\n    # SI/FF 識別用の追加特徴量\n    'vertical_rise',\n    'sink_rate',\n    'spin_axis_deviation_from_fastball'",
                    )
                    cell["source"] = [new_source]
                    print("  → 3つの新特徴量を feature_cols_fe に追加しました")

    # 保存
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

    print(f"\n✓ ノートブックを更新しました: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
