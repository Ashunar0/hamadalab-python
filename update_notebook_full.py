import json
import re

NOTEBOOK_PATH = "asao_step0_1219-with_features copy.ipynb"


def update_notebook_models():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    updated_count = 0

    # 1. 念のため、特徴量リスト定義セルも再確認・更新（前回スクリプトで行ったが、確実にするため）
    # 特に、もし古いリストが別の場所で再定義されていたら削除・統一する

    # 2. モデル訓練部分での特徴量指定を確認
    # 多くの場合、 `X_fe` や `feature_cols_fe` 変数が使われていれば自動的に反映されるはずだが、
    # 明示的にカラム名をリストしている箇所がないかチェックする。

    # パターン: GridSearchCVやモデル定義のセルを探す
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            source_lines = cell.get("source", [])
            source_text = "".join(source_lines)

            # RandomForest の調整
            if "RandomForestClassifier" in source_text and "estimators" in source_text:
                # これはモデル定義セルの可能性が高い。
                # ここで特徴量数を明示しているわけではないが、
                # 特徴量エンジニアリングのセルが正しく実行されていれば問題ない。
                pass

            # XGBoost / LightGBM の調整
            # これらも `X_train_fe` などを使っているはず。

            # チェック: `feature_cols` という変数が、古い8つのまま使われていないか？
            # `feature_cols_fe` (19個) を使うべき場所で `feature_cols` (8個) が使われていたら修正。

            # 例: 特徴量重要度の表示などで `feature_cols` を使っている場合 -> `feature_cols_fe` に変更
            new_source_lines = []
            modified = False
            for line in source_lines:
                # 特徴量重要度のプロットなどで、feature_cols (元8個) を参照している箇所を
                # feature_cols_fe (全19個) に置き換える候補を探す
                if (
                    "feature_names=feature_cols" in line
                    and "feature_cols_fe" not in line
                ):
                    line = line.replace(
                        "feature_names=feature_cols", "feature_names=feature_cols_fe"
                    )
                    modified = True

                # pd.DataFrame(..., columns=feature_cols) のような箇所
                if "columns=feature_cols" in line and "feature_cols_fe" not in line:
                    # 文脈によるが、モデル訓練後の重要度表示なら fe にすべき
                    if "importance" in source_text or "Importance" in source_text:
                        line = line.replace(
                            "columns=feature_cols", "columns=feature_cols_fe"
                        )
                        modified = True

                new_source_lines.append(line)

            if modified:
                cell["source"] = new_source_lines
                updated_count += 1
                print(
                    f"Updated cell {i} to use feature_cols_fe instead of feature_cols"
                )

    # 3. 3つの新特徴量の計算ロジックを追加するセルを挿入
    # 前回の試みで「特徴量計算セルが見つかりません」となったため、
    # より堅牢な検索ロジックで挿入場所を探す

    # ターゲット: `df_fe['spin_per_mph']` や `df_fe['normalized_spin_axis']` を計算しているあたり
    target_idx = -1
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "normalized_spin_axis" in src and "apply" in src:
                target_idx = i
                print(f"Found feature calculation cell at index {i}")
                break

    if target_idx != -1:
        # 新特徴量計算コード
        new_features_code = [
            "# === SI/FF 識別用の追加特徴量 (3つ) ===\n",
            "print('Adding SI/FF specific features...')\n",
            "\n",
            "# 1. vertical_rise (d=1.975) - 縦方向の浮き上がり成分\n",
            "if 'pfx_z' in df_fe.columns:\n",
            "    df_fe['vertical_rise'] = df_fe['pfx_z']\n",
            "\n",
            "# 2. sink_rate (d=0.544) - 沈み率\n",
            "if 'pfx_z' in df_fe.columns and 'pfx_x' in df_fe.columns:\n",
            "    df_fe['sink_rate'] = -df_fe['pfx_z'] / (np.abs(df_fe['pfx_x']) + 0.01)\n",
            "\n",
            "# 3. spin_axis_deviation_from_fastball (d=0.713) - 4シームからの回転軸のずれ\n",
            "if 'normalized_spin_axis' in df_fe.columns:\n",
            "    df_fe['spin_axis_deviation_from_fastball'] = np.abs(df_fe['normalized_spin_axis'] - 180)\n",
            "\n",
            "# NaNチェック\n",
            "print(df_fe[['vertical_rise', 'sink_rate', 'spin_axis_deviation_from_fastball']].isnull().sum())\n",
        ]

        # 既に存在するかチェック
        next_cell_src = (
            "".join(cells[target_idx + 1]["source"])
            if target_idx + 1 < len(cells)
            else ""
        )
        if "vertical_rise" not in next_cell_src:
            new_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": new_features_code,
            }
            cells.insert(target_idx + 1, new_cell)
            print(f"Inserted new feature calculation cell at index {target_idx + 1}")
            updated_count += 1
        else:
            print("Feature calculation cell already seems to exist.")

    if updated_count > 0:
        with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Successfully updated {updated_count} cells in {NOTEBOOK_PATH}")
    else:
        print("No changes needed.")


if __name__ == "__main__":
    update_notebook_models()
