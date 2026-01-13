import json
import os

NOTEBOOK_PATH = "asao_step0_1219-with_features_total_19.ipynb"


def fix_notebook_errors():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: {NOTEBOOK_PATH} not found")
        return

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])

    # 1. データの分割（train_test_split）を行っているセルを探し、変数が正しく定義されているか確認
    # そして、そのセルが欠損や誤記で機能していない場合は修正する

    split_cell_index = -1
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            # 既に正しい分割ロジックがあるか確認
            if "train_test_split" in source and "X_fe_train" in source:
                print(f"Found existing split cell at index {i}")
                split_cell_index = i
                break

    # 分割セルが見つからない、または不完全な場合
    if split_cell_index == -1:
        # 特徴量定義セルの直後に挿入するのが適切
        # 前回の修正で特徴量を定義したセルを探す
        target_idx = -1
        for i, cell in enumerate(cells):
            src = "".join(cell["source"])
            if "final_new_features =" in src:
                target_idx = i
                print(f"Found feature definition cell at index {i}")
                break

        if target_idx != -1:
            split_code = [
                "# データの分割\n",
                "X_fe_train, X_fe_valid, y_fe_train, y_fe_valid = train_test_split(\n",
                "    X_fe, y_fe_encoded, test_size=0.3, random_state=42, stratify=y_fe_encoded\n",
                ")\n",
                "\n",
                'print(f"Train data shape: {X_fe_train.shape}")\n',
                'print(f"Valid data shape: {X_fe_valid.shape}")\n',
            ]

            new_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": split_code,
            }

            cells.insert(target_idx + 1, new_cell)
            print("Inserted train_test_split cell.")

    # 2. X_fe_train, X_fe_valid を使用しているセルでのエラー回避
    # 基本的に分割セルさえ正しく実行されれば解決するはずだが、
    # もし `X_train_fe` という変数名を使っている場所があれば `X_fe_train` に統一する

    count_renamed = 0
    for cell in cells:
        if cell["cell_type"] == "code":
            new_source = []
            modified = False
            for line in cell["source"]:
                # 変数名の揺らぎを修正 (例: X_train_fe -> X_fe_train)
                # ただし、意図的に別の変数を使っている可能性もあるので慎重に。
                # ここではエラーメッセージにある `Undefined name X_fe_train` から、
                # ノートブック全体で `X_fe_train` を期待しているのに定義されていないことが問題と推測。
                # したがって、分割セルの追加で主原因は解消するはず。

                # 念のため、他の一般的な変数名間違いがあれば修正
                if "X_train_fe" in line and "X_fe_train" not in line:
                    # 文脈依存だが、ここでは安全策として print だけしておく
                    # print(f"Potential variable mismatch in line: {line.strip()}")
                    pass
                new_source.append(line)

            if modified:
                cell["source"] = new_source
                count_renamed += 1

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Updates saved to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    fix_notebook_errors()
