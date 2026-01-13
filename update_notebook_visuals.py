import nbformat
import json

notebook_path = "asao_step0_1219-with_features_total_19.ipynb"


def update_notebook():
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Define the new code string
    new_code = """# 混同行列の深掘り分析関数
def plot_binary_confusion_matrix(y_true, y_pred, le, class1, class2):
    \"\"\"特定の2クラス間の混同行列を可視化する\"\"\"
    print(f"\\n" + "=" * 60)
    print(f"{class1} vs {class2} の混同行列")
    print("=" * 60)

    # 全体の混同行列
    cm = confusion_matrix(y_true, y_pred)

    # インデックスを取得
    try:
        class_names = list(le.classes_)
        idx1 = class_names.index(class1)
        idx2 = class_names.index(class2)
    except ValueError:
        print(f"⚠ {class1}または{class2}が見つかりません")
        return

    # 2x2の混同行列を抽出
    # [ [True-1, Mistake-1-as-2],
    #   [Mistake-2-as-1, True-2] ]
    binary_cm = np.array([
        [cm[idx1, idx1], cm[idx1, idx2]],
        [cm[idx2, idx1], cm[idx2, idx2]]
    ])

    # 正規化
    binary_cm_norm = binary_cm.astype('float') / binary_cm.sum(axis=1)[:, np.newaxis]

    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 絶対値
    sns.heatmap(binary_cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=[class1, class2], yticklabels=[class1, class2])
    ax1.set_title(f'{class1} vs {class2} 混同行列 (絶対値)')
    ax1.set_ylabel('実際の球種')
    ax1.set_xlabel('予測された球種')

    # パーセンテージ
    sns.heatmap(binary_cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax2,
                xticklabels=[class1, class2], yticklabels=[class1, class2])
    ax2.set_title(f'{class1} vs {class2} 混同行列 (割合)')
    ax2.set_ylabel('実際の球種')
    ax2.set_xlabel('予測された球種')

    plt.tight_layout()
    plt.show()

    # 統計情報
    correct1 = binary_cm[0, 0]
    total1 = binary_cm[0, :].sum()
    mistake1 = binary_cm[0, 1]

    correct2 = binary_cm[1, 1]
    total2 = binary_cm[1, :].sum()
    mistake2 = binary_cm[1, 0]

    print(f"\\n{class1}:")
    print(f"  正解: {correct1}/{total1} ({correct1/total1*100:.2f}%)")
    print(f"  {class2}と誤分類: {mistake1} ({mistake1/total1*100:.2f}%)")

    print(f"\\n{class2}:")
    print(f"  正解: {correct2}/{total2} ({correct2/total2*100:.2f}%)")
    print(f"  {class1}と誤分類: {mistake2} ({mistake2/total2*100:.2f}%)")

# SL vs FC の分析
plot_binary_confusion_matrix(y_fe_valid, best_pred, le_fe, 'SL', 'FC')

# SI vs FF の分析
plot_binary_confusion_matrix(y_fe_valid, best_pred, le_fe, 'SI', 'FF')"""

    # Find the cell index where the general confusion matrix is plotted
    target_index = -1
    for i, cell in enumerate(nb.cells):
        if (
            cell.cell_type == "code"
            and "cm = confusion_matrix(y_fe_valid, best_pred)" in cell.source
        ):
            target_index = i
            break

    if target_index != -1:
        # Create a new code cell
        new_cell = nbformat.v4.new_code_cell(new_code)

        # Insert the new cell after the target cell
        nb.cells.insert(target_index + 1, new_cell)
        print(
            f"Successfully inserted new visualization cell at index {target_index + 1}"
        )
    else:
        print("Could not find the target confusion matrix cell.")
        # Fallback: find the classification report cell
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == "code" and "classification_report" in cell.source:
                target_index = i

        if target_index != -1:
            # Create a new code cell
            new_cell = nbformat.v4.new_code_cell(new_code)
            # Insert the new cell after the target cell
            nb.cells.insert(target_index + 1, new_cell)
            print(f"Inserted after classification report at index {target_index + 1}")
        else:
            print("Could not find any suitable location to insert.")
            return

    # Write the modified notebook back
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


if __name__ == "__main__":
    update_notebook()
