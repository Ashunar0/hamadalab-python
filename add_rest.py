
import json
import os

notebook_path = '/Users/ryugohanai/Documents/mlb_war_project/hamadalab-python/pitch_classification_hanai2copy.ipynb'

# New cells content for 8.2, 8.3, 8.4
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 8.2 ベースラインモデル（Best Model）の定義 (セクション 2)\n",
            "\n",
            "本ノートブックにおける「Best Model」としてのベースラインモデル (`rfc_baseline`) は、ランダムフォレスト分類器として以下のように定義されています。"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ベースラインモデルの定義（セクション2より）\n",
            "rfc_baseline = RandomForestClassifier(max_depth=10, min_samples_leaf=1, n_estimators=100, n_jobs=-1, random_state=42)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 8.3 ベースラインモデルの混同行列（％表示）\n",
            "\n",
            "各クラスごとの正解率をヒートマップで可視化します（数値はパーセンテージで表示）。"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 混同行列（％表示）の作成と表示\n",
            "cm = confusion_matrix(y_valid, baseline_pred)\n",
            "# 正規化（各行の合計で割ることで、正解ラベルごとの割合を算出）\n",
            "cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100\n",
            "\n",
            "plt.figure(figsize=(10, 8))\n",
            "sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', \n",
            "            xticklabels=le.classes_, yticklabels=le.classes_)\n",
            "plt.xlabel('Predicted Label')\n",
            "plt.ylabel('True Label')\n",
            "plt.title('Confusion Matrix (Baseline Model) %')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 8.4 ベースラインモデルの分類レポート\n",
            "\n",
            "各球種ごとの精度（Precision, Recall, F1-score）を一覧表示します。"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.metrics import classification_report\n",
            "\n",
            "print(\"Classification Report (Baseline Model):\")\n",
            "print(classification_report(y_valid, baseline_pred, target_names=le.classes_))"
        ]
    }
]

with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Append new cells
print("Appending new cells...")
notebook['cells'].extend(new_cells)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Successfully appended sections 8.2, 8.3, and 8.4.")
