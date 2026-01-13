import json
import os

NOTEBOOK_PATH = "asao_step0_1310-with_features_total_22.ipynb"


def update_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: {NOTEBOOK_PATH} not found.")
        return

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Update param_grid_xgb
    cell_found = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source_text = "".join(cell["source"])
            if "param_grid_xgb = {" in source_text:
                print("Found XGBoost param grid cell.")

                new_source = [
                    "# ハイパーパラメータチューニング - XGBoost\n",
                    "# 誤分類の多いSI->FF（中間的な球種）を救うため、min_child_weightを小さく、max_depthを深くする探索範囲を追加\n",
                    "param_grid_xgb = {\n",
                    "    'max_depth': [6, 8, 10],\n",
                    "    'learning_rate': [0.05, 0.1],\n",
                    "    'n_estimators': [200],\n",
                    "    'min_child_weight': [1, 0.5]\n",
                    "}\n",
                    "\n",
                    'print("XGBoost GridSearchCVを実行中...")\n',
                    "xgb_gs = GridSearchCV(\n",
                    "    xgb.XGBClassifier(random_state=42, n_jobs=-1),\n",
                    "    param_grid_xgb,\n",
                    "    cv=3,\n",
                    "    scoring='accuracy',\n",
                    "    n_jobs=-1\n",
                    ")\n",
                    "xgb_gs.fit(X_fe_train, y_fe_train)\n",
                    "\n",
                    "print('XGBoost Best Params:', xgb_gs.best_params_)\n",
                    "print('XGBoost Valid Score:', round(xgb_gs.score(X_fe_valid, y_fe_valid), 3))\n",
                    "\n",
                    "xgb_model = xgb_gs.best_estimator_",
                ]

                cell["source"] = new_source
                cell_found = True
                break

    if not cell_found:
        print("Could not find XGBoost param grid cell.")
        return

    # Save
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Successfully updated {NOTEBOOK_PATH}")


if __name__ == "__main__":
    update_notebook()
