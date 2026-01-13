import json
import nbformat

notebook_path = "/Users/asaoyushi/Documents/02_hamadalab/01_B3/03_sem-b3-python/sem-b3-python-project/asao_step0_1219-with_features.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)


# Helper to find cell index by content snippet
def find_cell_index(notebook, snippet):
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if snippet in source:
                return i
    return -1


# 1. Replace RF Baseline with RF GridSearch
rf_snippet = "rfc_fe = RandomForestClassifier(max_depth=7"
rf_idx = find_cell_index(nb, rf_snippet)
if rf_idx != -1:
    print(f"Index {rf_idx}: Updating RF to GridSearchCV")
    nb["cells"][rf_idx]["source"] = [
        "# ハイパーパラメータチューニング - Random Forest\n",
        "param_grid_rf = {\n",
        "    'max_depth': [7, 10, None],\n",
        "    'n_estimators': [100, 200],\n",
        "    'min_samples_leaf': [1, 2]\n",
        "}\n",
        "\n",
        'print("RandomForest GridSearchCVを実行中...")\n',
        "rfc_gs = GridSearchCV(\n",
        "    RandomForestClassifier(random_state=42, n_jobs=-1),\n",
        "    param_grid_rf,\n",
        "    cv=3,\n",
        "    scoring='accuracy',\n",
        "    n_jobs=-1\n",
        ")\n",
        "rfc_gs.fit(X_fe_train, y_fe_train)\n",
        "\n",
        "print('Random Forest Best Params:', rfc_gs.best_params_)\n",
        "print('Random Forest Valid Score:', round(rfc_gs.score(X_fe_valid, y_fe_valid), 3))\n",
    ]

# 2. Replace XGBoost with XGB GridSearch
xgb_snippet = "xgb_model = xgb.XGBClassifier("
xgb_idx = find_cell_index(nb, xgb_snippet)
if xgb_idx != -1:
    print(f"Index {xgb_idx}: Updating XGBoost to GridSearchCV")
    nb["cells"][xgb_idx]["source"] = [
        "# ハイパーパラメータチューニング - XGBoost\n",
        "param_grid_xgb = {\n",
        "    'max_depth': [5, 7],\n",
        "    'learning_rate': [0.05, 0.1],\n",
        "    'n_estimators': [100, 200]\n",
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
    ]

# 3. Update Ensemble to use Tuned Models
ensemble_snippet = (
    "ensemble_pred_proba = (rfc_pred_proba + xgb_pred_proba + lgb_pred_proba) / 3"
)
ens_idx = find_cell_index(nb, ensemble_snippet)
if ens_idx != -1:
    print(
        f"Index {ens_idx}: Updating Ensemble to use tuned models (gs.best_estimator_)"
    )
    nb["cells"][ens_idx]["source"] = [
        "# 各モデルの予測確率を取得 (Tuned Models)\n",
        "# Note: lgb_gs is defined in an existing cell. rfc_gs and xgb_gs defined above.\n",
        "rfc_pred_proba = rfc_gs.predict_proba(X_fe_valid)\n",
        "xgb_pred_proba = xgb_gs.predict_proba(X_fe_valid)\n",
        "lgb_pred_proba = lgb_gs.predict_proba(X_fe_valid)\n",
        "\n",
        "# 予測確率の平均（ブレンディング）\n",
        "ensemble_pred_proba = (rfc_pred_proba + xgb_pred_proba + lgb_pred_proba) / 3\n",
        "ensemble_pred = ensemble_pred_proba.argmax(axis=1)\n",
        "\n",
        "# アンサンブルモデルの評価\n",
        "ensemble_accuracy = accuracy_score(y_fe_valid, ensemble_pred)\n",
        "ensemble_f1 = f1_score(y_fe_valid, ensemble_pred, average='weighted')\n",
        "\n",
        "print('アンサンブルモデル（Tuned RandomForest + XGBoost + LightGBM）')\n",
        "print('Valid Accuracy: {}'.format(round(ensemble_accuracy, 3)))\n",
        "print('Valid F1 Score (weighted): {}'.format(round(ensemble_f1, 3)))\n",
    ]

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully.")
