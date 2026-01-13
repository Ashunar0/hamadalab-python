import nbformat
import json

notebook_path = "asao_step0_1310-with_features_total_19.ipynb"


def update_notebook():
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
    except FileNotFoundError:
        print(f"Error: {notebook_path} not found.")
        return

    # 1. Update Feature Importance Logic
    feature_imp_code = """# 各モデルの特徴量重要度を比較
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LGBM
lgb_importance = pd.DataFrame({
    'feature': feature_cols_fe,
    'lgb_importance': lgb_model.feature_importances_
}).sort_values('lgb_importance', ascending=False)

# XGBoost
xgb_importance = pd.DataFrame({
    'feature': feature_cols_fe,
    'xgb_importance': xgb_model.feature_importances_
}).sort_values('xgb_importance', ascending=False)

# RFC
rfc_importance = pd.DataFrame({
    'feature': feature_cols_fe,
    'rfc_importance': rfc_fe.feature_importances_
}).sort_values('rfc_importance', ascending=False)

# 重要度を正規化（0-1スケール）
# これにより、LightGBMの生スコアが平均を支配するのを防ぎます
scaler = MinMaxScaler()
lgb_importance['lgb_imp_norm'] = scaler.fit_transform(lgb_importance[['lgb_importance']])
xgb_importance['xgb_imp_norm'] = scaler.fit_transform(xgb_importance[['xgb_importance']])
rfc_importance['rfc_imp_norm'] = scaler.fit_transform(rfc_importance[['rfc_importance']])

# マージして比較
importance_comparison = lgb_importance.merge(xgb_importance, on='feature').merge(rfc_importance, on='feature')

# 正規化された重要度の平均を計算
importance_comparison['avg_importance'] = (
    importance_comparison['lgb_imp_norm'] + 
    importance_comparison['xgb_imp_norm'] + 
    importance_comparison['rfc_imp_norm']
) / 3

importance_comparison = importance_comparison.sort_values('avg_importance', ascending=False)

print("特徴量重要度の比較（上位20件）:")
print(importance_comparison[['feature', 'avg_importance', 'lgb_imp_norm', 'xgb_imp_norm', 'rfc_imp_norm']].head(20))

# 可視化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 各モデルの上位特徴量を取得
n_top = 15
lgb_top = lgb_importance.head(n_top)
xgb_top = xgb_importance.head(n_top)
rfc_top = rfc_importance.head(n_top)
avg_top = importance_comparison.head(n_top)

# LightGBM
axes[0, 0].barh(range(len(lgb_top)), lgb_top['lgb_imp_norm'].values)
axes[0, 0].set_yticks(range(len(lgb_top)))
axes[0, 0].set_yticklabels(lgb_top['feature'].values)
axes[0, 0].set_title('LightGBM (Normalized)')

# XGBoost
axes[0, 1].barh(range(len(xgb_top)), xgb_top['xgb_imp_norm'].values)
axes[0, 1].set_yticks(range(len(xgb_top)))
axes[0, 1].set_yticklabels(xgb_top['feature'].values)
axes[0, 1].set_title('XGBoost (Normalized)')

# RandomForest
axes[1, 0].barh(range(len(rfc_top)), rfc_top['rfc_imp_norm'].values)
axes[1, 0].set_yticks(range(len(rfc_top)))
axes[1, 0].set_yticklabels(rfc_top['feature'].values)
axes[1, 0].set_title('RandomForest (Normalized)')

# Average
axes[1, 1].barh(range(len(avg_top)), avg_top['avg_importance'].values)
axes[1, 1].set_yticks(range(len(avg_top)))
axes[1, 1].set_yticklabels(avg_top['feature'].values)
axes[1, 1].set_title('Average Importance (Normalized)')

plt.tight_layout()
plt.show()

# 重要度が低い特徴量を確認（削除候補）
print("\\n重要度が低い特徴量（下位10件）:")
print(importance_comparison.tail(10))"""

    # 2. Update Worst Cases Detail Logic
    worst_cases_code = """# Worst Casesの詳細データを表示
# 元のデータフレームから該当する行を取得
if 'X_fe_valid' in globals():
    # 検証データのインデックスを取得
    valid_indices = X_fe_valid.index if hasattr(X_fe_valid, 'index') else range(len(X_fe_valid))
    
    # Worst Casesの元データを取得
    worst_cases_data = []
    # データをより多く見て傾向を掴む
    for case in worst_cases[:50]:
        idx = case['index']
        if isinstance(valid_indices, pd.Index):
            original_idx = valid_indices[idx]
        else:
            original_idx = idx
        
        # 元のデータから該当行を取得（可能な場合）
        if 'data' in globals() and original_idx < len(data):
            row_data = data.iloc[original_idx].to_dict()
            row_data['worst_case_index'] = idx
            row_data['true_label'] = case['true_label']
            row_data['pred_label'] = case['pred_label']
            row_data['pred_proba'] = case['pred_proba']
            row_data['loss'] = case['loss']
            worst_cases_data.append(row_data)
    
    if worst_cases_data:
        worst_cases_detail_df = pd.DataFrame(worst_cases_data)
        
        # 表示するカラムを選択
        display_cols = ['worst_case_index', 'true_label', 'pred_label', 'pred_proba', 'loss']
        if 'p_throws' in worst_cases_detail_df.columns:
            display_cols.append('p_throws')
            
        print("Worst Casesの詳細データ（上位20件）:")
        print(worst_cases_detail_df[display_cols].head(20))
        
        # 主要な特徴量の統計を表示
        # 分析に有用なカラムを追加
        target_stats_cols = ['release_speed', 'release_spin_rate', 'pitch_type', 'pfx_x', 'pfx_z', 'spin_axis']
        available_cols = [c for c in target_stats_cols if c in worst_cases_detail_df.columns]
        
        if available_cols:
            print("\\nWorst Casesの主要特徴量の統計:")
            print(worst_cases_detail_df[available_cols].describe())
            
        if 'p_throws' in worst_cases_detail_df.columns:
            print("\\n投手の手（p_throws）の分布:")
            print(worst_cases_detail_df['p_throws'].value_counts())
else:
    print("元のデータフレームにアクセスできません。")
    print("Worst Casesのインデックス:", worst_indices[:20])"""

    # 3. SHAP Analysis Code (New)
    shap_code = """# SHAP値によるモデル解釈（Worst Casesの分析）
# 注: shapライブラリが必要です (pip install shap)
try:
    import shap
    print("SHAP値の計算を開始します（時間がかかる場合があります）...")
    
    # モデルの準備（XGBoostを使用）
    # lightgbmモデルを使用する場合は shap.TreeExplainer(lgb_model)
    model_to_explain = xgb_model 
    explainer = shap.TreeExplainer(model_to_explain)
    
    # Worst Casesの特徴量データを取得
    if hasattr(X_fe_valid, 'iloc'):
        worst_X = X_fe_valid.iloc[worst_indices[:50]] # 上位50件
    else:
        worst_X = pd.DataFrame(X_fe_valid[worst_indices[:50]], columns=feature_cols_fe)
    
    # SHAP値を計算
    shap_values = explainer.shap_values(worst_X)
    
    print("SHAP Summary Plot (Worst Cases):")
    # クラス分類の場合、shap_valuesはクラスごとのリストになることがある
    if isinstance(shap_values, list):
        # 予測されたクラス（間違った予測または正解など）に着目
        # ここでは全体の要約を表示
        shap.summary_plot(shap_values, worst_X, plot_type="bar")
    else:
        shap.summary_plot(shap_values, worst_X)
        
    print("\\nSHAP値の詳細プロット:")
    shap.summary_plot(shap_values, worst_X)

except ImportError:
    print("⚠ shapライブラリが見つかりません。詳細な分析には 'pip install shap' を実行してください。")
except Exception as e:
    print(f"⚠ SHAP分析中にエラーが発生しました: {e}")"""

    # Apply updates
    features_updated = False
    worst_cases_updated = False

    for cell in nb.cells:
        if cell.cell_type == "code":
            if (
                "lgb_importance = pd.DataFrame" in cell.source
                and "feature_importances_" in cell.source
            ):
                cell.source = feature_imp_code
                features_updated = True
                print("Updated Feature Importance cell.")

            if (
                "# Worst Casesの詳細データを表示" in cell.source
                or "worst_cases_detail_df =" in cell.source
            ):
                # 既存のセルを置換
                if "pd.DataFrame(worst_cases_data)" in cell.source:
                    cell.source = worst_cases_code
                    worst_cases_updated = True
                    print("Updated Worst Cases Detail cell.")

    if not features_updated:
        print("Warning: Could not find Feature Importance cell to update.")

    if not worst_cases_updated:
        print("Warning: Could not find Worst Cases Detail cell to update.")

    # Insert SHAP cell at the end of the notebook
    nb.cells.append(nbformat.v4.new_code_cell(shap_code))
    print("Appended SHAP analysis cell.")

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Notebook {notebook_path} updated successfully.")


if __name__ == "__main__":
    update_notebook()
