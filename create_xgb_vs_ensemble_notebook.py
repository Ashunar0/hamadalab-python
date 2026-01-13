import nbformat as nbf

nb = nbf.v4.new_notebook()

md_title = """# XGBoost単体モデル vs アンサンブル 比較
**目的**: XGBoost単体（95.2%）がアンサンブル（93.9%）より高精度なため、詳細比較を行う。
"""

code_imports = """import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("Libraries loaded.")
"""

code_load = """# データ読み込み
train_data = pd.read_csv('train_pitcher_v2.csv')
test_data = pd.read_csv('test_pitcher_v2.csv')

# 特徴量
all_features = [
    'release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z',
    'release_pos_x', 'release_pos_z',
    'normalized_spin_axis', 'movement_angle', 'abs_horizontal_movement',
    'movement_magnitude', 'spin_efficiency', 'speed_spin_ratio',
    'horizontal_vertical_ratio', 'release_position_magnitude',
    'vertical_rise', 'sink_rate', 'spin_axis_deviation_from_fastball',
    'velocity_times_pfx_z', 'velocity_abs_pfx_x_ratio', 'pfx_z_minus_abs_pfx_x',
    'speed_diff', 'spin_diff', 'pfx_x_diff', 'pfx_z_diff'
]

available_features = [f for f in all_features if f in train_data.columns]

le = LabelEncoder()
y_train = le.fit_transform(train_data['pitch_type'])
y_test = le.transform(test_data['pitch_type'])
X_train = train_data[available_features]
X_test = test_data[available_features]

print(f"Features: {len(available_features)}")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
"""

code_train_models = """# 最適化パラメータでモデル学習
print("Training models...")

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=10, learning_rate=0.1, min_child_weight=0.5,
    random_state=42, n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=100, max_depth=7, learning_rate=0.05,
    random_state=42, n_jobs=-1, verbose=-1
)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)

# RandomForest
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=None, min_samples_leaf=1,
    random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Ensemble
p_ensemble = (xgb_model.predict_proba(X_test) + lgb_model.predict_proba(X_test) + rf_model.predict_proba(X_test)) / 3
ensemble_pred = np.argmax(p_ensemble, axis=1)

print("Training complete.")
"""

code_compare = """# === 詳細比較 ===
print("="*70)
print("=== XGBoost単体 vs アンサンブル 詳細比較 ===")
print("="*70)

models = {
    'XGBoost': xgb_pred,
    'LightGBM': lgb_pred,
    'RandomForest': rf_pred,
    'Ensemble': ensemble_pred
}

results = []
for name, pred in models.items():
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    report = classification_report(y_test, pred, output_dict=True, target_names=le.classes_)
    fc_recall = report['FC']['recall']
    si_recall = report['SI']['recall']
    sl_recall = report['SL']['recall']
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'F1': f1,
        'FC Recall': fc_recall,
        'SI Recall': si_recall,
        'SL Recall': sl_recall
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Best model highlight
best_acc = results_df.loc[results_df['Accuracy'].idxmax()]
print(f"\\n★ 最高精度: {best_acc['Model']} (Accuracy: {best_acc['Accuracy']:.4f})")
"""

code_xgb_detail = """# === XGBoost単体の詳細レポート ===
print("\\n" + "="*70)
print("=== XGBoost単体モデル 詳細レポート ===")
print("="*70)

print(classification_report(y_test, xgb_pred, target_names=le.classes_))

# FC/SL誤分類
xgb_results = test_data.copy()
xgb_results['true'] = le.inverse_transform(y_test)
xgb_results['pred'] = le.inverse_transform(xgb_pred)

fc_data = xgb_results[xgb_results['true'] == 'FC']
fc_to_sl = len(fc_data[fc_data['pred'] == 'SL'])
print(f"\\nFC → SL 誤分類: {fc_to_sl} / {len(fc_data)} ({fc_to_sl/len(fc_data)*100:.1f}%)")

sl_data = xgb_results[xgb_results['true'] == 'SL']
sl_to_fc = len(sl_data[sl_data['pred'] == 'FC'])
print(f"SL → FC 誤分類: {sl_to_fc} / {len(sl_data)} ({sl_to_fc/len(sl_data)*100:.1f}%)")
"""

code_confusion_compare = """# === 混同行列比較 ===
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# XGBoost
cm_xgb = confusion_matrix(y_test, xgb_pred)
cm_xgb_norm = cm_xgb.astype('float') / cm_xgb.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_xgb_norm, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
axes[0].set_title('XGBoost単体（Recall）', fontsize=14)
axes[0].set_xlabel('予測')
axes[0].set_ylabel('真')

# Ensemble
cm_ens = confusion_matrix(y_test, ensemble_pred)
cm_ens_norm = cm_ens.astype('float') / cm_ens.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_ens_norm, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[1])
axes[1].set_title('アンサンブル（Recall）', fontsize=14)
axes[1].set_xlabel('予測')
axes[1].set_ylabel('真')

plt.tight_layout()
plt.show()
"""

code_summary = """# === 結論 ===
print("\\n" + "="*70)
print("=== 結論 ===")
print("="*70)

xgb_acc = results_df[results_df['Model']=='XGBoost']['Accuracy'].values[0]
ens_acc = results_df[results_df['Model']=='Ensemble']['Accuracy'].values[0]
xgb_fc = results_df[results_df['Model']=='XGBoost']['FC Recall'].values[0]
ens_fc = results_df[results_df['Model']=='Ensemble']['FC Recall'].values[0]

print(f"\\nXGBoost単体 vs アンサンブル:")
print(f"  Accuracy: {xgb_acc:.4f} vs {ens_acc:.4f} (差: {xgb_acc - ens_acc:+.4f})")
print(f"  FC Recall: {xgb_fc:.4f} vs {ens_fc:.4f} (差: {xgb_fc - ens_fc:+.4f})")

if xgb_acc > ens_acc:
    print(f"\\n★ XGBoost単体を最終モデルとして採用することを推奨")
else:
    print(f"\\n★ アンサンブルを最終モデルとして採用することを推奨")
"""

nb["cells"] = [
    nbf.v4.new_markdown_cell(md_title),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_code_cell(code_train_models),
    nbf.v4.new_code_cell(code_compare),
    nbf.v4.new_code_cell(code_xgb_detail),
    nbf.v4.new_code_cell(code_confusion_compare),
    nbf.v4.new_code_cell(code_summary),
]

with open("asao_1323_xgboost_vs_ensemble.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook 'asao_1323_xgboost_vs_ensemble.ipynb' created.")
