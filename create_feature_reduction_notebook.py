import nbformat as nbf

nb = nbf.v4.new_notebook()

# ========== MARKDOWN: Title ==========
md_title = """# 特徴量削減による精度への影響調査
**目的**: 高相関（冗長）な特徴量を削除することで、XGBoostモデルの精度や計算効率が向上するかを検証する。

## 削除対象の冗長な特徴量
1. `normalized_spin_axis` (vs `spin_axis`: 相関 1.000)
2. `velocity_times_pfx_z` (vs `pfx_z`: 相関 0.997)

## 比較条件
- **Model**: XGBoost (Optimized Params)
- **Dataset**: `train_pitcher_v2.csv` (同一)
"""

# ========== SECTION 0: Imports ==========
code_imports = """import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("Libraries loaded.")
"""

# ========== SECTION 1: Data Preparation ==========
code_load = """# データ読み込み
train_data = pd.read_csv('train_pitcher_v2.csv')
test_data = pd.read_csv('test_pitcher_v2.csv')

# --- 特徴量セットの定義 ---
full_features = [
    'release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z',
    'release_pos_x', 'release_pos_z',
    'normalized_spin_axis', 'movement_angle', 'abs_horizontal_movement',
    'movement_magnitude', 'spin_efficiency', 'speed_spin_ratio',
    'horizontal_vertical_ratio', 'release_position_magnitude',
    'vertical_rise', 'sink_rate', 'spin_axis_deviation_from_fastball',
    'velocity_times_pfx_z', 'velocity_abs_pfx_x_ratio', 'pfx_z_minus_abs_pfx_x',
    'speed_diff', 'spin_diff', 'pfx_x_diff', 'pfx_z_diff'
]

# 冗長なものを削除
reduced_features = [f for f in full_features if f not in ['normalized_spin_axis', 'velocity_times_pfx_z']]

# エンコード
le = LabelEncoder()
y_train = le.fit_transform(train_data['pitch_type'])
y_test = le.transform(test_data['pitch_type'])

print(f"Full Features: {len(full_features)}")
print(f"Reduced Features: {len(reduced_features)}")
"""

# ========== SECTION 2: Model Training ==========
code_train = """# 最適化されたパラメータ
params = {
    'n_estimators': 200, 
    'max_depth': 10, 
    'learning_rate': 0.1, 
    'min_child_weight': 0.5,
    'random_state': 42, 
    'n_jobs': -1
}

# 1. Full Feature Model
print("Training Full Feature Model...")
X_train_full = train_data[full_features]
X_test_full = test_data[full_features]
model_full = xgb.XGBClassifier(**params)
model_full.fit(X_train_full, y_train)
pred_full = model_full.predict(X_test_full)

# 2. Reduced Feature Model
print("Training Reduced Feature Model...")
X_train_red = train_data[reduced_features]
X_test_red = test_data[reduced_features]
model_red = xgb.XGBClassifier(**params)
model_red.fit(X_train_red, y_train)
pred_red = model_red.predict(X_test_red)

print("Training complete.")
"""

# ========== SECTION 3: Result Comparison ==========
code_compare = """# === 結果比較 ===
print("="*60)
print("=== 特徴量削減の比較結果 ===")
print("="*60)

# Full
acc_full = accuracy_score(y_test, pred_full)
f1_full = f1_score(y_test, pred_full, average='weighted')
rep_full = classification_report(y_test, pred_full, output_dict=True, target_names=le.classes_)

# Reduced
acc_red = accuracy_score(y_test, pred_red)
f1_red = f1_score(y_test, pred_red, average='weighted')
rep_red = classification_report(y_test, pred_red, output_dict=True, target_names=le.classes_)

comparison_df = pd.DataFrame({
    'Metric': ['Features Count', 'Accuracy', 'F1 Score', 'FC Recall', 'SI Recall', 'SL Recall'],
    'Full Feature': [len(full_features), acc_full, f1_full, rep_full['FC']['recall'], rep_full['SI']['recall'], rep_full['SL']['recall']],
    'Reduced Feature': [len(reduced_features), acc_red, f1_red, rep_red['FC']['recall'], rep_red['SI']['recall'], rep_red['SL']['recall']]
})

comparison_df['Diff'] = comparison_df['Reduced Feature'] - comparison_df['Full Feature']
print(comparison_df.round(4).to_string(index=False))

# 結論
if acc_red >= acc_full:
    print(f"\\n★ 結果: 精度は等しいか向上しました (Accuracy Diff: {acc_red - acc_full:+.6f})")
    print("モデルの簡素化（冗長性の排除）を推奨します。")
else:
    print(f"\\n★ 結果: わずかに精度が低下しました (Accuracy Diff: {acc_red - acc_full:+.6f})")
    print("しかし、計算効率と説明性の観点からは削減版も有力な候補です。")
"""

nb["cells"] = [
    nbf.v4.new_markdown_cell(md_title),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_code_cell(code_train),
    nbf.v4.new_code_cell(code_compare),
]

with open("asao_1325_feature_reduction.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook 'asao_1325_feature_reduction.ipynb' created.")
