import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# 新CSVでasao_1313モデルを評価
**目的**: `train_pitcher_v2.csv`を使って、asao_1313と同じ特徴量セットでモデルを学習し、比較する。
**仮説**: 投手相対特徴量の効果ではなく、データ処理パイプラインの違いが原因かを確認。
"""

code_imports = """import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgbm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')
print("Libraries loaded.")
"""

code_load = """# 新CSVを読み込み
train_df = pd.read_csv('train_pitcher_v2.csv')
test_df = pd.read_csv('test_pitcher_v2.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Columns: {list(train_df.columns)}")
"""

code_train_1313_features = """# === asao_1313と同じ特徴量のみを使用 ===
# 投手相対特徴量を除外

model_features_1313 = [
    # 基本特徴量（p_throwsは除外）
    'release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z',
    'release_pos_x', 'release_pos_z',
    # エンジニアリング特徴量（14個）
    'normalized_spin_axis', 'movement_angle', 'abs_horizontal_movement',
    'movement_magnitude', 'spin_efficiency', 'speed_spin_ratio',
    'horizontal_vertical_ratio', 'release_position_magnitude',
    'vertical_rise', 'sink_rate', 'spin_axis_deviation_from_fastball',
    'velocity_times_pfx_z', 'velocity_abs_pfx_x_ratio', 'pfx_z_minus_abs_pfx_x'
]

# 存在する特徴量のみ使用
available_features = [f for f in model_features_1313 if f in train_df.columns]
print(f"Using {len(available_features)} features (same as 1313):")
print(available_features)

target = 'pitch_type'

# データ準備
le = LabelEncoder()
y_train = le.fit_transform(train_df[target])
y_test = le.transform(test_df[target])
X_train = train_df[available_features]
X_test = test_df[available_features]

print(f"\\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
"""

code_train = """# === モデル学習（asao_1313のGridSearch最適パラメータ） ===

print("Training XGBoost (optimized params)...")
# asao_1313のGridSearch結果: learning_rate=0.1, max_depth=10, n_estimators=200
model_xgb = xgb.XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, min_child_weight=0.5, n_jobs=-1, random_state=42)
model_xgb.fit(X_train, y_train)

print("Training LightGBM (optimized params)...")
# asao_1313のGridSearch結果: learning_rate=0.05, max_depth=7, n_estimators=100
model_lgbm = lgbm.LGBMClassifier(n_estimators=100, max_depth=7, learning_rate=0.05, n_jobs=-1, random_state=42, verbose=-1)
model_lgbm.fit(X_train, y_train)

print("Training RandomForest (optimized params)...")
# asao_1313のGridSearch結果: max_depth=None, n_estimators=200
model_rf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=1, n_jobs=-1, random_state=42)
model_rf.fit(X_train, y_train)

# Ensemble
p_xgb = model_xgb.predict_proba(X_test)
p_lgbm = model_lgbm.predict_proba(X_test)
p_rf = model_rf.predict_proba(X_test)

p_ensemble = (p_xgb + p_lgbm + p_rf) / 3.0
y_pred = np.argmax(p_ensemble, axis=1)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\\n=== Results (1313 Features on New CSV) ===")
print(f"Accuracy: {acc:.4f}")
print(f"Weighted F1: {f1:.4f}")
print(f"\\nBaseline (asao_1313 on old CSV): Acc 0.924, F1 0.923")
print(f"1320 (with pitcher features): Acc 0.877, F1 0.875")
print(f"\\nThis run (1313 features on new CSV): Acc {acc:.4f}, F1 {f1:.4f}")
"""

code_eval = """# === 詳細評価 ===
print(classification_report(y_test, y_pred, target_names=le.classes_))

fc_report = classification_report(y_test, y_pred, output_dict=True, target_names=le.classes_)
print(f"\\n=== Key Metrics ===")
print(f"FC Recall: {fc_report['FC']['recall']:.4f} (Baseline: 0.76)")
print(f"SI Recall: {fc_report['SI']['recall']:.4f} (Baseline: 0.94)")
"""

nb["cells"] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_code_cell(code_train_1313_features),
    nbf.v4.new_code_cell(code_train),
    nbf.v4.new_code_cell(code_eval),
]

with open("asao_1321_compare_csv.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook 'asao_1321_compare_csv.ipynb' created.")
