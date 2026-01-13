import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# 投球分類モデル: 投手相対評価特徴量
**作成日**: 2026/01/14
**目的**: 投手ごとの平均値との差分（相対評価）を特徴量として追加し、SI/FF、FC/SLの識別精度を向上させる。
**仮説**: 「絶対的な球速92mph」ではなく「その投手の平均より5mph遅い」という情報が球種判定に有効。
"""

code_imports = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgbm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings('ignore')

# pybaseballからデータ取得
from pybaseball import statcast, cache
cache.enable()

plt.rcParams['font.family'] = 'sans-serif'
"""

code_fetch_data = """# === Step 1: pybaseballからデータ取得 ===
print("Fetching data from pybaseball... (約5-10分かかります)")
df_raw = statcast(start_dt='2023-04-01', end_dt='2023-10-01')
print(f"Raw data shape: {df_raw.shape}")

# 必要なカラムのみ抽出
cols_needed = [
    'pitcher', 'p_throws', 'pitch_type',
    'release_speed', 'release_spin_rate', 'spin_axis',
    'pfx_x', 'pfx_z', 'release_pos_x', 'release_pos_z'
]
df = df_raw[cols_needed].copy()
df = df.dropna()
print(f"After dropna: {df.shape}")
"""

code_normalize = """# === Step 2: 利き腕による反転（正規化） ===
# 左投手のpfx_x, release_pos_x, spin_axisを反転

def normalize_by_handedness(df):
    df = df.copy()
    
    # 左投手の横変化を反転
    left_mask = df['p_throws'] == 'L'
    df.loc[left_mask, 'pfx_x'] = -df.loc[left_mask, 'pfx_x']
    df.loc[left_mask, 'release_pos_x'] = -df.loc[left_mask, 'release_pos_x']
    
    # 回転軸も反転（360 - spin_axis）
    df.loc[left_mask, 'spin_axis'] = 360 - df.loc[left_mask, 'spin_axis']
    
    return df

df = normalize_by_handedness(df)
print("Handedness normalization applied.")
"""

code_pitcher_stats = """# === Step 3: 投手ごとの統計量を計算 ===
# 反転後のデータで計算することが重要

pitcher_stats = df.groupby('pitcher').agg({
    'release_speed': 'mean',
    'release_spin_rate': 'mean',
    'pfx_x': 'mean',
    'pfx_z': 'mean'
}).rename(columns={
    'release_speed': 'pitcher_avg_speed',
    'release_spin_rate': 'pitcher_avg_spin',
    'pfx_x': 'pitcher_avg_pfx_x',
    'pfx_z': 'pitcher_avg_pfx_z'
})

print(f"Calculated stats for {len(pitcher_stats)} pitchers")

# マージ
df = df.merge(pitcher_stats, on='pitcher', how='left')
"""

code_relative_features = """# === Step 4: 相対特徴量を作成 ===

# 投手平均との差分
df['speed_diff'] = df['release_speed'] - df['pitcher_avg_speed']
df['spin_diff'] = df['release_spin_rate'] - df['pitcher_avg_spin']
df['pfx_x_diff'] = df['pfx_x'] - df['pitcher_avg_pfx_x']
df['pfx_z_diff'] = df['pfx_z'] - df['pitcher_avg_pfx_z']

# 既存の特徴量エンジニアリング（asao_1313と同じ）
df['velocity_times_pfx_z'] = df['release_speed'] * df['pfx_z']
df['spin_per_mph'] = df['release_spin_rate'] / df['release_speed']
df['horizontal_vertical_ratio'] = df['pfx_x'] / (df['pfx_z'].abs() + 0.1)
df['speed_spin_ratio'] = df['release_speed'] / (df['release_spin_rate'] + 1)
df['movement_magnitude'] = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2)
df['normalized_spin_axis'] = df['spin_axis'] / 360.0

print("Relative features created.")
print(df[['speed_diff', 'pfx_x_diff', 'pfx_z_diff']].describe())
"""

code_prepare = """# === Step 5: 学習データ準備 ===

# ターゲットの絞り込み（asao_1313と同じ球種のみ）
valid_pitches = ['CH', 'CU', 'EP', 'FA', 'FC', 'FF', 'FO', 'FS', 'KC', 'KN', 'SI', 'SL', 'ST', 'SV']
df = df[df['pitch_type'].isin(valid_pitches)]

# 特徴量リスト（既存 + 新規相対特徴量）
features = [
    # 既存特徴量
    'release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z',
    'release_pos_x', 'release_pos_z', 'movement_magnitude',
    'velocity_times_pfx_z', 'spin_per_mph', 'normalized_spin_axis',
    'speed_spin_ratio', 'horizontal_vertical_ratio',
    # 新規：投手相対特徴量
    'speed_diff', 'spin_diff', 'pfx_x_diff', 'pfx_z_diff'
]

target = 'pitch_type'

# 欠損値処理
df_clean = df.dropna(subset=features + [target])
print(f"Final data shape: {df_clean.shape}")

# エンコード
le = LabelEncoder()
y = le.fit_transform(df_clean[target])
X = df_clean[features]

# 分割
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Valid: {X_valid.shape}")
"""

code_train = """# === Step 6: モデル学習（アンサンブル） ===

# XGBoost
print("Training XGBoost...")
model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
model_xgb.fit(X_train, y_train)

# LightGBM
print("Training LightGBM...")
model_lgbm = lgbm.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42, verbose=-1)
model_lgbm.fit(X_train, y_train)

# RandomForest
print("Training RandomForest...")
model_rf = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
model_rf.fit(X_train, y_train)

# Ensemble
p_xgb = model_xgb.predict_proba(X_valid)
p_lgbm = model_lgbm.predict_proba(X_valid)
p_rf = model_rf.predict_proba(X_valid)

p_ensemble = (p_xgb + p_lgbm + p_rf) / 3.0
y_pred = np.argmax(p_ensemble, axis=1)

acc = accuracy_score(y_valid, y_pred)
f1 = f1_score(y_valid, y_pred, average='weighted')

print(f"\\n=== Results with Pitcher-Relative Features ===")
print(f"Accuracy: {acc:.4f}")
print(f"Weighted F1: {f1:.4f}")
print(f"\\nBaseline (asao_1313): Acc 0.924, F1 0.923")
print(f"Improvement: Acc {acc - 0.924:+.4f}, F1 {f1 - 0.923:+.4f}")
"""

code_eval = """# === Step 7: 詳細評価 ===
print(classification_report(y_valid, y_pred, target_names=le.classes_))

# FC Recall Check
fc_report = classification_report(y_valid, y_pred, output_dict=True, target_names=le.classes_)
print(f"\\nFC Recall: {fc_report['FC']['recall']:.4f} (Baseline: 0.76)")
print(f"SI Recall: {fc_report['SI']['recall']:.4f} (Baseline: 0.94)")

# Feature Importance (XGBoost)
importance = pd.DataFrame({
    'feature': features,
    'importance': model_xgb.feature_importances_
}).sort_values('importance', ascending=False)

print("\\n=== Feature Importance (Top 10) ===")
print(importance.head(10))

# 新特徴量の重要度
new_features = ['speed_diff', 'spin_diff', 'pfx_x_diff', 'pfx_z_diff']
print("\\n=== New Features Importance ===")
print(importance[importance['feature'].isin(new_features)])
"""

nb["cells"] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_fetch_data),
    nbf.v4.new_code_cell(code_normalize),
    nbf.v4.new_code_cell(code_pitcher_stats),
    nbf.v4.new_code_cell(code_relative_features),
    nbf.v4.new_code_cell(code_prepare),
    nbf.v4.new_code_cell(code_train),
    nbf.v4.new_code_cell(code_eval),
]

with open("asao_1318_pitcher_relative_features.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook 'asao_1318_pitcher_relative_features.ipynb' created.")
