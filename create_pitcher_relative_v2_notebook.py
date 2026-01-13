import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# 投球分類モデル: 投手相対評価特徴量（修正版）
**作成日**: 2026/01/14
**目的**: 既存の`train_with_features.csv`の全特徴量を保持しつつ、投手相対評価特徴量を追加する。
**出力**: `train_with_pitcher_features.csv`, `test_with_pitcher_features.csv`
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
from sklearn.metrics import accuracy_score, classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')

from pybaseball import statcast, cache
cache.enable()

plt.rcParams['font.family'] = 'sans-serif'
"""

code_load_existing = """# === Step 1: 既存CSVと元データの読み込み ===

# 既存の特徴量付きCSV
df_train = pd.read_csv('train_with_features.csv')
df_test = pd.read_csv('test_with_features.csv')

print(f"Train shape: {df_train.shape}")
print(f"Test shape: {df_test.shape}")
print(f"Existing columns: {list(df_train.columns)}")
"""

code_fetch_pybaseball = """# === Step 2: pybaseballからデータ取得（投手ID用） ===
print("Fetching data from pybaseball...")
df_raw = statcast(start_dt='2023-04-01', end_dt='2023-10-01')
print(f"Raw data shape: {df_raw.shape}")

# 必要カラムのみ
cols_for_match = ['pitcher', 'p_throws', 'release_speed', 'release_spin_rate', 
                  'spin_axis', 'pfx_x', 'pfx_z', 'release_pos_x', 'release_pos_z', 'pitch_type']
df_pyb = df_raw[cols_for_match].copy()
df_pyb = df_pyb.dropna()
print(f"After dropna: {df_pyb.shape}")
"""

code_normalize_pyb = """# === Step 3: pybaseballデータを利き腕で正規化 ===
def normalize_by_handedness(df):
    df = df.copy()
    left_mask = df['p_throws'] == 'L'
    df.loc[left_mask, 'pfx_x'] = -df.loc[left_mask, 'pfx_x']
    df.loc[left_mask, 'release_pos_x'] = -df.loc[left_mask, 'release_pos_x']
    df.loc[left_mask, 'spin_axis'] = 360 - df.loc[left_mask, 'spin_axis']
    return df

df_pyb = normalize_by_handedness(df_pyb)
print("Handedness normalization applied to pybaseball data.")
"""

code_calc_pitcher_stats = """# === Step 4: 投手ごとの統計量を計算 ===
pitcher_stats = df_pyb.groupby('pitcher').agg({
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

# pybaseballデータにマージ
df_pyb = df_pyb.merge(pitcher_stats, on='pitcher', how='left')

# 相対特徴量を計算
df_pyb['speed_diff'] = df_pyb['release_speed'] - df_pyb['pitcher_avg_speed']
df_pyb['spin_diff'] = df_pyb['release_spin_rate'] - df_pyb['pitcher_avg_spin']
df_pyb['pfx_x_diff'] = df_pyb['pfx_x'] - df_pyb['pitcher_avg_pfx_x']
df_pyb['pfx_z_diff'] = df_pyb['pfx_z'] - df_pyb['pitcher_avg_pfx_z']

print(f"Pitcher stats calculated for {len(pitcher_stats)} pitchers")
"""

code_match_and_merge = """# === Step 5: 既存CSVとマッチングしてマージ ===
# マッチングキー: release_speed + spin_axis + pfx_x + pfx_z + release_pos_x + release_pos_z + p_throws + pitch_type
# 浮動小数点の微小誤差を考慮して丸める

def create_match_key(df, round_digits=4):
    key = (df['release_speed'].round(round_digits).astype(str) + '_' +
           df['spin_axis'].round(round_digits).astype(str) + '_' +
           df['pfx_x'].round(round_digits).astype(str) + '_' +
           df['pfx_z'].round(round_digits).astype(str) + '_' +
           df['release_pos_x'].round(round_digits).astype(str) + '_' +
           df['release_pos_z'].round(round_digits).astype(str) + '_' +
           df['p_throws'].astype(str) + '_' +
           df['pitch_type'].astype(str))
    return key

# マッチングキーを作成
df_train['match_key'] = create_match_key(df_train)
df_test['match_key'] = create_match_key(df_test)
df_pyb['match_key'] = create_match_key(df_pyb)

# 重複を除去（同じキーが複数ある場合は最初のものを使用）
df_pyb_unique = df_pyb.drop_duplicates(subset='match_key', keep='first')

# 新特徴量だけをマージ
new_features = ['match_key', 'pitcher', 'speed_diff', 'spin_diff', 'pfx_x_diff', 'pfx_z_diff']
df_pyb_for_merge = df_pyb_unique[new_features]

df_train_new = df_train.merge(df_pyb_for_merge, on='match_key', how='left')
df_test_new = df_test.merge(df_pyb_for_merge, on='match_key', how='left')

# マッチング結果確認
train_matched = df_train_new['speed_diff'].notna().sum()
test_matched = df_test_new['speed_diff'].notna().sum()

print(f"Train matched: {train_matched}/{len(df_train)} ({100*train_matched/len(df_train):.1f}%)")
print(f"Test matched: {test_matched}/{len(df_test)} ({100*test_matched/len(df_test):.1f}%)")

# マッチしなかった行の新特徴量は0で埋める（中立値）
for col in ['speed_diff', 'spin_diff', 'pfx_x_diff', 'pfx_z_diff']:
    df_train_new[col] = df_train_new[col].fillna(0)
    df_test_new[col] = df_test_new[col].fillna(0)

# match_keyとpitcherは不要なので削除
df_train_new = df_train_new.drop(columns=['match_key', 'pitcher'])
df_test_new = df_test_new.drop(columns=['match_key', 'pitcher'])

print(f"\\nNew train shape: {df_train_new.shape}")
print(f"New columns: {list(df_train_new.columns)}")
"""

code_save_csv = """# === Step 6: 新しいCSVを保存 ===
df_train_new.to_csv('train_with_pitcher_features.csv', index=False)
df_test_new.to_csv('test_with_pitcher_features.csv', index=False)
print("Saved: train_with_pitcher_features.csv, test_with_pitcher_features.csv")
"""

code_train_model = """# === Step 7: モデル学習と評価 ===

# 特徴量リスト（既存 + 新規）
features = [
    'release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z',
    'release_pos_x', 'release_pos_z', 
    'normalized_spin_axis', 'movement_angle', 'abs_horizontal_movement',
    'movement_magnitude', 'spin_efficiency', 'speed_spin_ratio',
    'horizontal_vertical_ratio', 'release_position_magnitude',
    'vertical_rise', 'sink_rate', 'spin_axis_deviation_from_fastball',
    # 新規：投手相対特徴量
    'speed_diff', 'spin_diff', 'pfx_x_diff', 'pfx_z_diff'
]
target = 'pitch_type'

# 存在する特徴量のみ使用
existing_features = [f for f in features if f in df_train_new.columns]
print(f"Using {len(existing_features)} features")

# 欠損値処理
df_clean = df_train_new.dropna(subset=existing_features + [target])
print(f"Clean data: {len(df_clean)}")

le = LabelEncoder()
y = le.fit_transform(df_clean[target])
X = df_clean[existing_features]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# アンサンブル学習
print("Training XGBoost...")
model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
model_xgb.fit(X_train, y_train)

print("Training LightGBM...")
model_lgbm = lgbm.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42, verbose=-1)
model_lgbm.fit(X_train, y_train)

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

code_eval = """# === Step 8: 詳細評価 ===
print(classification_report(y_valid, y_pred, target_names=le.classes_))

fc_report = classification_report(y_valid, y_pred, output_dict=True, target_names=le.classes_)
print(f"\\nFC Recall: {fc_report['FC']['recall']:.4f} (Baseline: 0.76)")
print(f"SI Recall: {fc_report['SI']['recall']:.4f} (Baseline: 0.94)")

# Feature Importance
importance = pd.DataFrame({
    'feature': existing_features,
    'importance': model_xgb.feature_importances_
}).sort_values('importance', ascending=False)

print("\\n=== Feature Importance (All) ===")
print(importance)

print("\\n=== New Features Importance ===")
new_feats = ['speed_diff', 'spin_diff', 'pfx_x_diff', 'pfx_z_diff']
print(importance[importance['feature'].isin(new_feats)])
"""

nb["cells"] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_load_existing),
    nbf.v4.new_code_cell(code_fetch_pybaseball),
    nbf.v4.new_code_cell(code_normalize_pyb),
    nbf.v4.new_code_cell(code_calc_pitcher_stats),
    nbf.v4.new_code_cell(code_match_and_merge),
    nbf.v4.new_code_cell(code_save_csv),
    nbf.v4.new_code_cell(code_train_model),
    nbf.v4.new_code_cell(code_eval),
]

with open("asao_1319_pitcher_relative_v2.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook 'asao_1319_pitcher_relative_v2.ipynb' created.")
