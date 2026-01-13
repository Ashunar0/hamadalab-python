import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# 1. データの読み込み
print("Loading data...")
train_data = pd.read_csv("train_pitcher_v2.csv")
test_data = pd.read_csv("test_pitcher_v2.csv")

# 特徴量リスト (Full Feature版 - 25個)
features = [
    "release_speed",
    "release_spin_rate",
    "spin_axis",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_z",
    "normalized_spin_axis",
    "movement_angle",
    "abs_horizontal_movement",
    "movement_magnitude",
    "spin_efficiency",
    "speed_spin_ratio",
    "horizontal_vertical_ratio",
    "release_position_magnitude",
    "vertical_rise",
    "sink_rate",
    "spin_axis_deviation_from_fastball",
    "velocity_times_pfx_z",
    "velocity_abs_pfx_x_ratio",
    "pfx_z_minus_abs_pfx_x",
    "speed_diff",
    "spin_diff",
    "pfx_x_diff",
    "pfx_z_diff",
]

# 2. モデルの学習 (最適化パラメータ)
print("Training final XGBoost model...")
le = LabelEncoder()
y_train = le.fit_transform(train_data["pitch_type"])
X_train = train_data[features]

# 最適化パラメータ (asao_1313/1322由来)
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    min_child_weight=0.5,
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(X_train, y_train)

# 3. 投手統計量 (Metadata) のエクスポート
# 推論時に相対特徴量を計算するために必要
print("Calculating pitcher metadata...")
# 元の生データがあれば正確だが、CSVから再構成する
# pipeline作成時のロジック: 全データでの平均を使用
all_data = pd.concat([train_data, test_data])
pitcher_stats = (
    all_data.groupby("pitcher")
    .agg(
        {
            "release_speed": "mean",
            "release_spin_rate": "mean",
            "pfx_x": "mean",
            "pfx_z": "mean",
        }
    )
    .rename(
        columns={
            "release_speed": "avg_speed",
            "release_spin_rate": "avg_spin",
            "pfx_x": "avg_pfx_x",
            "pfx_z": "avg_pfx_z",
        }
    )
)

# グローバル平均（未知の投手用）
global_stats = {
    "avg_speed": all_data["release_speed"].mean(),
    "avg_spin": all_data["release_spin_rate"].mean(),
    "avg_pfx_x": all_data["pfx_x"].mean(),
    "avg_pfx_z": all_data["pfx_z"].mean(),
}

# 4. モデル・成果物の保存
print("Saving artifacts...")
os.makedirs("models", exist_ok=True)

# モデル本体
with open("models/pitch_classifier_xgb.pkl", "wb") as f:
    pickle.dump(xgb_model, f)

# ラベルエンコーダー
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# 投手統計量
pitcher_stats.to_csv("models/pitcher_stats.csv")

# グローバル統計量 (JSON形式で保存)
import json

with open("models/global_stats.json", "w") as f:
    json.dump(global_stats, f, indent=4)

print("Finalization complete! All models saved in 'models/' directory.")
