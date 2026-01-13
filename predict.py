import pandas as pd
import numpy as np
import pickle
import json
import os


class PitchPredictor:
    def __init__(self, model_dir="models"):
        # 成果物の読み込み
        with open(os.path.join(model_dir, "pitch_classifier_xgb.pkl"), "rb") as f:
            self.model = pickle.load(f)

        with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
            self.le = pickle.load(f)

        self.pitcher_stats = pd.read_csv(
            os.path.join(model_dir, "pitcher_stats.csv"), index_col="pitcher"
        )

        with open(os.path.join(model_dir, "global_stats.json"), "r") as f:
            self.global_stats = json.load(f)

    def _feature_engineering(self, df):
        """14個のエンジニアリング特徴量を作成"""
        # 1. normalized_spin_axis
        df["normalized_spin_axis"] = df["spin_axis"] % 360

        # 2. movement_angle
        df["movement_angle"] = np.arctan2(df["pfx_z"], df["pfx_x"]) * 180 / np.pi

        # 3. abs_horizontal_movement
        df["abs_horizontal_movement"] = df["pfx_x"].abs()

        # 4. movement_magnitude
        df["movement_magnitude"] = np.sqrt(df["pfx_x"] ** 2 + df["pfx_z"] ** 2)

        # 5. spin_efficiency
        df["spin_efficiency"] = (
            df["movement_magnitude"] / (df["release_spin_rate"] + 1) * 1000
        )

        # 6. speed_spin_ratio
        df["speed_spin_ratio"] = df["release_speed"] / (df["release_spin_rate"] + 1)

        # 7. horizontal_vertical_ratio
        df["horizontal_vertical_ratio"] = df["pfx_x"] / (df["pfx_z"].abs() + 0.1)

        # 8. release_position_magnitude
        df["release_position_magnitude"] = np.sqrt(
            df["release_pos_x"] ** 2 + df["release_pos_z"] ** 2
        )

        # 9. vertical_rise
        df["vertical_rise"] = df["pfx_z"].clip(lower=0)

        # 10. sink_rate
        df["sink_rate"] = (-df["pfx_z"]).clip(lower=0)

        # 11. spin_axis_deviation_from_fastball (Simplified version used in pipeline)
        df["spin_axis_deviation_from_fastball"] = abs(df["spin_axis"] - 180)

        # 12. velocity_times_pfx_z
        df["velocity_times_pfx_z"] = df["release_speed"] * df["pfx_z"]

        # 13. velocity_abs_pfx_x_ratio
        df["velocity_abs_pfx_x_ratio"] = df["release_speed"] / (df["pfx_x"].abs() + 0.1)

        # 14. pfx_z_minus_abs_pfx_x
        df["pfx_z_minus_abs_pfx_x"] = df["pfx_z"] - df["pfx_x"].abs()

        return df

    def predict(self, input_data):
        """
        input_data: list of dict or pd.DataFrame
        Required columns: [pitcher, release_speed, release_spin_rate, spin_axis, pfx_x, pfx_z, release_pos_x, release_pos_z]
        """
        if isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        else:
            df = input_data.copy()

        # 1. 基本特徴量の準備
        # (ここではそのまま使用)

        # 2. 特徴量エンジニアリング (14個)
        df = self._feature_engineering(df)

        # 3. 投手相対特徴量 (4個)
        speed_diffs, spin_diffs, pfx_x_diffs, pfx_z_diffs = [], [], [], []

        for _, row in df.iterrows():
            pitcher_id = row["pitcher"]
            if pitcher_id in self.pitcher_stats.index:
                stats = self.pitcher_stats.loc[pitcher_id]
            else:
                stats = self.global_stats

            speed_diffs.append(row["release_speed"] - stats["avg_speed"])
            spin_diffs.append(row["release_spin_rate"] - stats["avg_spin"])
            pfx_x_diffs.append(row["pfx_x"] - stats["avg_pfx_x"])
            pfx_z_diffs.append(row["pfx_z"] - stats["avg_pfx_z"])

        df["speed_diff"] = speed_diffs
        df["spin_diff"] = spin_diffs
        df["pfx_x_diff"] = pfx_x_diffs
        df["pfx_z_diff"] = pfx_z_diffs

        # 4. 予測に使用するカラムのみ抽出
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

        X = df[features]

        # 5. 予測
        probs = self.model.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        labels = self.le.inverse_transform(preds)

        # 確率も返す
        results = []
        for i in range(len(labels)):
            results.append(
                {
                    "pitch_type": labels[i],
                    "confidence": float(np.max(probs[i])),
                    "probabilities": dict(zip(self.le.classes_, probs[i].tolist())),
                }
            )

        return results


if __name__ == "__main__":
    # 使用例
    predictor = PitchPredictor()

    # サンプルデータ (大谷翔平選手のスイーパーに近い数値など)
    sample_pitch = [
        {
            "pitcher": 660271,  # Shohei Ohtani
            "release_speed": 85.0,
            "release_spin_rate": 2500,
            "spin_axis": 270,
            "pfx_x": 1.5,
            "pfx_z": 0.1,
            "release_pos_x": -2.0,
            "release_pos_z": 5.8,
        }
    ]

    result = predictor.predict(sample_pitch)
    print(
        f"Prediction: {result[0]['pitch_type']} (Confidence: {result[0]['confidence']:.2%})"
    )
