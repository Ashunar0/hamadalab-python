"""
スライダー(SL)とカットボール(FC)の特性分析スクリプト
両球種の物理的特徴を比較し、識別に有効な特徴量を提案する
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import platform

# 日本語フォント設定
if platform.system() == "Darwin":  # macOS
    plt.rcParams["font.family"] = "Hiragino Sans"
elif platform.system() == "Windows":
    plt.rcParams["font.family"] = "Yu Gothic"
else:  # Linux
    plt.rcParams["font.family"] = "Noto Sans CJK JP"

plt.rcParams["axes.unicode_minus"] = False


def load_data():
    """補正済みデータを読み込む"""
    print("データを読み込んでいます...")
    train_df = pd.read_csv("train_corrected.csv")
    test_df = pd.read_csv("test_corrected.csv")

    # 全データを結合
    df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"総データ数: {len(df)}")

    return df


def analyze_sl_fc_characteristics(df):
    """SLとFCの基本統計量を比較"""
    print("\n" + "=" * 60)
    print("スライダー(SL)とカットボール(FC)の基本統計")
    print("=" * 60)

    # SLとFCのデータを抽出
    sl_data = df[df["pitch_type"] == "SL"]
    fc_data = df[df["pitch_type"] == "FC"]

    print(f"\nスライダー(SL): {len(sl_data)}球")
    print(f"カットボール(FC): {len(fc_data)}球")

    # 主要な特徴量の統計
    features = [
        "release_speed",
        "release_spin_rate",
        "spin_axis",
        "pfx_x",
        "pfx_z",
        "release_pos_x",
        "release_pos_z",
    ]

    comparison = pd.DataFrame()

    for feature in features:
        if feature in df.columns:
            comparison[f"SL_{feature}_mean"] = [sl_data[feature].mean()]
            comparison[f"SL_{feature}_std"] = [sl_data[feature].std()]
            comparison[f"FC_{feature}_mean"] = [fc_data[feature].mean()]
            comparison[f"FC_{feature}_std"] = [fc_data[feature].std()]
            comparison[f"{feature}_diff"] = [
                abs(sl_data[feature].mean() - fc_data[feature].mean())
            ]

            # t検定で有意差を確認
            t_stat, p_value = stats.ttest_ind(
                sl_data[feature].dropna(), fc_data[feature].dropna()
            )
            comparison[f"{feature}_p_value"] = [p_value]

    print("\n特徴量の比較:")
    print(comparison.T)

    return sl_data, fc_data, comparison


def calculate_new_features(df):
    """新しい特徴量を計算"""
    print("\n" + "=" * 60)
    print("新特徴量の計算")
    print("=" * 60)

    df_new = df.copy()

    # 1. 変化量の大きさ (movement magnitude)
    df_new["movement_magnitude"] = np.sqrt(df_new["pfx_x"] ** 2 + df_new["pfx_z"] ** 2)
    print("✓ movement_magnitude: 変化量の大きさ")

    # 2. 変化の角度 (movement angle)
    df_new["movement_angle"] = np.degrees(np.arctan2(df_new["pfx_z"], df_new["pfx_x"]))
    print("✓ movement_angle: 変化の角度")

    # 3. 横変化と縦変化の比率
    df_new["horizontal_vertical_ratio"] = df_new["pfx_x"] / (
        df_new["pfx_z"] + 0.001
    )  # ゼロ除算回避
    print("✓ horizontal_vertical_ratio: 横変化/縦変化の比率")

    # 4. 回転効率の推定 (spin efficiency)
    # 実際の変化量 / 理論的最大変化量の近似
    df_new["spin_efficiency"] = df_new["movement_magnitude"] / (
        df_new["release_spin_rate"] / 1000 + 0.001
    )
    print("✓ spin_efficiency: 回転効率の推定値")

    # 5. 球速と回転数の比率
    df_new["speed_spin_ratio"] = df_new["release_speed"] / (
        df_new["release_spin_rate"] / 100
    )
    print("✓ speed_spin_ratio: 球速/回転数の比率")

    # 6. リリースポイントの高さと横位置の組み合わせ
    df_new["release_position_magnitude"] = np.sqrt(
        df_new["release_pos_x"] ** 2 + df_new["release_pos_z"] ** 2
    )
    print("✓ release_position_magnitude: リリースポイントの距離")

    # 7. 横変化の絶対値（左右の変化の大きさ）
    df_new["abs_horizontal_movement"] = np.abs(df_new["pfx_x"])
    print("✓ abs_horizontal_movement: 横変化の絶対値")

    # 8. スピンアクシスの正規化（0-180度に変換）
    df_new["normalized_spin_axis"] = df_new["spin_axis"].apply(
        lambda x: x if x <= 180 else 360 - x
    )
    print("✓ normalized_spin_axis: 正規化されたスピンアクシス")

    # === SL/FC 専用の追加特徴量 ===
    print("\n" + "=" * 60)
    print("SL/FC 識別用の追加特徴量")
    print("=" * 60)

    # 9. 球速と横変化の比率 (Cutters are fast with less break)
    df_new["velocity_abs_pfx_x_ratio"] = df_new["release_speed"] / (
        np.abs(df_new["pfx_x"]) + 0.1
    )
    print("✓ velocity_abs_pfx_x_ratio: 球速 / (|横変化| + 0.1)")

    # 10. 球速と縦変化の積 (Cutters have more lift and speed)
    df_new["velocity_times_pfx_z"] = df_new["release_speed"] * df_new["pfx_z"]
    print("✓ velocity_times_pfx_z: 球速 * 縦変化")

    # 11. 縦変化と横変化の差 (Vertical lift vs Horizontal sweep)
    df_new["pfx_z_minus_abs_pfx_x"] = df_new["pfx_z"] - np.abs(df_new["pfx_x"])
    print("✓ pfx_z_minus_abs_pfx_x: 縦変化 - |横変化|")

    return df_new


def compare_new_features(sl_data, fc_data, features):
    """新特徴量でSLとFCを比較"""
    print("\n" + "=" * 60)
    print("新特徴量による識別力の評価")
    print("=" * 60)

    results = []

    for feature in features:
        if feature in sl_data.columns and feature in fc_data.columns:
            sl_mean = sl_data[feature].mean()
            fc_mean = fc_data[feature].mean()
            sl_std = sl_data[feature].std()
            fc_std = fc_data[feature].std()

            # 平均値の差
            mean_diff = abs(sl_mean - fc_mean)

            # 標準偏差の平均
            avg_std = (sl_std + fc_std) / 2

            # Cohen's d (効果量)
            cohens_d = mean_diff / avg_std if avg_std > 0 else 0

            # t検定
            t_stat, p_value = stats.ttest_ind(
                sl_data[feature].dropna(), fc_data[feature].dropna()
            )

            results.append(
                {
                    "feature": feature,
                    "SL_mean": sl_mean,
                    "FC_mean": fc_mean,
                    "mean_diff": mean_diff,
                    "cohens_d": cohens_d,
                    "p_value": p_value,
                    "significant": "***"
                    if p_value < 0.001
                    else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "")),
                }
            )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("cohens_d", ascending=False)

    print("\n識別力ランキング (Cohen's d: 効果量):")
    print("Cohen's d: 0.2=小, 0.5=中, 0.8=大")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(
            f"{row['feature']:30s} | d={row['cohens_d']:.3f} | p={row['p_value']:.6f} {row['significant']}"
        )

    return results_df


def visualize_distributions(sl_data, fc_data, top_features, output_dir="assets"):
    """上位特徴量の分布を可視化"""
    import os

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n分布図を作成中... (保存先: {output_dir}/)")

    n_features = len(top_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feature in enumerate(top_features):
        ax = axes[idx]

        # ヒストグラム
        ax.hist(
            sl_data[feature].dropna(),
            bins=50,
            alpha=0.5,
            label="SL",
            color="blue",
            density=True,
        )
        ax.hist(
            fc_data[feature].dropna(),
            bins=50,
            alpha=0.5,
            label="FC",
            color="red",
            density=True,
        )

        ax.set_xlabel(feature)
        ax.set_ylabel("密度")
        ax.set_title(f"{feature}の分布比較")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 余った軸を非表示
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/sl_fc_feature_distributions.png", dpi=150, bbox_inches="tight"
    )
    print(f"✓ 保存完了: {output_dir}/sl_fc_feature_distributions.png")
    plt.close()


def main():
    """メイン処理"""
    print("=" * 60)
    print("スライダーとカットボールの識別分析")
    print("=" * 60)

    # データ読み込み
    df = load_data()

    # 基本統計の比較
    sl_data, fc_data, basic_comparison = analyze_sl_fc_characteristics(df)

    # 新特徴量の計算
    df_with_new_features = calculate_new_features(df)
    sl_new = df_with_new_features[df_with_new_features["pitch_type"] == "SL"]
    fc_new = df_with_new_features[df_with_new_features["pitch_type"] == "FC"]

    # 新特徴量のリスト
    new_features = [
        "movement_magnitude",
        "movement_angle",
        "horizontal_vertical_ratio",
        "spin_efficiency",
        "speed_spin_ratio",
        "release_position_magnitude",
        "abs_horizontal_movement",
        "normalized_spin_axis",
        "velocity_abs_pfx_x_ratio",
        "velocity_times_pfx_z",
        "pfx_z_minus_abs_pfx_x",
    ]

    # 新特徴量の評価
    results_df = compare_new_features(sl_new, fc_new, new_features)

    # 上位5つの特徴量を可視化
    top_5_features = results_df.head(5)["feature"].tolist()
    visualize_distributions(sl_new, fc_new, top_5_features)

    # 推奨事項の出力
    print("\n" + "=" * 60)
    print("推奨される特徴量")
    print("=" * 60)
    print("\n効果量(Cohen's d)が大きい上位5つの特徴量:")
    for i, (_, row) in enumerate(results_df.head(5).iterrows(), 1):
        print(f"{i}. {row['feature']}")
        print(f"   - Cohen's d: {row['cohens_d']:.3f}")
        print(f"   - SL平均: {row['SL_mean']:.3f}, FC平均: {row['FC_mean']:.3f}")
        print(f"   - 有意性: {row['significant'] if row['significant'] else 'n.s.'}")
        print()

    print("これらの特徴量をモデルに追加することで、")
    print("スライダーとカットボールの識別精度が向上する可能性があります。")

    # 結果をCSVに保存
    results_df.to_csv("sl_fc_feature_analysis.csv", index=False)
    print(f"\n詳細な分析結果を 'sl_fc_feature_analysis.csv' に保存しました。")


if __name__ == "__main__":
    main()
