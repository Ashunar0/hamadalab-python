"""
シンカー(SI)と4シーム(FF)の特性分析スクリプト
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


def compare_basic_features(pitch1_data, pitch2_data):
    """基本統計量の比較"""
    # 分析対象の球種ペア
    TARGET_PITCHES = ["SI", "FF"]
    PITCH_NAMES = {"SI": "Sinker", "FF": "4-Seam Fastball"}

    print("\n" + "=" * 60)
    print(
        f"{PITCH_NAMES[TARGET_PITCHES[0]]}と{PITCH_NAMES[TARGET_PITCHES[1]]}の基本統計"
    )
    print("=" * 60)

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
        if feature in pitch1_data.columns:
            comparison[f"{TARGET_PITCHES[0]}_{feature}_mean"] = [
                pitch1_data[feature].mean()
            ]
            comparison[f"{TARGET_PITCHES[0]}_{feature}_std"] = [
                pitch1_data[feature].std()
            ]
            comparison[f"{TARGET_PITCHES[1]}_{feature}_mean"] = [
                pitch2_data[feature].mean()
            ]
            comparison[f"{TARGET_PITCHES[1]}_{feature}_std"] = [
                pitch2_data[feature].std()
            ]
            comparison[f"{feature}_diff"] = [
                abs(pitch1_data[feature].mean() - pitch2_data[feature].mean())
            ]

            # t検定で有意差を確認
            t_stat, p_value = stats.ttest_ind(
                pitch1_data[feature].dropna(), pitch2_data[feature].dropna()
            )
            comparison[f"{feature}_p_value"] = [p_value]

    print("\n特徴量の比較:")
    print(comparison.T)

    return comparison


def filter_target_pitches(df, pitch1_code, pitch2_code):
    """指定された2つの球種データを抽出"""
    pitch1_data = df[df["pitch_type"] == pitch1_code]
    pitch2_data = df[df["pitch_type"] == pitch2_code]

    print(f"\n{pitch1_code}: {len(pitch1_data)}球")
    print(f"{pitch2_code}: {len(pitch2_data)}球")

    return pitch1_data, pitch2_data


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

    # === SI/FF 専用の追加特徴量 ===
    print("\n" + "=" * 60)
    print("SI/FF 識別用の追加特徴量")
    print("=" * 60)

    # 9. 縦方向の浮き上がり成分（そのまま pfx_z を強調）
    df_new["vertical_rise"] = df_new["pfx_z"]
    print("✓ vertical_rise: 縦方向の浮き上がり成分 (SI=沈む, FF=浮く)")

    # 10. 沈み率（横変化に対する縦変化の比率、負=沈む）
    df_new["sink_rate"] = -df_new["pfx_z"] / (np.abs(df_new["pfx_x"]) + 0.01)
    print("✓ sink_rate: 沈み率 (負の値=沈む、SI特有)")

    # 11. 4シーム基準からの回転軸のずれ
    df_new["spin_axis_deviation_from_fastball"] = np.abs(
        df_new["normalized_spin_axis"] - 180
    )
    print("✓ spin_axis_deviation_from_fastball: 4シーム回転軸(180°)からのずれ")

    return df_new


def compare_new_features(pitch1_data, pitch2_data, features):
    """新特徴量でSIとFFを比較"""
    print("\n" + "=" * 60)
    # 分析対象の球種ペア
    TARGET_PITCHES = ["SI", "FF"]
    PITCH_NAMES = {"SI": "Sinker", "FF": "4-Seam Fastball"}

    print(
        f"新特徴量による{PITCH_NAMES[TARGET_PITCHES[0]]}と{PITCH_NAMES[TARGET_PITCHES[1]]}の識別力の評価"
    )
    print("=" * 60)

    results = []

    for feature in features:
        if feature in pitch1_data.columns and feature in pitch2_data.columns:
            pitch1_mean = pitch1_data[feature].mean()
            pitch2_mean = pitch2_data[feature].mean()
            pitch1_std = pitch1_data[feature].std()
            pitch2_std = pitch2_data[feature].std()

            # 平均値の差
            mean_diff = abs(pitch1_mean - pitch2_mean)

            # Cohen's d (効果量) の計算
            pooled_std = np.sqrt((pitch1_std**2 + pitch2_std**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

            # t検定
            t_stat, p_val = stats.ttest_ind(
                pitch1_data[feature].dropna(),
                pitch2_data[feature].dropna(),
                equal_var=False,
            )

            # 結果の判定
            significance = ""
            if p_val < 0.001:
                significance = "***"
            elif p_val < 0.01:
                significance = "**"
            elif p_val < 0.05:
                significance = "*"

            results.append(
                {
                    "feature": feature,
                    "pitch1_mean": pitch1_mean,
                    "pitch2_mean": pitch2_mean,
                    "mean_diff": mean_diff,
                    "cohens_d": cohens_d,
                    "p_value": p_val,
                    "significant": significance,
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


def visualize_distributions(
    pitch1_data, pitch2_data, top_features, output_dir="assets"
):
    """上位特徴量の分布を可視化"""
    import os

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n分布図を作成中... (保存先: {output_dir}/)")

    n_features = len(top_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    axes = axes.flatten()

    for i, feature in enumerate(top_features):
        ax = axes[i]

        # データの準備
        data1 = pitch1_data[feature].dropna()
        data2 = pitch2_data[feature].dropna()

        # ヒストグラムの描画
        sns.histplot(
            data=data1,
            ax=ax,
            kde=True,
            color="blue",
            label=TARGET_PITCHES[0],
            stat="density",
            alpha=0.5,
        )
        sns.histplot(
            data=data2,
            ax=ax,
            kde=True,
            color="red",
            label=TARGET_PITCHES[1],
            stat="density",
            alpha=0.5,
        )

        ax.set_title(f"{feature} Distribution")
        ax.set_xlabel(feature)
        ax.legend()

    # 余ったサブプロットを削除
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    # Define output file path if output_path is undefined (which it is, since we only pass output_dir)
    # Re-using the global OUTPUT_PREFIX from main logic is tricky here, so we default or need to pass filename.
    # To fix quickly, let's construct it.
    output_file_path = os.path.join(output_dir, "si_ff_feature_distributions.png")

    plt.savefig(output_file_path, dpi=150, bbox_inches="tight")
    print(f"✓ 保存完了: {output_file_path}")
    plt.close()


def main():
    # 分析対象の球種ペア
    global TARGET_PITCHES
    TARGET_PITCHES = ["SI", "FF"]
    PITCH_NAMES = {"SI": "Sinker", "FF": "4-Seam Fastball"}
    OUTPUT_PREFIX = "si_ff"

    print("=" * 60)
    print(
        f"{PITCH_NAMES[TARGET_PITCHES[0]]}と{PITCH_NAMES[TARGET_PITCHES[1]]}の識別分析"
    )
    print("=" * 60)
    print("データを読み込んでいます...")

    # データ読み込み
    df = load_data()
    print(f"総データ数: {len(df)}")

    # ターゲット球種の抽出
    pitch1_data, pitch2_data = filter_target_pitches(
        df, TARGET_PITCHES[0], TARGET_PITCHES[1]
    )

    # 基本統計の比較
    compare_basic_features(pitch1_data, pitch2_data)

    # 新特徴量の計算
    df_with_new_features = calculate_new_features(df)
    pitch1_new = df_with_new_features[
        df_with_new_features["pitch_type"] == TARGET_PITCHES[0]
    ]
    pitch2_new = df_with_new_features[
        df_with_new_features["pitch_type"] == TARGET_PITCHES[1]
    ]

    # SI/FF 専用の追加特徴量のみ（3つ）
    new_features = [
        "vertical_rise",
        "sink_rate",
        "spin_axis_deviation_from_fastball",
    ]

    # 新特徴量の評価
    results_df = compare_new_features(pitch1_new, pitch2_new, new_features)

    # 上位5つの特徴量を可視化
    top_5_features = results_df.head(5)["feature"].tolist()

    # 分布図の作成
    print("\n分布図を作成中... (保存先: assets/)")
    # visualize_distributions は元のままなので sl_fc 用のまま動くが、
    # 本来はファイル名も変えるべき。ここでは簡易的に呼び出しだけ修正
    visualize_distributions(
        pitch1_new,
        pitch2_new,
        top_5_features,
        output_dir="assets",  # output_dir expects a directory, not a file path in the original function
    )
    print(f"✓ 保存完了: assets/{OUTPUT_PREFIX}_feature_distributions.png")

    # 推奨事項の出力
    print("\n" + "=" * 60)
    print("推奨される特徴量")
    print("=" * 60)
    print("\n効果量(Cohen's d)が大きい上位5つの特徴量:")

    # main関数スコープでは直接アクセスできないため、ここでは仮に定義するか、
    # compare_new_featuresから返すように変更が必要。
    # 今回は、compare_new_featuresの内部定義を参考に直接記述する。
    PITCH_NAMES = {"SI": "Sinker", "FF": "4-Seam Fastball"}

    for i, (_, row) in enumerate(results_df.head(5).iterrows(), 1):
        print(f"{i}. {row['feature']}")
        print(f"   - Cohen's d: {row['cohens_d']:.3f}")
        print(
            f"   - {PITCH_NAMES['SI']}平均: {row['pitch1_mean']:.3f}, {PITCH_NAMES['FF']}平均: {row['pitch2_mean']:.3f}"
        )
        print(f"   - 有意性: {row['significant']}\n")

    print("これらの特徴量をモデルに追加することで、")
    print(
        f"{PITCH_NAMES['SI']}と{PITCH_NAMES['FF']}の識別精度が向上する可能性があります。"
    )

    # 結果をCSVに保存
    results_df.to_csv(f"{OUTPUT_PREFIX}_feature_analysis.csv", index=False)
    print(f"\n詳細な分析結果を '{OUTPUT_PREFIX}_feature_analysis.csv' に保存しました。")


if __name__ == "__main__":
    main()
