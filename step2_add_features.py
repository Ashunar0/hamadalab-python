"""
Step 2: 新特徴量の追加

スライダーとカットボールの識別を改善するため、
統計分析に基づいて8つの新特徴量を追加します。

入力: train_corrected.csv, test_corrected.csv
出力: train_with_features.csv, test_with_features.csv
"""

import pandas as pd
import numpy as np
import os


def load_corrected_data():
    """補正済みデータを読み込む"""
    print("=" * 60)
    print("補正済みデータの読み込み")
    print("=" * 60)

    if not os.path.exists("train_corrected.csv") or not os.path.exists(
        "test_corrected.csv"
    ):
        print(
            "エラー: train_corrected.csv または test_corrected.csv が見つかりません。"
        )
        print("先に step1_LH_correction.py を実行してください。")
        return None, None

    train_df = pd.read_csv("train_corrected.csv")
    test_df = pd.read_csv("test_corrected.csv")

    print(f"Train data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")

    return train_df, test_df


def add_new_features(df):
    """
    新特徴量を追加

    追加する特徴量:
    1. normalized_spin_axis (d=1.555) - 最重要
    2. movement_angle (d=1.215) - 最重要
    3. abs_horizontal_movement (d=0.634)
    4. spin_efficiency (d=0.568)
    5. movement_magnitude (d=0.532)
    6. speed_spin_ratio (d=0.423)
    7. horizontal_vertical_ratio (d=0.114)
    8. release_position_magnitude (d=0.068)
    """
    print("\n" + "=" * 60)
    print("新特徴量の計算")
    print("=" * 60)

    df_new = df.copy()

    # 1. normalized_spin_axis (最重要: d=1.555)
    print("\n[1/8] normalized_spin_axis を計算中...")
    if "spin_axis" in df_new.columns:
        df_new["normalized_spin_axis"] = df_new["spin_axis"].apply(
            lambda x: x if x <= 180 else 360 - x
        )
        print(
            f"  ✓ 完了 (範囲: {df_new['normalized_spin_axis'].min():.1f} - {df_new['normalized_spin_axis'].max():.1f})"
        )
    else:
        print("  ⚠ spin_axis列が見つかりません")

    # 2. movement_angle (最重要: d=1.215)
    print("\n[2/8] movement_angle を計算中...")
    if "pfx_x" in df_new.columns and "pfx_z" in df_new.columns:
        df_new["movement_angle"] = np.degrees(
            np.arctan2(df_new["pfx_z"], df_new["pfx_x"])
        )
        print(
            f"  ✓ 完了 (範囲: {df_new['movement_angle'].min():.1f}° - {df_new['movement_angle'].max():.1f}°)"
        )
    else:
        print("  ⚠ pfx_x または pfx_z列が見つかりません")

    # 3. abs_horizontal_movement (d=0.634)
    print("\n[3/8] abs_horizontal_movement を計算中...")
    if "pfx_x" in df_new.columns:
        df_new["abs_horizontal_movement"] = np.abs(df_new["pfx_x"])
        print(f"  ✓ 完了 (平均: {df_new['abs_horizontal_movement'].mean():.3f})")
    else:
        print("  ⚠ pfx_x列が見つかりません")

    # 4. movement_magnitude (d=0.532) - spin_efficiencyの計算に必要
    print("\n[4/8] movement_magnitude を計算中...")
    if "pfx_x" in df_new.columns and "pfx_z" in df_new.columns:
        df_new["movement_magnitude"] = np.sqrt(
            df_new["pfx_x"] ** 2 + df_new["pfx_z"] ** 2
        )
        print(f"  ✓ 完了 (平均: {df_new['movement_magnitude'].mean():.3f})")
    else:
        print("  ⚠ pfx_x または pfx_z列が見つかりません")

    # 5. spin_efficiency (d=0.568)
    print("\n[5/8] spin_efficiency を計算中...")
    if "movement_magnitude" in df_new.columns and "release_spin_rate" in df_new.columns:
        # ゼロ除算を避けるため、小さな値を加算
        df_new["spin_efficiency"] = df_new["movement_magnitude"] / (
            df_new["release_spin_rate"] / 1000 + 0.001
        )
        print(f"  ✓ 完了 (平均: {df_new['spin_efficiency'].mean():.3f})")
    else:
        print("  ⚠ movement_magnitude または release_spin_rate列が見つかりません")

    # 6. speed_spin_ratio (d=0.423)
    print("\n[6/8] speed_spin_ratio を計算中...")
    if "release_speed" in df_new.columns and "release_spin_rate" in df_new.columns:
        df_new["speed_spin_ratio"] = df_new["release_speed"] / (
            df_new["release_spin_rate"] / 100
        )
        print(f"  ✓ 完了 (平均: {df_new['speed_spin_ratio'].mean():.3f})")
    else:
        print("  ⚠ release_speed または release_spin_rate列が見つかりません")

    # 7. horizontal_vertical_ratio (d=0.114)
    print("\n[7/8] horizontal_vertical_ratio を計算中...")
    if "pfx_x" in df_new.columns and "pfx_z" in df_new.columns:
        # ゼロ除算を避けるため、小さな値を加算
        df_new["horizontal_vertical_ratio"] = df_new["pfx_x"] / (
            df_new["pfx_z"] + 0.001
        )
        print(f"  ✓ 完了 (平均: {df_new['horizontal_vertical_ratio'].mean():.3f})")
    else:
        print("  ⚠ pfx_x または pfx_z列が見つかりません")

    # 8. release_position_magnitude (d=0.068)
    print("\n[8/8] release_position_magnitude を計算中...")
    if "release_pos_x" in df_new.columns and "release_pos_z" in df_new.columns:
        df_new["release_position_magnitude"] = np.sqrt(
            df_new["release_pos_x"] ** 2 + df_new["release_pos_z"] ** 2
        )
        print(f"  ✓ 完了 (平均: {df_new['release_position_magnitude'].mean():.3f})")
    else:
        print("  ⚠ release_pos_x または release_pos_z列が見つかりません")

    # === SI/FF 専用の追加特徴量 ===
    print("\n" + "=" * 60)
    print("SI/FF 識別用の追加特徴量 (3つ)")
    print("=" * 60)

    # 9. vertical_rise (d=1.975) - 縦方向の浮き上がり成分
    print("\n[9/11] vertical_rise を計算中...")
    if "pfx_z" in df_new.columns:
        df_new["vertical_rise"] = df_new["pfx_z"]
        print(f"  ✓ 完了 (平均: {df_new['vertical_rise'].mean():.3f})")
        print("  → SI=沈む(小), FF=浮く(大)")
    else:
        print("  ⚠ pfx_z列が見つかりません")

    # 10. sink_rate (d=0.544) - 沈み率
    print("\n[10/11] sink_rate を計算中...")
    if "pfx_z" in df_new.columns and "pfx_x" in df_new.columns:
        df_new["sink_rate"] = -df_new["pfx_z"] / (np.abs(df_new["pfx_x"]) + 0.01)
        print(f"  ✓ 完了 (平均: {df_new['sink_rate'].mean():.3f})")
        print("  → 負の値=沈む(SI特有)")
    else:
        print("  ⚠ pfx_z または pfx_x列が見つかりません")

    # 11. spin_axis_deviation_from_fastball (d=0.713) - 4シームからの回転軸のずれ
    print("\n[11/11] spin_axis_deviation_from_fastball を計算中...")
    if "normalized_spin_axis" in df_new.columns:
        df_new["spin_axis_deviation_from_fastball"] = np.abs(
            df_new["normalized_spin_axis"] - 180
        )
        print(
            f"  ✓ 完了 (平均: {df_new['spin_axis_deviation_from_fastball'].mean():.3f})"
        )
        print("  → 4シーム回転軸(180°)からのずれ")
    else:
        print("  ⚠ normalized_spin_axis列が見つかりません")

    return df_new


def validate_features(df, feature_names):
    """新特徴量の品質チェック"""
    print("\n" + "=" * 60)
    print("特徴量の品質チェック")
    print("=" * 60)

    issues_found = False

    for feature in feature_names:
        if feature not in df.columns:
            print(f"⚠ {feature}: 列が存在しません")
            issues_found = True
            continue

        # NaNのチェック
        nan_count = df[feature].isna().sum()
        if nan_count > 0:
            print(
                f"⚠ {feature}: {nan_count}個のNaN値があります ({nan_count / len(df) * 100:.2f}%)"
            )
            issues_found = True

        # 無限大のチェック
        inf_count = np.isinf(df[feature]).sum()
        if inf_count > 0:
            print(f"⚠ {feature}: {inf_count}個の無限大値があります")
            issues_found = True

        # 正常な場合
        if nan_count == 0 and inf_count == 0:
            print(
                f"✓ {feature}: OK (範囲: {df[feature].min():.3f} - {df[feature].max():.3f})"
            )

    if not issues_found:
        print("\n全ての特徴量が正常です!")
    else:
        print("\n⚠ いくつかの問題が見つかりました。上記を確認してください。")

    return not issues_found


def save_data_with_features(train_df, test_df):
    """特徴量を追加したデータを保存"""
    print("\n" + "=" * 60)
    print("データの保存")
    print("=" * 60)

    train_output = "train_with_features.csv"
    test_output = "test_with_features.csv"

    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)

    print(f"✓ {train_output} を保存しました ({train_df.shape})")
    print(f"✓ {test_output} を保存しました ({test_df.shape})")

    # 列数の確認
    print("\n元の特徴量数: 9")
    print("追加した特徴量数: 11 (SL/FC用8つ + SI/FF用3つ)")
    print(f"合計特徴量数: {len(train_df.columns)}")


def print_summary(train_df, test_df):
    """処理のサマリーを表示"""
    print("\n" + "=" * 60)
    print("処理完了サマリー")
    print("=" * 60)

    new_features = [
        "normalized_spin_axis",
        "movement_angle",
        "abs_horizontal_movement",
        "movement_magnitude",
        "spin_efficiency",
        "speed_spin_ratio",
        "horizontal_vertical_ratio",
        "release_position_magnitude",
        # SI/FF 専用の追加特徴量
        "vertical_rise",
        "sink_rate",
        "spin_axis_deviation_from_fastball",
    ]

    print("\n追加された特徴量:")
    for i, feature in enumerate(new_features, 1):
        if feature in train_df.columns:
            print(f"  {i}. {feature}")

    print(f"\n出力ファイル:")
    print(
        f"  - train_with_features.csv ({train_df.shape[0]} rows, {train_df.shape[1]} columns)"
    )
    print(
        f"  - test_with_features.csv ({test_df.shape[0]} rows, {test_df.shape[1]} columns)"
    )

    print("\n次のステップ:")
    print("  1. これらの新特徴量を使ってモデルを再訓練")
    print("  2. 精度の向上を確認")
    print("  3. 特徴量重要度を分析")


def main():
    """メイン処理"""
    print("=" * 60)
    print("Step 2: 新特徴量の追加")
    print("=" * 60)

    # データ読み込み
    train_df, test_df = load_corrected_data()

    if train_df is None or test_df is None:
        return

    # 新特徴量の追加
    train_with_features = add_new_features(train_df)
    test_with_features = add_new_features(test_df)

    # 新特徴量のリスト
    new_feature_names = [
        "normalized_spin_axis",
        "movement_angle",
        "abs_horizontal_movement",
        "movement_magnitude",
        "spin_efficiency",
        "speed_spin_ratio",
        "horizontal_vertical_ratio",
        "release_position_magnitude",
        # SI/FF 専用の追加特徴量
        "vertical_rise",
        "sink_rate",
        "spin_axis_deviation_from_fastball",
    ]

    # 品質チェック
    print("\n--- Train Data ---")
    train_valid = validate_features(train_with_features, new_feature_names)

    print("\n--- Test Data ---")
    test_valid = validate_features(test_with_features, new_feature_names)

    if train_valid and test_valid:
        # データ保存
        save_data_with_features(train_with_features, test_with_features)

        # サマリー表示
        print_summary(train_with_features, test_with_features)
    else:
        print("\n⚠ 品質チェックで問題が見つかったため、データを保存しませんでした。")
        print("上記のエラーを確認してください。")


if __name__ == "__main__":
    main()
