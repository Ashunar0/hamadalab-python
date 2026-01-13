"""
Step 3: 新特徴量を使ったモデル訓練と評価

新しく追加した特徴量を使ってLightGBMモデルを訓練し、
精度の向上を確認します。

入力: train_with_features.csv, test_with_features.csv
出力: モデルの精度、特徴量重要度、混同行列
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import platform

# 日本語フォント設定
if platform.system() == "Darwin":  # macOS
    plt.rcParams["font.family"] = "Hiragino Sans"
elif platform.system() == "Windows":
    plt.rcParams["font.family"] = "Yu Gothic"
else:  # Linux
    plt.rcParams["font.family"] = "Noto Sans CJK JP"

plt.rcParams["axes.unicode_minus"] = False


def load_data_with_features():
    """特徴量追加済みデータを読み込む"""
    print("=" * 60)
    print("データの読み込み")
    print("=" * 60)

    train_df = pd.read_csv("train_with_features.csv")
    test_df = pd.read_csv("test_with_features.csv")

    print(f"Train data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    print(f"特徴量数: {train_df.shape[1] - 1}")  # pitch_typeを除く

    return train_df, test_df


def prepare_data(train_df, test_df, use_new_features=True):
    """
    データの準備

    Parameters:
    -----------
    use_new_features : bool
        Trueの場合、新特徴量を含む全特徴量を使用
        Falseの場合、元の8特徴量のみを使用(ベースライン比較用)
    """
    # 元の特徴量
    original_features = [
        "release_speed",
        "release_spin_rate",
        "spin_axis",
        "pfx_x",
        "pfx_z",
        "release_pos_x",
        "release_pos_z",
        "p_throws",
    ]

    # 新特徴量
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

    if use_new_features:
        feature_cols = original_features + new_features
        print(
            f"\n使用する特徴量: 元の8特徴量 + 新しい11特徴量 = 合計{len(feature_cols)}特徴量"
        )
    else:
        feature_cols = original_features
        print(f"\n使用する特徴量: 元の8特徴量のみ (ベースライン)")

    # 特徴量とターゲットの分離
    X_train = train_df[feature_cols]
    y_train = train_df["pitch_type"]

    X_test = test_df[feature_cols]
    y_test = test_df["pitch_type"]

    # ラベルエンコーディング
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    print(f"\nクラス数: {len(le.classes_)}")
    print(f"クラス: {list(le.classes_)}")

    return X_train, X_test, y_train_encoded, y_test_encoded, le, feature_cols


def train_model(X_train, y_train, X_test, y_test):
    """LightGBMモデルの訓練"""
    print("\n" + "=" * 60)
    print("モデルの訓練")
    print("=" * 60)

    # LightGBMのパラメータ
    params = {
        "objective": "multiclass",
        "num_class": len(np.unique(y_train)),
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    # データセットの作成
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # 訓練
    print("訓練中...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, test_data],
        valid_names=["train", "test"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    print(f"\n最適なイテレーション数: {model.best_iteration}")

    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, le):
    """モデルの評価"""
    print("\n" + "=" * 60)
    print("モデルの評価")
    print("=" * 60)

    # 予測
    y_train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    y_train_pred_class = np.argmax(y_train_pred, axis=1)

    y_test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_test_pred_class = np.argmax(y_test_pred, axis=1)

    # 精度計算
    train_accuracy = accuracy_score(y_train, y_train_pred_class)
    test_accuracy = accuracy_score(y_test, y_test_pred_class)

    print(f"\nTrain Accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
    print(f"Test Accuracy:  {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    # 詳細なレポート
    print("\n" + "-" * 60)
    print("Classification Report (Test Set)")
    print("-" * 60)
    print(classification_report(y_test, y_test_pred_class, target_names=le.classes_))

    return y_test_pred_class, test_accuracy


def plot_feature_importance(model, feature_names, top_n=15):
    """特徴量重要度のプロット"""
    print("\n" + "=" * 60)
    print("特徴量重要度の分析")
    print("=" * 60)

    # 重要度の取得
    importance = model.feature_importance(importance_type="gain")
    feature_importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values("importance", ascending=False)

    print(f"\nTop {top_n} 重要な特徴量:")
    print(feature_importance_df.head(top_n).to_string(index=False))

    # プロット
    plt.figure(figsize=(10, 8))
    top_features = feature_importance_df.head(top_n)

    plt.barh(range(len(top_features)), top_features["importance"])
    plt.yticks(range(len(top_features)), top_features["feature"])
    plt.xlabel("重要度 (Gain)")
    plt.title(f"特徴量重要度 Top {top_n}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("assets/feature_importance.png", dpi=150, bbox_inches="tight")
    print(f"\n✓ 特徴量重要度を 'assets/feature_importance.png' に保存しました")
    plt.close()

    return feature_importance_df


def plot_confusion_matrix_for_sl_fc(y_true, y_pred, le):
    """SLとFCに焦点を当てた混同行列"""
    print("\n" + "=" * 60)
    print("スライダー vs カットボールの混同行列")
    print("=" * 60)

    # 全体の混同行列
    cm = confusion_matrix(y_true, y_pred)

    # SLとFCのインデックスを取得
    try:
        sl_idx = list(le.classes_).index("SL")
        fc_idx = list(le.classes_).index("FC")
    except ValueError:
        print("⚠ SLまたはFCが見つかりません")
        return

    # SL vs FC の混同行列を抽出
    sl_fc_cm = np.array(
        [
            [cm[sl_idx, sl_idx], cm[sl_idx, fc_idx]],
            [cm[fc_idx, sl_idx], cm[fc_idx, fc_idx]],
        ]
    )

    # 正規化(行ごとのパーセンテージ)
    sl_fc_cm_norm = sl_fc_cm.astype("float") / sl_fc_cm.sum(axis=1)[:, np.newaxis]

    # プロット
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 絶対値
    sns.heatmap(
        sl_fc_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax1,
        xticklabels=["SL", "FC"],
        yticklabels=["SL", "FC"],
    )
    ax1.set_title("SL vs FC 混同行列 (絶対値)")
    ax1.set_ylabel("実際の球種")
    ax1.set_xlabel("予測された球種")

    # パーセンテージ
    sns.heatmap(
        sl_fc_cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        ax=ax2,
        xticklabels=["SL", "FC"],
        yticklabels=["SL", "FC"],
    )
    ax2.set_title("SL vs FC 混同行列 (割合)")
    ax2.set_ylabel("実際の球種")
    ax2.set_xlabel("予測された球種")

    plt.tight_layout()
    plt.savefig("assets/sl_fc_confusion_matrix.png", dpi=150, bbox_inches="tight")
    print(f"✓ 混同行列を 'assets/sl_fc_confusion_matrix.png' に保存しました")
    plt.close()

    # 統計情報
    sl_correct = sl_fc_cm[0, 0]
    sl_total = sl_fc_cm[0, :].sum()
    sl_to_fc_errors = sl_fc_cm[0, 1]

    fc_correct = sl_fc_cm[1, 1]
    fc_total = sl_fc_cm[1, :].sum()
    fc_to_sl_errors = sl_fc_cm[1, 0]

    print(f"\nスライダー(SL):")
    print(f"  正解: {sl_correct}/{sl_total} ({sl_correct / sl_total * 100:.2f}%)")
    print(f"  FCと誤分類: {sl_to_fc_errors} ({sl_to_fc_errors / sl_total * 100:.2f}%)")

    print(f"\nカットボール(FC):")
    print(f"  正解: {fc_correct}/{fc_total} ({fc_correct / fc_total * 100:.2f}%)")
    print(f"  SLと誤分類: {fc_to_sl_errors} ({fc_to_sl_errors / fc_total * 100:.2f}%)")


def compare_with_baseline(baseline_accuracy, new_accuracy):
    """ベースラインとの比較"""
    print("\n" + "=" * 60)
    print("ベースラインとの比較")
    print("=" * 60)

    improvement = (new_accuracy - baseline_accuracy) * 100
    relative_improvement = (
        (new_accuracy - baseline_accuracy) / baseline_accuracy
    ) * 100

    print(f"\nベースライン精度 (元の8特徴量): {baseline_accuracy * 100:.2f}%")
    print(f"新モデル精度 (16特徴量):        {new_accuracy * 100:.2f}%")
    print(f"\n改善:")
    print(f"  絶対値: +{improvement:.2f}ポイント")
    print(f"  相対値: +{relative_improvement:.2f}%")

    if improvement > 0:
        print(f"\n✓ 精度が向上しました!")
    elif improvement == 0:
        print(f"\n= 精度は変わりませんでした")
    else:
        print(f"\n⚠ 精度が低下しました")


def main():
    """メイン処理"""
    print("=" * 60)
    print("Step 3: 新特徴量を使ったモデル訓練と評価")
    print("=" * 60)

    # データ読み込み
    train_df, test_df = load_data_with_features()

    # ベースラインモデル(元の特徴量のみ)
    print("\n" + "#" * 60)
    print("# ベースラインモデル (元の8特徴量のみ)")
    print("#" * 60)
    X_train_base, X_test_base, y_train, y_test, le, base_features = prepare_data(
        train_df, test_df, use_new_features=False
    )
    model_baseline = train_model(X_train_base, y_train, X_test_base, y_test)
    _, baseline_accuracy = evaluate_model(
        model_baseline, X_train_base, y_train, X_test_base, y_test, le
    )

    # 新モデル(全特徴量)
    print("\n\n" + "#" * 60)
    print("# 新モデル (元の8特徴量 + 新しい8特徴量)")
    print("#" * 60)
    X_train_new, X_test_new, y_train, y_test, le, all_features = prepare_data(
        train_df, test_df, use_new_features=True
    )
    model_new = train_model(X_train_new, y_train, X_test_new, y_test)
    y_pred, new_accuracy = evaluate_model(
        model_new, X_train_new, y_train, X_test_new, y_test, le
    )

    # 特徴量重要度
    feature_importance_df = plot_feature_importance(model_new, all_features, top_n=16)

    # SL vs FC の混同行列
    plot_confusion_matrix_for_sl_fc(y_test, y_pred, le)

    # === SI vs FF の混同行列 ===
    print("\n" + "=" * 60)
    print("シンカー(SI) vs 4シーム(FF) の混同行列")
    print("=" * 60)

    try:
        si_idx = le.transform(["SI"])[0]
        ff_idx = le.transform(["FF"])[0]

        si_mask = y_test == si_idx
        ff_mask = y_test == ff_idx

        si_pred = y_pred[si_mask]
        ff_pred = y_pred[ff_mask]

        si_correct = np.sum(si_pred == si_idx)
        si_ff_mistake = np.sum(si_pred == ff_idx)

        ff_correct = np.sum(ff_pred == ff_idx)
        ff_si_mistake = np.sum(ff_pred == si_idx)

        print("\nシンカー(SI):")
        print(
            f"  正解: {si_correct}/{len(si_pred)} ({si_correct / len(si_pred) * 100:.2f}%)"
        )
        print(
            f"  FFと誤分類: {si_ff_mistake} ({si_ff_mistake / len(si_pred) * 100:.2f}%)"
        )

        print("\n4シーム(FF):")
        print(
            f"  正解: {ff_correct}/{len(ff_pred)} ({ff_correct / len(ff_pred) * 100:.2f}%)"
        )
        print(
            f"  SIと誤分類: {ff_si_mistake} ({ff_si_mistake / len(ff_pred) * 100:.2f}%)"
        )
    except Exception as e:
        print(f"SI/FF 混同行列の計算中にエラーが発生しました: {e}")

    # ベースラインとの比較
    compare_with_baseline(baseline_accuracy, new_accuracy)

    # 特徴量重要度をCSVに保存
    feature_importance_df.to_csv("feature_importance.csv", index=False)
    print(f"\n✓ 特徴量重要度を 'feature_importance.csv' に保存しました")

    print("\n" + "=" * 60)
    print("処理完了!")
    print("=" * 60)


if __name__ == "__main__":
    import os

    os.makedirs("assets", exist_ok=True)
    main()
