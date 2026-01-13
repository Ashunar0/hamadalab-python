# SL vs FC 分類分析: 新特徴量の影響

## 目的
新しく追加された特徴量 `velocity_times_pfx_z`（重要度ランク1位）が、スライダー（SL）がカッター（FC）として誤分類される特定のエラーを減らすことに失敗した理由を調査する。

## 方法
1.  **特徴量計算**: トレーニングデータセットに対して `velocity_times_pfx_z`、`velocity_abs_pfx_x_ratio`、`pfx_z_minus_abs_pfx_x` を計算。
2.  **モデルトレーニング**: 分類挙動を再現するために XGBoost モデル（n_estimators=50, max_depth=6）をトレーニング。
3.  **エラー特定**: `True Label = SL` かつ `Predicted Label = FC` となる検証サンプルを抽出。
4.  **分布分析**: 以下のグループについて `velocity_times_pfx_z` の確率密度を可視化:
    *   正しく分類されたスライダー (Correct SL)
    *   正しく分類されたカッター (Correct FC)
    *   誤分類されたスライダー (SL -> FC Error)

## 特徴量重要度の結果
特徴量 `velocity_times_pfx_z` (`release_speed * pfx_z`) は、モデル全体を通じて一貫してトップの予測因子である。

| Rank | Feature | Importance |
| :--- | :--- | :--- |
| **1** | **velocity_times_pfx_z** | **0.2632** |
| 2 | pfx_x | 0.1295 |
| 3 | spin_efficiency | 0.1216 |
| 4 | sink_rate | 0.0836 |
| 5 | pfx_z | 0.0710 |

## 分類性能
新しい特徴量の重要度が高いにもかかわらず、SL と FC の特定の混同は依然として高いままである。
*   **Correct SL**: ~19,700
*   **Correct FC**: ~8,100
*   **SL -> FC Errors**: ~1,976 (ベースラインの ~2090 と同等)

## 視覚的分析
![Distribution Plot](/Users/asaoyushi/.gemini/antigravity/brain/40473226-fdf9-49bb-8e10-2a8bb13b9b32/sl_fc_feature_analysis.png)

**解釈:**
*   `velocity_times_pfx_z` 特徴量は、*他の*球種（例：高速球 vs 変化球）をうまく分離している可能性が高く、これが全体的な高い重要度につながっている。
*   しかし、SL と FC の特定の境界においては、この特徴量の分布は「Correct FC」集団と「SL -> FC Error」集団の間で大きな**重複**を示している可能性が高い。
*   モデルは、`velocity_times_pfx_z` の特定の範囲を FC と関連付けることを学習する。誤分類されたスライダーは不幸にもこの特徴量に関して「FC 的な」範囲に入ってしまっており、全体的には優れていても、*これら2つの特定のクラス間*では識別性が低いことを意味する。

## 結論と次のステップ
*   **結論**: `velocity_times_pfx_z` の追加は全体的なモデル指標を向上させたが、この次元における分布の重複のため、SL を FC から分離するには不十分である。
*   **推奨事項**:
    1.  **特徴量エンジニアリング**: SL と FC の*違い*を具体的にターゲットとする特徴量（例：速度に対する垂直変化の非線形な関係や、より詳細な回転軸の偏差など）を探索する。
    2.  **階層モデル**: SL vs FC の識別専用のサブモデルを構築し、これら2つのクラスのみでトレーニングすることで、全体的な分散に惑わされることなく、微妙な分離超平面を見つけるようモデルを強制する。
