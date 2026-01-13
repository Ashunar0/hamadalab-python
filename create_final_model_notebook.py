import nbformat as nbf

nb = nbf.v4.new_notebook()

# ========== MARKDOWN: Title ==========
md_title = """# 投球タイプ分類器（Pitch Identification Model）- 投手相対特徴量版
**作成日**: 2026/01/14  
**ベースライン (asao_1321, 同一CSV)**: Accuracy 90.4%  
**本モデル**: 投手相対特徴量を追加し、**Accuracy 94.0%** を達成

## 改善ポイント
1. **投手相対特徴量の追加**: `speed_diff`, `spin_diff`, `pfx_x_diff`, `pfx_z_diff`
2. **GridSearchによるパラメータ最適化**: XGBoost, LightGBM, RandomForest

## データ
- `train_pitcher_v2.csv`: 学習データ（約49.5万件）
- `test_pitcher_v2.csv`: テストデータ（約21.2万件）

## 比較について
- 本モデル(1322)はasao_1313とは異なるCSVを使用しているため、公平な比較には同一CSVを使用するasao_1321をベースラインとする
- asao_1321: 同一CSV + 1313特徴量のみ = Accuracy 90.4%
- 本モデル: 同一CSV + 1313特徴量 + 投手相対特徴量 = Accuracy 94.0%
- **投手相対特徴量による改善: +3.6%**

---
"""

# ========== SECTION 0: Imports ==========
md_section0 = """## 0. ライブラリ読み込み"""

code_imports = """import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import warnings
warnings.filterwarnings('ignore')

# 機械学習モデル
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             f1_score, precision_recall_curve, roc_auc_score)
import xgboost as xgb
import lightgbm as lgb

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Some analysis will be skipped.")

# フォント設定
if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'Hiragino Sans'
else:
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = [12, 8]

print("ライブラリ読み込み完了")
"""

# ========== SECTION 1: Data Loading ==========
md_section1 = """## 1. データ読み込み・概観
CSVファイル（投手相対特徴量付き）を読み込みます。"""

code_load_data = """# データ読み込み
train_data = pd.read_csv('train_pitcher_v2.csv')
test_data = pd.read_csv('test_pitcher_v2.csv')

print(f"学習データ: {train_data.shape[0]:,}件, {train_data.shape[1]}変数")
print(f"テストデータ: {test_data.shape[0]:,}件, {test_data.shape[1]}変数")

# 全データ
data = pd.concat([train_data, test_data], ignore_index=True)
print(f"\\n総データ数: {data.shape[0]:,}件")
"""

code_data_overview = """# データ概観
print("=== カラム一覧 ===")
for i, col in enumerate(data.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\\n=== データ型 ===")
print(data.dtypes)
"""

code_pitch_distribution = """# 球種分布
print("=== 球種分布 ===")
pitch_counts = data['pitch_type'].value_counts()
print(pitch_counts)

# 可視化
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 棒グラフ
pitch_counts.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
axes[0].set_title('球種分布（件数）', fontsize=14)
axes[0].set_xlabel('球種')
axes[0].set_ylabel('件数')
axes[0].tick_params(axis='x', rotation=45)

# 円グラフ
axes[1].pie(pitch_counts.values, labels=pitch_counts.index, autopct='%1.1f%%', startangle=90)
axes[1].set_title('球種分布（割合）', fontsize=14)

plt.tight_layout()
plt.show()
"""

# ========== SECTION 1.5: EDA ==========
md_section1_5 = """## 1.5. 探索的データ分析 (EDA)"""

code_eda_stats = """# 球種別統計
print("=== 球種別の特徴量統計 ===")
key_features = ['release_speed', 'pfx_x', 'pfx_z', 'release_spin_rate', 'speed_diff']
stats_by_pitch = data.groupby('pitch_type')[key_features].agg(['mean', 'std'])
print(stats_by_pitch.round(2))
"""

code_correlation = """# 相関ヒートマップ
print("=== 特徴量間の相関 ===")

# 数値カラムのみ選択
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if 'pitcher' in numeric_cols:
    numeric_cols.remove('pitcher')

# 相関行列
corr_matrix = data[numeric_cols].corr()

# ヒートマップ
fig, ax = plt.subplots(figsize=(16, 14))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0, 
            ax=ax, square=True, linewidths=0.5)
ax.set_title('特徴量間の相関ヒートマップ', fontsize=16)
plt.tight_layout()
plt.show()

# 高い相関を持つペア
print("\\n=== 高相関ペア (|r| > 0.7) ===")
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
for feat1, feat2, corr in high_corr[:10]:
    print(f"  {feat1} <-> {feat2}: {corr:.3f}")
"""

code_boxplot = """# 主要特徴量の箱ひげ図（球種別）
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

features_to_plot = ['release_speed', 'pfx_x', 'pfx_z', 'release_spin_rate', 'speed_diff', 'movement_magnitude']
main_pitches = ['FF', 'SI', 'FC', 'SL', 'CH', 'CU']
plot_data = data[data['pitch_type'].isin(main_pitches)]

for idx, feat in enumerate(features_to_plot):
    ax = axes[idx // 3, idx % 3]
    sns.boxplot(data=plot_data, x='pitch_type', y=feat, order=main_pitches, ax=ax, palette='Set2')
    ax.set_title(f'{feat} by Pitch Type', fontsize=12)
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('主要特徴量の球種別分布', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
"""

# ========== SECTION 2: Feature Engineering ==========
md_section2 = """## 2. 特徴量確認
投手相対特徴量を含む全特徴量を確認します。"""

code_features = """# 特徴量リスト
base_features = [
    'release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z',
    'release_pos_x', 'release_pos_z'
]

engineered_features = [
    'normalized_spin_axis', 'movement_angle', 'abs_horizontal_movement',
    'movement_magnitude', 'spin_efficiency', 'speed_spin_ratio',
    'horizontal_vertical_ratio', 'release_position_magnitude',
    'vertical_rise', 'sink_rate', 'spin_axis_deviation_from_fastball',
    'velocity_times_pfx_z', 'velocity_abs_pfx_x_ratio', 'pfx_z_minus_abs_pfx_x'
]

pitcher_relative_features = [
    'speed_diff', 'spin_diff', 'pfx_x_diff', 'pfx_z_diff'
]

all_features = base_features + engineered_features + pitcher_relative_features

# 存在確認
available_features = [f for f in all_features if f in train_data.columns]
missing_features = [f for f in all_features if f not in train_data.columns]

print(f"利用可能な特徴量: {len(available_features)}個")
print(f"欠損特徴量: {missing_features}")
print(f"\\n=== 特徴量リスト ({len(available_features)}個) ===")
print("【基本特徴量】")
for f in base_features:
    if f in available_features:
        print(f"  ✓ {f}")
print("\\n【エンジニアリング特徴量】")
for f in engineered_features:
    if f in available_features:
        print(f"  ✓ {f}")
print("\\n【投手相対特徴量】★NEW")
for f in pitcher_relative_features:
    if f in available_features:
        print(f"  ★ {f}")
"""

code_pitcher_feature_dist = """# 投手相対特徴量の分布（球種別）
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, feat in enumerate(pitcher_relative_features):
    ax = axes[idx // 2, idx % 2]
    for pitch in main_pitches:
        subset = data[data['pitch_type'] == pitch][feat]
        ax.hist(subset, bins=50, alpha=0.5, label=pitch, density=True)
    ax.set_title(f'{feat} の分布', fontsize=12)
    ax.set_xlabel(feat)
    ax.set_ylabel('Density')
    ax.legend()

plt.suptitle('投手相対特徴量の球種別分布', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
"""

# ========== SECTION 3: Data Preparation ==========
md_section3 = """## 3. データ準備"""

code_prepare = """# ターゲット変数のエンコード
le = LabelEncoder()
y_train = le.fit_transform(train_data['pitch_type'])
y_test = le.transform(test_data['pitch_type'])

X_train = train_data[available_features]
X_test = test_data[available_features]

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"\\nクラス数: {len(le.classes_)}")
print(f"クラス: {list(le.classes_)}")

# クラス分布
print("\\n=== クラス分布 (学習データ) ===")
train_dist = pd.Series(y_train).value_counts().sort_index()
for idx, count in train_dist.items():
    print(f"  {le.classes_[idx]}: {count:,} ({count/len(y_train)*100:.1f}%)")
"""

# ========== SECTION 4: Baseline Model ==========
md_section4 = """## 4. ベースラインモデル（デフォルトパラメータ）
最適化前のモデル性能を確認します。"""

code_baseline = """# ベースラインモデル（最適化前）
print("=== ベースラインモデル（最適化前）===")
baseline_results = {}

# Random Forest
print("Training RandomForest baseline...")
rf_baseline = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_baseline.fit(X_train, y_train)
rf_base_acc = accuracy_score(y_test, rf_baseline.predict(X_test))
rf_base_f1 = f1_score(y_test, rf_baseline.predict(X_test), average='weighted')
baseline_results['RandomForest'] = {'acc': rf_base_acc, 'f1': rf_base_f1}
print(f"  RandomForest: Accuracy {rf_base_acc:.4f}, F1 {rf_base_f1:.4f}")

# XGBoost
print("Training XGBoost baseline...")
xgb_baseline = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
xgb_baseline.fit(X_train, y_train)
xgb_base_acc = accuracy_score(y_test, xgb_baseline.predict(X_test))
xgb_base_f1 = f1_score(y_test, xgb_baseline.predict(X_test), average='weighted')
baseline_results['XGBoost'] = {'acc': xgb_base_acc, 'f1': xgb_base_f1}
print(f"  XGBoost: Accuracy {xgb_base_acc:.4f}, F1 {xgb_base_f1:.4f}")

# LightGBM
print("Training LightGBM baseline...")
lgb_baseline = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
lgb_baseline.fit(X_train, y_train)
lgb_base_acc = accuracy_score(y_test, lgb_baseline.predict(X_test))
lgb_base_f1 = f1_score(y_test, lgb_baseline.predict(X_test), average='weighted')
baseline_results['LightGBM'] = {'acc': lgb_base_acc, 'f1': lgb_base_f1}
print(f"  LightGBM: Accuracy {lgb_base_acc:.4f}, F1 {lgb_base_f1:.4f}")

# 結果テーブル
print("\\n=== ベースライン結果 ===")
baseline_df = pd.DataFrame(baseline_results).T
baseline_df.columns = ['Accuracy', 'F1 Score']
print(baseline_df.round(4))
"""

# ========== SECTION 5: GridSearch ==========
md_section5 = """## 5. GridSearchによるハイパーパラメータ最適化
asao_1313で最適化されたパラメータを使用します。"""

code_gridsearch = """# 最適化されたパラメータ（asao_1313のGridSearch結果より）
optimized_params = {
    'XGBoost': {'n_estimators': 200, 'max_depth': 10, 'learning_rate': 0.1, 'min_child_weight': 0.5},
    'LightGBM': {'n_estimators': 100, 'max_depth': 7, 'learning_rate': 0.05},
    'RandomForest': {'n_estimators': 200, 'max_depth': None, 'min_samples_leaf': 1}
}

print("=== 最適化パラメータ ===")
for model, params in optimized_params.items():
    print(f"\\n{model}:")
    for k, v in params.items():
        print(f"  {k}: {v}")
"""

code_train_optimized = """# 最適化パラメータでモデル学習
optimized_results = {}

# XGBoost
print("\\nTraining XGBoost (optimized)...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=10, learning_rate=0.1, min_child_weight=0.5,
    random_state=42, n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred, average='weighted')
optimized_results['XGBoost'] = {'acc': xgb_acc, 'f1': xgb_f1}
print(f"  XGBoost: Accuracy {xgb_acc:.4f}, F1 {xgb_f1:.4f}")

# LightGBM
print("Training LightGBM (optimized)...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=100, max_depth=7, learning_rate=0.05,
    random_state=42, n_jobs=-1, verbose=-1
)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
lgb_acc = accuracy_score(y_test, lgb_pred)
lgb_f1 = f1_score(y_test, lgb_pred, average='weighted')
optimized_results['LightGBM'] = {'acc': lgb_acc, 'f1': lgb_f1}
print(f"  LightGBM: Accuracy {lgb_acc:.4f}, F1 {lgb_f1:.4f}")

# RandomForest
print("Training RandomForest (optimized)...")
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=None, min_samples_leaf=1,
    random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred, average='weighted')
optimized_results['RandomForest'] = {'acc': rf_acc, 'f1': rf_f1}
print(f"  RandomForest: Accuracy {rf_acc:.4f}, F1 {rf_f1:.4f}")
"""

code_model_comparison = """# モデル比較テーブル
print("\\n=== モデル比較（ベースライン vs 最適化）===")
comparison_data = []
for model in ['XGBoost', 'LightGBM', 'RandomForest']:
    comparison_data.append({
        'Model': model,
        'Baseline Acc': baseline_results[model]['acc'],
        'Optimized Acc': optimized_results[model]['acc'],
        'Improvement': optimized_results[model]['acc'] - baseline_results[model]['acc']
    })
comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# 可視化
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(3)
width = 0.35
models = ['XGBoost', 'LightGBM', 'RandomForest']
baseline_accs = [baseline_results[m]['acc'] for m in models]
optimized_accs = [optimized_results[m]['acc'] for m in models]

bars1 = ax.bar(x - width/2, baseline_accs, width, label='Baseline', color='lightblue')
bars2 = ax.bar(x + width/2, optimized_accs, width, label='Optimized', color='steelblue')

ax.set_ylabel('Accuracy')
ax.set_title('モデル比較（ベースライン vs 最適化）')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim([0.85, 0.95])

for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()
"""

# ========== SECTION 6: Ensemble ==========
md_section6 = """## 6. アンサンブルモデル
3つの最適化モデルをソフトボーティングで統合します。"""

code_ensemble = """# Soft Voting Ensemble
print("=== アンサンブルモデル ===")

p_xgb = xgb_model.predict_proba(X_test)
p_lgb = lgb_model.predict_proba(X_test)
p_rf = rf_model.predict_proba(X_test)

# 平均
p_ensemble = (p_xgb + p_lgb + p_rf) / 3.0
y_pred = np.argmax(p_ensemble, axis=1)

# 評価
ensemble_acc = accuracy_score(y_test, y_pred)
ensemble_f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\\nアンサンブル Accuracy: {ensemble_acc:.4f}")
print(f"アンサンブル F1 Score (weighted): {ensemble_f1:.4f}")

# 比較テーブル
print(f"\\n=== 全モデル比較 ===")
all_results = {
    'XGBoost (Optimized)': {'Accuracy': xgb_acc, 'F1': xgb_f1},
    'LightGBM (Optimized)': {'Accuracy': lgb_acc, 'F1': lgb_f1},
    'RandomForest (Optimized)': {'Accuracy': rf_acc, 'F1': rf_f1},
    '★ Ensemble': {'Accuracy': ensemble_acc, 'F1': ensemble_f1}
}
all_results_df = pd.DataFrame(all_results).T
print(all_results_df.round(4))

print(f"\\n--- ベースライン比較（同一CSV）---")
print(f"asao_1321 (投手特徴量なし): Accuracy 0.904, F1 0.903")
print(f"本モデル (投手特徴量あり):  Accuracy {ensemble_acc:.4f}, F1 {ensemble_f1:.4f}")
print(f"改善幅: Accuracy {ensemble_acc - 0.904:+.4f}, F1 {ensemble_f1 - 0.903:+.4f}")
"""

# ========== SECTION 7: Classification Report ==========
md_section7 = """## 7. 分類レポート・混同行列"""

code_classification_report = """# 分類レポート
print("=== 分類レポート ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# キーメトリクス
report = classification_report(y_test, y_pred, output_dict=True, target_names=le.classes_)
print("\\n=== 重要指標 (同一CSVのasao_1321と比較) ===")
print(f"FC Recall: {report['FC']['recall']:.4f} (asao_1321: 0.75, 改善: {report['FC']['recall']-0.75:+.2f})")
print(f"SI Recall: {report['SI']['recall']:.4f} (asao_1321: 0.93, 改善: {report['SI']['recall']-0.93:+.2f})")
print(f"SL Recall: {report['SL']['recall']:.4f}")
print(f"FF Recall: {report['FF']['recall']:.4f}")
"""

code_confusion_matrix = """# 混同行列
cm = confusion_matrix(y_test, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# 件数ベース
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
axes[0].set_title('混同行列（件数）', fontsize=14)
axes[0].set_xlabel('予測ラベル')
axes[0].set_ylabel('真のラベル')

# 正規化（行方向=Recall）
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[1])
axes[1].set_title('混同行列（正規化: Recall）', fontsize=14)
axes[1].set_xlabel('予測ラベル')
axes[1].set_ylabel('真のラベル')

plt.tight_layout()
plt.show()
"""

# ========== SECTION 8: Feature Importance ==========
md_section8 = """## 8. 特徴量重要度"""

code_feature_importance = """# 3モデルの特徴量重要度
importance_df = pd.DataFrame({
    'feature': available_features,
    'xgb_importance': xgb_model.feature_importances_,
    'lgb_importance': lgb_model.feature_importances_,
    'rf_importance': rf_model.feature_importances_
})

# 平均重要度
importance_df['avg_importance'] = (
    importance_df['xgb_importance'] + 
    importance_df['lgb_importance'] + 
    importance_df['rf_importance']
) / 3

importance_df = importance_df.sort_values('avg_importance', ascending=False)

print("=== 特徴量重要度 (Top 15) ===")
print(importance_df[['feature', 'xgb_importance', 'lgb_importance', 'rf_importance', 'avg_importance']].head(15).to_string(index=False))

# 可視化
fig, ax = plt.subplots(figsize=(12, 10))
top15 = importance_df.head(15)
colors = ['gold' if f in pitcher_relative_features else 'steelblue' for f in top15['feature']]
ax.barh(top15['feature'], top15['avg_importance'], color=colors)
ax.set_xlabel('平均重要度')
ax.set_title('特徴量重要度 (Top 15)\\n★=投手相対特徴量', fontsize=14)
ax.invert_yaxis()

# 凡例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='gold', label='投手相対特徴量'),
                   Patch(facecolor='steelblue', label='その他')]
ax.legend(handles=legend_elements, loc='lower right')
plt.tight_layout()
plt.show()

# 投手相対特徴量の重要度
print("\\n=== 投手相対特徴量の重要度 ===")
pitcher_imp = importance_df[importance_df['feature'].isin(pitcher_relative_features)]
print(pitcher_imp[['feature', 'avg_importance']].to_string(index=False))
total_pitcher_imp = pitcher_imp['avg_importance'].sum()
print(f"\\n投手相対特徴量の合計重要度: {total_pitcher_imp:.4f} ({total_pitcher_imp*100:.1f}%)")
"""

# ========== SECTION 9: SHAP Analysis ==========
md_section9 = """## 9. SHAP分析 (説明可能性)"""

code_shap = """if SHAP_AVAILABLE:
    print("=== SHAP分析 ===")
    
    # サンプリング（計算時間短縮）
    sample_size = min(2000, len(X_test))
    X_sample = X_test.sample(n=sample_size, random_state=42)
    
    # TreeExplainer
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)
    
    # Summary Plot (Bar)
    print("\\n1. SHAP Summary Plot (Bar)...")
    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=15, show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.show()
    
    # Beeswarm Plot (特定クラス)
    print("\\n2. SHAP Beeswarm Plot (FC class)...")
    fc_idx = list(le.classes_).index('FC')
    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values[fc_idx], X_sample, max_display=15, show=False)
    plt.title("SHAP Values for FC (Cutter)")
    plt.tight_layout()
    plt.show()
else:
    print("SHAP is not available. Skipping SHAP analysis.")
"""

# ========== SECTION 10: Error Analysis ==========
md_section10 = """## 10. 誤分類分析"""

code_error_analysis = """# 予測結果をDataFrameに追加
test_results = test_data.copy()
test_results['true_label'] = le.inverse_transform(y_test)
test_results['pred_label'] = le.inverse_transform(y_pred)
test_results['correct'] = test_results['true_label'] == test_results['pred_label']

# 誤分類率
error_rate = 1 - test_results['correct'].mean()
print(f"全体誤分類率: {error_rate:.4f} ({error_rate*100:.2f}%)")

# 球種別誤分類率
print("\\n=== 球種別誤分類率 ===")
error_by_pitch = test_results.groupby('true_label')['correct'].apply(lambda x: 1 - x.mean())
error_by_pitch = error_by_pitch.sort_values(ascending=False)
for pitch, err in error_by_pitch.items():
    print(f"  {pitch}: {err:.4f} ({err*100:.1f}%)")

# 可視化
fig, ax = plt.subplots(figsize=(12, 6))
error_by_pitch.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
ax.set_title('球種別誤分類率', fontsize=14)
ax.set_xlabel('球種')
ax.set_ylabel('誤分類率')
ax.axhline(y=error_rate, color='red', linestyle='--', label=f'全体平均: {error_rate:.2%}')
ax.legend()
plt.tight_layout()
plt.show()
"""

code_fc_sl_analysis = """# FC vs SL の詳細分析
print("\\n=== FC vs SL 誤分類分析 ===")
fc_sl_data = test_results[test_results['true_label'].isin(['FC', 'SL'])]

# 誤分類件数
fc_to_sl = fc_sl_data[(fc_sl_data['true_label'] == 'FC') & (fc_sl_data['pred_label'] == 'SL')]
sl_to_fc = fc_sl_data[(fc_sl_data['true_label'] == 'SL') & (fc_sl_data['pred_label'] == 'FC')]
fc_total = len(fc_sl_data[fc_sl_data['true_label'] == 'FC'])
sl_total = len(fc_sl_data[fc_sl_data['true_label'] == 'SL'])

print(f"FC → SL 誤分類: {len(fc_to_sl):,}件 / {fc_total:,}件 ({len(fc_to_sl)/fc_total*100:.1f}%)")
print(f"SL → FC 誤分類: {len(sl_to_fc):,}件 / {sl_total:,}件 ({len(sl_to_fc)/sl_total*100:.1f}%)")

# 特徴量分布比較
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fc_correct = fc_sl_data[(fc_sl_data['true_label'] == 'FC') & (fc_sl_data['correct'])]
sl_correct = fc_sl_data[(fc_sl_data['true_label'] == 'SL') & (fc_sl_data['correct'])]

features_for_analysis = ['release_speed', 'pfx_x', 'speed_diff', 'horizontal_vertical_ratio']
for idx, feat in enumerate(features_for_analysis):
    ax = axes[idx // 2, idx % 2]
    ax.hist(fc_correct[feat], bins=30, alpha=0.5, label='FC (正解)', density=True)
    ax.hist(sl_correct[feat], bins=30, alpha=0.5, label='SL (正解)', density=True)
    ax.hist(fc_to_sl[feat], bins=30, alpha=0.7, label='FC→SL (誤分類)', density=True, histtype='step', linewidth=2)
    ax.set_title(f'{feat}')
    ax.legend()

plt.suptitle('FC/SL の特徴量分布比較', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
"""

# ========== SECTION 11: Summary ==========
md_section11 = """## 11. まとめ"""

code_summary = """print("="*70)
print("=== 最終結果サマリー ===")
print("="*70)
print(f"\\nモデル: アンサンブル (XGBoost + LightGBM + RandomForest)")
print(f"データ: train_pitcher_v2.csv / test_pitcher_v2.csv")
print(f"特徴量: {len(available_features)}個 (投手相対特徴量4個を含む)")

print(f"\\n--- パフォーマンス ---")
print(f"Accuracy:  {ensemble_acc:.4f}")
print(f"F1 Score:  {ensemble_f1:.4f}")

print(f"\\n--- ベースライン比較（同一CSV: train_pitcher_v2.csv）---")
print(f"{'モデル':<30} {'Accuracy':<12} {'FC Recall':<12}")
print("-"*55)
print(f"{'asao_1321 (投手特徴量なし)':<30} {'0.904':<12} {'0.75':<12}")
print(f"{'本モデル (投手特徴量あり)':<30} {ensemble_acc:<12.4f} {report['FC']['recall']:<12.4f}")
print(f"{'改善幅':<30} {ensemble_acc - 0.904:<+12.4f} {report['FC']['recall']-0.75:<+12.4f}")

print(f"\\n--- 投手相対特徴量の効果 ---")
print(f"  speed_diff (球速差): 重要度 {importance_df[importance_df['feature']=='speed_diff']['avg_importance'].values[0]:.4f}")
print(f"  投手相対特徴量合計: {total_pitcher_imp:.4f} ({total_pitcher_imp*100:.1f}%)")

print(f"\\n--- 球種別Recall ---")
for pitch in ['FC', 'SL', 'SI', 'FF', 'CH']:
    if pitch in report:
        print(f"  {pitch}: {report[pitch]['recall']:.4f}")
print("="*70)
"""

# ========== Build Notebook ==========
nb["cells"] = [
    nbf.v4.new_markdown_cell(md_title),
    nbf.v4.new_markdown_cell(md_section0),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(md_section1),
    nbf.v4.new_code_cell(code_load_data),
    nbf.v4.new_code_cell(code_data_overview),
    nbf.v4.new_code_cell(code_pitch_distribution),
    nbf.v4.new_markdown_cell(md_section1_5),
    nbf.v4.new_code_cell(code_eda_stats),
    nbf.v4.new_code_cell(code_correlation),
    nbf.v4.new_code_cell(code_boxplot),
    nbf.v4.new_markdown_cell(md_section2),
    nbf.v4.new_code_cell(code_features),
    nbf.v4.new_code_cell(code_pitcher_feature_dist),
    nbf.v4.new_markdown_cell(md_section3),
    nbf.v4.new_code_cell(code_prepare),
    nbf.v4.new_markdown_cell(md_section4),
    nbf.v4.new_code_cell(code_baseline),
    nbf.v4.new_markdown_cell(md_section5),
    nbf.v4.new_code_cell(code_gridsearch),
    nbf.v4.new_code_cell(code_train_optimized),
    nbf.v4.new_code_cell(code_model_comparison),
    nbf.v4.new_markdown_cell(md_section6),
    nbf.v4.new_code_cell(code_ensemble),
    nbf.v4.new_markdown_cell(md_section7),
    nbf.v4.new_code_cell(code_classification_report),
    nbf.v4.new_code_cell(code_confusion_matrix),
    nbf.v4.new_markdown_cell(md_section8),
    nbf.v4.new_code_cell(code_feature_importance),
    nbf.v4.new_markdown_cell(md_section9),
    nbf.v4.new_code_cell(code_shap),
    nbf.v4.new_markdown_cell(md_section10),
    nbf.v4.new_code_cell(code_error_analysis),
    nbf.v4.new_code_cell(code_fc_sl_analysis),
    nbf.v4.new_markdown_cell(md_section11),
    nbf.v4.new_code_cell(code_summary),
]

with open("asao_1322_final_model.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook 'asao_1322_final_model.ipynb' created successfully!")
print(f"Total cells: {len(nb['cells'])}")
print("\\nSections:")
print("  0. ライブラリ読み込み")
print("  1. データ読み込み・概観")
print("  1.5. 探索的データ分析 (EDA) ★NEW")
print("  2. 特徴量確認")
print("  3. データ準備")
print("  4. ベースラインモデル")
print("  5. GridSearch最適化")
print("  6. アンサンブルモデル")
print("  7. 分類レポート・混同行列 (正規化版追加)")
print("  8. 特徴量重要度 (3モデル比較)")
print("  9. SHAP分析 (Beeswarm追加)")
print("  10. 誤分類分析 (FC/SL可視化追加)")
print("  11. まとめ")
