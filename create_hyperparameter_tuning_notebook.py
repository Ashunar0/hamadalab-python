import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# 投球分類モデル: ハイパーパラメータ最適化 (Optuna)
**作成日**: 2026/01/13
**目的**: Optuna (ベイズ最適化) を使用して、XGBoost, LightGBM, RandomForest の各モデルのハイパーパラメータを最適化し、アンサンブル精度を向上させる。
**ベースライン**: `asao_1313` (Accuracy 92.4%, F1 92.3%)
"""

code_imports = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgbm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
"""

code_load_data = """# データ読み込みと特徴量エンジニアリング
df = pd.read_csv('train_with_features.csv')

# Feature Engineering (Phase 1)
df['velocity_times_pfx_z'] = df['release_speed'] * df['pfx_z']
df['spin_per_mph'] = df['release_spin_rate'] / df['release_speed']
df['horizontal_vertical_ratio'] = df['pfx_x'] / (df['pfx_z'].abs() + 0.1)
df['speed_spin_ratio'] = df['release_speed'] / (df['release_spin_rate'] + 1)

features = [
    'release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z',
    'release_pos_x', 'release_pos_z', 'movement_magnitude', 
    'velocity_times_pfx_z', 'spin_per_mph', 'normalized_spin_axis',
    'speed_spin_ratio', 'horizontal_vertical_ratio',
    'spin_efficiency', 'abs_horizontal_movement', 'movement_angle'
]
target = 'pitch_type'

existing_features = [f for f in features if f in df.columns]
df_clean = df.dropna(subset=existing_features + [target]).copy()

le = LabelEncoder()
y = le.fit_transform(df_clean[target])
X = df_clean[existing_features]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train shape: {X_train.shape}, Valid shape: {X_valid.shape}")
"""

code_optuna_xgb = """# === Optuna Optimization: XGBoost ===
print("Optimizing XGBoost...")

def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'n_jobs': -1,
        'random_state': 42,
        'objective': 'multi:softprob'
    }
    
    model = xgb.XGBClassifier(**params)
    # 3-fold CV for robustness
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_weighted', n_jobs=-1)
    return scores.mean()

study_xgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=True)

print(f"Best XGBoost F1: {study_xgb.best_value:.4f}")
print(f"Best Params: {study_xgb.best_params}")

# Train final model with best params
best_xgb = xgb.XGBClassifier(**study_xgb.best_params, n_jobs=-1, random_state=42, objective='multi:softprob')
best_xgb.fit(X_train, y_train)
"""

code_optuna_lgbm = """# === Optuna Optimization: LightGBM ===
print("Optimizing LightGBM...")

def objective_lgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    }
    
    model = lgbm.LGBMClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_weighted', n_jobs=-1)
    return scores.mean()

study_lgbm = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study_lgbm.optimize(objective_lgbm, n_trials=50, show_progress_bar=True)

print(f"Best LightGBM F1: {study_lgbm.best_value:.4f}")
print(f"Best Params: {study_lgbm.best_params}")

best_lgbm = lgbm.LGBMClassifier(**study_lgbm.best_params, n_jobs=-1, random_state=42, verbose=-1)
best_lgbm.fit(X_train, y_train)
"""

code_optuna_rf = """# === Optuna Optimization: RandomForest ===
print("Optimizing RandomForest...")

def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'n_jobs': -1,
        'random_state': 42
    }
    
    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_weighted', n_jobs=-1)
    return scores.mean()

study_rf = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study_rf.optimize(objective_rf, n_trials=50, show_progress_bar=True)

print(f"Best RandomForest F1: {study_rf.best_value:.4f}")
print(f"Best Params: {study_rf.best_params}")

best_rf = RandomForestClassifier(**study_rf.best_params, n_jobs=-1, random_state=42)
best_rf.fit(X_train, y_train)
"""

code_ensemble = """# === Optimized Ensemble ===
print("Creating Optimized Ensemble...")

p_xgb = best_xgb.predict_proba(X_valid)
p_lgbm = best_lgbm.predict_proba(X_valid)
p_rf = best_rf.predict_proba(X_valid)

# Simple average (could also optimize weights, but adds complexity)
p_ensemble = (p_xgb + p_lgbm + p_rf) / 3.0
y_pred = np.argmax(p_ensemble, axis=1)

acc = accuracy_score(y_valid, y_pred)
f1 = f1_score(y_valid, y_pred, average='weighted')

print(f"\\n=== Optimized Ensemble Results ===")
print(f"Accuracy: {acc:.4f}")
print(f"Weighted F1: {f1:.4f}")
print(f"\\nBaseline (asao_1313): Acc 0.924, F1 0.923")
print(f"Improvement: Acc {acc - 0.924:+.4f}, F1 {f1 - 0.923:+.4f}")
"""

code_eval = """# === Detailed Evaluation ===
print(classification_report(y_valid, y_pred, target_names=le.classes_))

# FC Recall Check
fc_idx = le.transform(['FC'])[0]
fc_report = classification_report(y_valid, y_pred, output_dict=True, target_names=le.classes_)
print(f"\\nFC Recall: {fc_report['FC']['recall']:.4f} (Baseline: 0.76)")

# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_valid, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Optimized Ensemble Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
"""

nb["cells"] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_load_data),
    nbf.v4.new_code_cell(code_optuna_xgb),
    nbf.v4.new_code_cell(code_optuna_lgbm),
    nbf.v4.new_code_cell(code_optuna_rf),
    nbf.v4.new_code_cell(code_ensemble),
    nbf.v4.new_code_cell(code_eval),
]

with open("asao_1317_hyperparameter_tuning.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook 'asao_1317_hyperparameter_tuning.ipynb' created successfully.")
