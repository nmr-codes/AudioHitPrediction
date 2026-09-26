import time
import json
import warnings
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
# pyrefly: ignore [missing-import]
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import optuna

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Minimalist Matplotlib Theme Configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#EAEAEA'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

print("[1/6] Loading and Preprocessing Dataset...")
df = pd.read_csv('spotify_songs.csv')
df = df.dropna(subset=['track_name', 'track_artist']).reset_index(drop=True)

# Feature Engineering
df['release_year'] = pd.to_datetime(df['track_album_release_date'], errors='coerce').dt.year
median_year = df['release_year'].median()
df['release_year'] = df['release_year'].fillna(median_year).astype(int)
df['duration_min'] = df['duration_ms'] / 60000.0

# Target definition: Binary Hit Indicator (Popularity >= 50)
df['is_popular'] = (df['track_popularity'] >= 50).astype(int)

features = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 
    'speechiness', 'acousticness', 'instrumentalness', 
    'liveness', 'valence', 'tempo', 'duration_min', 
    'release_year', 'playlist_genre'
]

X = df[features].copy()
y = df['is_popular'].copy()

# Categorical encoding for playlist_genre
le_genre = LabelEncoder()
X['playlist_genre'] = le_genre.fit_transform(X['playlist_genre'])

# Stratified Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Dataset Size: {len(df)} samples | Features: {len(features)}")
print(f"Train Shape: {X_train.shape} | Test Shape: {X_test.shape}")
print(f"Class Balance: Train 1s={y_train.mean():.3f}, Test 1s={y_test.mean():.3f}")

# Storage for results
results = {}

# -------------------------------------------------------------
# 1. BASELINE LIGHTGBM MODEL
# -------------------------------------------------------------
print("\n[2/6] Training Baseline LightGBM Model...")
t0 = time.time()
lgbm_baseline = LGBMClassifier(random_state=42, verbose=-1)
lgbm_baseline.fit(X_train, y_train)
lgbm_base_time = time.time() - t0

train_preds_base = lgbm_baseline.predict(X_train)
test_preds_base = lgbm_baseline.predict(X_test)
train_proba_base = lgbm_baseline.predict_proba(X_train)[:, 1]
test_proba_base = lgbm_baseline.predict_proba(X_test)[:, 1]

results['baseline_lgbm'] = {
    'train_accuracy': float(accuracy_score(y_train, train_preds_base)),
    'test_accuracy': float(accuracy_score(y_test, test_preds_base)),
    'train_roc_auc': float(roc_auc_score(y_train, train_proba_base)),
    'test_roc_auc': float(roc_auc_score(y_test, test_proba_base)),
    'training_time_sec': round(lgbm_base_time, 4)
}

print(f"Baseline Train Acc: {results['baseline_lgbm']['train_accuracy']:.4f} | Test Acc: {results['baseline_lgbm']['test_accuracy']:.4f}")
print(f"Baseline Train AUC: {results['baseline_lgbm']['train_roc_auc']:.4f} | Test AUC: {results['baseline_lgbm']['test_roc_auc']:.4f}")
print(f"Training Time: {results['baseline_lgbm']['training_time_sec']:.4f}s")

# -------------------------------------------------------------
# 2. NUM_LEAVES TUNING EXPERIMENT (OVERFITTING ANALYSIS)
# -------------------------------------------------------------
print("\n[3/6] Running num_leaves Overfitting Experiment...")
num_leaves_list = [4, 8, 16, 31, 64, 128, 256, 512]
num_leaves_results = []

for nl in num_leaves_list:
    clf = LGBMClassifier(num_leaves=nl, random_state=42, verbose=-1)
    clf.fit(X_train, y_train)
    
    tr_acc = accuracy_score(y_train, clf.predict(X_train))
    te_acc = accuracy_score(y_test, clf.predict(X_test))
    tr_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
    te_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    
    num_leaves_results.append({
        'num_leaves': nl,
        'train_acc': tr_acc,
        'test_acc': te_acc,
        'train_auc': tr_auc,
        'test_auc': te_auc,
        'acc_gap': tr_acc - te_acc
    })
    print(f"num_leaves={nl:<3} | Train Acc: {tr_acc:.4f} | Test Acc: {te_acc:.4f} | Gap: {(tr_acc - te_acc):.4f}")

df_num_leaves = pd.DataFrame(num_leaves_results)
results['num_leaves_experiment'] = df_num_leaves.to_dict(orient='records')

# Minimalist Plot: num_leaves Overfitting Curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

ax1.plot(df_num_leaves['num_leaves'], df_num_leaves['train_acc'], marker='o', color='#1A365D', label='Train Accuracy', linewidth=2)
ax1.plot(df_num_leaves['num_leaves'], df_num_leaves['test_acc'], marker='s', color='#E53E3E', label='Test Accuracy', linewidth=2)
ax1.axvline(x=31, color='#718096', linestyle=':', label='Default (31)')
ax1.set_xscale('log', base=2)
ax1.set_title('Accuracy vs num_leaves (Overfitting Divergence)', fontsize=12, fontweight='bold', pad=12)
ax1.set_xlabel('num_leaves (log2 scale)', fontsize=10)
ax1.set_ylabel('Accuracy', fontsize=10)
ax1.grid(True)
ax1.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0')

ax2.fill_between(df_num_leaves['num_leaves'], 0, df_num_leaves['acc_gap'], color='#FEB2B2', alpha=0.5, label='Generalization Gap (Train - Test)')
ax2.plot(df_num_leaves['num_leaves'], df_num_leaves['acc_gap'], marker='d', color='#9B2C2C', linewidth=2)
ax2.set_xscale('log', base=2)
ax2.set_title('Generalization Penalty Gap', fontsize=12, fontweight='bold', pad=12)
ax2.set_xlabel('num_leaves (log2 scale)', fontsize=10)
ax2.set_ylabel('Overfitting Gap (Train - Test)', fontsize=10)
ax2.grid(True)
ax2.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0')

plt.tight_layout()
plt.savefig('assets/num_leaves_overfitting.png', bbox_inches='tight')
plt.close()
print("Saved assets/num_leaves_overfitting.png")

# -------------------------------------------------------------
# 3. OVERFITTING CONTROL: max_depth & min_child_samples
# -------------------------------------------------------------
print("\n[4/6] Grid Search: Overfitting Control (max_depth x min_child_samples)...")
depth_values = [3, 5, 7, 9, 12, -1]
child_values = [10, 20, 50, 100, 200]

grid_records = []
for md in depth_values:
    for mc in child_values:
        clf = LGBMClassifier(
            max_depth=md, 
            min_child_samples=mc, 
            random_state=42, 
            verbose=-1
        )
        clf.fit(X_train, y_train)
        tr_acc = accuracy_score(y_train, clf.predict(X_train))
        te_acc = accuracy_score(y_test, clf.predict(X_test))
        te_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        
        grid_records.append({
            'max_depth': str(md) if md != -1 else 'None',
            'min_child_samples': mc,
            'train_acc': tr_acc,
            'test_acc': te_acc,
            'test_auc': te_auc,
            'acc_gap': tr_acc - te_acc
        })

df_grid = pd.DataFrame(grid_records)
# Find best combination (highest test AUC with controlled gap)
best_grid_idx = df_grid['test_auc'].idxmax()
best_grid_row = df_grid.iloc[best_grid_idx]
results['best_overfitting_control'] = best_grid_row.to_dict()
print(f"Optimal Configuration: max_depth={best_grid_row['max_depth']}, min_child_samples={best_grid_row['min_child_samples']}")
print(f"Test Acc: {best_grid_row['test_acc']:.4f} | Test AUC: {best_grid_row['test_auc']:.4f} | Gap: {best_grid_row['acc_gap']:.4f}")

# Plot Heatmap
pivot_auc = df_grid.pivot(index='max_depth', columns='min_child_samples', values='test_auc')
pivot_gap = df_grid.pivot(index='max_depth', columns='min_child_samples', values='acc_gap')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
sns.heatmap(pivot_auc, annot=True, fmt='.4f', cmap='Blues', ax=ax1, cbar=False, linewidths=0.5)
ax1.set_title('Test ROC-AUC Heatmap', fontsize=12, fontweight='bold', pad=12)
ax1.set_xlabel('min_child_samples', fontsize=10)
ax1.set_ylabel('max_depth', fontsize=10)

sns.heatmap(pivot_gap, annot=True, fmt='.4f', cmap='Reds', ax=ax2, cbar=False, linewidths=0.5)
ax2.set_title('Overfitting Gap (Train Acc - Test Acc)', fontsize=12, fontweight='bold', pad=12)
ax2.set_xlabel('min_child_samples', fontsize=10)
ax2.set_ylabel('max_depth', fontsize=10)

plt.tight_layout()
plt.savefig('assets/overfitting_control_heatmap.png', bbox_inches='tight')
plt.close()
print("Saved assets/overfitting_control_heatmap.png")

# -------------------------------------------------------------
# 4. XGBOOST COMPARISON (METRICS + TRAINING TIME)
# -------------------------------------------------------------
print("\n[5/6] Training XGBoost Classifier for Benchmark Comparison...")
t0 = time.time()
xgb_model = XGBClassifier(
    n_estimators=100,
    random_state=42,
    eval_metric='logloss',
    tree_method='hist'
)
xgb_model.fit(X_train, y_train)
xgb_time = time.time() - t0

train_preds_xgb = xgb_model.predict(X_train)
test_preds_xgb = xgb_model.predict(X_test)
train_proba_xgb = xgb_model.predict_proba(X_train)[:, 1]
test_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

results['xgboost'] = {
    'train_accuracy': float(accuracy_score(y_train, train_preds_xgb)),
    'test_accuracy': float(accuracy_score(y_test, test_preds_xgb)),
    'train_roc_auc': float(roc_auc_score(y_train, train_proba_xgb)),
    'test_roc_auc': float(roc_auc_score(y_test, test_proba_xgb)),
    'test_f1': float(f1_score(y_test, test_preds_xgb)),
    'training_time_sec': round(xgb_time, 4)
}

print(f"XGBoost Train Acc: {results['xgboost']['train_accuracy']:.4f} | Test Acc: {results['xgboost']['test_accuracy']:.4f}")
print(f"XGBoost Train AUC: {results['xgboost']['train_roc_auc']:.4f} | Test AUC: {results['xgboost']['test_roc_auc']:.4f}")
print(f"XGBoost Training Time: {results['xgboost']['training_time_sec']:.4f}s")

# Side-by-Side Plot: LightGBM vs XGBoost
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

models_label = ['LightGBM (Baseline)', 'XGBoost (Hist)']
test_accs = [results['baseline_lgbm']['test_accuracy'], results['xgboost']['test_accuracy']]
test_aucs = [results['baseline_lgbm']['test_roc_auc'], results['xgboost']['test_roc_auc']]
times = [results['baseline_lgbm']['training_time_sec'], results['xgboost']['training_time_sec']]

x = np.arange(len(models_label))
width = 0.35

ax1.bar(x - width/2, test_accs, width, label='Test Accuracy', color='#2B6CB0')
ax1.bar(x + width/2, test_aucs, width, label='Test ROC-AUC', color='#4FD1C5')
ax1.set_ylim(0.65, 0.85)
ax1.set_ylabel('Score', fontsize=10)
ax1.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold', pad=12)
ax1.set_xticks(x)
ax1.set_xticklabels(models_label, fontsize=10)
ax1.grid(axis='y')
ax1.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0')

bars = ax2.bar(models_label, times, color=['#319795', '#DD6B20'], width=0.45)
ax2.set_ylabel('Training Time (seconds)', fontsize=10)
ax2.set_title('Training Speed (Lower is Faster)', fontsize=12, fontweight='bold', pad=12)
ax2.grid(axis='y')
for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('assets/lgbm_vs_xgboost.png', bbox_inches='tight')
plt.close()
print("Saved assets/lgbm_vs_xgboost.png")

# -------------------------------------------------------------
# 5. CHALLENGE: OPTUNA AUTOMATED HYPERPARAMETER TUNING
# -------------------------------------------------------------
print("\n[6/6] Executing Optuna Automated Optimization (30 Trials)...")

def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 150),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'n_estimators': 150,
        'random_state': 42,
        'verbose': -1
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        clf = LGBMClassifier(**params)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, proba))
        
    return np.mean(scores)

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30, show_progress_bar=False)

best_params = study.best_params
best_params.update({'n_estimators': 150, 'random_state': 42, 'verbose': -1})
print(f"Optuna Best CV ROC-AUC: {study.best_value:.4f}")
print("Optuna Best Parameters:", json.dumps(best_params, indent=2))

# Train final Optuna-tuned model
t0 = time.time()
lgbm_optuna = LGBMClassifier(**best_params)
lgbm_optuna.fit(X_train, y_train)
optuna_train_time = time.time() - t0

train_preds_opt = lgbm_optuna.predict(X_train)
test_preds_opt = lgbm_optuna.predict(X_test)
train_proba_opt = lgbm_optuna.predict_proba(X_train)[:, 1]
test_proba_opt = lgbm_optuna.predict_proba(X_test)[:, 1]

results['optuna_tuned_lgbm'] = {
    'best_params': best_params,
    'train_accuracy': float(accuracy_score(y_train, train_preds_opt)),
    'test_accuracy': float(accuracy_score(y_test, test_preds_opt)),
    'train_roc_auc': float(roc_auc_score(y_train, train_proba_opt)),
    'test_roc_auc': float(roc_auc_score(y_test, test_proba_opt)),
    'test_f1': float(f1_score(y_test, test_preds_opt)),
    'generalization_gap': float(accuracy_score(y_train, train_preds_opt) - accuracy_score(y_test, test_preds_opt)),
    'training_time_sec': round(optuna_train_time, 4)
}

# Plot Optuna Optimization History
trial_values = [t.value for t in study.trials if t.value is not None]
running_max = np.maximum.accumulate(trial_values)

plt.figure(figsize=(9, 4.5), dpi=300)
plt.scatter(range(1, len(trial_values) + 1), trial_values, color='#4A5568', alpha=0.6, label='Trial CV AUC', s=35)
plt.plot(range(1, len(running_max) + 1), running_max, color='#3182CE', linewidth=2.2, label='Best Cumulative AUC')
plt.title('Optuna Bayesian Hyperparameter Optimization Progression', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Trial Number', fontsize=10)
plt.ylabel('Validation ROC-AUC', fontsize=10)
plt.grid(True)
plt.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0')
plt.tight_layout()
plt.savefig('assets/optuna_optimization_history.png', bbox_inches='tight')
plt.close()
print("Saved assets/optuna_optimization_history.png")

# Save Models and Artifacts
joblib.dump(lgbm_optuna, 'models/lgbm_best.joblib')
joblib.dump(xgb_model, 'models/xgb_model.joblib')
joblib.dump({
    'features': features,
    'label_encoder': le_genre,
    'feature_medians': X.median().to_dict()
}, 'models/pipeline_meta.joblib')

# Feature Importances Plot
feature_imp = pd.Series(lgbm_optuna.feature_importances_, index=features).sort_values(ascending=True)
plt.figure(figsize=(10, 6), dpi=300)
plt.barh(feature_imp.index, feature_imp.values, color='#2B6CB0', height=0.65)
plt.title('Feature Importances (Optuna-Tuned LightGBM)', fontsize=12, fontweight='bold', pad=12)
plt.xlabel('Split Importance Count', fontsize=10)
plt.grid(axis='x')
plt.tight_layout()
plt.savefig('assets/feature_importances.png', bbox_inches='tight')
plt.close()
print("Saved assets/feature_importances.png")

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nALL PIPELINE EXPERIMENTS SUCCESSFULLY COMPLETED!")
