"""
Life Expectancy Prediction and Analysis
MA5755 – Data Analysis & Visualization

This script converts the analysis workflow from the accompanying notebook
into a reproducible Python script.

Dataset: WHO Life Expectancy dataset
Workflow:
    1. Data loading and cleaning
    2. Exploratory data analysis
    3. Preprocessing and standardization
    4. PCA
    5. Regression model comparison
    6. Lasso feature selection
    7. Random Forest feature importance
    8. Model diagnostics

Run with:
    python life_expectancy.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Plot style
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.bbox'] = 'tight'

SEED = 42
np.random.seed(SEED)

print('All imports successful.')

# Load dataset — place 'Life Expectancy Data.csv' in the same folder as this notebook
df = pd.read_csv('Life Expectancy Data.csv')

# Standardize column names: strip whitespace, replace spaces with underscores
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('/', '_')

print('Shape:', df.shape)
print('\nColumn names:')
print(df.columns.tolist())

df.head()

df.describe().T.round(2)

# Data types and dtypes
df.info()

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
print(missing_df)

# Visualize missing values
fig, ax = plt.subplots(figsize=(10, 5))
missing_df['Missing %'].plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
ax.set_xlabel('Missing Data (%)')
ax.set_title('Missing Values by Feature', fontweight='bold')
ax.axvline(x=10, color='red', linestyle='--', alpha=0.6, label='10% threshold')
ax.legend()
plt.tight_layout()
plt.savefig('fig_missing_values.png')
plt.show()

# Drop rows where the response variable (Life_expectancy) is missing
df = df.dropna(subset=['Life_expectancy'])
print(f'Rows after dropping missing response: {len(df)}')

# Encode Status (Developed=1, Developing=0)
df['Status'] = df['Status'].map({'Developed': 1, 'Developing': 0})

# Drop Country and Year for modeling (keep for EDA)
df_eda = df.copy()

# Select numeric features for modeling
drop_cols = ['Country', 'Year', 'Life_expectancy']
feature_cols = [c for c in df.columns if c not in drop_cols]

X_raw = df[feature_cols]
y = df['Life_expectancy']

print(f'Features: {len(feature_cols)}')
print(f'Response range: {y.min():.1f} — {y.max():.1f} years')

# Impute missing values with column median (robust to outliers)
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=feature_cols)

# Standardize features (required for Ridge, Lasso, PCA)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=feature_cols)

print('Preprocessing complete. No missing values remaining:', X_scaled.isnull().sum().sum())

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Overall distribution
axes[0].hist(y, bins=35, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(y.mean(), color='red', linestyle='--', label=f'Mean = {y.mean():.1f}')
axes[0].axvline(y.median(), color='orange', linestyle='--', label=f'Median = {y.median():.1f}')
axes[0].set_xlabel('Life Expectancy (years)')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Life Expectancy', fontweight='bold')
axes[0].legend()

# By development status
developed = df_eda[df_eda['Status'] == 1]['Life_expectancy'].dropna()
developing = df_eda[df_eda['Status'] == 0]['Life_expectancy'].dropna()
axes[1].hist(developed, bins=25, color='#2ecc71', alpha=0.7, label=f'Developed (n={len(developed)})', edgecolor='white')
axes[1].hist(developing, bins=25, color='#e74c3c', alpha=0.7, label=f'Developing (n={len(developing)})', edgecolor='white')
axes[1].set_xlabel('Life Expectancy (years)')
axes[1].set_ylabel('Count')
axes[1].set_title('Life Expectancy by Development Status', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('fig_life_expectancy_dist.png')
plt.show()

print(f"Developed countries — Mean: {developed.mean():.1f}, Std: {developed.std():.1f}")
print(f"Developing countries — Mean: {developing.mean():.1f}, Std: {developing.std():.1f}")

trend = df_eda.groupby(['Year', 'Status'])['Life_expectancy'].mean().reset_index()
trend['Status_label'] = trend['Status'].map({1: 'Developed', 0: 'Developing'})

fig, ax = plt.subplots(figsize=(10, 4))
for label, color in [('Developed', '#2ecc71'), ('Developing', '#e74c3c')]:
    subset = trend[trend['Status_label'] == label]
    ax.plot(subset['Year'], subset['Life_expectancy'], marker='o', label=label, color=color, linewidth=2)

ax.set_xlabel('Year')
ax.set_ylabel('Mean Life Expectancy (years)')
ax.set_title('Global Life Expectancy Trend (2000–2015)', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('fig_trend_over_time.png')
plt.show()

# Correlation with response
corr_with_y = df_eda[feature_cols + ['Life_expectancy']].corr()['Life_expectancy'].drop('Life_expectancy').sort_values()

fig, ax = plt.subplots(figsize=(8, 7))
colors = ['#e74c3c' if v < 0 else '#2980b9' for v in corr_with_y]
corr_with_y.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Pearson Correlation with Life Expectancy')
ax.set_title('Feature Correlations with Life Expectancy', fontweight='bold')
plt.tight_layout()
plt.savefig('fig_correlations.png')
plt.show()

# Full correlation heatmap among all variables
corr_matrix = X_imputed.corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
            linewidths=0.3, ax=ax, annot=False, vmin=-1, vmax=1,
            cbar_kws={'label': 'Pearson r'})
ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('fig_heatmap.png')
plt.show()

key_predictors = ['Schooling', 'GDP', 'HIV_AIDS', 'Adult_Mortality', 'BMI', 'Income_composition_of_resources']
key_predictors = [c for c in key_predictors if c in df_eda.columns]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, col in enumerate(key_predictors):
    axes[i].scatter(df_eda[col], df_eda['Life_expectancy'],
                    alpha=0.25, s=12, color='steelblue')
    axes[i].set_xlabel(col.replace('_', ' '))
    axes[i].set_ylabel('Life Expectancy')
    axes[i].set_title(f'Life Expectancy vs {col.replace("_", " ")}', fontweight='bold')

plt.tight_layout()
plt.savefig('fig_scatter_plots.png')
plt.show()

pca = PCA(random_state=SEED)
pca.fit(X_scaled)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Scree plot
axes[0].bar(range(1, len(explained)+1), explained, color='steelblue', alpha=0.8, edgecolor='white')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Variance Explained')
axes[0].set_title('Scree Plot', fontweight='bold')
axes[0].set_xlim(0.5, 15)

# Cumulative variance
axes[1].plot(range(1, len(cumulative)+1), cumulative, marker='o', color='steelblue', linewidth=2)
axes[1].axhline(0.90, color='red', linestyle='--', label='90% threshold')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance Explained')
axes[1].set_title('Cumulative Explained Variance', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('fig_pca_variance.png')
plt.show()

n90 = np.argmax(cumulative >= 0.90) + 1
print(f'Components needed to explain 90% of variance: {n90}')

# PCA biplot — PC1 vs PC2 colored by development status
pca2 = PCA(n_components=2, random_state=SEED)
X_pca = pca2.fit_transform(X_scaled)

status_colors = df_eda['Status'].map({1: '#2ecc71', 0: '#e74c3c'}).values

fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=status_colors, alpha=0.5, s=18, edgecolors='none')
ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}% variance)')
ax.set_title('PCA — First Two Principal Components', fontweight='bold')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ecc71', label='Developed'),
                   Patch(facecolor='#e74c3c', label='Developing')]
ax.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig('fig_pca_biplot.png')
plt.show()

# 5-fold cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

def cv_metrics(model, X, y, cv):
    """Return mean RMSE, MAE, R² from cross-validation."""
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error'))
    mae_scores  = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    r2_scores   =  cross_val_score(model, X, y, cv=cv, scoring='r2')
    return {
        'RMSE': rmse_scores.mean(),
        'RMSE_std': rmse_scores.std(),
        'MAE': mae_scores.mean(),
        'R2': r2_scores.mean()
    }

X_np = X_scaled.values
y_np = y.values

print('Cross-validation framework ready.')

ols = LinearRegression()
ols_metrics = cv_metrics(ols, X_np, y_np, kf)
print('OLS — 5-fold CV results:')
print(f"  RMSE: {ols_metrics['RMSE']:.3f} ± {ols_metrics['RMSE_std']:.3f}")
print(f"  MAE:  {ols_metrics['MAE']:.3f}")
print(f"  R²:   {ols_metrics['R2']:.3f}")

alphas = np.logspace(-3, 4, 100)

ridge_cv = RidgeCV(alphas=alphas, cv=kf)
ridge_cv.fit(X_np, y_np)
best_alpha_ridge = ridge_cv.alpha_
print(f'Best Ridge alpha: {best_alpha_ridge:.4f}')

ridge = Ridge(alpha=best_alpha_ridge)
ridge_metrics = cv_metrics(ridge, X_np, y_np, kf)
print(f"  RMSE: {ridge_metrics['RMSE']:.3f} ± {ridge_metrics['RMSE_std']:.3f}")
print(f"  MAE:  {ridge_metrics['MAE']:.3f}")
print(f"  R²:   {ridge_metrics['R2']:.3f}")

lasso_cv = LassoCV(alphas=alphas, cv=kf, max_iter=10000, random_state=SEED)
lasso_cv.fit(X_np, y_np)
best_alpha_lasso = lasso_cv.alpha_
print(f'Best Lasso alpha: {best_alpha_lasso:.4f}')

lasso = Lasso(alpha=best_alpha_lasso, max_iter=10000)
lasso_metrics = cv_metrics(lasso, X_np, y_np, kf)
print(f"  RMSE: {lasso_metrics['RMSE']:.3f} ± {lasso_metrics['RMSE_std']:.3f}")
print(f"  MAE:  {lasso_metrics['MAE']:.3f}")
print(f"  R²:   {lasso_metrics['R2']:.3f}")

# Lasso feature selection — which coefficients survive?
lasso.fit(X_np, y_np)
lasso_coefs = pd.Series(lasso.coef_, index=feature_cols)
selected = lasso_coefs[lasso_coefs != 0].sort_values(key=abs, ascending=False)
zeroed = lasso_coefs[lasso_coefs == 0]

print(f'Features retained by Lasso: {len(selected)} / {len(feature_cols)}')
print(f'Features zeroed out:        {len(zeroed)}')
print('\nTop retained features:')
print(selected.head(10))

# Plot Lasso coefficients
fig, ax = plt.subplots(figsize=(9, 6))
colors = ['#2980b9' if v > 0 else '#e74c3c' for v in selected.values]
selected.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Lasso Coefficient')
ax.set_title(f'Lasso Selected Features (α = {best_alpha_lasso:.4f})', fontweight='bold')
plt.tight_layout()
plt.savefig('fig_lasso_coefs.png')
plt.show()

# Lasso regularization path
from sklearn.linear_model import lasso_path

alphas_path, coefs_path, _ = lasso_path(X_np, y_np, alphas=np.logspace(0, -3, 100))

fig, ax = plt.subplots(figsize=(10, 5))
for i in range(coefs_path.shape[0]):
    ax.plot(np.log10(alphas_path), coefs_path[i], linewidth=1, alpha=0.7)
ax.axvline(np.log10(best_alpha_lasso), color='red', linestyle='--', label=f'Best α = {best_alpha_lasso:.4f}')
ax.set_xlabel('log10(α)')
ax.set_ylabel('Coefficient')
ax.set_title('Lasso Regularization Path', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('fig_lasso_path.png')
plt.show()

rf = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=2,
                           n_jobs=-1, random_state=SEED)
rf_metrics = cv_metrics(rf, X_np, y_np, kf)
print('Random Forest — 5-fold CV results:')
print(f"  RMSE: {rf_metrics['RMSE']:.3f} ± {rf_metrics['RMSE_std']:.3f}")
print(f"  MAE:  {rf_metrics['MAE']:.3f}")
print(f"  R²:   {rf_metrics['R2']:.3f}")

# Feature importances from Random Forest
rf.fit(X_np, y_np)
rf_importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 6))
rf_importance.head(15).sort_values().plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
ax.set_xlabel('Feature Importance (Mean Decrease in Impurity)')
ax.set_title('Top 15 Features — Random Forest', fontweight='bold')
plt.tight_layout()
plt.savefig('fig_rf_importance.png')
plt.show()

gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                               max_depth=4, subsample=0.8, random_state=SEED)
gb_metrics = cv_metrics(gb, X_np, y_np, kf)
print('Gradient Boosting — 5-fold CV results:')
print(f"  RMSE: {gb_metrics['RMSE']:.3f} ± {gb_metrics['RMSE_std']:.3f}")
print(f"  MAE:  {gb_metrics['MAE']:.3f}")
print(f"  R²:   {gb_metrics['R2']:.3f}")

results = pd.DataFrame({
    'Model': ['OLS', 'Ridge', 'Lasso', 'Random Forest', 'Gradient Boosting'],
    'RMSE': [ols_metrics['RMSE'], ridge_metrics['RMSE'], lasso_metrics['RMSE'],
             rf_metrics['RMSE'], gb_metrics['RMSE']],
    'MAE': [ols_metrics['MAE'], ridge_metrics['MAE'], lasso_metrics['MAE'],
            rf_metrics['MAE'], gb_metrics['MAE']],
    'R2': [ols_metrics['R2'], ridge_metrics['R2'], lasso_metrics['R2'],
           rf_metrics['R2'], gb_metrics['R2']]
}).sort_values('RMSE')

results = results.set_index('Model')
results = results.round(4)
print(results.to_string())

# Visual comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics_to_plot = ['RMSE', 'MAE', 'R2']
colors = ['#2980b9', '#27ae60', '#8e44ad', '#e67e22', '#e74c3c']

for i, metric in enumerate(metrics_to_plot):
    vals = results[metric]
    bars = axes[i].bar(vals.index, vals.values, color=colors, edgecolor='white')
    axes[i].set_title(metric, fontweight='bold')
    axes[i].set_xticklabels(vals.index, rotation=20, ha='right')
    for bar, val in zip(bars, vals.values):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('5-Fold Cross-Validation Model Comparison', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('fig_model_comparison.png')
plt.show()

# Fit best model on full data and plot residuals
best_model = gb  # update if another model wins
best_model.fit(X_np, y_np)
y_pred = best_model.predict(X_np)
residuals = y_np - y_pred

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Residuals vs Fitted
axes[0].scatter(y_pred, residuals, alpha=0.3, s=12, color='steelblue')
axes[0].axhline(0, color='red', linestyle='--', linewidth=1)
axes[0].set_xlabel('Fitted Values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Fitted (Gradient Boosting)', fontweight='bold')

# Residual distribution
axes[1].hist(residuals, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
axes[1].axvline(0, color='red', linestyle='--')
axes[1].set_xlabel('Residual (years)')
axes[1].set_ylabel('Count')
axes[1].set_title('Residual Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig('fig_residuals.png')
plt.show()

print(f'Residual mean: {residuals.mean():.4f}')
print(f'Residual std:  {residuals.std():.4f}')

print('='*60)
print('SUMMARY OF RESULTS')
print('='*60)
print(f'\nDataset: {len(df)} observations, {len(feature_cols)} features')
print(f'Response: Life Expectancy (years), range {y.min():.1f}–{y.max():.1f}')
print(f'\nLasso feature selection: retained {len(selected)} of {len(feature_cols)} features')
print(f'Top 5 Lasso features: {list(selected.head(5).index)}')
print(f'\nTop 5 Random Forest features: {list(rf_importance.head(5).index)}')
print(f'\nModel Performance (5-fold CV):')
print(results[['RMSE', 'R2']].to_string())
print('='*60)
