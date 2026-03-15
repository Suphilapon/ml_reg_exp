"""
正则化对比脚本第二阶段：Lasso / Ridge / 无正则化 对比
- 数据生成、预处理、模型训练（均匀采样的 lambda）
- 原有 4 类图 + 新增 5 类图（全英文 + lambda 命名）
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ========== 1. 数据生成（固定种子 42）==========
np.random.seed(42)
m, n = 150, 50
beta_true = np.zeros(n)
beta_true[:5] = [1.5, -2.0, 0.8, 3.2, -1.7]

X = np.random.randn(m, n)
y = X @ beta_true + np.random.randn(m) * 0.5

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ========== 2. 数据预处理（仅对 X 标准化）==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 3. Lambda 列表 ==========
# MSE 曲线用均匀采样的 lambda
lambda_list = np.linspace(0.001, 10, 30)
# 系数路径用较密的 lambda（对数间隔，便于平滑曲线）
lambda_path = np.logspace(-3, 1.5, 50)

# ========== 4. 无正则化 ==========
lr = LinearRegression().fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
coef_lr = lr.coef_

# ========== 5. Lasso / Ridge 在 lambda_list 下训练，选最佳 lambda ==========
mse_lasso_list = []
coef_lasso_list = []
for lam in lambda_list:
    model = Lasso(alpha=lam).fit(X_train_scaled, y_train)
    mse_lasso_list.append(mean_squared_error(y_test, model.predict(X_test_scaled)))
    coef_lasso_list.append(model.coef_)

mse_ridge_list = []
coef_ridge_list = []
for lam in lambda_list:
    model = Ridge(alpha=lam).fit(X_train_scaled, y_train)
    mse_ridge_list.append(mean_squared_error(y_test, model.predict(X_test_scaled)))
    coef_ridge_list.append(model.coef_)

idx_best_lasso = np.argmin(mse_lasso_list)
idx_best_ridge = np.argmin(mse_ridge_list)
lambda_best_lasso = lambda_list[idx_best_lasso]
lambda_best_ridge = lambda_list[idx_best_ridge]

lasso_best = Lasso(alpha=lambda_best_lasso).fit(X_train_scaled, y_train)
ridge_best = Ridge(alpha=lambda_best_ridge).fit(X_train_scaled, y_train)
y_pred_lasso = lasso_best.predict(X_test_scaled)
y_pred_ridge = ridge_best.predict(X_test_scaled)
coef_lasso_best = lasso_best.coef_
coef_ridge_best = ridge_best.coef_

# 系数路径（密 lambda）
coef_lasso_path = []
coef_ridge_path = []
for lam in lambda_path:
    coef_lasso_path.append(Lasso(alpha=lam).fit(X_train_scaled, y_train).coef_)
    coef_ridge_path.append(Ridge(alpha=lam).fit(X_train_scaled, y_train).coef_)
coef_lasso_path = np.array(coef_lasso_path)  # (n_path, n)
coef_ridge_path = np.array(coef_ridge_path)

# 残差（用于图5、图8等）
resid_lr = y_test - y_pred_lr
resid_lasso = y_test - y_pred_lasso
resid_ridge = y_test - y_pred_ridge

n_test = len(y_test)
n_show = 10   # 前 10 个特征用于系数图
n_sample_bar = 20  # 图9 抽样样本数

# ========== 图1：MSE vs lambda（均匀采样）==========
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(lambda_list, mse_lasso_list, 'b-', label='Lasso', linewidth=2)
ax1.plot(lambda_list, mse_ridge_list, 'r-', label='Ridge', linewidth=2)
ax1.axhline(mse_lr, color='gray', linestyle='--', label='No regularization')
ax1.set_xlabel('lambda')
ax1.set_ylabel('Test MSE')
ax1.set_title('Test MSE vs lambda')
ax1.legend()
ax1.set_xscale('log')
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig1_mse_vs_lambda.png', dpi=150, bbox_inches='tight')
plt.close()

# ========== 图2：Lasso 系数路径（前 10 个特征）==========
fig2, ax2 = plt.subplots(figsize=(8, 5))
for j in range(n_show):
    ax2.plot(lambda_path, coef_lasso_path[:, j], label=f'Feature {j}')
ax2.set_xlabel('lambda')
ax2.set_ylabel('Coefficient')
ax2.set_title('Lasso coefficient path (first 10 features)')
ax2.legend()
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_lasso_coef_path.png', dpi=150, bbox_inches='tight')
plt.close()

# ========== 图3：Ridge 系数路径（前 10 个特征）==========
fig3, ax3 = plt.subplots(figsize=(8, 5))
for j in range(n_show):
    ax3.plot(lambda_path, coef_ridge_path[:, j], label=f'Feature {j}')
ax3.set_xlabel('lambda')
ax3.set_ylabel('Coefficient')
ax3.set_title('Ridge coefficient path (first 10 features)')
ax3.legend()
ax3.set_xscale('log')
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig3_ridge_coef_path.png', dpi=150, bbox_inches='tight')
plt.close()

# ========== 图4：最佳 lambda 下系数对比（前 10 个特征）==========
x4 = np.arange(n_show)
w = 0.2
fig4, ax4 = plt.subplots(figsize=(9, 5))
ax4.bar(x4 - w, coef_lr[:n_show], width=w, label='No reg')
ax4.bar(x4, coef_lasso_best[:n_show], width=w, label='Lasso')
ax4.bar(x4 + w, coef_ridge_best[:n_show], width=w, label='Ridge')
ax4.axhline(0, color='black', linewidth=0.5)
ax4.plot(x4, beta_true[:n_show], 'k*', markersize=10, label='True coefficients')
ax4.set_xlabel('Feature index')
ax4.set_ylabel('Coefficient')
ax4.set_title('Coefficient comparison at best lambda (first 10 features)')
ax4.set_xticks(x4)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('fig4_coef_compare.png', dpi=150, bbox_inches='tight')
plt.close()

# ========== 图5：残差分布箱线图 ==========
fig5, ax5 = plt.subplots(figsize=(6, 5))
bp = ax5.boxplot(
    [resid_lr, resid_lasso, resid_ridge],
    tick_labels=['No reg', 'Lasso', 'Ridge'],
    patch_artist=True
)
colors = ['lightblue', 'lightgreen', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax5.axhline(0, color='gray', linestyle='--', alpha=0.7)
ax5.set_ylabel('Residual (y_test - y_pred)')
ax5.set_title('Residual distribution (test set)')
ax5.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('fig5_residual_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()

# ========== 图6：预测值 vs 真实值散点图 ==========
fig6, ax6 = plt.subplots(figsize=(6, 6))
ax6.scatter(y_test, y_pred_lr, alpha=0.6, s=30, label='No reg', c='C0')
ax6.scatter(y_test, y_pred_lasso, alpha=0.6, s=30, label='Lasso', c='C1')
ax6.scatter(y_test, y_pred_ridge, alpha=0.6, s=30, label='Ridge', c='C2')
ymin = min(y_test.min(), y_pred_lr.min(), y_pred_lasso.min(), y_pred_ridge.min())
ymax = max(y_test.max(), y_pred_lr.max(), y_pred_lasso.max(), y_pred_ridge.max())
ax6.plot([ymin, ymax], [ymin, ymax], 'k--', label='y=x')
ax6.set_xlabel('True y')
ax6.set_ylabel('Predicted y')
ax6.set_title('Predicted vs True (test set)')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.set_aspect('equal')
plt.tight_layout()
plt.savefig('fig6_pred_vs_true.png', dpi=150, bbox_inches='tight')
plt.close()

# ========== 图7：误差热图（样本 × 模型）==========
err_matrix = np.column_stack([
    np.abs(resid_lr),
    np.abs(resid_lasso),
    np.abs(resid_ridge)
])
fig7, ax7 = plt.subplots(figsize=(5, 8))
im = ax7.imshow(err_matrix, aspect='auto', cmap='YlOrRd')
ax7.set_xticks([0, 1, 2])
ax7.set_xticklabels(['No reg', 'Lasso', 'Ridge'])
ax7.set_ylabel('Sample index')
ax7.set_xlabel('Model')
ax7.set_title('Absolute error (sample × model)')
plt.colorbar(im, ax=ax7, label='Absolute error')
plt.tight_layout()
plt.savefig('fig7_error_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# ========== 图8：累积误差分布图 ==========
def cumulative_error_proportion(abs_errors):
    sorted_err = np.sort(abs_errors)
    n = len(sorted_err)
    return sorted_err, np.arange(1, n + 1) / n

e_lr, p_lr = cumulative_error_proportion(np.abs(resid_lr))
e_lasso, p_lasso = cumulative_error_proportion(np.abs(resid_lasso))
e_ridge, p_ridge = cumulative_error_proportion(np.abs(resid_ridge))

fig8, ax8 = plt.subplots(figsize=(7, 5))
ax8.step(e_lr, p_lr, where='post', label='No reg')
ax8.step(e_lasso, p_lasso, where='post', label='Lasso')
ax8.step(e_ridge, p_ridge, where='post', label='Ridge')
ax8.set_xlabel('Absolute error')
ax8.set_ylabel('Cumulative proportion')
ax8.set_title('Cumulative error distribution (test set)')
ax8.legend()
ax8.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig8_cumulative_error.png', dpi=150, bbox_inches='tight')
plt.close()

# ========== 图9：抽样预测条形图（前 20 个样本）==========
n_bar = min(n_sample_bar, n_test)
x9 = np.arange(n_bar)
w9 = 0.2
y_true_bar = y_test[:n_bar]
y_pred_lr_bar = y_pred_lr[:n_bar]
y_pred_lasso_bar = y_pred_lasso[:n_bar]
y_pred_ridge_bar = y_pred_ridge[:n_bar]

fig9, ax9 = plt.subplots(figsize=(12, 5))
ax9.bar(x9 - 1.5 * w9, y_true_bar, width=w9, label='True')
ax9.bar(x9 - 0.5 * w9, y_pred_lr_bar, width=w9, label='No reg')
ax9.bar(x9 + 0.5 * w9, y_pred_lasso_bar, width=w9, label='Lasso')
ax9.bar(x9 + 1.5 * w9, y_pred_ridge_bar, width=w9, label='Ridge')
ax9.set_xlabel('Sample index')
ax9.set_ylabel('Value')
ax9.set_title('Predicted vs True (first {} samples, test set)'.format(n_bar))
ax9.set_xticks(x9)
ax9.legend()
ax9.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('fig9_sample_predictions.png', dpi=150, bbox_inches='tight')
plt.close()

# ========== 6. 文本输出：非零系数个数与简要分析 ==========
thresh = 1e-6
nz_lasso = np.sum(np.abs(coef_lasso_best) > thresh)
nz_ridge = np.sum(np.abs(coef_ridge_best) > thresh)

print('=' * 50)
print('Best lambda:')
print(f'  Lasso: {lambda_best_lasso:.4f}')
print(f'  Ridge: {lambda_best_ridge:.4f}')
print()
print('Non-zero coefficients (|coef| > 1e-6):')
print(f'  Lasso: {nz_lasso} / {n}')
print(f'  Ridge: {nz_ridge} / {n}')
print()
print('Brief analysis:')
print('  - Lasso (L1) tends to drive irrelevant feature coefficients to exactly 0,')
print('    so the number of non-zero coefficients is small; it performs feature selection.')
print('  - Ridge (L2) shrinks all coefficients but rarely sets them to zero,')
print('    so most coefficients remain non-zero; it only regularizes magnitude.')
print('=' * 50)
print('Figures saved: fig1_mse_vs_lambda.png ~ fig9_sample_predictions.png')
