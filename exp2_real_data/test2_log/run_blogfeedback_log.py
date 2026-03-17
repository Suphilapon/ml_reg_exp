"""
BlogFeedback 正则化对比实验 —— 目标变量对数变换版 (test2_log)
- 使用 y_log = log(1 + y) 作为训练目标，缓解评论数右偏分布
- 预测时反变换：y_pred = exp(pred_log) - 1，评估与画图均在原始尺度
- 数据与流程同 run_blogfeedback.py，仅增加对 y 的 log1p / expm1 变换
- 输出：与主脚本相同的 9 张图，保存到本脚本所在目录 (test2_log)
"""

import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ---------- 路径与常量（输出到 test2_log，数据从上级 data 读取）----------
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_PREFIX = 'exp2_log_'  # 图片文件名前缀，与 test1_no_Log 区分，避免混用歧义
# 数据目录：上级目录的 data（与 test1_no_Log 共用）
DATA_DIR = os.path.join(os.path.dirname(OUT_DIR), 'data')
TRAIN_CSV = 'blogData_train.csv'
DATA_PATH = os.path.join(DATA_DIR, TRAIN_CSV)
UCI_ZIP_URL = 'https://archive.ics.uci.edu/static/public/304/blogfeedback.zip'
SUBSAMPLE_SIZE = 5000
RANDOM_STATE = 42

N_FEATURES = 280
TARGET_COL = 280
lambda_list = np.linspace(0.001, 10, 30)
lambda_path = np.logspace(-3, 1.5, 50)
n_show = 10
n_sample_bar = 20


def ensure_data():
    """若本地没有 blogData_train.csv，尝试从 UCI 下载并解压到 DATA_DIR。"""
    if os.path.isfile(DATA_PATH):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, 'blogfeedback.zip')
    print(f'Data not found at {DATA_PATH}. Attempting download from UCI...')
    try:
        urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(DATA_DIR)
        if os.path.isfile(DATA_PATH):
            print('Download and extract OK.')
        else:
            for root, _, files in os.walk(DATA_DIR):
                for f in files:
                    if f == TRAIN_CSV:
                        src = os.path.join(root, f)
                        if src != DATA_PATH:
                            import shutil
                            shutil.move(src, DATA_PATH)
                        break
        if not os.path.isfile(DATA_PATH):
            raise FileNotFoundError(f'After extract, {TRAIN_CSV} not found in {DATA_DIR}')
    except Exception as e:
        raise FileNotFoundError(
            f'Missing {DATA_PATH}. Place blogData_train.csv in exp2_real_data/data/ '
            f'or run with network to auto-download. Error: {e}'
        ) from e


def main():
    ensure_data()
    df = pd.read_csv(DATA_PATH, header=None)
    X = df.iloc[:, :N_FEATURES].values
    y = df.iloc[:, TARGET_COL].values

    # 先按原始 y 划分，再对训练集做 log 变换；测试集始终用原始 y 做评估与画图
    if SUBSAMPLE_SIZE is not None and len(X) > SUBSAMPLE_SIZE:
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_STATE
        )
        n_train = min(SUBSAMPLE_SIZE, len(X_train_full))
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train_full), n_train, replace=False)
        X_train = X_train_full[idx]
        y_train = y_train_full[idx]
        y_train_log = np.log1p(y_train)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_STATE
        )
        y_train_log = np.log1p(y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    n = X_train_scaled.shape[1]

    # ---------- 无正则化（在 log 尺度上拟合，预测后反变换到原始尺度）----------
    lr = LinearRegression().fit(X_train_scaled, y_train_log)
    pred_lr_log = lr.predict(X_test_scaled)
    y_pred_lr = np.expm1(pred_lr_log)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    coef_lr = lr.coef_

    # ---------- Lasso / Ridge：在 lambda_list 上训练，选最佳 lambda（按原始尺度 MSE）----------
    mse_lasso_list = []
    coef_lasso_list = []
    for lam in lambda_list:
        model = Lasso(alpha=lam, max_iter=10000).fit(X_train_scaled, y_train_log)
        pred_log = model.predict(X_test_scaled)
        y_pred_orig = np.expm1(pred_log)
        mse_lasso_list.append(mean_squared_error(y_test, y_pred_orig))
        coef_lasso_list.append(model.coef_)

    mse_ridge_list = []
    coef_ridge_list = []
    for lam in lambda_list:
        model = Ridge(alpha=lam).fit(X_train_scaled, y_train_log)
        pred_log = model.predict(X_test_scaled)
        y_pred_orig = np.expm1(pred_log)
        mse_ridge_list.append(mean_squared_error(y_test, y_pred_orig))
        coef_ridge_list.append(model.coef_)

    idx_best_lasso = np.argmin(mse_lasso_list)
    idx_best_ridge = np.argmin(mse_ridge_list)
    lambda_best_lasso = lambda_list[idx_best_lasso]
    lambda_best_ridge = lambda_list[idx_best_ridge]

    lasso_best = Lasso(alpha=lambda_best_lasso, max_iter=10000).fit(X_train_scaled, y_train_log)
    ridge_best = Ridge(alpha=lambda_best_ridge).fit(X_train_scaled, y_train_log)
    y_pred_lasso = np.expm1(lasso_best.predict(X_test_scaled))
    y_pred_ridge = np.expm1(ridge_best.predict(X_test_scaled))
    coef_lasso_best = lasso_best.coef_
    coef_ridge_best = ridge_best.coef_

    # ---------- 系数路径（log 尺度下的系数，仅用于画图）----------
    coef_lasso_path = []
    coef_ridge_path = []
    for lam in lambda_path:
        coef_lasso_path.append(Lasso(alpha=lam, max_iter=10000).fit(X_train_scaled, y_train_log).coef_)
        coef_ridge_path.append(Ridge(alpha=lam).fit(X_train_scaled, y_train_log).coef_)
    coef_lasso_path = np.array(coef_lasso_path)
    coef_ridge_path = np.array(coef_ridge_path)

    resid_lr = y_test - y_pred_lr
    resid_lasso = y_test - y_pred_lasso
    resid_ridge = y_test - y_pred_ridge
    n_test = len(y_test)

    # ----- 图 1 -----
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(lambda_list, mse_lasso_list, 'b-', label='Lasso', linewidth=2)
    ax1.plot(lambda_list, mse_ridge_list, 'r-', label='Ridge', linewidth=2)
    ax1.axhline(mse_lr, color='gray', linestyle='--', label='No regularization')
    ax1.set_xlabel('lambda')
    ax1.set_ylabel('Test MSE (original scale)')
    ax1.set_title('Test MSE vs lambda (log-target model)')
    ax1.legend()
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, FIG_PREFIX + 'fig1_mse_vs_lambda.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ----- 图 2 -----
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for j in range(n_show):
        ax2.plot(lambda_path, coef_lasso_path[:, j], label=f'Feature {j}')
    ax2.set_xlabel('lambda')
    ax2.set_ylabel('Coefficient')
    ax2.set_title('Lasso coefficient path (first 10 features, log-target)')
    ax2.legend()
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, FIG_PREFIX + 'fig2_lasso_coef_path.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ----- 图 3 -----
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for j in range(n_show):
        ax3.plot(lambda_path, coef_ridge_path[:, j], label=f'Feature {j}')
    ax3.set_xlabel('lambda')
    ax3.set_ylabel('Coefficient')
    ax3.set_title('Ridge coefficient path (first 10 features, log-target)')
    ax3.legend()
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, FIG_PREFIX + 'fig3_ridge_coef_path.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ----- 图 4 -----
    x4 = np.arange(n_show)
    w = 0.2
    fig4, ax4 = plt.subplots(figsize=(9, 5))
    ax4.bar(x4 - w, coef_lr[:n_show], width=w, label='No reg')
    ax4.bar(x4, coef_lasso_best[:n_show], width=w, label='Lasso')
    ax4.bar(x4 + w, coef_ridge_best[:n_show], width=w, label='Ridge')
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.set_xlabel('Feature index')
    ax4.set_ylabel('Coefficient')
    ax4.set_title('Coefficient comparison at best lambda (first 10, log-target)')
    ax4.set_xticks(x4)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, FIG_PREFIX + 'fig4_coef_compare.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ----- 图 5 -----
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
    ax5.set_ylabel('Residual (y_test - y_pred, original scale)')
    ax5.set_title('Residual distribution (test set, log-target)')
    ax5.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, FIG_PREFIX + 'fig5_residual_boxplot.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ----- 图 6 -----
    fig6, ax6 = plt.subplots(figsize=(6, 6))
    ax6.scatter(y_test, y_pred_lr, alpha=0.6, s=30, label='No reg', c='C0')
    ax6.scatter(y_test, y_pred_lasso, alpha=0.6, s=30, label='Lasso', c='C1')
    ax6.scatter(y_test, y_pred_ridge, alpha=0.6, s=30, label='Ridge', c='C2')
    ymin = min(y_test.min(), y_pred_lr.min(), y_pred_lasso.min(), y_pred_ridge.min())
    ymax = max(y_test.max(), y_pred_lr.max(), y_pred_lasso.max(), y_pred_ridge.max())
    ax6.plot([ymin, ymax], [ymin, ymax], 'k--', label='y=x')
    ax6.set_xlabel('True y')
    ax6.set_ylabel('Predicted y (original scale)')
    ax6.set_title('Predicted vs True (test set, log-target)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, FIG_PREFIX + 'fig6_pred_vs_true.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ----- 图 7 -----
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
    ax7.set_title('Absolute error (sample × model, log-target)')
    plt.colorbar(im, ax=ax7, label='Absolute error')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, FIG_PREFIX + 'fig7_error_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ----- 图 8 -----
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
    ax8.set_title('Cumulative error distribution (test set, log-target)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, FIG_PREFIX + 'fig8_cumulative_error.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ----- 图 9 -----
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
    ax9.set_ylabel('Value (original scale)')
    ax9.set_title('Predicted vs True (first {} samples, test set, log-target)'.format(n_bar))
    ax9.set_xticks(x9)
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, FIG_PREFIX + 'fig9_sample_predictions.png'), dpi=150, bbox_inches='tight')
    plt.close()

    thresh = 1e-6
    nz_lasso = np.sum(np.abs(coef_lasso_best) > thresh)
    nz_ridge = np.sum(np.abs(coef_ridge_best) > thresh)
    print('=' * 50)
    print('Log-target model (y_log = log(1+y), predict then expm1)')
    print('Best lambda:')
    print(f'  Lasso: {lambda_best_lasso:.4f}')
    print(f'  Ridge: {lambda_best_ridge:.4f}')
    print()
    print('Non-zero coefficients (|coef| > 1e-6):')
    print(f'  Lasso: {nz_lasso} / {n}')
    print(f'  Ridge: {nz_ridge} / {n}')
    print()
    print('Test MSE (original scale):')
    print(f'  No reg:  {mse_lr:.2f}')
    print(f'  Lasso:   {mean_squared_error(y_test, y_pred_lasso):.2f}')
    print(f'  Ridge:   {mean_squared_error(y_test, y_pred_ridge):.2f}')
    print('=' * 50)
    print('Figures saved to:', OUT_DIR)
    print('  ' + FIG_PREFIX + 'fig1_mse_vs_lambda.png ~ ' + FIG_PREFIX + 'fig9_sample_predictions.png')


if __name__ == '__main__':
    main()
