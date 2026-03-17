import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# =============================================================================
#  参数与路径配置
# =============================================================================

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

RANDOM_STATE = 42

N_SAMPLES = 150
N_FEATURES = 200
N_INFORMATIVE = 10
NOISE_STD = 0.5

LAMBDA_LIST = np.linspace(0.001, 5, 30)
LAMBDA_PATH = np.logspace(-3, 1, 50)
N_SHOW = 10


# =============================================================================
#  数据生成：高维少数有效特征的制造业场景模拟
# =============================================================================

def generate_synthetic_manufact_data(
    n_samples: int,
    n_features: int,
    n_informative: int,
    noise_std: float,
    random_state: int = 42,
):
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    beta_true = np.zeros(n_features)
    beta_true[:n_informative] = rng.uniform(1, 5, size=n_informative) * rng.choice(
        [-1, 1], size=n_informative
    )
    y = X @ beta_true + rng.randn(n_samples) * noise_std
    return X, y, beta_true


# =============================================================================
#  主流程
# =============================================================================

def main():
    np.random.seed(RANDOM_STATE)

    X, y, beta_true = generate_synthetic_manufact_data(
        N_SAMPLES, N_FEATURES, N_INFORMATIVE, NOISE_STD, RANDOM_STATE
    )

    print(f"样本数: {N_SAMPLES}, 特征数: {N_FEATURES}")
    print(f"真正有用的特征: {N_INFORMATIVE} 个")
    print(f"真实非零系数: {beta_true[:N_INFORMATIVE].round(2)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n = X_train_scaled.shape[1]

    # ----------------- 无正则化基线 -----------------
    lr = LinearRegression().fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    coef_lr = lr.coef_

    # ----------------- 在 LAMBDA_LIST 上扫描 L1/L2 -----------------
    mse_lasso_list = []
    coef_lasso_list = []
    for lam in LAMBDA_LIST:
        model = Lasso(alpha=lam, max_iter=10000).fit(X_train_scaled, y_train)
        mse_lasso_list.append(
            mean_squared_error(y_test, model.predict(X_test_scaled))
        )
        coef_lasso_list.append(model.coef_)

    mse_ridge_list = []
    coef_ridge_list = []
    for lam in LAMBDA_LIST:
        model = Ridge(alpha=lam).fit(X_train_scaled, y_train)
        mse_ridge_list.append(
            mean_squared_error(y_test, model.predict(X_test_scaled))
        )
        coef_ridge_list.append(model.coef_)

    idx_best_lasso = int(np.argmin(mse_lasso_list))
    idx_best_ridge = int(np.argmin(mse_ridge_list))
    lambda_best_lasso = LAMBDA_LIST[idx_best_lasso]
    lambda_best_ridge = LAMBDA_LIST[idx_best_ridge]

    lasso_best = Lasso(alpha=lambda_best_lasso, max_iter=10000).fit(
        X_train_scaled, y_train
    )
    ridge_best = Ridge(alpha=lambda_best_ridge).fit(X_train_scaled, y_train)

    y_pred_lasso = lasso_best.predict(X_test_scaled)
    y_pred_ridge = ridge_best.predict(X_test_scaled)

    coef_lasso_best = lasso_best.coef_
    coef_ridge_best = ridge_best.coef_

    # ----------------- 系数路径（LAMBDA_PATH） -----------------
    coef_lasso_path = []
    coef_ridge_path = []
    for lam in LAMBDA_PATH:
        coef_lasso_path.append(
            Lasso(alpha=lam, max_iter=10000).fit(X_train_scaled, y_train).coef_
        )
        coef_ridge_path.append(
            Ridge(alpha=lam).fit(X_train_scaled, y_train).coef_
        )
    coef_lasso_path = np.array(coef_lasso_path)
    coef_ridge_path = np.array(coef_ridge_path)

    # ----------------- 残差 -----------------
    resid_lr = y_test - y_pred_lr
    resid_lasso = y_test - y_pred_lasso
    resid_ridge = y_test - y_pred_ridge

    # =========================================================================
    #  图 1：MSE vs lambda
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(LAMBDA_LIST, mse_lasso_list, "b-", label="Lasso", linewidth=2)
    ax1.plot(LAMBDA_LIST, mse_ridge_list, "r-", label="Ridge", linewidth=2)
    ax1.axhline(mse_lr, color="gray", linestyle="--", label="No regularization")
    ax1.set_xlabel("lambda")
    ax1.set_ylabel("Test MSE")
    ax1.set_title("Test MSE vs lambda (synthetic manufacturing data)")
    ax1.legend()
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "fig1_mse_vs_lambda.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # =========================================================================
    #  图 2：Lasso 系数路径（前 N_SHOW 个特征）
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for j in range(min(N_SHOW, n)):
        ax2.plot(LAMBDA_PATH, coef_lasso_path[:, j], label=f"Feature {j}")
    ax2.set_xlabel("lambda")
    ax2.set_ylabel("Coefficient")
    ax2.set_title("Lasso coefficient path (first features)")
    ax2.legend()
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "fig2_lasso_coef_path.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # =========================================================================
    #  图 3：Ridge 系数路径（前 N_SHOW 个特征）
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for j in range(min(N_SHOW, n)):
        ax3.plot(LAMBDA_PATH, coef_ridge_path[:, j], label=f"Feature {j}")
    ax3.set_xlabel("lambda")
    ax3.set_ylabel("Coefficient")
    ax3.set_title("Ridge coefficient path (first features)")
    ax3.legend()
    ax3.set_xscale("log")
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "fig3_ridge_coef_path.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # =========================================================================
    #  图 4：最佳 lambda 下三种方法系数对比（前 N_SHOW 个特征）
    # =========================================================================
    x4 = np.arange(min(N_SHOW, n))
    w = 0.25
    fig4, ax4 = plt.subplots(figsize=(9, 5))
    ax4.bar(x4 - w, coef_lr[: len(x4)], width=w, label="No reg")
    ax4.bar(x4, coef_lasso_best[: len(x4)], width=w, label="Lasso")
    ax4.bar(x4 + w, coef_ridge_best[: len(x4)], width=w, label="Ridge")
    ax4.axhline(0, color="black", linewidth=0.5)
    ax4.set_xlabel("Feature index")
    ax4.set_ylabel("Coefficient")
    ax4.set_title(
        "Coefficient comparison at best lambda (first features)"
    )
    ax4.set_xticks(x4)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "fig4_coef_compare.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # =========================================================================
    #  图 5：残差箱线图
    # =========================================================================
    fig5, ax5 = plt.subplots(figsize=(6, 5))
    bp = ax5.boxplot(
        [resid_lr, resid_lasso, resid_ridge],
        tick_labels=["No reg", "Lasso", "Ridge"],
        patch_artist=True,
    )
    colors = ["lightblue", "lightgreen", "lightcoral"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax5.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax5.set_ylabel("Residual (y_test - y_pred)")
    ax5.set_title("Residual distribution (test set)")
    ax5.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "fig5_residual_boxplot.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # =========================================================================
    #  图 6：预测值 vs 真实值散点图
    # =========================================================================
    fig6, ax6 = plt.subplots(figsize=(6, 6))
    ax6.scatter(y_test, y_pred_lr, alpha=0.6, s=30, label="No reg", c="C0")
    ax6.scatter(
        y_test, y_pred_lasso, alpha=0.6, s=30, label="Lasso", c="C1"
    )
    ax6.scatter(
        y_test, y_pred_ridge, alpha=0.6, s=30, label="Ridge", c="C2"
    )
    ymin = min(
        y_test.min(),
        y_pred_lr.min(),
        y_pred_lasso.min(),
        y_pred_ridge.min(),
    )
    ymax = max(
        y_test.max(),
        y_pred_lr.max(),
        y_pred_lasso.max(),
        y_pred_ridge.max(),
    )
    ax6.plot([ymin, ymax], [ymin, ymax], "k--", label="y=x")
    ax6.set_xlabel("True y")
    ax6.set_ylabel("Predicted y")
    ax6.set_title("Predicted vs True (test set)")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "fig6_pred_vs_true.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # =========================================================================
    #  控制台关键指标与简要分析
    # =========================================================================
    thresh = 1e-6
    nz_lasso = int(np.sum(np.abs(coef_lasso_best) > thresh))
    nz_ridge = int(np.sum(np.abs(coef_ridge_best) > thresh))

    l1_lasso = float(np.sum(np.abs(coef_lasso_best)))
    l1_ridge = float(np.sum(np.abs(coef_ridge_best)))
    l2_lasso = float(np.sqrt(np.sum(coef_lasso_best ** 2)))
    l2_ridge = float(np.sqrt(np.sum(coef_ridge_best ** 2)))

    coef_mse_lasso = float(np.mean((coef_lasso_best - beta_true) ** 2))
    coef_mse_ridge = float(np.mean((coef_ridge_best - beta_true) ** 2))

    print("\n" + "=" * 60)
    print("合成制造业场景下 L1 / L2 正则化对比")
    print("=" * 60)
    print(f"最佳 lambda: Lasso = {lambda_best_lasso:.4f}, Ridge = {lambda_best_ridge:.4f}")
    print()
    print("非零系数个数 (|coef| > 1e-6):")
    print(f"  Lasso: {nz_lasso} / {n} (真实有用特征 {N_INFORMATIVE} 个)")
    print(f"  Ridge: {nz_ridge} / {n}")
    print()
    print("系数范数与系数恢复误差:")
    print(f"  ||beta_hat||_1: Lasso = {l1_lasso:.2f}, Ridge = {l1_ridge:.2f}")
    print(f"  ||beta_hat||_2: Lasso = {l2_lasso:.2f}, Ridge = {l2_ridge:.2f}")
    print(f"  与真实系数的 MSE: Lasso = {coef_mse_lasso:.4f}, Ridge = {coef_mse_ridge:.4f}")
    print()
    print("预测 MSE (test set):")
    print(f"  No reg: {mse_lr:.4f}")
    print(
        f"  Lasso:  {mean_squared_error(y_test, y_pred_lasso):.4f}"
    )
    print(
        f"  Ridge:  {mean_squared_error(y_test, y_pred_ridge):.4f}"
    )
    print()
    print("简要分析：")
    print("  - L1 (Lasso) 使用向量 L1 范数作为惩罚，更容易把不重要特征系数压缩到 0，")
    print("    在本实验中可以看到非零系数个数大幅少于 Ridge，体现出稀疏性和特征选择能力。")
    print("  - L2 (Ridge) 使用向量 L2 范数惩罚，系数整体被缩小但很少变成 0，")
    print("    在存在噪声和特征相关性时，往往能获得更稳定的预测和更平滑的系数路径。")
    print(
        "  - 虽然在有限维空间中不同范数在拓扑上是等价的，但在正则化优化问题中，"
    )
    print("    由于约束几何形状不同（L1 的菱形 vs L2 的圆球），会导致解的结构出现显著差异。")
    print("=" * 60)
    print("图像已保存到：", OUT_DIR)
    print("  fig1_mse_vs_lambda.png ~ fig6_pred_vs_true.png")


if __name__ == "__main__":
    main()