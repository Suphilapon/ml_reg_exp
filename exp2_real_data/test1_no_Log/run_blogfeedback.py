"""
===============================================================================
  BlogFeedback (UCI) 正则化对比实验脚本
===============================================================================
  本脚本用 UCI 的 BlogFeedback 数据集，对比三种线性回归方式：
    1. 无正则化（普通最小二乘）
    2. Lasso（L1 正则化，会令部分系数变为 0，做特征选择）
    3. Ridge（L2 正则化，缩小系数但不为 0）
  数据：blogData_train.csv，280 个特征，1 个目标（未来 24 小时评论数）。
  输出：与 exp1 相同的 9 张图，保存到本脚本所在目录。
===============================================================================
"""


# =============================================================================
#  模块一：导入库与常量配置
# =============================================================================
#
#  下面会用到：
#  - 文件和路径、下载解压 → os, zipfile, urllib.request
#  - 数值和表格 → numpy, pandas
#  - 模型与评估 → sklearn 的线性模型、预处理、划分、MSE
#  - 画图 → matplotlib
#
# -----------------------------------------------------------------------------

import os
# os：和操作系统打交道，例如判断文件是否存在、创建目录、拼接路径

import zipfile
# zipfile：读写 .zip 压缩包，用于解压从 UCI 下载的数据

import urllib.request
# urllib.request：从网址下载文件到本地

import numpy as np
# numpy：做数值计算，例如生成等间隔的 lambda、数组运算；通常简写为 np

import pandas as pd
# pandas：读 CSV、按列取数据；通常简写为 pd

from sklearn.linear_model import LinearRegression, Lasso, Ridge
# sklearn.linear_model：线性模型
#   LinearRegression：无正则化的普通线性回归
#   Lasso：带 L1 正则化的线性回归（系数可变为 0）
#   Ridge：带 L2 正则化的线性回归（系数缩小但不为 0）

from sklearn.preprocessing import StandardScaler
# StandardScaler：把特征缩放到均值 0、方差 1，避免某些特征数值过大影响训练

from sklearn.model_selection import train_test_split
# train_test_split：把数据随机分成“训练集”和“测试集”，例如 70% 训练、30% 测试

from sklearn.metrics import mean_squared_error
# mean_squared_error：计算预测与真实值的均方误差（MSE），越小说明模型越好

import matplotlib.pyplot as plt
# matplotlib.pyplot：画图（折线、散点、柱状图等），通常简写为 plt

# -----------------------------------------------------------------------------
#  路径与数据相关常量（可根据需要修改）
# -----------------------------------------------------------------------------

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
# __file__ 是当前脚本的文件路径；abspath 转成绝对路径；dirname 取所在目录
# 结果：OUT_DIR = 本脚本所在的文件夹路径（即 test1_no_Log），图会保存到这里
FIG_PREFIX = 'exp2_noLog_'  # 图片文件名前缀，与 test2_log 区分，避免混用歧义

DATA_DIR = os.path.join(OUT_DIR, 'data')
# 数据文件夹：OUT_DIR 下的 data 子文件夹（即 exp2_real_data/data）

TRAIN_CSV = 'blogData_train.csv'
# 训练数据文件名（UCI 提供的 60,021 行 × 281 列）

DATA_PATH = os.path.join(DATA_DIR, TRAIN_CSV)
# 训练数据的完整路径：exp2_real_data/data/blogData_train.csv

UCI_ZIP_URL = 'https://archive.ics.uci.edu/static/public/304/blogfeedback.zip'
# UCI 上 BlogFeedback 数据集的 zip 下载地址；若本地没有 CSV 可自动下载并解压

SUBSAMPLE_SIZE = 5000
# 子采样数量：只用 5000 条数据做实验，加快运行；设为 None 则使用全部数据（约 6 万条，较慢）

RANDOM_STATE = 42
# 随机种子：固定为 42 后，每次运行的“随机划分”“随机抽样”结果一致，便于复现

# -----------------------------------------------------------------------------
#  数据格式常量（BlogFeedback 数据集约定）
# -----------------------------------------------------------------------------
#  CSV 无表头，共 281 列：前 280 列是特征（如博客统计、词频等），第 281 列是目标（评论数）
#  列索引从 0 开始：特征 = 第 0～279 列，目标 = 第 280 列

N_FEATURES = 280
# 特征个数

TARGET_COL = 280
# 目标所在列索引（第 281 列，因为从 0 开始所以是 280）

# -----------------------------------------------------------------------------
#  正则化与画图用常量（与 exp1 保持一致）
# -----------------------------------------------------------------------------

lambda_list = np.linspace(0.001, 10, 30)
# 用于“选最佳 lambda”的一组值：从 0.001 到 10 均匀取 30 个
# 我们会在这 30 个 lambda 上分别训练 Lasso/Ridge，看哪个 lambda 在测试集上 MSE 最小

lambda_path = np.logspace(-3, 1.5, 50)
# 用于“画系数路径”的一组值：从 10^-3 到 10^1.5，对数间隔取 50 个
# 系数路径：随 lambda 变化，每个特征的系数如何变化（用于画图 2、图 3）

n_show = 10
# 系数图里只画前 10 个特征的系数（共 280 个，画 10 个便于看清）

n_sample_bar = 20
# 图 9 条形图里展示测试集前 20 个样本的预测值对比


# =============================================================================
#  模块二：确保数据存在（若没有则自动下载并解压）
# =============================================================================
#
#  流程简图：
#
#      DATA_PATH 是否存在？
#           |
#     是 ---+--- 直接 return，不做任何事
#           |
#     否 ---+--- 创建 data 目录
#           |    下载 UCI 的 zip
#           |    解压到 data
#           |    若解压后 CSV 在子文件夹里，则移动到 data 根目录
#           |    若仍没有 CSV，抛出 FileNotFoundError 提示用户
#
# -----------------------------------------------------------------------------

def ensure_data():
    """若本地没有 blogData_train.csv，则尝试从 UCI 下载 zip 并解压到 DATA_DIR。"""
    if os.path.isfile(DATA_PATH):
        # 若 DATA_PATH 指向的文件已存在，说明数据准备好了，直接返回
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    # 若 data 目录不存在则创建；exist_ok=True 表示已存在也不报错
    zip_path = os.path.join(DATA_DIR, 'blogfeedback.zip')
    # 下载后的 zip 打算放在 data 文件夹下，命名为 blogfeedback.zip
    print(f'Data not found at {DATA_PATH}. Attempting download from UCI...')
    # 提示用户：正在尝试下载
    try:
        urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)
        # 从 UCI_ZIP_URL 下载文件，保存到 zip_path
        with zipfile.ZipFile(zip_path, 'r') as z:
            # 以只读方式打开 zip 文件，z 是 ZipFile 对象
            z.extractall(DATA_DIR)
            # 把 zip 里所有文件解压到 DATA_DIR（即 data 文件夹）
        if os.path.isfile(DATA_PATH):
            # 解压后若 blogData_train.csv 已经在 data 根目录
            print('Download and extract OK.')
        else:
            # 有些 zip 解压后会在子文件夹里（例如 data/blogfeedback/blogData_train.csv）
            # 下面遍历 data 下所有文件，找到 blogData_train.csv 并移到 data 根目录
            for root, _, files in os.walk(DATA_DIR):
                # os.walk 遍历 DATA_DIR 及其所有子目录；root 是当前目录，files 是当前目录下的文件名列表
                for f in files:
                    if f == TRAIN_CSV:
                        src = os.path.join(root, f)
                        # 当前找到的 CSV 的完整路径
                        if src != DATA_PATH:
                            import shutil
                            # shutil 提供文件复制、移动等操作
                            shutil.move(src, DATA_PATH)
                            # 把 CSV 从 src 移动到 DATA_PATH（即 data/blogData_train.csv）
                        break
        if not os.path.isfile(DATA_PATH):
            # 若移动后仍然不存在，说明 zip 里可能没有这个文件，报错
            raise FileNotFoundError(f'After extract, {TRAIN_CSV} not found in {DATA_DIR}')
    except Exception as e:
        # 下载或解压过程中任何错误（如网络问题），都转成明确的提示
        raise FileNotFoundError(
            f'Missing {DATA_PATH}. Place blogData_train.csv in exp2_real_data/data/ '
            f'or run with network to auto-download. Error: {e}'
        ) from e


# =============================================================================
#  模块三：主流程 main() —— 从读数据到画图、打印结果
# =============================================================================
#
#  整体流程：
#
#   [1] 确保数据存在
#        ↓
#   [2] 读 CSV → 得到 X（特征矩阵）、y（目标向量）
#        ↓
#   [3] 若启用子采样：先 70/30 划分，再对训练集随机抽 SUBSAMPLE_SIZE 条
#        否则：直接 70/30 划分
#        ↓
#   [4] 对特征 X 做标准化（训练集 fit_transform，测试集 transform）
#        ↓
#   [5] 无正则化：用 LinearRegression 拟合，得到预测和 MSE、系数
#        ↓
#   [6] 在 lambda_list 上训练 Lasso/Ridge，选测试集 MSE 最小的 lambda
#        ↓
#   [7] 用最佳 lambda 再训练一次，得到最佳模型的预测和系数
#        ↓
#   [8] 在 lambda_path 上训练，得到系数路径（用于画图 2、图 3）
#        ↓
#   [9] 计算三组残差（真实 - 预测）
#        ↓
#   [10] 画图 1～9 并保存
#        ↓
#   [11] 打印最佳 lambda、非零系数个数、简要分析
#
# -----------------------------------------------------------------------------

def main():
    # ----------  [1] 确保数据文件存在（没有则下载解压）  ----------
    ensure_data()

    # ----------  [2] 读取 CSV，拆成特征 X 和目标 y  ----------
    df = pd.read_csv(DATA_PATH, header=None)
    # 读 CSV；header=None 表示第一行也是数据而不是列名（该数据集无表头）
    # df 是一个“表格”对象（DataFrame），行=样本，列=281 列
    X = df.iloc[:, :N_FEATURES].values
    # iloc[:, :280]：所有行，第 0 到 279 列（即前 280 列）
    # .values 转成 numpy 数组，方便后面给 sklearn 用
    # X 的形状：(样本数, 280)
    y = df.iloc[:, TARGET_COL].values
    # 取第 280 列（目标：未来 24 小时评论数）
    # y 的形状：(样本数,)，一维数组

    # ----------  [3] 划分训练集 / 测试集（可选子采样）  ----------
    if SUBSAMPLE_SIZE is not None and len(X) > SUBSAMPLE_SIZE:
        # 若设置了子采样且数据量大于 SUBSAMPLE_SIZE，则先划分再对训练集子采样
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_STATE
        )
        # 随机把 X,y 分成 70% 训练、30% 测试；random_state 固定随机性
        # 得到：训练部分 (X_train_full, y_train_full)，测试部分 (X_test, y_test)
        n_train = min(SUBSAMPLE_SIZE, len(X_train_full))
        # 实际使用的训练样本数：不超过 SUBSAMPLE_SIZE，也不超过当前训练集大小
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train_full), n_train, replace=False)
        # 从 0 到 len(X_train_full)-1 中无放回随机抽 n_train 个索引
        # RandomState(RANDOM_STATE) 保证每次运行抽到的索引一致
        X_train = X_train_full[idx]
        # 用索引取出对应的样本，作为最终训练集特征
        y_train = y_train_full[idx]
        # 对应的目标值，作为最终训练集目标
    else:
        # 不子采样：直接 70/30 划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_STATE
        )

    # ----------  [4] 特征标准化（只对 X，不对 y）  ----------
    scaler = StandardScaler()
    # 创建一个“标准化器”对象：会把数据变成均值 0、标准差 1
    X_train_scaled = scaler.fit_transform(X_train)
    # 在训练集上“拟合”均值和标准差，并用它们对训练集做变换
    # 这样训练集特征尺度统一，模型训练更稳定
    X_test_scaled = scaler.transform(X_test)
    # 测试集只用训练集已经拟合好的均值和标准差来变换（不能重新 fit，否则会“偷看”测试集）
    n = X_train_scaled.shape[1]
    # 特征维度（列数），即 280，后面打印非零系数个数时会用到

    # ----------  [5] 无正则化：普通线性回归  ----------
    lr = LinearRegression().fit(X_train_scaled, y_train)
    # 创建普通线性回归模型，并用训练数据拟合（学习系数）
    y_pred_lr = lr.predict(X_test_scaled)
    # 用拟合好的模型对测试集做预测
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    # 预测与真实值的均方误差（MSE），作为“无正则化”的基准
    coef_lr = lr.coef_
    # 学到的系数（280 维），用于后面画系数对比图

    # ----------  [6] 在 lambda_list 上训练 Lasso / Ridge，记录每个 lambda 的测试 MSE  ----------
    mse_lasso_list = []
    # 每个 lambda 下 Lasso 在测试集上的 MSE，后面用来找“最佳 lambda”
    coef_lasso_list = []
    # 每个 lambda 下 Lasso 的系数（画图或分析用）
    for lam in lambda_list:
        # 遍历 30 个 lambda 值
        model = Lasso(alpha=lam, max_iter=10000).fit(X_train_scaled, y_train)
        # Lasso：alpha 就是正则化强度 lambda；max_iter 最大迭代次数，避免未收敛警告
        mse_lasso_list.append(mean_squared_error(y_test, model.predict(X_test_scaled)))
        # 用当前模型预测测试集，计算 MSE，加入列表
        coef_lasso_list.append(model.coef_)
        # 当前模型的系数加入列表

    mse_ridge_list = []
    coef_ridge_list = []
    for lam in lambda_list:
        model = Ridge(alpha=lam).fit(X_train_scaled, y_train)
        # Ridge 没有 Lasso 那么难收敛，一般不需要特意设 max_iter
        mse_ridge_list.append(mean_squared_error(y_test, model.predict(X_test_scaled)))
        coef_ridge_list.append(model.coef_)

    # ----------  选出测试集 MSE 最小的 lambda（最佳 lambda）  ----------
    idx_best_lasso = np.argmin(mse_lasso_list)
    # argmin：返回使 mse_lasso_list 取最小值的那个下标（0～29 之一）
    idx_best_ridge = np.argmin(mse_ridge_list)
    lambda_best_lasso = lambda_list[idx_best_lasso]
    # 该下标对应的 lambda 值，即“最佳 Lasso lambda”
    lambda_best_ridge = lambda_list[idx_best_ridge]

    # ----------  [7] 用最佳 lambda 再训练一次，得到最终模型与预测  ----------
    lasso_best = Lasso(alpha=lambda_best_lasso, max_iter=10000).fit(X_train_scaled, y_train)
    ridge_best = Ridge(alpha=lambda_best_ridge).fit(X_train_scaled, y_train)
    y_pred_lasso = lasso_best.predict(X_test_scaled)
    y_pred_ridge = ridge_best.predict(X_test_scaled)
    coef_lasso_best = lasso_best.coef_
    coef_ridge_best = ridge_best.coef_

    # ----------  [8] 系数路径：在 lambda_path 上训练，用于画系数随 lambda 变化的曲线  ----------
    coef_lasso_path = []
    coef_ridge_path = []
    for lam in lambda_path:
        # 这里用另一组更密的 lambda（50 个，对数间隔），专门为了画平滑的系数路径
        coef_lasso_path.append(Lasso(alpha=lam, max_iter=10000).fit(X_train_scaled, y_train).coef_)
        coef_ridge_path.append(Ridge(alpha=lam).fit(X_train_scaled, y_train).coef_)
    coef_lasso_path = np.array(coef_lasso_path)
    # 转成二维数组：行 = 50 个 lambda，列 = 280 个特征
    coef_ridge_path = np.array(coef_ridge_path)

    # ----------  [9] 残差 = 真实值 - 预测值（用于图 5、7、8 等）  ----------
    resid_lr = y_test - y_pred_lr
    resid_lasso = y_test - y_pred_lasso
    resid_ridge = y_test - y_pred_ridge

    n_test = len(y_test)
    # 测试集样本数，后面画图时可能用到

    # =========================================================================
    #  模块三之「画图部分」：图 1～图 9
    # =========================================================================
    #  每张图：先创建图与坐标轴 → 画内容 → 设置标签/标题/图例 → 保存到 OUT_DIR → 关闭图
    #  保存参数：dpi=150 清晰度，bbox_inches='tight' 避免边缘被裁掉
    # -------------------------------------------------------------------------

    # ----- 图 1：测试集 MSE 随 lambda 变化（Lasso / Ridge + 无正则化水平线） -----
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    # 创建一个 8×5 英寸的图，ax1 是用于画线的坐标轴
    ax1.plot(lambda_list, mse_lasso_list, 'b-', label='Lasso', linewidth=2)
    # 横轴 lambda_list，纵轴 mse_lasso_list，蓝色实线，图例名 'Lasso'
    ax1.plot(lambda_list, mse_ridge_list, 'r-', label='Ridge', linewidth=2)
    ax1.axhline(mse_lr, color='gray', linestyle='--', label='No regularization')
    # 画一条水平线，高度为无正则化的 MSE，表示基准
    ax1.set_xlabel('lambda')
    ax1.set_ylabel('Test MSE')
    ax1.set_title('Test MSE vs lambda')
    ax1.legend()
    # 显示图例（Lasso / Ridge / No regularization）
    ax1.set_xscale('log')
    # 横轴用对数刻度，因为 lambda 跨度大，对数更易读
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    # 自动调整子图参数，避免标签被裁
    plt.savefig(os.path.join(OUT_DIR, FIG_PREFIX + 'fig1_mse_vs_lambda.png'), dpi=150, bbox_inches='tight')
    plt.close()
    # 关闭图，释放内存，否则多张图会叠在一起

    # ----- 图 2：Lasso 系数路径（前 10 个特征随 lambda 的变化） -----
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for j in range(n_show):
        # j = 0,1,...,9，即前 10 个特征
        ax2.plot(lambda_path, coef_lasso_path[:, j], label=f'Feature {j}')
        # 横轴 lambda_path，纵轴第 j 个特征在 50 个 lambda 下的系数
        # coef_lasso_path[:, j] 表示所有行的第 j 列
    ax2.set_xlabel('lambda')
    ax2.set_ylabel('Coefficient')
    ax2.set_title('Lasso coefficient path (first 10 features)')
    ax2.legend()
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, FIG_PREFIX + 'fig2_lasso_coef_path.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ----- 图 3：Ridge 系数路径（前 10 个特征） -----
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
    plt.savefig(os.path.join(OUT_DIR, FIG_PREFIX + 'fig3_ridge_coef_path.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ----- 图 4：最佳 lambda 下三种方法的系数对比（前 10 个特征，柱状图） -----
    #  真实数据没有“真实系数”，所以不画 True coefficients，只画 No reg / Lasso / Ridge
    x4 = np.arange(n_show)
    # 横轴刻度位置：0,1,...,9
    w = 0.2
    # 柱宽，三组柱并排时用偏移 w 错开
    fig4, ax4 = plt.subplots(figsize=(9, 5))
    ax4.bar(x4 - w, coef_lr[:n_show], width=w, label='No reg')
    # 第一组柱：无正则化的前 10 个系数，放在 x4-w 位置
    ax4.bar(x4, coef_lasso_best[:n_show], width=w, label='Lasso')
    ax4.bar(x4 + w, coef_ridge_best[:n_show], width=w, label='Ridge')
    ax4.axhline(0, color='black', linewidth=0.5)
    # 画 y=0 的参考线
    ax4.set_xlabel('Feature index')
    ax4.set_ylabel('Coefficient')
    ax4.set_title('Coefficient comparison at best lambda (first 10 features)')
    ax4.set_xticks(x4)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, FIG_PREFIX + 'fig4_coef_compare.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ----- 图 5：残差分布箱线图（三组：No reg / Lasso / Ridge） -----
    fig5, ax5 = plt.subplots(figsize=(6, 5))
    bp = ax5.boxplot(
        [resid_lr, resid_lasso, resid_ridge],
        # 三组数据，每组是测试集上该模型的残差
        tick_labels=['No reg', 'Lasso', 'Ridge'],
        patch_artist=True
        # 允许后面给箱子填色
    )
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        # 给每个箱子的矩形块上色
        patch.set_facecolor(color)
    ax5.axhline(0, color='gray', linestyle='--', alpha=0.7)
    # 残差 0 的参考线
    ax5.set_ylabel('Residual (y_test - y_pred)')
    ax5.set_title('Residual distribution (test set)')
    ax5.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, FIG_PREFIX + 'fig5_residual_boxplot.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ----- 图 6：预测值 vs 真实值散点图（理想情况应沿 y=x 直线） -----
    fig6, ax6 = plt.subplots(figsize=(6, 6))
    ax6.scatter(y_test, y_pred_lr, alpha=0.6, s=30, label='No reg', c='C0')
    # 横轴真实值，纵轴预测值；alpha 透明度，s 点大小，c 颜色
    ax6.scatter(y_test, y_pred_lasso, alpha=0.6, s=30, label='Lasso', c='C1')
    ax6.scatter(y_test, y_pred_ridge, alpha=0.6, s=30, label='Ridge', c='C2')
    ymin = min(y_test.min(), y_pred_lr.min(), y_pred_lasso.min(), y_pred_ridge.min())
    ymax = max(y_test.max(), y_pred_lr.max(), y_pred_lasso.max(), y_pred_ridge.max())
    # 取所有数据的最小/最大值为坐标轴范围，使 y=x 线能画满
    ax6.plot([ymin, ymax], [ymin, ymax], 'k--', label='y=x')
    # 画对角线：预测=真实时落在线上，越准越贴近这条线
    ax6.set_xlabel('True y')
    ax6.set_ylabel('Predicted y')
    ax6.set_title('Predicted vs True (test set)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal')
    # 横纵轴等比例，对角线呈 45°
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, FIG_PREFIX + 'fig6_pred_vs_true.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ----- 图 7：误差热力图（样本 × 模型，颜色表示绝对误差大小） -----
    err_matrix = np.column_stack([
        np.abs(resid_lr),
        np.abs(resid_lasso),
        np.abs(resid_ridge)
    ])
    # 三列分别对应三个模型的每个样本的绝对残差；行=样本，列=模型
    fig7, ax7 = plt.subplots(figsize=(5, 8))
    im = ax7.imshow(err_matrix, aspect='auto', cmap='YlOrRd')
    # 热力图：YlOrRd 为黄-橙-红，越红误差越大
    ax7.set_xticks([0, 1, 2])
    ax7.set_xticklabels(['No reg', 'Lasso', 'Ridge'])
    ax7.set_ylabel('Sample index')
    ax7.set_xlabel('Model')
    ax7.set_title('Absolute error (sample × model)')
    plt.colorbar(im, ax=ax7, label='Absolute error')
    # 右侧颜色条，表示数值与颜色的对应
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, FIG_PREFIX + 'fig7_error_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ----- 图 8：累积误差分布（横轴=绝对误差，纵轴=累积比例） -----
    def cumulative_error_proportion(abs_errors):
        # 输入：一维数组，每个样本的绝对误差
        sorted_err = np.sort(abs_errors)
        # 将误差从小到大排序
        n = len(sorted_err)
        return sorted_err, np.arange(1, n + 1) / n
        # 返回：排序后的误差、以及对应的累积比例 (1/n, 2/n, ..., 1)

    e_lr, p_lr = cumulative_error_proportion(np.abs(resid_lr))
    e_lasso, p_lasso = cumulative_error_proportion(np.abs(resid_lasso))
    e_ridge, p_ridge = cumulative_error_proportion(np.abs(resid_ridge))
    # 三组 (误差, 累积比例)，用于画阶梯线

    fig8, ax8 = plt.subplots(figsize=(7, 5))
    ax8.step(e_lr, p_lr, where='post', label='No reg')
    # step：阶梯图，横轴误差、纵轴累积比例
    ax8.step(e_lasso, p_lasso, where='post', label='Lasso')
    ax8.step(e_ridge, p_ridge, where='post', label='Ridge')
    ax8.set_xlabel('Absolute error')
    ax8.set_ylabel('Cumulative proportion')
    ax8.set_title('Cumulative error distribution (test set)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, FIG_PREFIX + 'fig8_cumulative_error.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ----- 图 9：前 20 个测试样本的预测值条形图（True / No reg / Lasso / Ridge 并排） -----
    n_bar = min(n_sample_bar, n_test)
    # 若测试集不足 20 个则全部画
    x9 = np.arange(n_bar)
    w9 = 0.2
    y_true_bar = y_test[:n_bar]
    y_pred_lr_bar = y_pred_lr[:n_bar]
    y_pred_lasso_bar = y_pred_lasso[:n_bar]
    y_pred_ridge_bar = y_pred_ridge[:n_bar]
    # 取前 n_bar 个样本的真实值和三种预测值

    fig9, ax9 = plt.subplots(figsize=(12, 5))
    ax9.bar(x9 - 1.5 * w9, y_true_bar, width=w9, label='True')
    ax9.bar(x9 - 0.5 * w9, y_pred_lr_bar, width=w9, label='No reg')
    ax9.bar(x9 + 0.5 * w9, y_pred_lasso_bar, width=w9, label='Lasso')
    ax9.bar(x9 + 1.5 * w9, y_pred_ridge_bar, width=w9, label='Ridge')
    # 每个样本位置画四根小柱，便于逐样本对比
    ax9.set_xlabel('Sample index')
    ax9.set_ylabel('Value')
    ax9.set_title('Predicted vs True (first {} samples, test set)'.format(n_bar))
    ax9.set_xticks(x9)
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, FIG_PREFIX + 'fig9_sample_predictions.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    #  模块三之「控制台输出」：最佳 lambda、非零系数个数、简要分析
    # =========================================================================

    thresh = 1e-6
    # 系数绝对值大于此阈值视为“非零”（Lasso 会把很多系数压成接近 0）
    nz_lasso = np.sum(np.abs(coef_lasso_best) > thresh)
    # 统计 Lasso 最佳模型中非零系数的个数
    nz_ridge = np.sum(np.abs(coef_ridge_best) > thresh)
    # Ridge 通常很少严格为 0，所以非零个数一般接近 280

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
    print('Figures saved to:', OUT_DIR)
    print('  ' + FIG_PREFIX + 'fig1_mse_vs_lambda.png ~ ' + FIG_PREFIX + 'fig9_sample_predictions.png')


# =============================================================================
#  程序入口：只有“直接运行本脚本”时才执行 main()
# =============================================================================
#  if __name__ == '__main__' 含义：
#  当你在命令行执行  python run_blogfeedback.py  时，__name__ 会被设为 '__main__'，于是执行 main()
#  当别的脚本  import run_blogfeedback  时，__name__ 是模块名，不会自动执行 main()，避免误跑
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    main()
