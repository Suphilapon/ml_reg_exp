---
name: 线性回归正则化对比脚本
overview: 在现有空文件 `regularization_compare.py` 中实现一个完整的 Python 脚本，生成合成数据、训练无正则化/ Lasso / Ridge 三种线性模型，并绘制 MSE 曲线、系数路径和系数对比图，最后输出非零系数统计与简要分析。
todos: []
isProject: false
---

# 线性回归正则化对比脚本实现计划

## 目标文件

- 主脚本：[regularization_compare.py](C:\Users\YuYan\Desktop\newtest\regularization_compare.py)（当前为空，将在此文件中实现全部逻辑）

## 1. 依赖与结构

- **依赖**：`numpy`、`sklearn`（`LinearRegression`、`Lasso`、`Ridge`、`StandardScaler`、`train_test_split`、`mean_squared_error`）、`matplotlib.pyplot`。
- **结构**：单文件脚本，按“数据生成 → 预处理 → 训练与记录 → 可视化 → 结果输出”顺序组织，关键步骤加中文注释。

## 2. 数据生成（固定种子 42）

- `m=150`，`n=50`。
- `beta_true`：长度 50，前 5 个为 `[1.5, -2.0, 0.8, 3.2, -1.7]`，其余为 0。
- `X`：`np.random.randn(m, n)`。
- `y = X @ beta_true + noise`，噪声 `np.random.randn(m) * 0.5`。
- `train_test_split(X, y, test_size=0.3, random_state=42)` 得到训练/测试集。

## 3. 数据预处理

- 用 `StandardScaler` 拟合并变换训练集 `X_train`，再对 `X_test` 只做 `transform`，保证测试集不参与拟合，避免信息泄露。

## 4. 模型训练与指标记录

- **无正则化**：`LinearRegression().fit(X_train_scaled, y_train)`，在测试集上算一次 MSE，无 alpha 维度。
- **Lasso**：对 `alpha_list = [0.001, 0.01, 0.1, 1, 10]` 循环训练，每个 alpha 保存：测试 MSE、拟合后的 `coef`_（用于系数路径和最佳 alpha 下的系数）。
- **Ridge**：同上 alpha 列表，同样保存每个 alpha 的测试 MSE 和 `coef`_。

根据测试集 MSE 选出 Lasso 和 Ridge 的**最佳 alpha**（MSE 最小的那个），用于后续“最佳 alpha 下”的系数对比和非零系数统计。

## 5. 可视化（matplotlib，4 张图）

- **图1：MSE vs alpha（对数横轴）**
  - 横轴：`alpha`（用 `plt.xscale('log')`）。
  - 纵轴：测试集 MSE。
  - 三条线：无正则化（水平线，无 alpha）、Lasso、Ridge；图例、标题、坐标轴标签清晰。
- **图2：Lasso 系数路径**
  - 横轴：alpha（对数尺度）。
  - 纵轴：系数值。
  - 只画前 10 个特征的系数随 alpha 的变化（或所有非零系数变化，二者择一即可，计划中可写“前 10 个特征”以简化）。
  - 每条线一个特征，可标注或图例区分。
- **图3：Ridge 系数路径**
  - 与图2结构相同，横轴 alpha（对数）、纵轴系数，展示前 10 个特征的系数路径。
- **图4：最佳 alpha 下系数对比（前 10 个特征）**
  - 柱状图：横轴为特征索引 0～9，纵轴为系数值。
  - 三组柱子：无正则化、Lasso（最佳 alpha）、Ridge（最佳 alpha）。
  - 用水平线（`axhline`）标出真实非零系数：前 5 个为 1.5, -2.0, 0.8, 3.2, -1.7，便于对比稀疏性与收缩效果。

所有图需设置标题、坐标轴标签，必要时 `plt.tight_layout()` 或 `plt.subplots` 排版，最后 `plt.show()`（或保存图片，按你偏好二选一在计划里说明）。

## 6. 结果输出（print）

- 在**最佳 alpha** 下：
  - Lasso 非零系数个数：`np.sum(np.abs(coef) > 1e-6)`。
  - Ridge 非零系数个数：同上判定。
- 简要文字分析（可用 print 多行）：
  - Lasso 是否将无关特征压成 0（理想情况应接近 5 个非零）。
  - Ridge 是否主要缩小系数但几乎不变成 0（非零个数接近 50）。

## 7. 实现要点与注意事项

- 标准化只在 `X` 上做，`y` 不标准化，以便系数和真实 `beta_true` 可比。
- 系数路径需要在对数尺度 alpha 上足够细才能平滑：可用更密的 alpha 列表（如 `np.logspace(-3, 1, 50)`）仅用于画路径，而“最佳 alpha”仍从 [0.001, 0.01, 0.1, 1, 10] 中选。
- 图4 若用 `x = np.arange(10)` 与 `width` 错开三组柱子，避免重叠。
- 保证脚本从头到尾可单独运行（无交互输入），固定 `random_state=42` 保证可重复。

## 8. 交付物

- 一份完整、带注释、可运行的 `regularization_compare.py`，包含：数据生成、预处理、三种模型训练与 MSE/系数记录、四张图、非零系数统计与简要分析输出。

