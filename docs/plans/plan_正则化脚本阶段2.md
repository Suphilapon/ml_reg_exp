---
name: 正则化对比脚本第二阶段
overview: 在新工作区中实现完整的正则化对比脚本：沿用相同的数据生成与模型逻辑，图表全部使用英文与 lambda 符号，MSE 曲线采用均匀采样的 lambda，并新增残差箱线/小提琴图、预测值 vs 真实值散点图、误差热图、累积误差分布图、抽样预测条形图共 5 类可视化。
todos: []
isProject: false
---

# 正则化对比脚本第二阶段（自包含计划）

本计划面向**新工作区**（含 git），脚本可从零实现。下文先复述与第一阶段相同的核心逻辑（交接），再说明第二阶段改动与新增图表。

---

## 一、与第一阶段相同的逻辑（交接）

### 1. 数据生成（固定种子 42）

- 样本数 `m=150`，特征数 `n=50`。
- 真实参数 `beta_true`：长度 50，前 5 个为 `[1.5, -2.0, 0.8, 3.2, -1.7]`，其余为 0。
- 特征矩阵 `X = np.random.randn(m, n)`；标签 `y = X @ beta_true + np.random.randn(m) * 0.5`。
- `train_test_split(X, y, test_size=0.3, random_state=42)` 得到训练/测试集。

### 2. 数据预处理

- 仅对特征 `X` 使用 `StandardScaler`：`fit_transform` 训练集，`transform` 测试集；`y` 不标准化。

### 3. 模型训练与记录

- **无正则化**：`LinearRegression().fit(X_train_scaled, y_train)`，记录测试集 MSE 与 `coef_`。
- **Lasso / Ridge**：在若干 `lambda`（对应 sklearn 的 `alpha` 参数）下循环训练，每个 lambda 记录测试 MSE 与 `coef_`。
- 根据测试 MSE 选出 Lasso、Ridge 的**最佳 lambda**，用于系数对比与非零个数统计。
- 系数路径可仍用较密的 lambda 序列（如 `np.logspace`）以得到平滑曲线；**最佳 lambda** 从用于 MSE 曲线的 lambda 列表中选取。

### 4. 文本输出

- 在最佳 lambda 下，打印 Lasso / Ridge 的非零系数个数（`|系数| > 1e-6`）。
- 简要分析：Lasso 是否将无关特征压成 0，Ridge 是否仅缩小系数而不为 0。

---

## 二、第二阶段改动

### 1. 图表与变量命名：仅英文 + 使用 lambda

- 所有图标题、轴标签、图例、注释**仅使用英文**，避免字体/编码乱码（可不设置中文字体）。
- 脚本内变量与图表中的正则化强度**统一称为 lambda**（例如 `lambda_list`、横轴标签 `lambda`）。调用 sklearn 时仍使用参数 `alpha=...`（sklearn API 不变），即：`Lasso(alpha=lam)`、`Ridge(alpha=lam)`，其中 `lam` 为“lambda”的取值。

### 2. MSE vs lambda：均匀采样

- 使用**均匀采样**的 lambda：例如 `lambda_list = np.linspace(0.001, 10, 30)` 或类似区间与点数（避免仅用对数间隔的 5 个点）。
- 横轴为 **lambda**；纵轴为**测试集 MSE**。
- 图中包含：Lasso 曲线、Ridge 曲线、无正则化水平线（无 lambda 维度）。若横轴范围较宽，可考虑对横轴取对数（`plt.xscale('log')`）以更好展示；计划要求是“lambda 均匀采样”，曲线数据点均匀即可，横轴用线性或对数由实现自定。

---

## 三、原有 4 类图（保持结构，仅改为英文 + lambda）


| 图   | 内容                         | 注意                                                                 |
| --- | -------------------------- | ------------------------------------------------------------------ |
| 图1  | MSE vs lambda（lambda 均匀采样） | 横轴标签 `lambda`，纵轴 "Test MSE"；三条线：Lasso, Ridge, No regularization。   |
| 图2  | Lasso 系数路径（前 10 个特征）       | 横轴 lambda（可用对数尺度），纵轴系数；图例 "Feature 0" … "Feature 9"。               |
| 图3  | Ridge 系数路径（前 10 个特征）       | 同图2。                                                               |
| 图4  | 最佳 lambda 下系数对比（前 10 个特征）  | 柱状图：No reg / Lasso / Ridge + 真实系数水平线；轴与图例英文，如 "True coefficients"。 |


---

## 四、新增 5 类图（实现要点）

### 图5：残差分布箱线图 / 小提琴图

- **目的**：比较三模型在测试集上预测残差的分布（集中趋势、离散、异常值）。
- **数据**：残差 = `y_test - y_pred`，分别对无正则化、Lasso（最佳 lambda）、Ridge（最佳 lambda）计算，得到三组一维数组。
- **实现**：`plt.boxplot([resid_lr, resid_lasso, resid_ridge], labels=['No reg', 'Lasso', 'Ridge'])`，或 `sns.violinplot`（若引入 seaborn）。标题/轴用英文，如 "Residual distribution (test set)"。
- **解读**：箱体越窄、越对称、中位数越接近 0、离群越少，说明误差越稳定；正则化若使中位数更近 0、离群更少，可说明抑制过拟合。

### 图6：预测值 vs 真实值散点图（三模型叠加）

- **目的**：对比预测准确性与一致性。
- **数据**：同一张图上画三组点：`(y_test, y_pred_lr)`、`(y_test, y_pred_lasso)`、`(y_test, y_pred_ridge)`，不同颜色/标记区分。
- **实现**：`plt.scatter(y_test, y_pred_*, ...)`，并画参考线 `y=x`（如 `plt.plot([ymin, ymax], [ymin, ymax], 'k--')`）。图例 "No reg", "Lasso", "Ridge", "y=x"。轴标签 "True y" / "Predicted y"，标题如 "Predicted vs True (test set)"。
- **解读**：点越贴近 y=x 越准；不同颜色点群可看出高估/低估与系统性偏差。

### 图7：误差热图（样本 × 模型）

- **目的**：查看每个测试样本在三模型下的绝对误差，识别“困难样本”。
- **数据**：矩阵行 = 测试样本索引（可取前 N 个或全部），列 = 三模型（No reg, Lasso, Ridge），值为该样本在该模型下的 `|y_test - y_pred|`。
- **实现**：`plt.imshow(err_matrix, aspect='auto', cmap='...')`，横轴为模型，纵轴为样本；`plt.colorbar(label='Absolute error')`；轴刻度标签英文。
- **解读**：深色表示误差大；某行整行深色表示该样本在各模型都难；某列在部分样本上更浅表示该模型在这些样本上更优。

### 图8：累积误差分布图

- **目的**：比较“误差整体分布”，看哪个模型在更多样本上达到给定精度。
- **数据**：对每个模型，取测试集上绝对误差 `|y_test - y_pred|`，排序后计算累积比例（横轴为误差阈值或排序后的误差值，纵轴为“绝对误差 ≤ 当前阈值”的样本比例）。
- **实现**：对每个模型的绝对误差排序，如 `np.sort(np.abs(residual))`，纵轴为 `np.arange(1, n+1)/n` 或类似；在同一图中画三条阶梯/折线，图例 "No reg", "Lasso", "Ridge"。横轴 "Absolute error"，纵轴 "Cumulative proportion" 或 "Fraction of samples"。
- **解读**：曲线越高表示在相同误差阈值下，达到该精度的样本比例越大；若 Lasso 曲线在 Ridge 上方，表示 Lasso 在多数样本上误差更小。

### 图9：预测误差条形图（抽样展示）

- **目的**：对少量样本展示真实值与三模型预测值的对比。
- **数据**：取测试集前 20 个样本（或 15），每个样本 4 个值：真实值、无正则化预测、Lasso 预测、Ridge 预测。
- **实现**：横轴为样本索引（或编号），纵轴为数值；每组 4 根并排柱子（真实值 + 三模型），不同颜色区分，图例 "True", "No reg", "Lasso", "Ridge"。可用 `np.arange(20)` 为 x，`width` 控制柱宽避免重叠。
- **解读**：哪根柱子越接近“真实值”柱子，该模型在该样本上越准。

---

## 五、文件与依赖

- **单脚本**：如 `regularization_compare.py`，包含：数据生成 → 预处理 → 训练与记录（含均匀采样的 lambda 列表、最佳 lambda、系数路径用密 lambda）→ 原有 4 图（英文 + lambda）→ 新增 5 图 → 控制台输出非零系数个数与简要分析。
- **依赖**：`numpy`、`sklearn`（`LinearRegression`、`Lasso`、`Ridge`、`StandardScaler`、`train_test_split`、`mean_squared_error`）、`matplotlib`；图5 若用小提琴图则加 `seaborn`。
- **注释**：关键步骤保留中文或英文注释，保证可读与可维护。

---

## 六、实现顺序建议

1. 数据生成、预处理、模型训练（使用 `lambda_list` 均匀采样，并保留用于系数路径的密 lambda）。
2. 计算并保存三组预测与残差（无正则化、Lasso 最佳、Ridge 最佳）。
3. 绘制图1～4（英文 + lambda）。
4. 绘制图5～9（残差箱线/小提琴、Pred vs True、误差热图、累积误差、抽样条形图）。
5. 打印最佳 lambda 下非零系数个数与简要分析。

按此计划在新工作区从零实现即可得到与第一阶段一致的核心逻辑，并完成全部第二阶段改动与新增图表。