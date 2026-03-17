# ml_reg_exp — Norms in Machine Learning Regularization

Course project for **Matrix Computation and Optimization**: *Norms in L1/L2 Regularization for Linear Regression*. The repo contains theory notes, numerical experiments (synthetic & real data), and a Beamer presentation (LaTeX).

## Topics

- **Vector/Matrix norms** and norm equivalence in finite dimensions  
- **L1 (Lasso)** vs **L2 (Ridge)** regularization: sparsity, geometry, and optimization  
- **Frobenius (F) norm** for matrices: numerical stability (rotation invariance, smoothness, condition number)

## Repository structure

| Path | Description |
|------|-------------|
| `file_3_16/` | Beamer slides: `demo.tex` (XeLaTeX), `figs/` (figures), `out/demo.pdf` |
| `exp1_manufact_data/` | Exp1: low‑dimensional synthetic data, Lasso/Ridge/no-reg comparison (`regularization_compare.py`) |
| `exp2_real_data/` | Exp2: UCI BlogFeedback (280 features); `test1_no_Log/` (raw target), `test2_log/` (log-target) |
| `exp3_manufact_data/` | Exp3: high‑dim sparse synthetic (150×200, 10 informative); main figures for the slides (`test3.py`) |
| `角度1_范数意义与稀疏性.md` | Angle 1: norm meaning and sparsity (CN) |
| `角度2_范数等价性证明.md` | Angle 2: norm equivalence proof (CN) |
| `角度4_F范数数值稳定性.md` | Angle 4: F-norm and numerical stability (CN) |
| `实验与汇报说明.md` | Experiment & presentation guide (CN) |
| `docs/plans/` | Planning notes for regularization scripts |

## Experiments

- **Exp1** (`exp1_manufact_data/`): 150×50 synthetic data, 5 non-zero coefficients; MSE vs λ, coefficient paths, residuals, pred vs true.  
- **Exp2** (`exp2_real_data/`): BlogFeedback; no-log and log-target runs; shows need for preprocessing and robust metrics.  
- **Exp3** (`exp3_manufact_data/`): 150×200, 10 informative features; Lasso recovers sparsity and achieves lower test MSE; produces `fig1_mse_vs_lambda.png` … `fig6_pred_vs_true.png` used in the slides.

## Building the slides

From the project root, in `file_3_16/`:

```bash
cd file_3_16
xelatex -output-directory=out demo.tex
xelatex -output-directory=out demo.tex
```

---


# ml_reg_exp — 机器学习正则化中的范数应用

**矩阵计算与优化** 课程题目 1：线性回归中 L1/L2 正则化的范数意义、等价性证明、数值实验与 F 范数稳定性分析。本仓库包含理论推导笔记、数值实验脚本（合成数据与真实数据）及课程汇报用 Beamer 幻灯片（LaTeX）。

## 涉及知识点

- **向量/矩阵范数** 及有限维范数等价性  
- **L1（Lasso）与 L2（Ridge）正则化**：稀疏性、几何直观与优化性质  
- **Frobenius（F）范数**：酉不变性、平滑性、条件数改善等数值稳定性

## 仓库结构

| 路径 | 说明 |
|------|------|
| `file_3_16/` | 汇报用 Beamer：`demo.tex`（XeLaTeX）、`figs/`（图片）、`out/demo.pdf` |
| `exp1_manufact_data/` | 实验 1：低维合成数据，Lasso/Ridge/无正则对比（`regularization_compare.py`） |
| `exp2_real_data/` | 实验 2：UCI BlogFeedback（280 维）；`test1_no_Log/`（原始目标）、`test2_log/`（对目标取对数） |
| `exp3_manufact_data/` | 实验 3：高维稀疏合成（150×200，10 个有效特征）；汇报主图来源（`test3.py`） |
| `角度1_范数意义与稀疏性.md` | 角度一：范数意义与稀疏性 |
| `角度2_范数等价性证明.md` | 角度二：范数等价性证明 |
| `角度4_F范数数值稳定性.md` | 角度四：F 范数数值稳定性 |
| `实验与汇报说明.md` | 实验与汇报结构、用图说明及脚本位置 |
| `docs/plans/` | 正则化脚本实现与阶段计划 |

## 实验概览

- **实验 1**（`exp1_manufact_data/`）：150×50 合成数据、5 个非零系数；MSE 随 λ、系数路径、残差、预测 vs 真实等图。  
- **实验 2**（`exp2_real_data/`）：BlogFeedback 真实数据；未取对数与取对数两种设定，用“丑图”说明预处理与评估指标的重要性。  
- **实验 3**（`exp3_manufact_data/`）：150×200、10 个有效特征；Lasso 产生稀疏解、测试 MSE 更低，产出汇报用 6 张图（`fig1_mse_vs_lambda.png`～`fig6_pred_vs_true.png`）。

## 编译幻灯片

在项目根目录下进入 `file_3_16/` 后执行：

```bash
cd file_3_16
xelatex -output-directory=out demo.tex
xelatex -output-directory=out demo.tex
