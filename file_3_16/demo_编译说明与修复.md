# demo.tex 渲染修复说明

## 一、为何渲染失败

1. **Beamer 中在 `\begin{column}` 里不能使用 `\begin{figure}`**  
   LaTeX 报错：`Not allowed in LR mode`。column 是横向盒子，figure 是浮动体，不能嵌套。

2. **图片路径**  
   所有图片使用 `figs/xxx.png`，**必须在 `file_3_16` 目录下编译**，否则会找不到图（例如在项目根目录编译会找成 `ml_reg_exp/figs/`）。

3. **数学模式里的中文**  
   第 70 行公式里的「价格」「面积」等未用 `\text{...}`，xelatex 会报 Missing character，建议改为 `\text{价格}` 等。

---

## 二、需要你手动改的内容

请**关闭**可能正在打开 `demo.tex` 的编辑器/预览后，对 `demo.tex` 做如下修改。

### 1. 在导言区增加图片路径（约第 9 行后）

在 `\usepackage{graphicx} % 用于插入图片` 下面加一行：

```latex
\graphicspath{{figs/}}
```

这样写 `\includegraphics{fig1_mse_vs_lambda.png}` 即可，不必写 `figs/` 前缀（可选，见下）。

### 2. 第 70 行：数学公式里的中文

把：

```latex
$$价格 = \beta_1 \cdot 面积 + \beta_2 \cdot 地段 + \beta_3 \cdot 楼层 + \beta_4 \cdot 星座 \dots$$
```

改成：

```latex
$$\text{价格} = \beta_1 \cdot \text{面积} + \beta_2 \cdot \text{地段} + \beta_3 \cdot \text{楼层} + \beta_4 \cdot \text{星座} \dots$$
```

### 3. 第 280–287 行：MSE 随 λ 变化那一页

把整段：

```latex
\begin{frame}{测试集 MSE 随正则化强度 $\lambda$ 的变化}
    \begin{figure}[h]
        \centering
        \includegraphics[height=0.65\textheight]{figs/fig1_mse_vs_lambda.png}
        \caption{Lasso（蓝色）在某个最佳 $\lambda$ 处达到了远低于 Ridge（橙色）的测试均方误差。}
    \end{figure}
\end{frame}
```

换成：

```latex
\begin{frame}{测试集 MSE 随正则化强度 $\lambda$ 的变化}
    \centering
    \includegraphics[height=0.65\textheight]{figs/fig1_mse_vs_lambda.png}\\
    \footnotesize Lasso（蓝）在某个最佳 $\lambda$ 处达到远低于 Ridge（橙）的测试 MSE。
\end{frame}
```

### 4. 第 289–304 行：「正则化路径对比」那一页

把从 `\begin{frame}{正则化路径对比：稀疏性 vs 整体收缩}` 到 `\end{frame}` 的整段，替换为：

```latex
\begin{frame}{正则化路径对比：稀疏性 vs 整体收缩}
    \begin{columns}[T]
        \begin{column}{0.48\textwidth}
            \centering
            \includegraphics[width=\linewidth]{figs/fig2_lasso_coef_path.png}\\
            \footnotesize \textbf{Lasso：}随 $\lambda$ 增大，无关系数被压成 0。
        \end{column}
        \begin{column}{0.48\textwidth}
            \centering
            \includegraphics[width=\linewidth]{figs/fig3_ridge_coef_path.png}\\
            \footnotesize \textbf{Ridge：}系数整体变小，但不为 0。
        \end{column}
    \end{columns}
\end{frame}
```

### 5. 第 306–312 行：「最佳 λ 下的最终系数分布对比」

把：

```latex
\begin{frame}{最佳 $\lambda$ 下的最终系数分布对比}
    \begin{figure}[h]
        \centering
        \includegraphics[height=0.6\textheight]{figs/fig4_coef_compare.png}
        \caption{在各自最佳的 $\lambda$ 下，Lasso（橙点）完美贴合了真实稀疏特征（黑星），而 Ridge（绿点）的系数依然杂乱且密集存在。}
    \end{figure}
\end{frame}
```

换成：

```latex
\begin{frame}{最佳 $\lambda$ 下的最终系数分布对比}
    \centering
    \includegraphics[height=0.6\textheight]{figs/fig4_coef_compare.png}\\
    \footnotesize Lasso 贴合真实稀疏特征，Ridge 系数密集非零。
\end{frame}
```

### 6. 第 314–328 行：「模型预测性能深度评估」

把整段（含 `\begin{columns}` 和两个 `\begin{figure}`）换成：

```latex
\begin{frame}{模型预测性能深度评估}
    \begin{columns}[T]
        \begin{column}{0.48\textwidth}
            \centering
            \includegraphics[width=\linewidth]{figs/fig5_residual_boxplot.png}\\
            \footnotesize 残差箱线图：Lasso 更集中于 0 附近。
        \end{column}
        \begin{column}{0.48\textwidth}
            \centering
            \includegraphics[width=\linewidth]{figs/fig6_pred_vs_true.png}\\
            \footnotesize 预测 vs 真实：Lasso 更贴对角线。
        \end{column}
    \end{columns}
\end{frame}
```

### 7. 第 330–351 行：「辅助实验：BlogFeedback」

把整段（含 `\begin{columns}` 和两个 `\begin{figure}`）换成：

```latex
\begin{frame}{辅助实验：真实数据分布中的教训（BlogFeedback）}
    \textbf{实验背景：} BlogFeedback 真实数据（280 维，预测评论数），$y$ 分布极偏、长尾。
    \begin{columns}[T]
        \begin{column}{0.48\textwidth}
            \centering
            \includegraphics[width=\linewidth]{figs/exp2_ugly_heatmap.png}\\
            \footnotesize 误差热力图：极端样本主导损失。
        \end{column}
        \begin{column}{0.48\textwidth}
            \centering
            \includegraphics[width=\linewidth]{figs/exp2_ugly_cumulative.png}\\
            \footnotesize 累积误差「丑图」：形态不均衡。
        \end{column}
    \end{columns}
    \vspace{0.2cm}
    \footnotesize \textbf{教训：} 真实场景需数据预处理（如 $\log(1+y)$）与鲁棒指标。
\end{frame}
```

---

## 三、编译方式

1. 打开终端，进入 **`file_3_16`** 目录（和 `demo.tex`、`figs` 同级）。
2. 执行（输出到 `out` 目录）：
   ```bash
   xelatex -output-directory=out demo.tex
   ```
   再执行一次（生成目录等）：
   ```bash
   xelatex -output-directory=out demo.tex
   ```
3. PDF 路径：`file_3_16/out/demo.pdf`。

**注意**：不要在项目根目录 `ml_reg_exp` 下直接执行 `xelatex file_3_16/demo.tex`，否则 `figs/` 会从根目录找，会找不到图。

---

## 四、文件夹结构检查

应保证：

- `file_3_16/demo.tex` 存在  
- `file_3_16/figs/` 存在，且其下有：  
  `fig1_mse_vs_lambda.png`, `fig2_lasso_coef_path.png`, `fig3_ridge_coef_path.png`, `fig4_coef_compare.png`, `fig5_residual_boxplot.png`, `fig6_pred_vs_true.png`, `exp2_ugly_heatmap.png`, `exp2_ugly_cumulative.png`

若 `figs` 被放在别处，请要么移回 `file_3_16/figs/`，要么在 `demo.tex` 里把所有 `figs/xxx.png` 改成相对 `demo.tex` 所在目录的正确路径。
