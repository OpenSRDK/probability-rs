# infinite mixture linear regression

$$
\begin{aligned}
y_t | f(x_t), \sigma & \sim \mathcal{N}(f(x_t),\sigma^2) \\
f(x_t) | s_t, \bm{\alpha}, \bm{\beta} & \sim \mathcal{D}(\alpha_{s_t} + \beta_{s_t} x_t) \\
\mathbf{s} & \sim \mathcal{PYP}(\alpha, d) \\
(\alpha_k, \beta_k) & \sim G_0
\end{aligned}
$$
