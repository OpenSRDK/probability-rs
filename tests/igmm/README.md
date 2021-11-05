# infinite gaussian mixture model

$$
\begin{aligned}
y_t | f(x_t), \sigma & \sim \mathcal{N}(f(x_t),\sigma^2) \\
x_t | s_i, \bm{\mu}, \bm{\Sigma}, & \sim \mathcal{N}(\mu_{s_i},\Sigma_{s_i}) \\
\mathbf{s} & \sim \mathcal{PYP}(\alpha, d) \\
(\mu_{s_i},\Sigma_{s_i}) & \sim G_0
\end{aligned}
$$
