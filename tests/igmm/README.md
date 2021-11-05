# infinite gaussian mixture model

$$
  \begin{aligned}
    y_t \mid f(x_t), \sigma & \sim \mathcal{N}(f(x_t), \sigma^2) \\
    x_t \mid \theta_{s_i}, & \sim \mathcal{N}(\bm{\mu}_{s_i}, \bm{\Sigma}_{s_i}) \\
    \mathbf{s} \mid \alpha, d & \sim \mathcal{PYP}(\alpha, d) \\
    \theta_k & \sim G_0 \\
    \theta_k & = (\bm{\mu}_k, \bm{\Sigma}_k)
  \end{aligned}
$$
