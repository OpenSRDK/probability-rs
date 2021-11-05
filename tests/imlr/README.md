# infinite mixture linear regression

$$
  \begin{aligned}
    y_t \mid f(x_t), \sigma & \sim \mathcal{N}(f(x_t),\sigma^2) \\
    f(x_t) \mid \theta_{s_t} & \sim \mathcal{D}(\alpha_{s_t} + \beta_{s_t} x_t) \\
    \mathbf{s} \mid \alpha, d & \sim \mathcal{PYP}(\alpha, d) \\
    \theta_k & \sim G_0 \\
    \theta_k & = (\alpha_k, \beta_k)
  \end{aligned}
$$
