# sparse gp

$$
  \begin{aligned}
    y_i \mid f(\mathbf{x}_i), \varepsilon_i                                & \sim \mathcal{D}(f(\mathbf{x}_i) + \varepsilon_i)                                                                                                                       \\
    f(\mathbf{x}_i) \mid \mathbf{x}_i, \mathbf{v}, \mathbf{U}, \bm{\theta} & \sim \mathcal{N}(\mathbf{k}_{\mathbf{x}_i\mathbf{U}} \mathbf{K}_{\mathbf{U}\mathbf{U}}^{-1} \mathbf{v}, k_{\mathbf{x}_i \mathbf{x}_i} - q_{\mathbf{x}_i \mathbf{x}_i} ) \\
    \mathbf{v} \mid \mathbf{U}                                             & \sim \mathcal{N}(\mathbf{0}, \mathbf{K}_{\mathbf{U}\mathbf{U}})                                                                                                         \\
    \varepsilon_i                                                          & \sim \mathcal{N}(0, \sigma^2)
  \end{aligned}
$$

$$
  \mathbf{f}^* \mid \mathbf{X}, \mathbf{X}^*, \mathbf{y} \sim \mathcal{N}(\bm{\mu}, \bm{\Sigma})
$$

$$
\begin{aligned}
  \mathbf{Q}_{\mathbf{A}\mathbf{B}}
  & = \mathbf{k}_{\mathbf{A}\mathbf{U}} \mathbf{K}_{\mathbf{U}\mathbf{U}}^{-1} \mathbf{k}_{\mathbf{U}\mathbf{B}} \\
  \bm{\Omega} &= \mathrm{diag}(\mathbf{K}_{\mathbf{X}\mathbf{X}} - \mathbf{Q}_{\mathbf{X}\mathbf{X}}) + \sigma^2 \mathbf{I} \\
  \mathbf{S} & = \mathbf{K}_{\mathbf{U}\mathbf{U}} + \mathbf{K}_{\mathbf{U}\mathbf{X}} \bm{\Omega}^{-1} \mathbf{K}_{\mathbf{X}\mathbf{U}} \\
  \bm{\mu} & = \mathbf{K}_{*\mathbf{U}} S^{-1} \mathbf{K}_{\mathbf{U}\mathbf{X}} \bm{\Omega} \mathbf{y} \\
  \bm{\Sigma} & = \mathbf{K}_{**} - \mathbf{Q}_{**} + \mathbf{K}_{*\mathbf{U}} S^{-1} \mathbf{K}_{\mathbf{U}*}
\end{aligned}
$$
