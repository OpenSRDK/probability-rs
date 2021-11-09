# exact gp

$$
\begin{aligned}
  y_i \mid f(\mathbf{x}_i), \varepsilon_i & \sim \mathcal{D}(f(\mathbf{x}_i) + \varepsilon) \\
  \mathbf{f} \mid \mathbf{X} & \sim \mathcal{N}(\mathbf{0}, \mathbf{K}_{\mathbf{X}\mathbf{X}}) \\
  \varepsilon_i & \sim \mathcal{N}(0, \sigma^2)
\end{aligned}
$$

$$
  \mathbf{f}^* \mid \mathbf{X}, \mathbf{X}^*, \mathbf{y} \sim \mathcal{N}(\bm{\mu}, \bm{\Sigma})
$$

$$
\begin{aligned}
  \bm{\mu} & = \mathbf{K}_{*\mathbf{X}} (\mathbf{K}_{\mathbf{X}\mathbf{X}} + \sigma^2 \mathbf{I})^{-1} \mathbf{y} \\
  \bm{\Sigma} & = \mathbf{K}_{**} - \mathbf{K}_{*\mathbf{X}} (\mathbf{K}_{\mathbf{X}\mathbf{X}} + \sigma^2 \mathbf{I})^{-1} \mathbf{K}_{\mathbf{X}*}
\end{aligned}
$$
