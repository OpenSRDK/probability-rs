$$
\begin{aligned}
  \hat{Q}(z_i, h_i, a_i; \theta_i, \theta_{-i}^i, \mathbf{w}_Q) &= \hat{V}(z_i, h_i; \theta_i, \theta_{-i}^i, \mathbf{w}_V) + f(z_i, h_i, a_i; \theta_i, \theta_{-i}^i, \mathbf{w}_Q) \\
  f(z_i, h_i, a_i; \theta_i, \theta_{-i}^i, \mathbf{w}_Q) &= \mathbf{w}_Q^\top (\nabla_{\theta_i} \log{p_i(a_i \mid z_i, h_i, \theta_i, \theta_{-i}^i)})
\end{aligned}
$$

という形で近似行動価値関数を定義すると、

$$
\begin{aligned}
  \mathbf{w}_V' &= \mathbf{w}_V + \alpha_V \delta \nabla_{\mathbf{w}_V} \hat{V}(z_i, h_i) \\
  \mathbf{w}_Q' &= \left( \mathbf{I} - \alpha \mathbf{G} \bm{\psi}^\top \bm{\psi} \right) \mathbf{w}_Q - \alpha \delta \mathbf{G} \mathbf{e}^\delta \\
  \delta &= r + \gamma \hat{V}_i (z_i', h_i') - \hat{V}_i (z_i, h_i) \\
  \mathbf{G} &= \mathbf{I} - \frac{l}{1 + l \|\mathbf{e}\|^2} \mathbf{e}^\top \mathbf{e} \\
  \bm{\psi} &= \nabla_{\theta_i} \log{p(a_i \mid z_i, h_i, \theta_i, \theta_{-i}^i)} \\
  \mathbf{e} &= \mathbf{e}^\delta = \mathbf{e}^f = \bm{\psi}
\end{aligned}
$$

という形で$\mathbf{w}_{Q_i}$を更新していけば、

$$
  \theta_i' = \theta_i + \beta w_{Q_i}
$$

と極めて単純な形で方策パラメータを更新できることが知られている。
ただし$\alpha_V, \alpha_Q, \beta$と$ l > \alpha$はハイパーパラメータである。