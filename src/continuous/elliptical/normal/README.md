# Normal

## Mutivariate

$$\mathbf{L} \mathbf{L}^{ \top} = \bm{\Sigma}$$

$$
\log{p(\mathbf{x} \mid \bm{\mu}, \mathbf{L})}=-\frac{1}{2}n\log{2\pi} -\frac{1}{2}\log{|\Sigma|}-\frac{1}{2}(\mathbf{x}-\bm{\mu})^T\Sigma^{-1}(\mathbf{x}-\bm{\mu})
$$

### mu

$$
\begin{align*}
\frac{\partial \log{p(\mathbf{x} \mid \bm{\mu}, \mathbf{L})}}{\partial \bm{\mu}}
&=-\frac{1}{2}\Sigma^{-1}(2\mathbf{x}-2\bm{\mu})
\\ &= -(\mathbf{L}{ \mathbf{L}^\top})^{-1}(\mathbf{x}-\bm{\mu})
\end{align*}
$$

### lsigma

$$
\begin{align*}
\frac{\partial \log{p(\mathbf{x} \mid \bm{\mu}, \mathbf{L})}}{\partial \mathbf{L}}
&= -\frac{1}{2}(\Sigma^T)^{-1}-\frac{1}{2}(-\Sigma^T\Sigma^T)(\mathbf{x}-\bm{\mu})(\mathbf{x}-\bm{\mu})^T
\\ &= \frac{1}{2}(\mathbf{L} \mathbf{L}^{\top}\mathbf{L} \mathbf{L}^{\top}(\mathbf{x}-\bm{\mu})(\mathbf{x}-\bm{\mu})^{\top}-(\mathbf{L} \mathbf{L}^{\top})^{-1})
\end{align*}
$$
