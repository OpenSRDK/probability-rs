# Normal

## Mutivariate

$$\mathbf{L} \mathbf{L}^{ \top} = \bm{\Sigma}$$

$$
\log{p(\mathbf{x} \mid \bm{\mu}, \mathbf{L})}=-\frac{1}{2}n\log{2\pi} -\frac{1}{2}\log{|\Sigma|}-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)
$$

### mu

$$
\frac{\partial \log{p(\mathbf{x} \mid \bm{\mu}, \mathbf{L})}}{\partial \bm{\mu}}
=-\frac{1}{2}\Sigma^{-1}(2x-2\mu)= -(\mathbf{L}{ \mathbf{L}^\top})^{-1}(\mathbf{x}-\bm{\mu})
$$

### lsigma

$$
\frac{\partial \log{p(\mathbf{x} \mid \bm{\mu}, \mathbf{L})}}{\partial \mathbf{L}}
= -\frac{1}{2}(\Sigma^T)^{-1}-\frac{1}{2}(-\Sigma^T\Sigma^T)(x-\mu)(x-\mu)^T
= \frac{1}{2}(\mathbf{L} \mathbf{L}^{\top}\mathbf{L} \mathbf{L}^{\top}(x-\mu)(x-\mu)^{\top}-(\mathbf{L} \mathbf{L}^{\top})^{-1})
$$
