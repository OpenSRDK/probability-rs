# Normal

## Univariate

$$
  \log{p(x \mid \mu, \sigma)} = -\frac{1}{2} \log{2\pi} -\frac{1}{2\sigma^{2}} (x-\mu)^2
$$

### Diff x

$$
  \frac{\partial \log{p(x \mid \mu, \sigma)}}{\partial x} = -\frac{1}{\sigma^2} (x-\mu)
$$

### Diff mu

$$
  \frac{\partial \log{p(x \mid \mu, \sigma)}}{\partial \mu} = \frac{1}{\sigma^2} (x-\mu)
$$

### Diff sigma

$$
  \frac{\partial \log{p(x \mid \mu, \sigma)}}{\partial \sigma} = \frac{1}{\sigma^3} (x-\mu)^2
$$

## Mutivariate

$$ \mathbf{L} \mathbf{L}^{ \top} = \bm{\Sigma} $$

$$
  \log{p(\mathbf{x} \mid \bm{\mu}, \mathbf{L})} = -\frac{1}{2}n\log{2\pi} -\frac{1}{2}\log{|\Sigma|}-\frac{1}{2}(\mathbf{x}-\bm{\mu})^T\Sigma^{-1}(\mathbf{x}-\bm{\mu})
$$

### Diff x

$$
  \begin{aligned}
    \frac{\partial \log{p(\mathbf{x} \mid \bm{\mu}, \mathbf{L})}}{\partial \mathbf{x}}
    & = -\frac{1}{2} \Sigma^{-1} (2\mathbf{x}-2\bm{\mu})
    \\ & = -(\mathbf{L}{ \mathbf{L}^\top})^{-1} (\mathbf{x}-\bm{\mu})
  \end{aligned}
$$

### Diff mu

$$
  \begin{aligned}
    \frac{\partial \log{p(\mathbf{x} \mid \bm{\mu}, \mathbf{L})}}{\partial \bm{\mu}}
    & = \frac{1}{2} \Sigma^{-1} (2\mathbf{x}-2\bm{\mu})
    \\ & = (\mathbf{L}{ \mathbf{L}^\top})^{-1} (\mathbf{x}-\bm{\mu})
  \end{aligned}
$$

### Diff lsigma

$$
  \begin{aligned}
    \frac{\partial \log{p(\mathbf{x} \mid \bm{\mu}, \mathbf{L})}}{\partial \mathbf{L}}
    & = -\frac{1}{2}(\Sigma^T)^{-1} - \frac{1}{2}(-\Sigma^T\Sigma^T) (\mathbf{x}-\bm{\mu}) (\mathbf{x}-\bm{\mu})^T
    \\ & = \frac{1}{2} (\mathbf{L} \mathbf{L}^{\top} \mathbf{L} \mathbf{L}^{\top} (\mathbf{x}-\bm{\mu}) (\mathbf{x}-\bm{\mu})^{\top} - (\mathbf{L} \mathbf{L}^{\top})^{-1})
  \end{aligned}
$$
