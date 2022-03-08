# Student T

## Univariate

$$
  \log{p(x \mid \mu, \sigma)} = -\frac{\nu + 1}{2} (2 \log{\frac{x - \mu}{\sigma}}  - \log{\nu})
$$

### Diff x

$$
  \frac{\partial \log{p(x \mid \mu, \sigma)}}{\partial x} = - \frac{\nu + 1}{2} 2 \frac{\sigma}{x - \mu} \frac{1}{\sigma} = - \frac{\nu + 1}{x - \mu}
$$

### Diff mu

$$
  \frac{\partial \log{p(x \mid \mu, \sigma)}}{\partial \mu} = - \frac{\nu + 1}{2} 2 \frac{\sigma}{x - \mu} (- \frac{1}{\sigma}) = \frac{\nu + 1}{x - \mu}
$$

### Diff sigma

$$
  \frac{\partial \log{p(x \mid \mu, \sigma)}}{\partial \sigma} = - \frac{\nu + 1}{2} 2 \frac{\sigma}{x - \mu} (- \frac{x - \mu}{\sigma^2}) = \frac{\nu + 1}{\sigma}
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
