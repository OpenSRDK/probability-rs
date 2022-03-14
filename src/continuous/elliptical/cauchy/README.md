# Cauchye

$$
  {|\Sigma|}^{\frac{1}{2}} = |\mathbf{L}\mathbf{L}^\top|^{\frac{1}{2}} = (|\mathbf{L}||\mathbf{L}^\top|)^{\frac{1}{2}} = | \mathbf{L} |
$$

$$
  |\mathbf{L}| = |\mathbf{L}^\top|
$$

## Univariate

$$
  \begin{aligned}
    \log{p(x \mid \mu, \sigma)} =&  - \log{\pi} - \log{\sigma} \\
    & - \log{\left(1 + \frac{(x - \mu)^2} {\sigma^2}\right)}
  \end{aligned}
$$

### Diff x

$$
  \begin{aligned}
    \frac{\partial \log{p(x \mid \mu, \sigma)}}{\partial x}
    &= - \left(1 + \frac{(x - \mu)^2}{\sigma^2}\right)^{-1} \frac{2x - 2\mu}{\sigma^2} \\
    &= - \left(\frac{\sigma^2 + (x - \mu)^2}{\sigma^2}\right)^{-1} 2 \frac{x - \mu}{\sigma^2} \\
    &= -\frac{2 (x - \mu)}{\sigma^2 +(x - \mu)^2}
  \end{aligned}
$$

### Diff mu

$$
  \begin{aligned}
    \frac{\partial \log{p(x \mid \mu, \sigma)}}{\partial \mu}
    &= - \left(1 + \frac{(x - \mu)^2}{\sigma^2}\right)^{-1} \left( - \frac{2x - 2\mu}{\sigma^2} \right) \\
    &= \left(\frac{\sigma^2 + (x - \mu)^2}{\sigma^2}\right)^{-1} 2 \frac{x - \mu}{\sigma^2} \\
    &= \frac{2 (x - \mu)}{\sigma^2 +(x - \mu)^2}
  \end{aligned}
$$

### Diff sigma

$$
  \begin{aligned}
    \frac{\partial \log{p(x \mid \mu, \sigma)}}{\partial \sigma}
    &= -\frac{1}{\sigma} - \left(1 + \frac{(x - \mu)^2}{\sigma^2}\right)^{-1} \left( -\frac{2 (x - \mu)^2}{\sigma^3} \right) \\
    &= \left(\frac{\sigma^2 + (x - \mu)^2}{\sigma^2}\right)^{-1} \frac{2 (x - \mu)^2}{\sigma^3} -\frac{1}{\sigma} \\
    &= \frac{2}{\sigma} \frac{(x - \mu)^2}{\sigma^2 +(x - \mu)^2} -\frac{1}{\sigma}
  \end{aligned}
$$

## Mutivariate

$$ \mathbf{L} \mathbf{L}^{ \top} = \bm{\Sigma} $$
$$ d = (\mathbf{x} - \bm{\mu})^\top \bm{\Sigma}^{-1} (\mathbf{x} - \bm{\mu}) $$

$$
  \begin{aligned}
    \log{p(\mathbf{x} \mid \bm{\mu}, \mathbf{L})} =& \log{\Gamma\left(\frac{1 + n}{2}\right)} \\
    &- \log{\Gamma \left(\frac{1}{2} \right)} - \frac{n}{2} \log{\pi} - \log{|\mathbf{L}|} \\
    & - \frac{1 + n}{2}  \log{\left(1 + (\mathbf{x} - \bm{\mu})^\top \bm{\Sigma}^{-1} (\mathbf{x} - \bm{\mu})\right)}
  \end{aligned}
$$

### Diff x

$$
  \begin{aligned}
    \frac{\partial \log{p(\mathbf{x} \mid \bm{\mu}, \mathbf{L})}}{\partial \mathbf{x}}
    &= - \frac{1 + n}{2} (1 + d)^{-1} 2(\mathbf{x} - \bm{\mu})^\top \bm{\Sigma}^{-1} \\
    &= - (1 + n) (1 + d)^{-1} (\mathbf{x} - \bm{\mu})^\top \bm{\Sigma}^{-1}
  \end{aligned}
$$

### Diff mu

$$
  \begin{aligned}
    \frac{\partial \log{p(\mathbf{x} \mid \bm{\mu}, \mathbf{L})}}{\partial \bm{\mu}}
    &= - \frac{1 + n}{2} (1 + d)^{-1} \left(-2(\mathbf{x} - \bm{\mu})^\top \bm{\Sigma}^{-1} \right) \\
    &= (1 + n) (1 + d)^{-1} (\mathbf{x} - \bm{\mu})^\top \bm{\Sigma}^{-1}
  \end{aligned}
$$

### Diff lsigma

$$
  \begin{aligned}
    \frac{\partial \log{p(\mathbf{x} \mid \bm{\mu}, \mathbf{L})}}{\partial \mathbf{L}}
    &= - \frac{1 + n}{2} \left(1 + d\right)^{-1} \left(-2 (\mathbf{x} - \bm{\mu})^\top M (\mathbf{x} - \bm{\mu}) \right) \\
    &= (1 + n) (1 + d)^{-1} (\mathbf{x} - \bm{\mu})^\top M (\mathbf{x} - \bm{\mu})
  \end{aligned}
$$

$$
  k_{ji}k_{ij} =
  \begin{cases}
\frac{1}{l_{ij^2}} & \ \text{if} \ i = j \\
0 & \ \text{others}
  \end{cases}
$$

$$
  M = [l_{ab}^{-3}]_{ab}
$$
