# Student T

$$
 \psi(z) = \frac{\partial \log{\Gamma(z)}}{\partial z}
$$

$$
  {|\Sigma|}^{\frac{1}{2}} = |\mathbf{L}\mathbf{L}^\top|^{\frac{1}{2}} = (|\mathbf{L}||\mathbf{L}^\top|)^{\frac{1}{2}} = | \mathbf{L} |
$$

$$
  |\mathbf{L}| = |\mathbf{L}^\top|
$$

## Univariate

$$
  \begin{aligned}
    \log{p(x \mid \mu, \sigma, \nu)} =& \log{\Gamma\left(\frac{\nu+1}{2}\right)} \\
    &- \log{\Gamma \left(\frac{\nu}{2} \right)} - \frac{1}{2} \log{\nu} - \frac{1}{2} \log{\pi} - \log{\sigma} \\
    & - \frac{\nu + 1}{2}  \log{\left(1 + \frac{(x - \mu)^2} {\nu \sigma^2}\right)}
  \end{aligned}
$$

### Diff x

$$
  \begin{aligned}
    \frac{\partial \log{p(x \mid \mu, \sigma, \nu)}}{\partial x}
    &= - \frac{\nu + 1}{2} \left(1 + \frac{(x - \mu)^2}{\nu \sigma^2}\right)^{-1} \frac{2x - 2\mu}{\nu \sigma^2} \\
    &= -(\nu + 1) \left(\frac{\nu \sigma^2 + (x - \mu)^2}{\nu \sigma^2}\right)^{-1} \frac{x - \mu}{\nu \sigma^2} \\
    &= -\frac{(\nu + 1)(x - \mu)}{\nu \sigma^2 +(x - \mu)^2}
  \end{aligned}
$$

### Diff mu

$$
  \begin{aligned}
    \frac{\partial \log{p(x \mid \mu, \sigma, \nu)}}{\partial \mu}
    &= - \frac{\nu + 1}{2} \left(1 + \frac{(x - \mu)^2}{\nu \sigma^2}\right)^{-1} \left( -\frac{2x - 2\mu}{\nu \sigma^2} \right) \\
    &= (\nu + 1) \left(\frac{\nu \sigma^2 + (x - \mu)^2}{\nu \sigma^2}\right)^{-1} \frac{x - \mu}{\nu \sigma^2} \\
    &= \frac{(\nu + 1)(x - \mu)}{\nu \sigma^2 +(x - \mu)^2}
  \end{aligned}
$$

### Diff sigma

$$
  \begin{aligned}
    \frac{\partial \log{p(x \mid \mu, \sigma, \nu)}}{\partial \sigma}
    &= -\frac{1}{\sigma} - \frac{\nu + 1}{2} \left(1 + \frac{(x - \mu)^2}{\nu \sigma^2}\right)^{-1} \left( -\frac{2 (x - \mu)^2}{\nu \sigma^3} \right) \\
    &= (\nu + 1) \left(\frac{\nu \sigma^2 + (x - \mu)^2}{\nu \sigma^2}\right)^{-1} \frac{(x - \mu)^2}{\nu \sigma^3} -\frac{1}{\sigma} \\
    &= \frac{(\nu + 1)(x - \mu)^2 \sigma}{\nu \sigma^2 +(x - \mu)^2} -\frac{1}{\sigma}
  \end{aligned}
$$

### Diff nu

$$
  \begin{aligned}
    \frac{\partial \log{p(x \mid \mu, \sigma, \nu)}}{\partial \nu}
    =& \frac{1}{2} \psi \left(\frac{\nu + 1}{2} \right) - \frac{1}{2} \psi \left(\frac{\nu}{2}\right) - \frac{1}{2 \nu} \\
    &+ \frac{\nu + 1}{2} \left(1 + \frac{(x - \mu)^2}{\nu \sigma^2}\right)^{-1} \frac{(x - \mu)^2}{\nu^2 \sigma^2} \\
    &- \frac{1}{2} \log \left(1 + \frac{(x - \mu)^2}{\nu \sigma^2} \right)
  \end{aligned}
$$

## Mutivariate

$$ \mathbf{L} \mathbf{L}^{ \top} = \bm{\Sigma} $$
$$ d = (\mathbf{x} - \bm{\mu})^\top \bm{\Sigma}^{-1} (\mathbf{x} - \bm{\mu}) $$

$$
  \begin{aligned}
    \log{p(\mathbf{x} \mid \bm{\mu}, \mathbf{L}, \nu)} =& \log{\Gamma\left(\frac{\nu + n}{2}\right)} \\
    &- \log{\Gamma \left(\frac{\nu}{2} \right)} - \frac{n}{2} \log{\nu} - \frac{n}{2} \log{\pi} - \log{|\mathbf{L}|} \\
    & - \frac{\nu + n}{2}  \log{\left(1 + \frac{1}{\nu}(\mathbf{x} - \bm{\mu})^\top \bm{\Sigma}^{-1} (\mathbf{x} - \bm{\mu})\right)}
  \end{aligned}
$$

### Diff x

$$
  \begin{aligned}
    \frac{\partial \log{p(\mathbf{x} \mid \bm{\mu}, \mathbf{L}, \nu)}}{\partial \mathbf{x}}
    &= - \frac{\nu + n}{2} (1 + d)^{-1} \frac{2}{\nu}(\mathbf{x} - \bm{\mu})^\top \bm{\Sigma}^{-1} \\
    &= - \frac{\nu + n}{\nu} (1 + d)^{-1} (\mathbf{x} - \bm{\mu})^\top \bm{\Sigma}^{-1}
  \end{aligned}
$$

### Diff mu

$$
  \begin{aligned}
    \frac{\partial \log{p(\mathbf{x} \mid \bm{\mu}, \mathbf{L}, \nu)}}{\partial \bm{\mu}}
    &= - \frac{\nu + n}{2} (1 + d)^{-1} \left(-\frac{2}{\nu}(\mathbf{x} - \bm{\mu})^\top \bm{\Sigma}^{-1} \right) \\
    &= \frac{\nu + n}{\nu} (1 + d)^{-1} (\mathbf{x} - \bm{\mu})^\top \bm{\Sigma}^{-1}
  \end{aligned}
$$

### Diff lsigma

$$
  \begin{aligned}
    \frac{\partial \log{p(\mathbf{x} \mid \bm{\mu}, \mathbf{L}, \nu)}}{\partial \mathbf{L}}
    &= - \frac{\nu + n}{2} \left(1 + \frac{1}{\nu} d\right)^{-1} \left(-\frac{2}{\nu} (\mathbf{x} - \bm{\mu}) M (\mathbf{x} - \bm{\mu})^\top \right) \\
    &= \frac{\nu + n}{\nu} (1 + \frac{1}{\nu} d)^{-1} (\mathbf{x} - \bm{\mu})^\top M (\mathbf{x} - \bm{\mu})
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

### Diff nu

$$
  \begin{aligned}
    \frac{\partial \log{p(\mathbf{x} \mid \bm{\mu}, \mathbf{L}, \nu)}}{\partial \nu}
    =& \frac{1}{2} \psi \left(\frac{\nu + n}{2} \right) - \frac{n}{2 \nu} - \frac{1}{2} \psi \left(\frac{\nu}{2}\right) \\
    &+ \frac{\nu + n}{2} \left(1 + \frac{1}{\nu} d \right)^{-1} \left(-\frac{1}{\nu^2}d \right) \\
    &- \frac{1}{2} \log \left(1 + \frac{1}{\nu} d \right) \\
    =& \frac{1}{2} \left( \psi \left(\frac{\nu + n}{2} \right) - \frac{n}{\nu} - \psi \left(\frac{\nu}{2}\right) - \frac{(\nu + n) d}{\nu^2} \left(1 + \frac{1}{\nu} d \right)^{-1} -  \log \left(1 + \frac{1}{\nu} d\right) \right)
  \end{aligned}
$$
