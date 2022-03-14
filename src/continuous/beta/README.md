# Beta

## Univariate

$$
  \begin{aligned}
    \log{p(x \mid \alpha, \beta)} =& (\alpha - 1) \log{x} + (\beta - 1) \log{(1 - x)} \\
    &- \log{\Beta (\alpha, \beta)} \\
  \end{aligned}
$$

$$
\begin{aligned}
  \Beta(\alpha, \beta) &= \int_0^1 x^{\alpha - 1} (1 - x)^{\beta - 1}dx \\
  &=\frac{\Gamma(\alpha) + \Gamma(\beta)}{\Gamma(\alpha + \beta)}
\end{aligned}
$$

$$
 \psi(z) = \frac{\partial \log{\Gamma(z)}}{\partial z}
$$

### Diff x

$$
  \begin{aligned}
    \frac{\partial \log{p(x \mid \alpha, \beta)}}{\partial x}
    &= (\alpha - 1) \frac{1}{x}+(\beta - 1) \left( - \frac{1}{1 - x} \right) \\
    &= \frac{\alpha - 1}{x} - \frac{\beta -1}{1 - x}
  \end{aligned}
$$

### Diff alpha

$$
  \begin{aligned}
    \frac{\partial \log{p(x \mid \alpha, \beta)}}{\partial \alpha}
    &= \log{x} - \frac{\partial \log{\Beta(\alpha, \beta)}}{\partial \alpha} \\
    &= \log{x} - \frac{\partial \log{\Gamma(\alpha)}}{\partial \alpha} + \frac{\partial \log{\Gamma(\alpha + \beta)}}{\partial \alpha} \\
    &= \log{x} - \psi(\alpha) + \psi(\alpha + \beta)
  \end{aligned}
$$

## Diff beta

$$
  \begin{aligned}
    \frac{\partial \log{p(x \mid \alpha, \beta)}}{\partial \beta}
    &= \log{(1 -x)} - \frac{\partial \log{\Beta(\alpha, \beta)}}{\partial \beta} \\
    &= \log{(1 -x)} - \frac{\partial \log{\Gamma(\beta)}}{\partial \beta} + \frac{\partial \log{\Gamma(\alpha + \beta)}}{\partial \beta} \\
    &= \log{(1 -x)} - \psi(\beta) + \psi(\alpha + \beta)
  \end{aligned}
$$
