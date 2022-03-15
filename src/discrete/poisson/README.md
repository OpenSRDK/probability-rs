# Poisson

## Probability density function

$$
  p(x|\lambda) = \frac{\lambda^x}{x!}\exp(-\lambda)
$$

$$
  \log{p(x|\lambda)} = x \log{\lambda} - \log{x!} - \lambda
$$

## Diff x

$$
  \frac{\partial \log{p(x|\lambda)}}{\partial x} = \log{\lambda} -\frac{\partial \log{x!}}{\partial x}
$$

x の階乗は連続関数ではないため、微分不可

## Diff lambda

$$
  \frac{\partial \log{p(x|\lambda)}}{\partial \lambda} = \frac{x}{\lambda} - 1
$$
