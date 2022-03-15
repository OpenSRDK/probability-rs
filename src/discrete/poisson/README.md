# Poisson

## Probability mass function

$$
  p^{m(x \mid \lambda)} = P(X = x \mid \lambda) = \frac{\lambda^x}{x!}\exp(-\lambda)
$$

$$
  \log{P(X = x \mid \lambda)} = x \log{\lambda} - \log{x!} - \lambda
$$

## Diff x

離散確率分布は value による微分不可

## Diff lambda

$$
  \frac{\partial \log{P(X = x \mid \lambda)}}{\partial \lambda} = \frac{x}{\lambda} - 1
$$
