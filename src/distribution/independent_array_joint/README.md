# `independent_array_joint`

$$
  p(\mathbf{y} \mid \mathbf{x}) = \prod_{i=1}^n p(y_i \mid x_i)
$$

## Log Value Difference

$$
  \left( \frac{\partial \log {p(\mathbf{y}_1 \mid \mathbf{x}_1)}}{\partial \mathbf{x}_1} ,\cdots , \frac{\partial \log {p(\mathbf{y}_i \mid \mathbf{x}_i)}}{\partial \mathbf{x}_i} ,\cdots ,\frac{\partial \log {p(\mathbf{y}_n \mid \mathbf{x}_n)}}{\partial \mathbf{x}_n} \right)
$$

## Log Condition Difference

$$
 \frac{\partial \log {p(\mathbf{y}_1 \mid \mathbf{x}_1)}}{\partial \bm{\theta}}  + \cdots + \frac{\partial \log{p(\mathbf{y}_i \mid \mathbf{x}_i)}}{\partial \bm{\theta}}+ \cdots + \frac{\partial \log {p(\mathbf{y}_n \mid \mathbf{x}_n)}}{\partial \bm{\theta}}
$$
