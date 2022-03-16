# `independent_array_joint`

$$
  p(\mathbf{y} \mid \mathbf{x}) = \prod_{i=1}^n p(y_i \mid x_i)
$$

Log

$$
 \log{p(\mathbf{y} \mid \mathbf{x})} = \log {p(y_1 \mid x_1)} + \cdots + \log{p(y_i \mid x_1)}+ \cdots + \log {p(y_n \mid x_1)}
$$

Log Value Difference

$$
  \frac{\partial \log{p(\mathbf{y} \mid \mathbf{x})}}{\partial \mathbf{y}}
  = \left( \frac{\partial \log {p(y_1 \mid x_1)}}{\partial y_1} + \cdots + \frac{\partial \log {p(y_n \mid x_n)}}{\partial y_1},\cdots , \frac{\partial \log {p(y_1 \mid x_1)}}{\partial y_i} + \cdots + \frac{\partial \log {p(y_n \mid x_n)}}{\partial y_i},\cdots ,\frac{\partial \log {p(y_1 \mid x_1)}}{\partial y_n} + \cdots + \frac{\partial \log {p(y_n \mid x_n)}}{\partial y_n}, \right) 

  \\

  = \left( \frac{\partial \log {p(y_1 \mid x_1)}}{\partial y_1} + 0 + \cdots,\cdots , \cdots + 0 + \frac{\partial \log {p(y_i \mid x_i)}}{\partial y_i} + 0 + \cdots ,\cdots ,0 + \cdots + \frac{\partial \log {p(y_n \mid x_n)}}{\partial y_n} \right) 

  \\

  = \left( \frac{\partial \log {p(y_1 \mid x_1)}}{\partial y_1} ,\cdots , \frac{\partial \log {p(y_i \mid x_i)}}{\partial y_i} ,\cdots ,\frac{\partial \log {p(\mathbf{y}_n \mid x_n)}}{\partial y_n} \right)
$$

Log Condition Difference

$$
  \frac{\partial \log{p(\mathbf{y} \mid \mathbf{x})}}{\partial \mathbf{x}}
  = \left( \frac{\partial \log {p(y_1 \mid x_1)}}{\partial x_1} + \cdots + \frac{\partial \log {p(y_n \mid x_n)}}{\partial x_1},\cdots , \frac{\partial \log {p(y_1 \mid x_1)}}{\partial x_i} + \cdots + \frac{\partial \log {p(y_n \mid x_n)}}{\partial x_i},\cdots ,\frac{\partial \log {p(y_1 \mid x_1)}}{\partial x_n} + \cdots + \frac{\partial \log {p(y_n \mid x_n)}}{\partial x_n}, \right) 

  \\

  = \left( \frac{\partial \log {p(y_1 \mid x_1)}}{\partial x_1} + 0 + \cdots,\cdots , \cdots + 0 + \frac{\partial \log {p(y_i \mid x_i)}}{\partial x_i} + 0 + \cdots ,\cdots ,0 + \cdots + \frac{\partial \log {p(y_n \mid x_n)}}{\partial x_n} \right) 

  \\
  =\left( \frac{\partial \log {p(y_1 \mid x_1)}}{\partial x_1} ,\cdots , \frac{\partial \log {p(y_i \mid x_i)}}{\partial x_i} ,\cdots ,\frac{\partial \log {p(\mathbf{y}_n \mid x_n)}}{\partial x_n} \right)
$$
