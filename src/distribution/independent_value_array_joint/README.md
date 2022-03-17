# `independent_value_array_joint`

$$
  p(\mathbf{y} \mid x) = \prod_{i=1}^n p(y_i \mid x)
$$


Log

$$
 \log{p(\mathbf{y} \mid x)} = \log {p(y_1 \mid x)} + \cdots + \log{p(y_i \mid x)}+ \cdots + \log {p(y_n \mid x)}
$$


Log Value Difference

$$
  \frac{\partial \log{p(\mathbf{y} \mid x)}}{\partial \mathbf{y}}
  = \begin{bmatrix} \frac{\partial \log {p(y_1 \mid x)}}{\partial y_1} + \cdots + \frac{\partial \log {p(y_n \mid x)}}{\partial y_1} \\ \vdots  \\ \frac{\partial \log {p(y_1 \mid x)}}{\partial y_i} + \cdots + \frac{\partial \log {p(y_n \mid x)}}{\partial y_i} \\ \vdots  \\ \frac{\partial \log {p(y_1 \mid x)}}{\partial y_n} + \cdots + \frac{\partial \log {p(y_n \mid x)}}{\partial y_n} \end{bmatrix}

  \\

  = \begin{bmatrix} \frac{\partial \log {p(y_1 \mid x)}}{\partial y_1} + 0 + \cdots \\ \vdots  \\ \cdots + 0 + \frac{\partial \log {p(y_i \mid x)}}{\partial y_i} + 0 + \cdots \\ \vdots  \\ 0 + \cdots + \frac{\partial \log {p(y_n \mid x)}}{\partial y_n} \end{bmatrix}

  \\

  =\begin{bmatrix} \frac{\partial \log {p(y_1 \mid x)}}{\partial y_1} \\ \vdots  \\ \frac{\partial \log {p(y_i \mid x)}}{\partial y_i} \\ \vdots  \\ \frac{\partial \log {p(y_n \mid x)}}{\partial y_n} \end{bmatrix}
$$

Log Condition Difference

$$
  \frac{\partial \log{p(\mathbf{y} \mid x)}}{\partial x}
  = \frac{\partial \log {p(y_1 \mid x)}}{\partial x}  + \cdots + \frac{\partial \log{p(y_i \mid x)}}{\partial x}+ \cdots + \frac{\partial \log {p(y_n \mid x)}}{\partial x}
$$
