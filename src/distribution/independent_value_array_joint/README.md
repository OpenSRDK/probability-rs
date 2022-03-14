# `independent_value_array_joint`

$$
  p(\mathbf{y} \mid x) = \prod_{i=1}^n p(y_i \mid x)
$$


Log Value Difference

$$
  \left( \frac{\partial \log ( p(\mathbf{y}_1 \mid x_1))}{\partial x_1} ,\cdot \cdot \cdot , \frac{\partial \log ( p(\mathbf{y}_i \mid x_i))}{\partial x_i} ,\cdot \cdot \cdot ,\frac{\partial \log ( p(\mathbf{y}_n \mid x_n))}{\partial x_n} \right)
$$

Log Condition Difference

$$
 \frac{\partial \log ( p(\mathbf{y}_1 \mid x_1))}{\partial \theta}  + \cdot \cdot \cdot + \frac{\partial \log ( p(\mathbf{y}_i \mid x_i))}{\partial \theta}+ \cdot \cdot \cdot + \frac{\partial \log ( p(\mathbf{y}_n \mid x_n))}{\partial \theta}
$$
