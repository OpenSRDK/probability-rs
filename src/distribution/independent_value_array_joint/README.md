# `independent_value_array_joint`

$$
  p(\mathbf{y} \mid x) = \prod_{i=1}^n p(y_i \mid x)
$$


Log Value Difference

$$
  \left( \frac{\partial log ( p(\mathbf{y_1} \mid x_1))}{\partial x_1} , ...\,, \frac{\partial log ( p(\mathbf{y_i} \mid x_i))}{\partial x_i} ,...\,,\frac{\partial log ( p(\mathbf{y_n} \mid x_n))}{\partial x_n} \right)
$$

Log Condition Difference

$$
 \frac{\partial log ( p(\mathbf{y_1} \mid x_1))}{\partial \theta}  + ... +\, \frac{\partial log ( p(\mathbf{y_i} \mid x_i))}{\partial \theta}+...+\,d \,\frac{\partial log ( p(\mathbf{y_n} \mid x_n))}{\partial \theta}
$$
