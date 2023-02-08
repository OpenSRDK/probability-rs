# `independent_joint`

$$
  p(a, b \mid c) = p(a \mid c) p(b \mid c)
$$

TODO:

$$
  p(a, b \mid c, d) = p(a \mid c) p(b \mid d)
$$

Log

$$
 \log{p(a, b \mid c)} = \log{p(a \mid c)} + \log{p(b \mid c)}
$$

Log Value Difference

$$
  \frac{\partial \log{p(a, b \mid c)}}{\partial (a, b)} 
  =\left( \frac{\partial \log{p(a \mid c)}}{\partial a} + \frac{\partial \log{p(b \mid c)}}{\partial a}, \, \frac{\partial \log{p(a \mid c)}}{\partial b} + \frac{\partial \log{p(b \mid c)}}{\partial b} \right) \\
  =\left( \frac{\partial \log{p(a \mid c)}}{\partial a} + 0 , \, \frac{\partial \log{p(b \mid c)}}{\partial b} + 0 \right) \\
  =\left( \frac{\partial \log{p(a \mid c)}}{\partial a}, \, \frac{\partial \log{p(b \mid c)}}{\partial b} \right)
$$

Log Condition Difference

$$
  \frac{\partial \log{p(a, b \mid c)}}{\partial c} 
  =\frac{\partial \log{p(a \mid c)}}{\partial c} + \frac{\partial \log{p(b \mid c)}}{\partial c}
$$
