# conditioned

`conditioned` method is like the mapping $\mathcal{F}$ such that

$$
  \mathcal{F}: p(a \mid b) \mapsto p(a \mid f(c))
$$

with mapping $f$ such that

$$
  f: C \mapsto B
$$

Log

$$
 \log{p(\mathbf{a} \mid f(c))}
$$

Log Value Difference

$$
  \frac{\partial \log{p(\mathbf{a} \mid f(c))}}{\partial \mathbf{a}}
$$

Log Condition Difference

（ use `ConditionDifferentiableConditionedDistribution` ）

$$
 \frac{\partial \log {p(\mathbf{a} \mid f(c))}}{\partial c} =  \frac{\partial \log {p(\mathbf{a} \mid f(c))}}{\partial f(c)} \times \frac{\partial f(c)}{\partial c}
$$

`ConditionDifferentiableConditionedDistribution` has 

Conditioned Distribution : $ p(\mathbf{a} \mid f(c))$ ,

and 

Differentiated Condition ( not $\log$ ) : $\frac{\partial f(c)}{\partial c}$
