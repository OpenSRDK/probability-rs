# valued

`valued` method is like the mapping $\mathcal{F}$ such that

$$
  \mathcal{F}: p(a \mid b) \mapsto p(f(c) \mid b)
$$

with mapping $f$ such that

$$
  f: C \mapsto A
$$

Log

$$
 \log{p(\mathbf{a} \mid f(c))}
$$

Log Value Difference

（ use `ConditionDifferentiableConditionedDistribution` ）

$$
 \frac{\partial \log {p(f(c) \mid \mathbf{b})}}{\partial c} =  \frac{\partial \log {p(f(c) \mid \mathbf{b})}}{\partial f(c)} \times \frac{\partial f(c)}{\partial c}
$$

`ConditionDifferentiableConditionedDistribution` has 

Valued Distribution : $ p(f(c) \mid \mathbf{b})$ ,

and 

Differentiated Value ( not $\log$ ) : $\frac{\partial f(c)}{\partial c}$ ( Matrix )

Log Condition Difference

$$
  \frac{\partial \log{p(f(c) \mid \mathbf{b})}}{\partial \mathbf{b}}
$$
