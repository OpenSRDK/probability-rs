# conditioned

`conditioned` method is like the mapping $\mathcal{F}$ such that

$$
  \mathcal{F}: p(a \mid b) \mapsto p(a \mid f(c))
$$

with mapping $f$ such that

$$
  f: C \mapsto B
$$

Log Value Difference

$$
  d\, log ( p(\mathbf{a} \mid f(c))/da
$$

Log Condition Difference

（ use `ConditionDifferentiableConditionedDistribution` ）

$$
  d\, log ( p(\mathbf{a} \mid f(c))/d\, c =  d\, log ( p(\mathbf{a} \mid f(c))/d\, f(c) \, * \, d\, f(c) / d\, c
$$

`ConditionDifferentiableConditionedDistribution` has 

Conditioned Distribution : $ p(\mathbf{a} \mid f(c)$ ,

and 

Differentiated Condition ( not $log$ ) : $d\, f(c) / d\, c$
