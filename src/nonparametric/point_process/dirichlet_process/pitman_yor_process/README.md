# Pitman Yor Process

## `sample_s`

For $i$,

$$
  p(s_i \mid x_i, \{s\}_{\setminus i}) = \frac{p(x_i \mid s_i) p(s_i \mid \{s\}_{\setminus i})}{p(x_i)} =  \frac{p(x_i \mid s_i) p(s_i \mid \{s\}_{\setminus i})}{\sum_{s_i} p(x_i \mid s_i)}
$$

## `sample_theta`

For parallel $k$,

$$
  p(\theta_k \mid \{x_i : s_i = k\}) \propto \prod_{x \in \{x_i : s_i = k\}} p(x \mid \theta_k) p(\theta_k)
$$
