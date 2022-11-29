An implementation of rotated monogenic decomposition of 2-D signal $`g`$.

Let $`\theta \in [0, 2 \pi)`$. Denote the Riesz kernel $`r_l`$ and Riesz transform $`R_l`$ as  
```math
r_l({\bf x}) = c_l \dfrac{x_l}{\Vert {\bf x} \Vert^3}
```
and
```math
R_l g({\bf x}) = \tilde{g}^{(l)}({\bf x}) = (r_l ** g)({\bf x}).
```

Denote the rotated signal as
```math
\tilde{g}_{-\theta}({\bf x}) = g({\bf R}_\theta {\bf x}).
```

The Risez tranform of the rotated signal is
```math
\tilde{g}^{(l)}_{-\theta}({\bf x}) = R_l \tilde{g}_{-\theta}({\bf x}).
```

The rotated monogenic decomposition is 
```math
\tilde{g}^{\pm}_{-\theta, \theta}({\bf x}) = g({\bf x}) \pm \bigg( {\bf i} \tilde{g}^{(1)}_{-\theta}({\bf R}_{-\theta} {\bf x}) + {\bf j} \tilde{g}^{(2)}_{-\theta}({\bf R}_{-\theta} {\bf x} ) \bigg).
```

Before running the code, load packages
```
using CUDA
using ScatteringTransform
using MonogenicFilterFlux
```


