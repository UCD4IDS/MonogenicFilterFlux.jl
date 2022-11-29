An implementation of rotated monogenic scattering transform network of 2-D signal $g$.

Let $\theta \in [0, 2 \pi)$. Denote the Riesz kernel $r_l$ and Riesz transform $R_l$ as  
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

Suppose the feature dimension is `(nTrain_x, nTrain_y, 1, nSubsample)`.

The $l$-th layer monogenic scattering transform network with maximum scale $s$ is 
```
scale = s; 
st = stFlux((nTrain_x, nTrain_y, 1, nSubsample), 2, σ=abs, outputPool = 1, scale = scale); 
st = cu(st);
```

Suppose `input_data` is the input of the 2-D signal, and set $\theta$ as `ang`. 

We can find the rotated monogenic decomposition of the scattering output by
```
img_rot = rotate_image(input_data, ang);
output_rot = st(cu(img_rot))
output_rot = inv_rotate_out(output_rot, img_rot, -ang);
```
