# Bates-Granger Optimal Combination

## Framework

The optimal combination weights minimize the variance of the combined forecast error.

$$w^* = \arg\min_w \text{Var}\left(\sum_{i=1}^K w_i e_{it}\right)$$

subject to $\sum_i w_i = 1$.

## Solution

$$w^* = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}' \Sigma^{-1} \mathbf{1}}$$

## References

- Bates, J.M. & Granger, C.W.J. (1969). "The Combination of Forecasts."

<!-- TODO: Add shrinkage estimator discussion -->
<!-- TODO: Add empirical examples -->
