# AutoARIMA Theory

## ARIMA Models

An ARIMA(p,d,q) model is defined as:

$$\phi(B)(1-B)^d y_t = \theta(B) \epsilon_t$$

where $\phi(B)$ is the AR polynomial and $\theta(B)$ is the MA polynomial.

## Automatic Model Selection

The Hyndman-Khandakar algorithm:

1. Determine $d$ using KPSS test
2. Search over $(p,q)$ using AIC/BIC
3. Apply stepwise search for efficiency

## References

- Hyndman, R.J. & Khandakar, Y. (2008). "Automatic Time Series Forecasting: The forecast Package for R."

<!-- TODO: Expand mathematical details -->
<!-- TODO: Add implementation notes -->
