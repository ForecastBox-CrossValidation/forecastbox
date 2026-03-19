# Model Confidence Set

## Overview

The MCS identifies a set of models that contains the best model with a given confidence level.

## Algorithm

1. Start with all models $M_0 = \{1, \ldots, m\}$
2. Test $H_0$: equal predictive ability
3. If rejected, remove worst model
4. Repeat until $H_0$ is not rejected

## References

- Hansen, P.R., Lunde, A. & Nason, J.M. (2011). "The Model Confidence Set."

<!-- TODO: Add bootstrap details -->
<!-- TODO: Add implementation notes -->
