---
title: "Contributing Guide"
description: "How to contribute to forecastbox — setup, code standards, templates, and PR process."
---

# Contributing to forecastbox

Thank you for your interest in contributing to forecastbox! Whether you are reporting a bug, proposing a feature, improving documentation, or submitting code, your help is welcome and appreciated.

## Types of Contributions

| Type | Where | Description |
|------|-------|-------------|
| Bug reports | [GitHub Issues](https://github.com/nodesecon/forecastbox/issues) | Reproducible problem with expected vs. actual behavior |
| Feature requests | [GitHub Issues](https://github.com/nodesecon/forecastbox/issues) | Proposals with `[Feature]` label |
| Code (PR) | [Pull Requests](https://github.com/nodesecon/forecastbox/pulls) | New models, methods, bug fixes |
| Documentation | `docs/` directory | Tutorials, API docs, examples |
| Test additions | `tests/` directory | Unit, integration, and validation tests |

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/nodesecon/forecastbox.git
cd forecastbox
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
# venv\Scripts\activate    # Windows
```

### 3. Install in Development Mode

```bash
pip install -e ".[dev,docs]"
```

### 4. Install Pre-Commit Hooks

```bash
pre-commit install
```

### 5. Verify Setup

```bash
# Run tests
pytest tests/ -v

# Serve documentation locally
mkdocs serve
```

### Development Dependencies

| Tool | Purpose |
|------|---------|
| pytest | Testing framework |
| pytest-cov | Coverage measurement |
| black | Code formatting |
| ruff | Linting |
| mypy | Static type checking |
| mkdocs-material | Documentation |
| mkdocstrings[python] | API reference generation |

## Code Standards

### Style

- **Formatter**: black (line length 88)
- **Linter**: ruff
- **Type checker**: mypy (strict mode)
- **Docstrings**: NumPy-style for all public classes and methods
- **Python version**: 3.9+ compatibility

```bash
# Format
black forecastbox/

# Lint
ruff check forecastbox/ --fix

# Type check
mypy forecastbox/

# Run all hooks manually
pre-commit run --all-files
```

### Branch Naming

Use descriptive branch names:

- `feature/add-nbeats-model` — New features
- `fix/combination-weights` — Bug fixes
- `docs/update-tutorial` — Documentation changes
- `test/add-mcs-tests` — Test additions

### Commit Messages (Conventional Commits)

```text
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `style`, `chore`

**Example**:

```text
feat(combination): Add time-varying weight combination

Implements the time-varying combination method with
exponentially decaying weights for forecast combination.

Closes #45
```

## Adding a New Model to the ModelZoo

Every forecasting model in forecastbox follows a consistent interface. To add a new model:

### 1. Create the Model Class

Place your code in the appropriate module. Every model must:

1. Inherit from the appropriate base class
2. Implement `fit()` and `predict()` methods
3. Return standardized result objects
4. Have comprehensive tests
5. Be exported from the package `__init__.py`
6. Have a documentation page

### Model Template

```python
"""My forecasting model.

Implements the Author (Year) forecasting method.
"""

import numpy as np
import pandas as pd

from forecastbox.core.base import BaseForecaster


class MyForecaster(BaseForecaster):
    """Short description of the forecasting model.

    Longer description explaining the model, its assumptions,
    and when to use it.

    Parameters
    ----------
    endog : array_like
        Time series to forecast.
    exog : array_like, optional
        Exogenous variables.
    horizon : int
        Forecast horizon.

    References
    ----------
    .. [1] Author, A. (Year). Title. *Journal*, vol(issue), pages.
    """

    def __init__(self, endog, exog=None, horizon=1, **kwargs):
        super().__init__(endog, exog=exog, horizon=horizon)
        # Model-specific initialization

    def fit(self, **kwargs):
        """Fit the model to the data.

        Returns
        -------
        self
            Fitted model instance.
        """
        # 1. Estimate parameters
        # 2. Store fitted values and residuals
        return self

    def predict(self, horizon=None, **kwargs):
        """Generate forecasts.

        Parameters
        ----------
        horizon : int, optional
            Forecast horizon. Defaults to self.horizon.

        Returns
        -------
        pd.DataFrame
            Forecasts with point estimates and intervals.
        """
        # Generate point forecasts and prediction intervals
        return forecasts
```

### 2. Register in the ModelZoo

```python
from forecastbox.auto.model_zoo import ModelZoo

ModelZoo.register("my_model", MyForecaster, category="univariate")
```

### 3. Add Tests

```python
import pytest
import numpy as np
from forecastbox.models import MyForecaster


class TestMyForecaster:
    """Tests for MyForecaster."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series."""
        np.random.seed(42)
        return np.random.randn(100).cumsum()

    def test_basic_fit(self, sample_data):
        """Test that fit runs without error."""
        model = MyForecaster(sample_data, horizon=12)
        result = model.fit()
        assert result is not None

    def test_forecast_shape(self, sample_data):
        """Test forecast output shape."""
        model = MyForecaster(sample_data, horizon=12)
        model.fit()
        forecasts = model.predict()
        assert len(forecasts) == 12

    def test_forecast_reasonable(self, sample_data):
        """Test forecasts are within reasonable range."""
        model = MyForecaster(sample_data, horizon=12)
        model.fit()
        forecasts = model.predict()
        assert np.all(np.isfinite(forecasts.values))
```

### 4. Add Documentation

Create a page in `docs/user-guide/auto-forecast/` following the existing pattern with:

- Description and mathematical formulation
- Usage examples
- Parameter reference
- Comparison with similar models

## Adding a New Combination Method

Combination methods merge forecasts from multiple models. To add a new method:

### 1. Create the Combination Class

```python
"""My combination method.

Implements the Author (Year) forecast combination approach.
"""

import numpy as np
import pandas as pd

from forecastbox.combination.base import BaseCombiner


class MyCombiner(BaseCombiner):
    """Short description of the combination method.

    Parameters
    ----------
    forecasts : pd.DataFrame
        Matrix of forecasts (rows=time, columns=models).
    actuals : pd.Series
        Actual observed values.

    References
    ----------
    .. [1] Author, A. (Year). Title. *Journal*, vol(issue), pages.
    """

    def __init__(self, forecasts, actuals, **kwargs):
        super().__init__(forecasts, actuals)

    def fit(self, **kwargs):
        """Estimate combination weights.

        Returns
        -------
        self
            Fitted combiner with weights.
        """
        # 1. Estimate optimal weights
        # 2. Store self.weights_
        return self

    def combine(self, new_forecasts=None, **kwargs):
        """Generate combined forecast.

        Parameters
        ----------
        new_forecasts : pd.DataFrame, optional
            New forecasts to combine. Uses training forecasts if None.

        Returns
        -------
        pd.Series
            Combined forecast.
        """
        if new_forecasts is None:
            new_forecasts = self.forecasts
        return new_forecasts @ self.weights_
```

### 2. Register the Method

Export from the combination module's `__init__.py` and add to the method registry.

### 3. Add Tests and Documentation

Follow the same pattern as model additions — tests with known analytical results and a documentation page in `docs/user-guide/combination/`.

## Writing Documentation

forecastbox uses MkDocs with Material theme. Key conventions:

### Admonitions

```markdown
!!! note "Important"
    Use admonitions for callouts.

!!! warning
    Highlight potential issues.

!!! example "Usage"
    Show code examples.

!!! tip
    Provide helpful suggestions.
```

### Code Blocks with Tabs

```markdown
=== "Python"

    ```python
    from forecastbox import AutoARIMA
    model = AutoARIMA(data)
    model.fit()
    ```

=== "CLI"

    ```bash
    forecastbox forecast --method auto-arima --data data.csv
    ```
```

### Math (MathJax)

Use `$...$` for inline math and `$$...$$` for display math:

```markdown
The combined forecast is $\hat{y}_t = \sum_{i=1}^{K} w_i \hat{y}_{it}$

$$
\hat{y}_{t+h|t} = \sum_{i=1}^{K} w_{i,t} \hat{y}_{i,t+h|t}
$$
```

### API Documentation

Use mkdocstrings directives:

```markdown
::: forecastbox.auto.AutoARIMA
    options:
      show_root_heading: true
      show_source: true
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/auto/ -v
pytest tests/combination/ -v

# With coverage
pytest tests/ --cov=forecastbox --cov-report=html --cov-branch

# Specific test
pytest tests/evaluation/test_dm.py::test_dm_basic -v
```

## Pull Request Process

### Step-by-Step

1. **Create a feature branch**:
    ```bash
    git checkout -b feature/my-new-feature
    ```

2. **Make changes**: code, tests, documentation.

3. **Run checks locally**:
    ```bash
    pytest tests/ -v
    pre-commit run --all-files
    ```

4. **Commit with a clear message**:
    ```bash
    git commit -m "feat(auto): Add MyForecaster model

    - Implements Author (Year) forecasting method
    - Adds validation against R forecast package
    - Includes 15 unit tests

    Closes #123"
    ```

5. **Push and open a PR**:
    ```bash
    git push origin feature/my-new-feature
    ```

6. **Fill out the PR template and address review comments.**

### PR Checklist

- [ ] Tests pass locally (`pytest tests/ -v`)
- [ ] Pre-commit hooks pass (`pre-commit run --all-files`)
- [ ] New code has tests
- [ ] Public API has docstrings (NumPy style)
- [ ] Documentation updated (if applicable)
- [ ] Exports added to `__init__.py` (if applicable)

## Reporting Issues

File issues on [GitHub](https://github.com/nodesecon/forecastbox/issues) with:

1. A clear title describing the problem
2. **Minimal reproducible example** (MRE)
3. Expected vs. actual behavior
4. forecastbox version: `pip show forecastbox`
5. Python version: `python --version`

## Recognition

Contributors are recognized in:

- The [Changelog](changelog.md)
- Release notes
- The AUTHORS file

Significant contributions may result in co-authorship on methodological papers.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Questions?

- **General questions**: [GitHub Discussions](https://github.com/nodesecon/forecastbox/discussions)
- **Bug reports**: [GitHub Issues](https://github.com/nodesecon/forecastbox/issues)
- **Feature requests**: [GitHub Issues](https://github.com/nodesecon/forecastbox/issues) with `[Feature]` label

## See Also

- [Code of Conduct](code-of-conduct.md) — Community standards
- [Changelog](changelog.md) — Version history
- [Roadmap](roadmap.md) — Planned features
- [API Reference](../api/index.md) — Full API documentation
