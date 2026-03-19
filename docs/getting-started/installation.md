# Installation

## Requirements

- Python >= 3.11
- NumPy >= 1.24
- pandas >= 2.0
- matplotlib >= 3.7
- click >= 8.0

## Install from PyPI

```bash
pip install forecastbox
```

## Install from source

```bash
git clone https://github.com/nodesecon/forecastbox.git
cd forecastbox
pip install -e ".[dev]"
```

## Development dependencies

```bash
pip install -e ".[dev]"
```

This installs pytest, ruff, pyright, and other development tools.

## Verify installation

```python
import forecastbox
print(forecastbox.__version__)
```

<!-- TODO: Add conda installation instructions -->
<!-- TODO: Add Docker instructions -->
