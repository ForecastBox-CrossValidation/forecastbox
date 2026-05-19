---
title: Instalacao
description: Instale o forecastbox e suas dependencias em qualquer ambiente Python
---

# Instalacao

## Quick Install

```bash
pip install forecastbox
```

Pronto. O forecastbox e todas as dependencias obrigatorias serao instaladas automaticamente.

---

## Requisitos

**Python**: >= 3.11 (3.11 e 3.12 suportados)

**Dependencias obrigatorias** (instaladas automaticamente):

| Pacote | Versao Minima | Finalidade |
|--------|:------------:|------------|
| NumPy | >= 1.24 | Operacoes numericas |
| Pandas | >= 2.0 | Manipulacao de dados |
| Matplotlib | >= 3.7 | Visualizacao |
| Click | >= 8.0 | Interface de linha de comando |

!!! warning "Versao do Python"

    O forecastbox requer **Python >= 3.11**. Versoes anteriores nao sao suportadas.
    Verifique sua versao com:

    ```bash
    python --version
    ```

---

## Opcoes de Instalacao

=== "Base"

    ```bash
    pip install forecastbox
    ```

    Inclui auto-forecast, combinacao, avaliacao e cenarios.

=== "Nowcasting"

    ```bash
    pip install forecastbox[nowcasting]
    ```

    Adiciona **kalmanbox** para Dynamic Factor Models, bridge equations e MIDAS.

=== "Visualizacao"

    ```bash
    pip install forecastbox[viz]
    ```

    Adiciona **plotly** para graficos interativos alem do matplotlib.

=== "Completa"

    ```bash
    pip install forecastbox[full]
    ```

    Todas as dependencias, incluindo **archbox** para modelos de volatilidade.

=== "Desenvolvimento"

    ```bash
    git clone https://github.com/nodesecon/forecastbox.git
    cd forecastbox
    pip install -e ".[dev]"
    ```

    Inclui pytest, ruff, pyright e ferramentas de documentacao.

### Tabela de Extras

| Extra | Pacotes Incluidos | Caso de Uso |
|-------|-------------------|-------------|
| `[nowcasting]` | kalmanbox | DFM, MIDAS, bridge equations |
| `[viz]` | plotly | Graficos interativos |
| `[full]` | kalmanbox, archbox, plotly | Todas as funcionalidades |
| `[dev]` | pytest, ruff, pyright | Desenvolvimento e testes |
| `[docs]` | mkdocs-material, mkdocstrings | Geracao de documentacao |

---

## Verificacao

Verifique que o forecastbox esta instalado corretamente:

```python
import forecastbox

# Verificar versao
print(f"forecastbox version: {forecastbox.__version__}")
```

```text
forecastbox version: 0.1.0
```

Teste rapido com dados de exemplo:

```python
from forecastbox.datasets import load_gdp

data = load_gdp()
print(f"Serie: {data.name}")
print(f"Periodos: {len(data)}")
print(data.tail())
```

```text
Serie: gdp_growth
Periodos: 80
2019Q1    1.1
2019Q2    2.0
2019Q3    2.1
2019Q4    2.1
2020Q1   -5.0
Freq: QS, Name: gdp_growth, dtype: float64
```

---

## Ambientes Virtuais

Recomendamos usar um ambiente virtual para evitar conflitos de dependencias:

=== "venv"

    ```bash
    python -m venv forecastbox_env
    source forecastbox_env/bin/activate   # Linux/macOS
    forecastbox_env\Scripts\activate      # Windows
    pip install forecastbox
    ```

=== "conda"

    ```bash
    conda create -n forecastbox_env python=3.11
    conda activate forecastbox_env
    pip install forecastbox
    ```

---

## Jupyter e Google Colab

### Local Jupyter

```bash
pip install forecastbox
python -m ipykernel install --user --name=forecastbox_env
```

### Google Colab

Na primeira celula do notebook:

```python
!pip install forecastbox -q
import forecastbox
print(forecastbox.__version__)
```

!!! tip "Notebooks prontos"

    Os tutoriais do forecastbox estao disponiveis como notebooks prontos para executar.
    Veja a secao [Tutorials](../tutorials/index.md) para links.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'forecastbox'`

Verifique se o forecastbox esta instalado no ambiente ativo:

```bash
pip list | grep forecastbox
```

Se nao encontrado, instale-o. Em Jupyter, verifique se o kernel do notebook
corresponde ao ambiente onde o forecastbox esta instalado.

### Conflitos de Dependencias

Crie um ambiente virtual limpo:

```bash
python -m venv fresh_env
source fresh_env/bin/activate
pip install forecastbox
```

### `ImportError` ao importar kalmanbox ou archbox

Esses pacotes sao **dependencias opcionais**. Instale o extra correspondente:

```bash
# Para nowcasting (kalmanbox)
pip install forecastbox[nowcasting]

# Para volatilidade (archbox)
pip install forecastbox[full]
```

### Versao do Python incompativel

O forecastbox requer Python >= 3.11. Verifique sua versao:

```bash
python --version
```

Se necessario, instale uma versao compativel ou use `pyenv`:

```bash
pyenv install 3.11
pyenv local 3.11
```

### Informacoes do sistema para bug reports

```python
import sys, platform
import forecastbox

print(f"Python:       {sys.version}")
print(f"Platform:     {platform.platform()}")
print(f"forecastbox:  {forecastbox.__version__}")
```

Inclua essa saida ao [abrir issues](https://github.com/nodesecon/forecastbox/issues).

---

## Proximos Passos

- **[Quickstart](quickstart.md)** -- Faca sua primeira previsao em 5 minutos
- **[Conceitos Fundamentais](core-concepts.md)** -- Entenda a arquitetura do forecastbox
- **[Escolhendo o Metodo](choosing-method.md)** -- Guia para selecionar a abordagem certa
