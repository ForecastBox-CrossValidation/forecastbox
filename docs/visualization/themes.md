---
title: Temas e Customizacao
description: Sistema de temas built-in, customizacao de paletas, fontes e estilos, export para publicacao e acessibilidade
---

# Temas e Customizacao

O forecastbox inclui um sistema de temas flexivel que garante consistencia visual
em todos os graficos. Cinco temas built-in cobrem os casos de uso mais comuns,
e a API de customizacao permite criar temas para qualquer contexto institucional.

---

## Temas Built-in

| Tema | Estilo | Caso de Uso | Fonte | Background |
|:-----|:-------|:------------|:------|:-----------|
| `light` | Limpo, moderno | Uso geral, exploracao | Sans-serif | Branco |
| `dark` | Escuro, contrastante | Dashboards, apresentacoes escuras | Sans-serif | #1a1a2e |
| `publication` | Academico, minimalista | Papers, dissertacoes | Serif | Branco |
| `presentation` | Bold, alto contraste | Slides, apresentacoes | Sans-serif (grande) | Branco |
| `bcb` | Institucional | Relatorios do Banco Central | Calibri | Branco |

---

### Light (Padrao)

Tema padrao com paleta moderna e fundo limpo. Ideal para exploracao interativa
e relatorios internos.

```python
from forecastbox.plot import set_theme

set_theme("light")
```

**Paleta de cores** (10 cores):

| Cor | Hex | Uso |
|:----|:----|:----|
| :material-circle:{ style="color: #1f77b4" } Azul | `#1f77b4` | Primeiro modelo / serie principal |
| :material-circle:{ style="color: #ff7f0e" } Laranja | `#ff7f0e` | Segundo modelo / contraste |
| :material-circle:{ style="color: #2ca02c" } Verde | `#2ca02c` | Terceiro modelo / positivo |
| :material-circle:{ style="color: #d62728" } Vermelho | `#d62728` | Quarto modelo / negativo / erro |
| :material-circle:{ style="color: #9467bd" } Roxo | `#9467bd` | Quinto modelo |
| :material-circle:{ style="color: #8c564b" } Marrom | `#8c564b` | Sexto modelo |
| :material-circle:{ style="color: #e377c2" } Rosa | `#e377c2` | Setimo modelo |
| :material-circle:{ style="color: #7f7f7f" } Cinza | `#7f7f7f` | Referencia / benchmark |
| :material-circle:{ style="color: #bcbd22" } Amarelo-verde | `#bcbd22` | Oitavo modelo |
| :material-circle:{ style="color: #17becf" } Ciano | `#17becf` | Nono modelo |

**Configuracao**: fonte Arial 12pt, grid sutil (#e0e0e0), eixos cinza escuro.

---

### Dark

Tema escuro para dashboards e apresentacoes em ambientes com pouca luz.
Paleta com cores mais saturadas para contraste com fundo escuro.

```python
set_theme("dark")
```

**Paleta de cores** (10 cores):

| Cor | Hex | Uso |
|:----|:----|:----|
| :material-circle:{ style="color: #4fc3f7" } Azul claro | `#4fc3f7` | Serie principal |
| :material-circle:{ style="color: #ffb74d" } Laranja claro | `#ffb74d` | Contraste |
| :material-circle:{ style="color: #81c784" } Verde claro | `#81c784` | Positivo |
| :material-circle:{ style="color: #e57373" } Vermelho claro | `#e57373` | Negativo / erro |
| :material-circle:{ style="color: #ba68c8" } Roxo claro | `#ba68c8` | Quinto modelo |
| :material-circle:{ style="color: #a1887f" } Marrom claro | `#a1887f` | Sexto modelo |
| :material-circle:{ style="color: #f06292" } Rosa claro | `#f06292` | Setimo modelo |
| :material-circle:{ style="color: #b0bec5" } Cinza claro | `#b0bec5` | Referencia |
| :material-circle:{ style="color: #dce775" } Lima | `#dce775` | Oitavo modelo |
| :material-circle:{ style="color: #4dd0e1" } Ciano claro | `#4dd0e1` | Nono modelo |

**Configuracao**: fundo #1a1a2e, texto branco, grid sutil (#333355), fonte
Arial 12pt. Otimizado para contraste WCAG AA.

---

### Publication

Tema minimalista para publicacoes academicas. Fonte serif, eixos limpos,
paleta restrita. Segue convencoes de journals de econometria.

```python
set_theme("publication")
```

**Paleta de cores** (6 cores):

| Cor | Hex | Uso |
|:----|:----|:----|
| :material-circle:{ style="color: #000000" } Preto | `#000000` | Serie principal |
| :material-circle:{ style="color: #404040" } Cinza escuro | `#404040` | Segundo modelo |
| :material-circle:{ style="color: #808080" } Cinza medio | `#808080` | Terceiro modelo |
| :material-circle:{ style="color: #a0a0a0" } Cinza claro | `#a0a0a0` | Quarto modelo |
| :material-circle:{ style="color: #1f77b4" } Azul (acento) | `#1f77b4` | Destaque |
| :material-circle:{ style="color: #d62728" } Vermelho (acento) | `#d62728` | Alerta |

**Configuracao**: fonte Times New Roman 11pt, fundo branco puro, eixos com
ticks externos, sem grid (ou grid muito sutil). Legendas posicionadas fora
do grafico quando possivel. Otimizado para impressao P&B.

---

### Presentation

Tema bold para slides e apresentacoes. Fontes grandes, linhas espessas,
cores vibrantes. Projetado para legibilidade a distancia.

```python
set_theme("presentation")
```

**Paleta de cores** (8 cores):

| Cor | Hex | Uso |
|:----|:----|:----|
| :material-circle:{ style="color: #e63946" } Vermelho | `#e63946` | Destaque principal |
| :material-circle:{ style="color: #1d3557" } Azul marinho | `#1d3557` | Serie principal |
| :material-circle:{ style="color: #2a9d8f" } Teal | `#2a9d8f` | Positivo |
| :material-circle:{ style="color: #e9c46a" } Dourado | `#e9c46a` | Acento |
| :material-circle:{ style="color: #f4a261" } Laranja | `#f4a261` | Quinto modelo |
| :material-circle:{ style="color: #a8dadc" } Azul claro | `#a8dadc` | Sexto modelo |
| :material-circle:{ style="color: #457b9d" } Azul medio | `#457b9d` | Setimo modelo |
| :material-circle:{ style="color: #264653" } Azul escuro | `#264653` | Oitavo modelo |

**Configuracao**: fonte Helvetica 14pt (titulos 18pt), linewidth 2.5, markersize 10,
fundo branco, grid cinza claro com linewidth 0.8. Bordas espessas.

---

### BCB (Banco Central do Brasil)

Tema institucional seguindo as diretrizes visuais do Banco Central do Brasil.
Cores e fontes alinhadas com publicacoes oficiais.

```python
set_theme("bcb")
```

**Paleta de cores** (8 cores):

| Cor | Hex | Uso |
|:----|:----|:----|
| :material-circle:{ style="color: #003399" } Azul BCB | `#003399` | Serie principal |
| :material-circle:{ style="color: #006633" } Verde BCB | `#006633` | Segundo modelo |
| :material-circle:{ style="color: #cc0000" } Vermelho | `#cc0000` | Terceiro modelo / alerta |
| :material-circle:{ style="color: #ff9900" } Laranja | `#ff9900` | Quarto modelo |
| :material-circle:{ style="color: #666666" } Cinza | `#666666` | Referencia |
| :material-circle:{ style="color: #0066cc" } Azul claro | `#0066cc` | Quinto modelo |
| :material-circle:{ style="color: #339966" } Verde claro | `#339966` | Sexto modelo |
| :material-circle:{ style="color: #993366" } Vinho | `#993366` | Setimo modelo |

**Configuracao**: fonte Calibri 11pt, fundo branco, grid cinza (#d0d0d0),
bordas finas. Legendas posicionadas externamente. Compativel com templates
de relatorios do BCB (Relatorio de Inflacao, Focus).

---

## Usando Temas

### Aplicar Globalmente

```python
from forecastbox.plot import set_theme, get_theme, reset_theme

# Definir tema para todas as figuras subsequentes
set_theme("publication")

# Consultar tema atual
current = get_theme()
print(current.name)  # "publication"

# Restaurar tema padrao (light)
reset_theme()
```

### Por Grafico

Todos os graficos aceitam o parametro `style` para override pontual:

```python
from forecastbox.plot import plot_forecast

# Tema global e "light", mas este grafico usa "publication"
plot_forecast(forecast, style="publication")
```

### Context Manager

Use como context manager para aplicar um tema temporariamente:

```python
from forecastbox.plot import theme_context

with theme_context("publication"):
    plot_forecast(forecast)
    plot_comparison(forecasts)
    # Tema revertido automaticamente ao sair do bloco
```

---

## Customizacao de Temas

### Criar Tema Customizado

```python
from forecastbox.plot import Theme

my_theme = Theme(
    name="institutional",
    palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    font_family="serif",
    font_size=12,
    title_size=14,
    grid=True,
    grid_alpha=0.3,
    grid_color="#cccccc",
    background="#fafafa",
    text_color="#333333",
    linewidth=1.5,
    markersize=6,
    legend_position="best",
    tick_direction="out",
    spine_visible=True,
    dpi=150,
)

# Aplicar
from forecastbox.plot import set_theme
set_theme(my_theme)
```

### Parametros do Theme

| Parametro | Tipo | Default | Descricao |
|:----------|:-----|:--------|:----------|
| `name` | `str` | `"custom"` | Nome do tema |
| `palette` | `list[str]` | *Tab10* | Lista de cores hex |
| `font_family` | `str` | `"sans-serif"` | Familia de fonte |
| `font_size` | `int` | `12` | Tamanho de fonte base |
| `title_size` | `int` | `14` | Tamanho do titulo |
| `grid` | `bool` | `True` | Exibir grid |
| `grid_alpha` | `float` | `0.3` | Transparencia do grid |
| `grid_color` | `str` | `"#cccccc"` | Cor do grid |
| `background` | `str` | `"#ffffff"` | Cor de fundo da area do grafico |
| `paper_color` | `str` | `"#ffffff"` | Cor de fundo da figura |
| `text_color` | `str` | `"#333333"` | Cor do texto |
| `linewidth` | `float` | `1.5` | Espessura padrao de linhas |
| `markersize` | `float` | `6` | Tamanho padrao de marcadores |
| `legend_position` | `str` | `"best"` | Posicao da legenda |
| `tick_direction` | `str` | `"out"` | Direcao dos ticks: `"in"`, `"out"`, `"inout"` |
| `spine_visible` | `bool` | `True` | Exibir bordas do grafico |
| `dpi` | `int` | `150` | Resolucao padrao |

### Herdar e Modificar

Crie um tema baseado em um existente, modificando apenas o necessario:

```python
from forecastbox.plot import get_theme, Theme

# Partir do tema publication e modificar
base = get_theme("publication")
my_theme = base.copy(
    name="my_publication",
    palette=["#003399", "#006633", "#cc0000"],  # cores institucionais
    font_family="Calibri",
    font_size=11,
)

set_theme(my_theme)
```

### Registrar Tema

Registre temas customizados para reutilizacao por nome:

```python
from forecastbox.plot import register_theme

register_theme(my_theme)

# Agora pode usar por nome
set_theme("my_publication")

# Listar temas disponiveis
from forecastbox.plot import list_themes
print(list_themes())
# ["light", "dark", "publication", "presentation", "bcb", "my_publication"]
```

---

## Temas por Backend

Os temas sao traduzidos automaticamente para as configuracoes nativas de cada
backend.

### Matplotlib

O tema e aplicado via `rcParams`:

```python
import matplotlib as mpl

# O forecastbox traduz o tema para rcParams
theme = get_theme()

# Equivalente manual:
mpl.rcParams.update({
    "font.family": theme.font_family,
    "font.size": theme.font_size,
    "axes.grid": theme.grid,
    "axes.facecolor": theme.background,
    "figure.facecolor": theme.paper_color,
    "axes.prop_cycle": mpl.cycler(color=theme.palette),
    "lines.linewidth": theme.linewidth,
    "lines.markersize": theme.markersize,
    "figure.dpi": theme.dpi,
})
```

### Plotly

O tema e traduzido para um Plotly template:

```python
import plotly.graph_objects as go
import plotly.io as pio

# Traduzao automatica para template Plotly
template = theme.to_plotly_template()

# Equivalente:
pio.templates["forecastbox"] = go.layout.Template(
    layout=dict(
        font=dict(family=theme.font_family, size=theme.font_size),
        plot_bgcolor=theme.background,
        paper_bgcolor=theme.paper_color,
        colorway=theme.palette,
    )
)
pio.templates.default = "forecastbox"
```

---

## Export para Publicacao

### Formatos Suportados

| Formato | Extensao | Uso | Backend |
|:--------|:---------|:----|:--------|
| PNG | `.png` | Web, apresentacoes | matplotlib, plotly |
| SVG | `.svg` | Vetorial, web | matplotlib, plotly |
| PDF | `.pdf` | Papers, relatorios | matplotlib |
| HTML | `.html` | Dashboards interativos | plotly |
| EPS | `.eps` | Journals (LaTeX) | matplotlib |

### Configuracao de Export

```python
from forecastbox.plot import plot_forecast, export_config

# Configurar export global
export_config(
    dpi=300,
    format="pdf",
    bbox_inches="tight",
    transparent=False,
    pad_inches=0.1,
)

# Export individual
fig = plot_forecast(forecast, show=False)
fig.savefig(
    "forecast.pdf",
    dpi=300,
    bbox_inches="tight",
    transparent=False,
)
```

### Tamanhos Recomendados

| Contexto | Tamanho (polegadas) | DPI | Formato |
|:---------|:--------------------|:----|:--------|
| Journal (1 coluna) | (3.5, 2.5) | 300 | PDF/EPS |
| Journal (2 colunas) | (7.0, 4.0) | 300 | PDF/EPS |
| Apresentacao (16:9) | (10, 5.6) | 150 | PNG |
| Relatorio A4 | (6.5, 4.0) | 200 | PNG/PDF |
| Dashboard | (12, 8) | 100 | HTML |

```python
# Exemplo: figura para journal (1 coluna)
set_theme("publication")

plot_forecast(
    forecast,
    figsize=(3.5, 2.5),
    title="",  # journals frequentemente pedem caption separado
    show=False,
).savefig("fig1.pdf", dpi=300, bbox_inches="tight")
```

---

## Acessibilidade

### Paletas Colorblind-Safe

O forecastbox inclui paletas otimizadas para daltonismo, testadas com
simuladores de deuteranopia, protanopia e tritanopia.

```python
from forecastbox.plot import Theme

# Paleta colorblind-safe (8 cores)
cb_theme = Theme(
    name="colorblind_safe",
    palette=[
        "#0072B2",  # azul
        "#E69F00",  # laranja
        "#009E73",  # verde-azulado
        "#CC79A7",  # rosa
        "#F0E442",  # amarelo
        "#56B4E9",  # azul claro
        "#D55E00",  # vermelho-laranja
        "#000000",  # preto
    ],
)

set_theme(cb_theme)
```

!!! info "Paleta Wong (2011)"

    A paleta colorblind-safe do forecastbox e baseada na recomendacao de
    [Bang Wong (Nature Methods, 2011)](https://www.nature.com/articles/nmeth.1618),
    amplamente usada em publicacoes cientificas. Ela e distinguivel por
    pessoas com os tres tipos mais comuns de daltonismo.

### Verificar Acessibilidade

```python
from forecastbox.plot import check_accessibility

# Verifica se o tema atual e acessivel
report = check_accessibility()
print(report)
# AccessibilityReport(
#     colorblind_safe=True,
#     min_contrast_ratio=4.8,
#     wcag_level="AA",
#     suggestions=[]
# )
```

### Boas Praticas de Acessibilidade

!!! tip "Alem das cores"

    Nao dependa apenas de cores para distinguir elementos. Combine:

    - **Cores** diferentes para cada serie
    - **Estilos de linha** (solida, tracejada, pontilhada) para distinguir sem cor
    - **Marcadores** (circulo, quadrado, triangulo) para pontos individuais
    - **Anotacoes** textuais para destaque

```python
# Exemplo acessivel: combinando cor + estilo de linha + marcador
from forecastbox.plot import plot_comparison

plot_comparison(
    forecasts,
    line_styles=["solid", "dashed", "dotted"],
    markers=["o", "s", "^"],
    style="publication",
)
```

---

## Exemplo Completo: Tema Institucional

Criando um tema customizado para relatorios de uma instituicao financeira:

```python
from forecastbox.plot import Theme, set_theme, register_theme

# Definir tema institucional
bank_theme = Theme(
    name="bank_reports",
    palette=[
        "#003366",  # azul corporativo (primario)
        "#006699",  # azul medio
        "#669933",  # verde institucional
        "#CC3333",  # vermelho (alertas)
        "#FF9900",  # laranja (avisos)
        "#666666",  # cinza (referencias)
        "#333399",  # azul escuro
        "#009999",  # teal
    ],
    font_family="Calibri",
    font_size=11,
    title_size=13,
    grid=True,
    grid_alpha=0.2,
    grid_color="#d0d0d0",
    background="#ffffff",
    paper_color="#ffffff",
    text_color="#333333",
    linewidth=1.5,
    markersize=5,
    legend_position="upper right",
    tick_direction="out",
    spine_visible=True,
    dpi=200,
)

# Registrar para uso por nome
register_theme(bank_theme)

# Aplicar globalmente
set_theme("bank_reports")

# Gerar relatorio completo
from forecastbox.plot import (
    plot_forecast,
    plot_comparison,
    plot_weights,
    plot_monitor_dashboard,
)

# Todos os graficos usam o tema "bank_reports"
plot_forecast(forecast, title="Projecao IPCA - 12 meses")
plot_comparison(forecasts, title="Comparacao de Modelos")
plot_weights(combination, title="Pesos da Combinacao BMA")
```

**Output**: Todos os graficos com identidade visual consistente: azul
corporativo como cor principal, fonte Calibri, fundo branco, grid
discreto. Pronto para inclusao em relatorio institucional.

---

## Referencia Rapida

| Funcao | Descricao |
|:-------|:----------|
| `set_theme(name_or_theme)` | Definir tema global |
| `get_theme(name=None)` | Obter tema atual ou por nome |
| `reset_theme()` | Restaurar tema padrao (`light`) |
| `list_themes()` | Listar temas registrados |
| `register_theme(theme)` | Registrar tema customizado |
| `theme_context(name)` | Context manager para tema temporario |
| `export_config(**kwargs)` | Configurar parametros de export |
| `check_accessibility()` | Verificar acessibilidade do tema |
| `Theme(...)` | Criar tema customizado |
| `theme.copy(**overrides)` | Copiar tema com modificacoes |
| `theme.to_plotly_template()` | Converter para template Plotly |

---

## See Also

- :material-school: [Tutorial: Fundamentos](../tutorials/fundamentals.md) — primeiros graficos com forecastbox
- :material-school: [Tutorial: Workflow Completo](../tutorials/complete-workflow.md) — dashboard completo com temas
- [Graficos de Previsao](forecast-plots.md) — funcoes de visualizacao de previsoes
- [Graficos de Comparacao](comparison-plots.md) — funcoes de comparacao de modelos
- [API Reference - Visualization](../api/visualization.md) — referencia completa da API
