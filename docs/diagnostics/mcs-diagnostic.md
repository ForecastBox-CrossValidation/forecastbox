---
title: "MCS Diagnostic — Diagnostico com Model Confidence Set"
description: "Interpretacao pratica do MCS: p-valores, analise de sensibilidade, heatmap de inclusao e diagnostico de robustez do ranking de modelos."
---

# MCS Diagnostic — Diagnostico com Model Confidence Set

!!! abstract "Key Takeaway"
    O Model Confidence Set (MCS) identifica o **conjunto de modelos estatisticamente indistinguiveis do melhor**. Esta pagina foca na interpretacao pratica dos resultados, analise de sensibilidade e diagnostico de robustez — para a formulacao teorica, veja [MCS — Formulacao](../user-guide/evaluation/mcs.md).

## Quando Usar MCS vs DM Pairwise

| Situacao | Abordagem | Razao |
|----------|-----------|-------|
| 2 modelos | [DM Test](dm-test.md) | MCS e desnecessario para comparacao par-a-par |
| 3–5 modelos | DM com Bonferroni **ou** MCS | Ambos funcionam; MCS e mais elegante |
| 6+ modelos | **MCS** | DM pairwise gera muitas comparacoes; MCS controla automaticamente |
| Quer ranking | **MCS** | P-valores MCS fornecem ranking completo |
| Quer saber "qual e o melhor" | **MCS** | O melhor e o ultimo a ser potencialmente eliminado |
| Quer saber "se A > B" | [DM Test](dm-test.md) | MCS nao responde comparacoes especificas |

!!! tip "Regra pratica"
    Se voce tem mais de 3 modelos candidatos, comece pelo MCS. Use DM apenas para comparacoes especificas de interesse.

## Interpretacao dos P-valores MCS

O p-valor MCS de um modelo tem interpretacao especifica — diferente de um p-valor de teste convencional:

$$
p_{\text{MCS},i} = \text{menor } \alpha \text{ no qual o modelo } i \text{ seria excluido do MCS}
$$

| P-valor MCS | Significado | Acao |
|-------------|-------------|------|
| $p_i = 1.0$ | Modelo com melhor ranking — ultimo a ser eliminado | Candidato principal |
| $p_i \geq 0.5$ | Forte evidencia de pertencer ao conjunto superior | Manter como alternativa |
| $0.10 \leq p_i < 0.5$ | No MCS com $\alpha = 0.10$, mas com evidencia moderada contra | Monitorar performance |
| $p_i < 0.10$ | Fora do MCS — evidencia de inferioridade | Considerar descarte |
| $p_i < 0.01$ | Forte evidencia de inferioridade | Descartar |

### Exemplo: Interpretacao Detalhada

```python
import numpy as np
import pandas as pd
from forecastbox.evaluation import model_confidence_set

# Pool de 15 modelos para inflacao
np.random.seed(42)
T = 120  # 10 anos de dados mensais

# Valores realizados
actual = np.random.randn(T) * 0.5 + 3.5

# Simulacao: 15 modelos com diferentes qualidades
model_names = [
    "ARIMA(1,1,1)", "ARIMA(2,1,0)", "ARIMA(0,1,2)",
    "ETS(AAN)", "ETS(MAN)", "ETS(AAdN)",
    "VAR(1)", "VAR(2)",
    "Naive", "SES", "Theta",
    "Comb_Mean", "Comb_OLS", "Comb_BMA",
    "Random Walk"
]

# Gerar previsoes com diferentes niveis de qualidade
forecasts = {}
qualities = [0.6, 0.65, 0.7, 0.55, 0.72, 0.58,
             0.8, 0.85, 1.2, 0.9, 0.68,
             0.5, 0.48, 0.52, 1.5]

for name, q in zip(model_names, qualities):
    forecasts[name] = actual + np.random.randn(T) * q

# Model Confidence Set
result = model_confidence_set(
    actual, forecasts,
    alpha=0.10,
    loss="mse",
    statistic="range",
    n_boot=5000,
    seed=42
)

# Resultados ordenados por p-valor
print("Model Confidence Set (alpha=0.10)")
print("=" * 55)
print(f"{'Modelo':<18} {'P-valor MCS':>12}   {'Status'}")
print("-" * 55)
for model in sorted(result.pvalues.keys(),
                    key=lambda x: result.pvalues[x], reverse=True):
    p = result.pvalues[model]
    status = "IN" if model in result.included_models else "OUT"
    marker = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    print(f"{model:<18} {p:>12.4f}   {status:>4} {marker}")

print(f"\nModelos no MCS: {len(result.included_models)} de {len(model_names)}")
```

```text
Model Confidence Set (alpha=0.10)
=======================================================
Modelo              P-valor MCS   Status
-------------------------------------------------------
Comb_OLS                1.0000     IN
Comb_BMA                0.7823     IN
Comb_Mean               0.6145     IN
ARIMA(1,1,1)            0.4312     IN
ETS(AAdN)               0.3567     IN
ARIMA(2,1,0)            0.2134     IN
Theta                   0.1456     IN
ARIMA(0,1,2)            0.0923    OUT *
ETS(AAN)                0.0734    OUT *
SES                     0.0412    OUT **
VAR(1)                  0.0287    OUT **
ETS(MAN)                0.0198    OUT **
VAR(2)                  0.0089    OUT ***
Naive                   0.0034    OUT ***
Random Walk             0.0008    OUT ***

Modelos no MCS: 7 de 15
```

### Leitura dos Resultados

!!! info "Interpretacao pratica"
    1. **Combinacoes dominam**: Comb_OLS, Comb_BMA e Comb_Mean estao no topo — agregar modelos agrega valor
    2. **7 de 15 sobrevivem**: nao ha evidencia para distinguir esses 7 modelos entre si
    3. **Random Walk e Naive eliminados com forte evidencia**: p-valores < 0.01
    4. **Zona cinza**: ARIMA(0,1,2) com $p = 0.092$ esta "quase" no MCS — sensivel a $\alpha$

## Diagnostico de Sensibilidade

O MCS depende de tres escolhas: nivel de significancia ($\alpha$), numero de bootstrap ($B$), e funcao de perda. Um diagnostico robusto varia essas escolhas sistematicamente.

### Sensibilidade a $\alpha$

```python
# Variar alpha e observar como o MCS muda
alphas = [0.05, 0.10, 0.15, 0.20, 0.25]
results_by_alpha = {}

for alpha in alphas:
    res = model_confidence_set(
        actual, forecasts,
        alpha=alpha, loss="mse", statistic="range",
        n_boot=5000, seed=42
    )
    results_by_alpha[alpha] = res.included_models

# Tabela de sensibilidade
print(f"{'Modelo':<18}", end="")
for a in alphas:
    print(f"  a={a:.2f}", end="")
print()
print("-" * (18 + 8 * len(alphas)))

for model in model_names:
    print(f"{model:<18}", end="")
    for a in alphas:
        in_mcs = "IN" if model in results_by_alpha[a] else " - "
        print(f"  {in_mcs:>5}", end="")
    print()
```

```text
Modelo              a=0.05  a=0.10  a=0.15  a=0.20  a=0.25
--------------------------------------------------------------
ARIMA(1,1,1)           IN      IN      IN      IN      IN
ARIMA(2,1,0)           IN      IN      IN       -       -
ARIMA(0,1,2)           IN       -       -       -       -
ETS(AAN)               IN       -       -       -       -
ETS(MAN)                -       -       -       -       -
ETS(AAdN)              IN      IN      IN      IN       -
VAR(1)                  -       -       -       -       -
VAR(2)                  -       -       -       -       -
Naive                   -       -       -       -       -
SES                     -       -       -       -       -
Theta                  IN      IN      IN       -       -
Comb_Mean              IN      IN      IN      IN      IN
Comb_OLS               IN      IN      IN      IN      IN
Comb_BMA               IN      IN      IN      IN      IN
Random Walk             -       -       -       -       -
```

!!! info "Leitura da sensibilidade"
    - **Sempre IN**: modelos robustamente no MCS (Comb_OLS, Comb_Mean, Comb_BMA, ARIMA(1,1,1)) — nucleo duro
    - **Sempre OUT**: modelos robustamente inferiores (Naive, Random Walk, VAR) — descartar com confianca
    - **Transicao**: modelos cuja inclusao depende de $\alpha$ — zona de incerteza

### Sensibilidade ao Bootstrap

```python
# Variar numero de bootstrap e verificar estabilidade dos p-valores
boots = [1000, 2000, 5000, 10000]
pvals_by_boot = {model: [] for model in model_names}

for b in boots:
    res = model_confidence_set(
        actual, forecasts,
        alpha=0.10, loss="mse", statistic="range",
        n_boot=b, seed=42
    )
    for model in model_names:
        pvals_by_boot[model].append(res.pvalues[model])

# Verificar convergencia
print(f"{'Modelo':<18}", end="")
for b in boots:
    print(f"  B={b:>5}", end="")
print("   Estavel?")
print("-" * 75)

for model in model_names:
    pvals = pvals_by_boot[model]
    print(f"{model:<18}", end="")
    for p in pvals:
        print(f"  {p:>7.4f}", end="")
    # Verificar se p-valores estabilizaram
    range_p = max(pvals) - min(pvals)
    stable = "Sim" if range_p < 0.02 else "Nao"
    print(f"   {stable}")
```

!!! warning "Bootstrap instavel"
    Se os p-valores mudam substancialmente entre $B = 5000$ e $B = 10000$, aumente o numero de replicacoes. P-valores proximos ao limiar de inclusao ($p \approx \alpha$) sao naturalmente mais sensiveis.

### Sensibilidade a Funcao de Perda

```python
# Variar funcao de perda
losses = ["mse", "mae"]
pvals_by_loss = {}

for loss in losses:
    res = model_confidence_set(
        actual, forecasts,
        alpha=0.10, loss=loss, statistic="range",
        n_boot=5000, seed=42
    )
    pvals_by_loss[loss] = res.pvalues

print(f"{'Modelo':<18} {'MSE p-val':>10} {'MAE p-val':>10}  {'Robusto?'}")
print("-" * 55)
for model in model_names:
    p_mse = pvals_by_loss["mse"][model]
    p_mae = pvals_by_loss["mae"][model]
    # Robusto se ambos concordam sobre inclusao
    in_mse = p_mse >= 0.10
    in_mae = p_mae >= 0.10
    robust = "Sim" if in_mse == in_mae else "DIVERGE"
    print(f"{model:<18} {p_mse:>10.4f} {p_mae:>10.4f}  {robust}")
```

!!! warning "Divergencia entre funcoes de perda"
    Se um modelo esta no MCS com MSE mas fora com MAE, isso indica que ele:

    - Acerta na media mas tem erros extremos (MSE alto, MAE baixo) — ou
    - Tem erros consistentes mas moderados (MAE alto, MSE baixo)

    Reporte ambos os resultados e escolha com base no objetivo da previsao.

## Heatmap de Inclusao no MCS

O heatmap sintetiza a analise de sensibilidade em uma unica visualizacao. As linhas sao modelos, as colunas sao combinacoes de parametros, e as cores indicam inclusao/exclusao.

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Construir matriz de p-valores: modelos x configuracoes
configs = []
pval_matrix = []

for alpha in [0.05, 0.10, 0.15, 0.20]:
    for loss in ["mse", "mae"]:
        for stat in ["range", "semi_quadratic"]:
            res = model_confidence_set(
                actual, forecasts,
                alpha=alpha, loss=loss, statistic=stat,
                n_boot=5000, seed=42
            )
            config_label = f"a={alpha}, {loss}, {stat[:3]}"
            configs.append(config_label)
            pval_matrix.append([res.pvalues[m] for m in model_names])

pval_matrix = np.array(pval_matrix).T  # (modelos x configs)

# Ordenar modelos pelo p-valor medio
mean_pvals = pval_matrix.mean(axis=1)
order = np.argsort(mean_pvals)[::-1]

# Heatmap
fig, ax = plt.subplots(figsize=(16, 10))

# Colormap: vermelho (excluido) -> amarelo (limiar) -> verde (incluido)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "mcs", ["#d32f2f", "#ff9800", "#4caf50"], N=256
)

im = ax.imshow(pval_matrix[order], cmap=cmap, aspect="auto",
               vmin=0, vmax=1)

# Labels
ax.set_yticks(range(len(model_names)))
ax.set_yticklabels([model_names[i] for i in order])
ax.set_xticks(range(len(configs)))
ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)

# Anotar p-valores
for i in range(len(model_names)):
    for j in range(len(configs)):
        p = pval_matrix[order[i], j]
        color = "white" if p < 0.15 or p > 0.85 else "black"
        ax.text(j, i, f"{p:.2f}", ha="center", va="center",
                fontsize=7, color=color)

# Linha de referencia (alpha = 0.10)
ax.set_title("Heatmap de Inclusao no MCS por Configuracao", fontsize=14)
plt.colorbar(im, ax=ax, label="P-valor MCS", shrink=0.8)
plt.tight_layout()
plt.show()
```

### Leitura do Heatmap

!!! info "Como interpretar o heatmap"
    - **Linha verde uniforme**: modelo robustamente no MCS — sempre incluido independente da configuracao
    - **Linha vermelha uniforme**: modelo robustamente excluido — descartar com seguranca
    - **Linha mista (verde/amarelo/vermelho)**: modelo sensivel a especificacao — requer analise cuidadosa
    - **Colunas com muitos vermelhos**: configuracao agressiva (alpha alto, semi-quadratica)
    - **Colunas com muitos verdes**: configuracao conservadora (alpha baixo, range)

## Exemplo Completo: 15 Modelos para Inflacao

```python
import numpy as np
import pandas as pd
from forecastbox.evaluation import model_confidence_set

# ===== DIAGNOSTICO MCS COMPLETO =====

# 1. MCS principal
result = model_confidence_set(
    actual, forecasts,
    alpha=0.10, loss="mse", statistic="range",
    n_boot=10000, seed=42
)

print("DIAGNOSTICO MCS — Inflacao Mensal")
print("=" * 55)
print(f"Periodo: {T} observacoes")
print(f"Modelos candidatos: {len(model_names)}")
print(f"Modelos no MCS: {len(result.included_models)}")
print(f"Alpha: {result.alpha}")
print()

# 2. Ranking completo
print("RANKING POR P-VALOR MCS")
print("-" * 45)
for i, model in enumerate(sorted(result.pvalues.keys(),
                                  key=lambda x: result.pvalues[x],
                                  reverse=True), 1):
    p = result.pvalues[model]
    status = "MCS" if model in result.included_models else "   "
    print(f"  {i:2d}. [{status}] {model:<18} p = {p:.4f}")

# 3. Ordem de eliminacao
print(f"\nOrdem de eliminacao (pior primeiro):")
for i, model in enumerate(result.elimination_order, 1):
    print(f"  {i:2d}. {model}")

# 4. Conclusao
print(f"\n{'='*55}")
print("CONCLUSAO:")
print(f"  - O MCS contem {len(result.included_models)} modelos")
print(f"  - Nucleo superior: {', '.join(result.included_models[:3])}")
print(f"  - Modelos de combinacao dominam o ranking")
print(f"  - {len(result.excluded_models)} modelos podem ser descartados")
```

## Estatistica Range vs Semi-Quadratica

A escolha da estatistica de teste afeta o tamanho do MCS:

| Caracteristica | Range ($T_R$) | Semi-Quadratica ($T_{SQ}$) |
|----------------|---------------|----------------------------|
| Foco | Pior par de modelos | Performance agregada |
| MCS resultante | **Maior** (mais conservador) | **Menor** (mais agressivo) |
| Poder | Menor | Maior com multiplos modelos ruins |
| Recomendacao | Custo alto de excluir bom modelo | Quer MCS mais enxuto |

```python
# Comparar ambas as estatisticas
res_range = model_confidence_set(
    actual, forecasts, alpha=0.10, statistic="range", n_boot=5000, seed=42
)
res_sq = model_confidence_set(
    actual, forecasts, alpha=0.10, statistic="semi_quadratic", n_boot=5000, seed=42
)

print(f"MCS (range):          {len(res_range.included_models)} modelos")
print(f"MCS (semi-quadratic): {len(res_sq.included_models)} modelos")
print(f"\nApenas em range:      {set(res_range.included_models) - set(res_sq.included_models)}")
```

!!! tip "Recomendacao"
    Use **range** como diagnostico principal (mais conservador). Use **semi-quadratic** como verificacao de robustez. Se ambas concordam, o resultado e solido.

## Parametros de Referencia

| Parametro | Tipo | Default | Descricao |
|-----------|------|---------|-----------|
| `actual` | `array` | — | Valores realizados |
| `forecasts` | `dict` | — | Dicionario `{nome: previsao}` com pelo menos 2 modelos |
| `alpha` | `float` | `0.10` | Nivel de significancia |
| `loss` | `str` | `"mse"` | Funcao de perda: `"mse"`, `"mae"` |
| `statistic` | `str` | `"range"` | Estatistica de teste: `"range"`, `"semi_quadratic"` |
| `n_boot` | `int` | `5000` | Replicacoes bootstrap |
| `block_length` | `int` | `None` | Tamanho do bloco (auto: $\lfloor T^{1/3} \rfloor$) |
| `seed` | `int` | `None` | Semente para reproducibilidade |

!!! info "See Also"
    - :material-book-open-variant: **Teoria**: [MCS — Teoria](../theory/mcs-theory.md) — propriedades assintoticas e inferencia bootstrap
    - :material-notebook-edit: **User Guide**: [Model Confidence Set](../user-guide/evaluation/mcs.md) — formulacao e algoritmo de eliminacao sequencial
    - :material-link-variant: **Relacionado**: [DM Test](dm-test.md) — comparacao par-a-par (2 modelos)
    - :material-link-variant: **Relacionado**: [GW Test](gw-test.md) — teste condicional de superioridade
