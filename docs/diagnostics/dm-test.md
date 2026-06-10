---
title: "DM Test — Diagnostico de Superioridade Preditiva"
description: "Workflow pratico do teste Diebold-Mariano como ferramenta de diagnostico: interpretacao de loss differentials, armadilhas comuns e visualizacao."
---

# DM Test — Diagnostico de Superioridade Preditiva

!!! abstract "Key Takeaway"
    O teste Diebold-Mariano responde a pergunta mais fundamental da avaliacao de previsoes: **a diferenca de performance entre dois modelos e estatisticamente significativa, ou apenas ruido amostral?** Esta pagina foca no uso pratico como ferramenta de diagnostico — para a formulacao teorica, veja [Teste Diebold-Mariano](../user-guide/evaluation/diebold-mariano.md).

## Workflow de Diagnostico

O DM test deve ser aplicado como um processo estruturado, nao como um numero isolado. Siga este workflow:

```text
1. Definir funcao de perda adequada ao problema
2. Calcular loss differentials d_t
3. Verificar estacionariedade de d_t
4. Aplicar teste com correcao de horizonte
5. Interpretar p-valor E tamanho do efeito
```

### Passo 1: Escolher a Funcao de Perda

A escolha da funcao de perda **define o que "melhor" significa**. Diferentes funcoes podem inverter o ranking de modelos.

=== "Perda Quadratica (MSE)"

    $$
    L(e_t) = e_t^2
    $$

    Penaliza erros grandes proporcionalmente mais. Adequada quando erros grandes sao desproporcionalmente custosos.

    ```python
    from forecastbox.evaluation import diebold_mariano

    dm = diebold_mariano(actual, forecast1, forecast2, loss="mse")
    ```

=== "Perda Absoluta (MAE)"

    $$
    L(e_t) = |e_t|
    $$

    Robusta a outliers. Use quando o custo do erro e proporcional ao tamanho.

    ```python
    dm = diebold_mariano(actual, forecast1, forecast2, loss="mae")
    ```

=== "Perda MAPE"

    $$
    L(e_t) = \left|\frac{e_t}{y_t}\right|
    $$

    Penaliza erros relativos ao nivel da serie. Use quando a escala varia ao longo do tempo.

    ```python
    dm = diebold_mariano(actual, forecast1, forecast2, loss="mape")
    ```

!!! warning "A funcao de perda muda a conclusao"
    Dois modelos podem ter MSE muito diferente mas MAE similar — especialmente quando um modelo acerta o nivel mas erra mais em periodos de alta volatilidade. **Sempre relate qual funcao de perda foi usada.**

### Passo 2: Calcular e Inspecionar Loss Differentials

O loss differential $d_t = L(e_{1,t}) - L(e_{2,t})$ e a serie fundamental do teste. Antes de olhar a estatistica, **inspecione visualmente** a serie $d_t$:

```python
import numpy as np
import matplotlib.pyplot as plt
from forecastbox.evaluation import diebold_mariano

# Dados: actual, forecast_arima, forecast_ets
dm = diebold_mariano(actual, forecast_arima, forecast_ets, loss="mse", h=1)

# Extrair loss differentials
d_t = dm.loss_differential
T = len(d_t)

# Visualizacao dos loss differentials
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

# Painel 1: serie d_t com media e bandas de confianca
mean_d = np.mean(d_t)
se_d = np.std(d_t, ddof=1) / np.sqrt(T)
ci_upper = mean_d + 1.96 * se_d
ci_lower = mean_d - 1.96 * se_d

axes[0].plot(d_t, color="steelblue", linewidth=0.8, alpha=0.8)
axes[0].axhline(mean_d, color="red", linewidth=1.5, label=f"Media = {mean_d:.4f}")
axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[0].fill_between(
    range(T), ci_lower, ci_upper,
    color="red", alpha=0.15, label=f"IC 95%: [{ci_lower:.4f}, {ci_upper:.4f}]"
)
axes[0].set_ylabel("$d_t = L(e_1) - L(e_2)$")
axes[0].set_title("Loss Differentials ao Longo do Tempo")
axes[0].legend()

# Painel 2: media movel de d_t (estabilidade temporal)
window = min(12, T // 4)
rolling_mean = np.convolve(d_t, np.ones(window)/window, mode="valid")
axes[1].plot(range(window-1, T), rolling_mean, color="teal", linewidth=1.5)
axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[1].set_ylabel(f"Media movel ({window} periodos)")
axes[1].set_xlabel("Periodo")
axes[1].set_title("Estabilidade Temporal da Superioridade")

plt.tight_layout()
plt.show()
```

!!! tip "O que procurar no grafico"
    - **$d_t$ centrado em zero**: modelos equivalentes — DM provavelmente nao rejeita
    - **$d_t$ consistentemente positivo/negativo**: superioridade clara
    - **$d_t$ muda de sinal ao longo do tempo**: superioridade instavel — considere [GW Test](gw-test.md)
    - **Outliers em $d_t$**: poucos periodos extremos podem dominar a media — considere MAE

### Passo 3: Verificar Estacionariedade

O DM test assume que $\{d_t\}$ e **estacionario com covariancia**. Se a serie de loss differentials tem tendencia ou quebra estrutural, o teste pode ser invalido.

```python
from chronobox.tests import adf_test

# Testar estacionariedade dos loss differentials
adf = adf_test(d_t)
print(f"ADF statistic: {adf.statistic:.4f}")
print(f"p-valor:       {adf.pvalue:.4f}")
```

!!! warning "Se $d_t$ nao e estacionario"
    Se o ADF nao rejeita a raiz unitaria em $d_t$, a superioridade relativa esta **mudando ao longo do tempo**. Neste caso:

    1. Considere o [Teste Giacomini-White](gw-test.md) com instrumentos
    2. Aplique o DM em sub-amostras (split-sample)
    3. Nao confie no p-valor do DM padrao

### Passo 4: Aplicar o Teste com Correcao Adequada

```python
from forecastbox.evaluation import diebold_mariano

# Horizonte h=1: previsao um passo a frente
dm = diebold_mariano(
    actual, forecast_arima, forecast_ets,
    h=1,
    loss="mse",
    one_sided=False,      # bilateral: testar se sao diferentes
    hln_correction=True,  # correcao para amostra finita
)

print(f"DM* statistic: {dm.statistic:.4f}")
print(f"p-valor:       {dm.pvalue:.4f}")
print(f"Media d_t:     {dm.mean_loss_diff:.6f}")
print(f"Conclusao:     {dm.conclusion()}")
```

```text
DM* statistic: -2.147
p-valor:       0.0384
Media d_t:     -0.001234
Conclusao:     Rejeita H0 a 5%. Modelo 2 tem desempenho
               significativamente diferente do Modelo 1.
```

### Passo 5: Interpretar P-valor E Tamanho do Efeito

O p-valor sozinho **nao e suficiente**. Com amostra grande, diferencas minusculas sao "significativas". Sempre avalie o **tamanho do efeito**:

$$
\text{Efeito relativo} = \frac{|\bar{d}|}{\bar{L}_1} \times 100\%
$$

onde $\bar{L}_1$ e a perda media do modelo de referencia.

```python
# Tamanho do efeito
loss_1 = np.mean((actual - forecast_arima) ** 2)  # MSE do modelo 1
loss_2 = np.mean((actual - forecast_ets) ** 2)     # MSE do modelo 2
efeito_relativo = abs(dm.mean_loss_diff) / loss_1 * 100

print(f"MSE modelo 1 (ARIMA): {loss_1:.6f}")
print(f"MSE modelo 2 (ETS):   {loss_2:.6f}")
print(f"Diferenca relativa:   {efeito_relativo:.2f}%")
```

| P-valor | Efeito relativo | Interpretacao |
|---------|-----------------|---------------|
| $< 0.05$ | $> 10\%$ | Diferenca significativa e relevante |
| $< 0.05$ | $< 2\%$ | Estatisticamente significativa mas economicamente irrelevante |
| $> 0.10$ | qualquer | Sem evidencia de diferenca |
| $0.05 - 0.10$ | $> 5\%$ | Zona cinza — coletar mais dados ou usar [MCS](mcs-diagnostic.md) |

## Armadilhas Comuns

### Comparacoes Multiplas

!!! warning "Problema: inflar falsos positivos"
    Se voce compara 10 modelos par-a-par, sao $\binom{10}{2} = 45$ testes. Com $\alpha = 0.05$, espera-se ~2 rejeicoes **por acaso**. Isso e o problema de comparacoes multiplas.

**Solucao: Correcao de Bonferroni**

$$
\alpha_{\text{corrigido}} = \frac{\alpha}{k}
$$

onde $k$ e o numero de comparacoes.

```python
import itertools

models = {
    "ARIMA": forecast_arima,
    "ETS": forecast_ets,
    "VAR": forecast_var,
    "Naive": forecast_naive,
    "Theta": forecast_theta,
}

pairs = list(itertools.combinations(models.keys(), 2))
n_comparisons = len(pairs)
alpha_bonferroni = 0.05 / n_comparisons

print(f"Comparacoes: {n_comparisons}")
print(f"Alpha corrigido (Bonferroni): {alpha_bonferroni:.4f}")
print()

for m1, m2 in pairs:
    dm = diebold_mariano(actual, models[m1], models[m2], loss="mse", h=1)
    sig = "***" if dm.pvalue < alpha_bonferroni else ""
    print(f"{m1:8s} vs {m2:8s}: DM*={dm.statistic:+.3f}, p={dm.pvalue:.4f} {sig}")
```

!!! tip "Alternativa: use o MCS"
    Para mais de 2-3 modelos, o [Model Confidence Set](mcs-diagnostic.md) ja controla comparacoes multiplas internamente. E a abordagem recomendada.

### Horizonte Longo e Correlacao Serial

!!! warning "Problema: erros autocorrelacionados em $h > 1$"
    Para previsoes multi-step ($h > 1$), os erros de previsao sao autocorrelacionados por construcao (overlapping forecasts). Isso inflaciona a estatistica DM se nao corrigido.

O DM original usa estimador HAC (Newey-West) com truncamento em $h - 1$:

$$
\hat{\sigma}_{\bar{d}}^2 = \frac{1}{T} \left[ \hat{\gamma}_0 + 2 \sum_{k=1}^{h-1} \hat{\gamma}_k \right]
$$

```python
# Para horizonte h=4 (trimestral)
dm_h4 = diebold_mariano(
    actual, forecast1, forecast2,
    h=4,               # ajuste HAC automatico com lag = h-1 = 3
    loss="mse",
    hln_correction=True  # essencial para h grande com amostra finita
)

print(f"DM* (h=4): {dm_h4.statistic:.4f}")
print(f"p-valor:   {dm_h4.pvalue:.4f}")
```

!!! warning "Cuidado com horizontes muito longos"
    Para $h > T/4$, o estimador HAC pode ter poucos graus de liberdade. Neste caso, o poder do teste e baixo e os resultados sao pouco confiaveis.

### Poucos Dados e Baixo Poder

!!! warning "Problema: nao-rejeicao nao significa igualdade"
    Com $T < 30$, o teste DM tem **baixo poder**: pode nao rejeitar $H_0$ mesmo quando um modelo e substancialmente melhor. A nao-rejeicao significa "evidencia insuficiente", nao "modelos iguais".

**Diagnostico de poder:**

```python
# Quanto dado seria necessario para detectar a diferenca observada?
from scipy import stats

d_bar = np.mean(d_t)
s_d = np.std(d_t, ddof=1)
effect_size = abs(d_bar) / s_d  # Cohen's d

# Poder do teste atual
z_alpha = stats.norm.ppf(0.975)  # bilateral, alpha=0.05
power = stats.norm.cdf(effect_size * np.sqrt(T) - z_alpha)

print(f"Tamanho do efeito (Cohen's d): {effect_size:.4f}")
print(f"Poder do teste (T={T}):        {power:.2%}")
print(f"T necessario para poder 80%:   {int(np.ceil((z_alpha + 0.84)**2 / effect_size**2))}")
```

| Tamanho da amostra | Efeito detectavel (Cohen's d) | Recomendacao |
|--------------------|-------------------------------|--------------|
| $T < 30$ | Apenas efeitos grandes ($> 0.5$) | Nao confie em nao-rejeicao |
| $30 \leq T < 100$ | Efeitos moderados ($> 0.3$) | Resultados indicativos |
| $T \geq 100$ | Efeitos pequenos ($> 0.15$) | Resultados confiaveis |

## Exemplo Completo: ARIMA vs ETS para Inflacao

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from forecastbox.evaluation import diebold_mariano

# Dados simulados: 60 observacoes de previsao fora da amostra
np.random.seed(42)
T = 60
actual = np.cumsum(np.random.randn(T) * 0.5) + 100

# Previsoes de dois modelos
forecast_arima = actual + np.random.randn(T) * 0.8 - 0.05
forecast_ets = actual + np.random.randn(T) * 0.9 + 0.02

# ===== WORKFLOW COMPLETO =====

# 1. Teste bilateral: os modelos diferem?
dm = diebold_mariano(actual, forecast_arima, forecast_ets, h=1, loss="mse")
print("=" * 50)
print("DIAGNOSTICO DM: ARIMA vs ETS")
print("=" * 50)
print(f"DM* statistic: {dm.statistic:+.4f}")
print(f"p-valor:       {dm.pvalue:.4f}")
print(f"Media d_t:     {dm.mean_loss_diff:+.6f}")
print(f"Conclusao:     {dm.conclusion()}")

# 2. Tamanho do efeito
mse_arima = np.mean((actual - forecast_arima)**2)
mse_ets = np.mean((actual - forecast_ets)**2)
efeito = abs(dm.mean_loss_diff) / mse_arima * 100
print(f"\nMSE ARIMA:     {mse_arima:.4f}")
print(f"MSE ETS:       {mse_ets:.4f}")
print(f"Efeito relat.: {efeito:.1f}%")

# 3. Visualizacao
d_t = dm.loss_differential
fig, axes = plt.subplots(3, 1, figsize=(12, 9))

# Loss differentials
mean_d = np.mean(d_t)
se_d = np.std(d_t, ddof=1) / np.sqrt(T)
axes[0].bar(range(T), d_t, color=["steelblue" if d > 0 else "coral" for d in d_t],
            alpha=0.7, width=0.8)
axes[0].axhline(mean_d, color="red", linewidth=2, label=f"$\\bar{{d}}$ = {mean_d:.4f}")
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].fill_between(range(T), mean_d - 1.96*se_d, mean_d + 1.96*se_d,
                     color="red", alpha=0.1)
axes[0].set_ylabel("$d_t$")
axes[0].set_title("Loss Differentials: $d_t = L(e_{ARIMA}) - L(e_{ETS})$")
axes[0].legend()

# Soma acumulada (CUSUM de d_t)
cusum = np.cumsum(d_t - mean_d)
axes[1].plot(cusum, color="teal", linewidth=1.5)
axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[1].set_ylabel("CUSUM de $d_t$")
axes[1].set_title("Estabilidade: CUSUM dos Loss Differentials")

# Distribuicao de d_t
axes[2].hist(d_t, bins=20, color="steelblue", alpha=0.7, edgecolor="white", density=True)
axes[2].axvline(mean_d, color="red", linewidth=2, label=f"Media = {mean_d:.4f}")
axes[2].axvline(0, color="black", linewidth=0.8, linestyle="--")
axes[2].set_xlabel("$d_t$")
axes[2].set_ylabel("Densidade")
axes[2].set_title("Distribuicao dos Loss Differentials")
axes[2].legend()

plt.tight_layout()
plt.show()
```

### Interpretacao do Exemplo

!!! info "Leitura dos resultados"
    - **$d_t > 0$**: ARIMA teve perda maior que ETS naquele periodo (ETS melhor)
    - **$d_t < 0$**: ARIMA teve perda menor que ETS naquele periodo (ARIMA melhor)
    - **$\bar{d} < 0$ com $p < 0.05$**: ARIMA e significativamente melhor na media
    - **CUSUM estavel**: a superioridade e consistente ao longo do tempo
    - **CUSUM com tendencia**: a superioridade pode estar mudando — investigar com [GW Test](gw-test.md)

## Teste Unilateral vs Bilateral

=== "Bilateral (padrao)"

    $$
    H_0: E[d_t] = 0 \quad \text{vs} \quad H_1: E[d_t] \neq 0
    $$

    Use quando nao tem expectativa previa de qual modelo e melhor.

    ```python
    dm = diebold_mariano(actual, f1, f2, one_sided=False)
    ```

=== "Unilateral (modelo 1 melhor)"

    $$
    H_0: E[d_t] \geq 0 \quad \text{vs} \quad H_1: E[d_t] < 0
    $$

    Use quando quer testar se modelo 1 tem perda **menor** que modelo 2.

    ```python
    dm = diebold_mariano(actual, f1, f2, one_sided=True)
    ```

!!! tip "Quando usar unilateral"
    O teste unilateral tem **mais poder** (metade do p-valor bilateral). Use quando:

    - Voce tem hipotese previa clara (e.g., modelo complexo vs naive)
    - O objetivo e confirmar superioridade, nao apenas diferenca
    - A direcao da alternativa foi definida **antes** de ver os dados

## Parametros de Referencia

| Parametro | Tipo | Default | Descricao |
|-----------|------|---------|-----------|
| `actual` | `array` | — | Valores realizados |
| `forecast1` | `array` | — | Previsoes do modelo 1 |
| `forecast2` | `array` | — | Previsoes do modelo 2 |
| `h` | `int` | `1` | Horizonte de previsao (truncamento HAC = $h - 1$) |
| `loss` | `str` | `"mse"` | Funcao de perda: `"mse"`, `"mae"`, `"mape"` |
| `one_sided` | `bool` | `False` | `True` para teste unilateral ($H_1$: modelo 1 melhor) |
| `hln_correction` | `bool` | `True` | Correcao Harvey-Leybourne-Newbold para amostra finita |

## Checklist de Diagnostico

Antes de reportar o resultado do DM test, verifique:

- [ ] **Funcao de perda**: adequada ao problema? Resultado muda com outra funcao?
- [ ] **Estacionariedade**: $d_t$ e estacionario? (ADF test)
- [ ] **Horizonte**: correcao HAC adequada para $h > 1$?
- [ ] **Tamanho do efeito**: a diferenca e economicamente relevante, nao apenas estatistica?
- [ ] **Poder**: com $T$ pequeno, nao-rejeicao e confiavel?
- [ ] **Comparacoes multiplas**: se mais de 2 modelos, corrigiu alpha ou usou [MCS](mcs-diagnostic.md)?
- [ ] **Estabilidade temporal**: a superioridade e constante? (inspecionar CUSUM)

!!! info "See Also"
    - :material-book-open-variant: **Teoria**: [Teste Diebold-Mariano](../theory/evaluation-theory.md) — formulacao teorica e propriedades assintoticas
    - :material-notebook-edit: **User Guide**: [Diebold-Mariano — Formulacao](../user-guide/evaluation/diebold-mariano.md) — formulacao e implementacao
    - :material-link-variant: **Relacionado**: [MCS Diagnostic](mcs-diagnostic.md) — comparacao multipla com controle de tamanho
    - :material-link-variant: **Relacionado**: [GW Test](gw-test.md) — quando a superioridade e condicional ou instavel
    - :material-link-variant: **Relacionado**: [Metricas](../user-guide/evaluation/metrics.md) — funcoes de perda disponiveis
