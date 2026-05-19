---
title: Troubleshooting
description: Solucoes para problemas comuns ao usar o forecastbox - erros de importacao, convergencia, performance e visualizacao
---

# Troubleshooting

Solucoes para problemas comuns ao usar o **forecastbox**. Cada entrada segue o formato: **Sintoma**, **Causa**, **Solucao** e **Codigo**.

---

## 1. ImportError: No module named 'kalmanbox'

**Sintoma**

```text
ImportError: No module named 'kalmanbox'
```

Ocorre ao tentar usar DFM, pesos time-varying ou qualquer recurso que dependa do filtro de Kalman.

**Causa**

O `kalmanbox` e uma dependencia opcional e nao e instalado automaticamente com `pip install forecastbox`.

**Solucao**

```bash
# Instalar kalmanbox separadamente
pip install kalmanbox

# Ou instalar forecastbox com extras
pip install forecastbox[kalman]
```

**Verificacao**

```python
import forecastbox as fb
deps = fb.check_dependencies()
print(deps)
# {'chronobox': True, 'kalmanbox': True, 'archbox': False}
```

!!! tip "Ambiente virtual"
    Certifique-se de instalar no mesmo ambiente virtual onde o forecastbox esta instalado. Use `pip list | grep kalmanbox` para verificar.

---

## 2. AutoARIMA nao converge

**Sintoma**

```text
ConvergenceWarning: Maximum likelihood optimization did not converge.
```

Ou o `AutoARIMA` retorna ordens inesperadas (ex: ARIMA(0,0,0)).

**Causa**

- Serie muito curta para a grade de busca
- Dados com outliers extremos ou quebras estruturais
- Metodo de otimizacao inadequado para a serie

**Solucao**

```python
import forecastbox as fb

# 1. Reduzir a grade de busca
model = fb.AutoARIMA(
    max_p=3,       # Reduzir de 5 (padrao) para 3
    max_q=3,
    max_d=1,
    max_P=1,       # Parte sazonal
    max_Q=1,
    stepwise=True  # Busca stepwise (mais rapida)
)

# 2. Mudar metodo de otimizacao
model = fb.AutoARIMA(
    method="css-mle",    # Iniciar com CSS, refinar com MLE
    solver="nm"          # Nelder-Mead (mais robusto)
)

# 3. Tratar outliers antes
from chronobox import detect_outliers
y_clean = detect_outliers(y, method="iqr", replace="interpolate")
model.fit(y_clean)
```

!!! warning "Diagnostico"
    Sempre verifique os residuos apos o ajuste:
    ```python
    model.fit(y)
    model.plot_diagnostics()
    ```

---

## 3. Pesos BMA sao todos iguais

**Sintoma**

Apos ajustar `BMA()`, todos os modelos recebem peso aproximadamente igual ($w_k \approx 1/K$).

**Causa**

- **Prior uniforme** com modelos de performance similar
- **Marginal likelihood** numericamente identica para todos os modelos
- Dados insuficientes para discriminar entre modelos

**Solucao**

```python
import forecastbox as fb

# 1. Usar janela de avaliacao maior
combiner = fb.BMA(
    train_window=60  # Usar ultimos 60 periodos para avaliar
)

# 2. Ajustar o prior
combiner = fb.BMA(
    prior="performance",  # Prior proporcional ao historico
    prior_window=24       # Baseado em 24 periodos
)

# 3. Verificar marginal likelihood
combiner.fit(forecasts_dict, y_actual)
print(combiner.marginal_likelihoods_)
# Se valores muito proximos, os modelos sao equivalentes

# 4. Considerar subsets com modelos mais distintos
# Remover modelos redundantes antes de combinar
```

!!! info "Interpretacao"
    Pesos iguais nao sao necessariamente um problema. A literatura mostra que a media simples frequentemente supera metodos sofisticados de combinacao (Timmermann, 2006).

---

## 4. MCS elimina todos os modelos

**Sintoma**

Apos rodar `ModelConfidenceSet`, o conjunto sobrevivente esta vazio ou contem apenas 1 modelo.

**Causa**

- **Alpha muito alto** (ex: $\alpha = 0.25$): criterio muito exigente
- **Poucos dados**: poder estatistico insuficiente
- **Um modelo claramente dominante**: resultado esperado

**Solucao**

```python
import forecastbox as fb

# 1. Reduzir alpha (menos eliminacoes)
mcs = fb.ModelConfidenceSet(alpha=0.05)  # Menos restritivo
mcs.fit(forecasts_dict, y_actual)
print(f"Sobreviventes: {mcs.surviving_models()}")

# 2. Verificar MCS p-values para entender a hierarquia
print(mcs.pvalues_)
# Modelo     p-value
# arima      1.000    <- sempre sobrevive
# ets        0.342    <- sobrevive com alpha=0.10
# theta      0.087    <- eliminado com alpha=0.10
# naive      0.003    <- claramente inferior

# 3. Aumentar janela de avaliacao
# Mais dados = mais poder para diferenciar modelos

# 4. Verificar se dados estao corretos
# Erros de alinhamento temporal sao causa comum
assert len(y_actual) == len(list(forecasts_dict.values())[0])
```

!!! tip "Regra pratica"
    Use $\alpha = 0.10$ para aplicacoes praticas e $\alpha = 0.05$ ou $\alpha = 0.01$ para publicacoes academicas.

---

## 5. Pipeline trava em modelo X

**Sintoma**

A execucao do `Pipeline` para de responder em um modelo especifico, sem erro aparente.

**Causa**

- Modelo com otimizacao lenta (grade grande, muitos parametros)
- Dados com sazonalidade complexa que aumenta tempo de busca
- Deadlock em processamento paralelo

**Solucao**

```python
import forecastbox as fb

# 1. Definir timeout por modelo
pipeline = fb.Pipeline(
    models=["arima", "ets", "theta"],
    timeout=120  # Maximo 120 segundos por modelo
)

# 2. Ativar logs para identificar onde trava
import logging
logging.basicConfig(level=logging.INFO)
fb.set_log_level("DEBUG")

pipeline.run(y)
# INFO: Fitting AutoARIMA... (started 10:00:01)
# INFO: AutoARIMA completed in 3.2s
# INFO: Fitting AutoETS... (started 10:00:04)
# DEBUG: AutoETS grid search: 45/128 combinations...

# 3. Usar modo sequencial para debug
pipeline = fb.Pipeline(
    models=["arima", "ets", "theta"],
    n_jobs=1,       # Sequencial
    verbose=True    # Progresso detalhado
)

# 4. Desativar modelos problematicos
pipeline = fb.Pipeline(
    models=["arima", "theta"],  # Remover ETS se problematico
    fallback="naive"            # Fallback se modelo falhar
)
```

---

## 6. Graficos nao aparecem no Jupyter

**Sintoma**

Chamadas como `fb.plot_forecast()` ou `results.plot()` nao exibem graficos no Jupyter Notebook.

**Causa**

- Backend do matplotlib nao configurado para inline
- Conflito com outros backends (Qt, Tk)
- Versao do `ipykernel` incompativel

**Solucao**

```python
# No inicio do notebook, executar:
%matplotlib inline

# Ou para graficos interativos:
%matplotlib widget  # Requer ipympl

# Verificar backend atual
import matplotlib
print(matplotlib.get_backend())
# Deve ser: 'module://matplotlib_inline.backend_inline'

# Forcar backend
matplotlib.use('Agg')  # Para salvar em arquivo
# OU
matplotlib.use('module://matplotlib_inline.backend_inline')  # Para Jupyter
```

Se o problema persistir:

```bash
# Reinstalar ipykernel
pip install --upgrade ipykernel matplotlib

# Para graficos interativos
pip install ipympl
```

!!! tip "VS Code"
    No VS Code com Jupyter, use `%matplotlib inline`. O backend `widget` pode nao funcionar em todas as versoes.

---

## 7. Erro de memoria com muitas series

**Sintoma**

```text
MemoryError: Unable to allocate X.XX GiB for an array
```

Ocorre ao processar centenas ou milhares de series simultaneamente.

**Causa**

- Todas as series carregadas na memoria simultaneamente
- Historico muito longo sem necessidade
- Modelos armazenando dados intermediarios

**Solucao**

```python
import forecastbox as fb

# 1. Processar em lotes (batch processing)
pipeline = fb.Pipeline(
    models=["arima", "ets"],
    batch_size=50,   # Processar 50 series por vez
    n_jobs=4         # 4 processos paralelos
)
results = pipeline.run_batch(series_dict)

# 2. Limitar historico
# Usar apenas ultimos N periodos
for name, series in series_dict.items():
    series_dict[name] = series.iloc[-120:]  # Ultimos 10 anos mensais

# 3. Liberar memoria entre modelos
pipeline = fb.Pipeline(
    models=["arima", "ets"],
    gc_between_models=True  # Garbage collection entre modelos
)

# 4. Usar dtype mais eficiente
import numpy as np
y = y.astype(np.float32)  # 32-bit em vez de 64-bit
```

!!! info "Estimativa de memoria"
    Regra pratica: cada serie com $T=240$ observacoes e $K=5$ modelos requer ~2MB. Para 10.000 series: ~20GB.

---

## 8. Previsao condicional da resultados absurdos

**Sintoma**

Previsoes condicionais com valores extremos, explosivos ou negativos quando nao deveriam ser.

**Causa**

- Condicoes incompativeis com a dinamica do modelo
- Horizonte condicional muito longo
- Modelo VAR instavel (raizes proximas do circulo unitario)

**Solucao**

```python
import forecastbox as fb

# 1. Verificar estabilidade do modelo
model = fb.AutoVAR()
model.fit(df)
print(model.is_stable())       # Deve ser True
print(model.max_eigenvalue())   # Deve ser < 1.0

# 2. Verificar se condicoes sao razoaveis
# Condicoes nao devem implicar mudancas abruptas
conditions = {"selic": [10.5, 10.5, 10.5, 10.5]}
# Verificar: o nivel atual da selic esta proximo de 10.5?
print(f"Selic atual: {df['selic'].iloc[-1]}")

# 3. Usar restricoes soft em vez de hard
cond_forecast = fb.ConditionalForecast(
    model=model,
    conditions=conditions,
    conditions_std={"selic": [0.5, 0.5, 0.5, 0.5]},
    method="soft"
)

# 4. Limitar horizonte condicional
# Condicoes para mais de 4-8 periodos tendem a ser instáveis
cond_forecast = fb.ConditionalForecast(
    model=model,
    conditions={"selic": [10.5, 10.5]},  # Apenas 2 periodos
    h=4  # Previsao total de 4, condicional nos 2 primeiros
)
```

!!! warning "Regra de ouro"
    Se a previsao condicional difere muito da incondicional, questione as condicoes antes de questionar o modelo.

---

## 9. Resultados diferentes entre execucoes

**Sintoma**

Rodar o mesmo codigo duas vezes produz resultados ligeiramente diferentes.

**Causa**

- Componentes estocasticos (bootstrap, MCMC no BMA)
- Inicializacao aleatoria de otimizadores
- Paralelismo nao-deterministico

**Solucao**

```python
import forecastbox as fb
import numpy as np

# Fixar seed global
np.random.seed(42)
fb.set_seed(42)

# Fixar seed por componente
mcs = fb.ModelConfidenceSet(
    alpha=0.10,
    n_boot=5000,
    random_state=42  # Seed para bootstrap
)

bma = fb.BMA(random_state=42)

# Pipeline com seed fixa
pipeline = fb.Pipeline(
    models=["arima", "ets", "theta"],
    combination="bma",
    random_state=42,
    n_jobs=1  # Sequencial para reproducibilidade total
)
```

!!! info "Paralelismo e reproducibilidade"
    Com `n_jobs > 1`, a ordem de execucao pode variar entre rodadas. Para reproducibilidade exata, use `n_jobs=1`. Com `n_jobs > 1`, os resultados serao quase identicos mas podem diferir na ultima casa decimal.

---

## 10. Erro ao combinar previsoes de modelos com horizontes diferentes

**Sintoma**

```text
ValueError: All forecasts must have the same length. Got lengths: [12, 8, 6]
```

**Causa**

Modelos produzindo previsoes de comprimentos diferentes, geralmente porque alguns modelos nao suportam horizontes longos.

**Solucao**

```python
import forecastbox as fb

# 1. Forcar mesmo horizonte para todos
forecasts = {}
for name, model in models.items():
    try:
        forecasts[name] = model.forecast(h=12)
    except Exception as e:
        print(f"Modelo {name} falhou para h=12: {e}")

# 2. Usar Pipeline que alinha automaticamente
pipeline = fb.Pipeline(
    models=["arima", "ets", "theta"],
    h=12,
    align_forecasts=True  # Trunca ou preenche para alinhar
)

# 3. Combinar por horizonte
combiner = fb.OLSCombination()
for h in range(1, 13):
    h_forecasts = {k: v[:h] for k, v in forecasts.items()
                   if len(v) >= h}
    combiner.fit(h_forecasts, y_actual[-len(list(h_forecasts.values())[0]):])
```

---

## 11. Dados com valores faltantes causam erro

**Sintoma**

```text
ValueError: Input contains NaN. AutoARIMA does not support missing values.
```

**Causa**

Series temporais com valores `NaN` ou lacunas no indice temporal.

**Solucao**

```python
import forecastbox as fb
import pandas as pd

# 1. Verificar NaNs
print(f"NaNs: {y.isna().sum()}")
print(f"Posicoes: {y[y.isna()].index.tolist()}")

# 2. Interpolar valores faltantes
y_filled = y.interpolate(method="time")  # Interpolacao temporal

# 3. Preencher com chronobox
from chronobox import fill_missing
y_filled = fill_missing(y, method="spline", order=3)

# 4. Usar modelos que aceitam NaN
# DFM e modelos estado-espaco lidam com NaN nativamente
model = fb.DFM(n_factors=2)  # Aceita NaN via filtro de Kalman
model.fit(X_with_nans)
```

!!! tip "Verificacao de indice"
    Gaps no `DatetimeIndex` sao diferentes de NaN nos valores. Verifique ambos:
    ```python
    expected = pd.date_range(y.index[0], y.index[-1], freq="MS")
    missing_dates = expected.difference(y.index)
    print(f"Datas faltantes: {missing_dates}")
    ```

---

## 12. Previsao sazonal incorreta

**Sintoma**

O modelo ignora a sazonalidade ou aplica padrao sazonal errado (ex: sazonal mensal em dados trimestrais).

**Causa**

- Frequencia do indice nao detectada corretamente
- Parametro `seasonal_period` incorreto
- Serie muito curta para capturar sazonalidade (< 2 ciclos completos)

**Solucao**

```python
import forecastbox as fb

# 1. Verificar frequencia detectada
print(f"Frequencia: {y.index.freq}")
# Se None, definir explicitamente:
y.index.freq = "MS"  # Mensal

# 2. Especificar periodo sazonal
model = fb.AutoARIMA(
    seasonal=True,
    m=12  # 12 para mensal, 4 para trimestral
)

# 3. Verificar se ha sazonalidade
from chronobox import seasonal_decompose
decomp = seasonal_decompose(y, model="additive", period=12)
decomp.plot()
# Se componente sazonal e flat, a serie nao tem sazonalidade

# 4. Testar com e sem sazonalidade
model_s = fb.AutoARIMA(seasonal=True, m=12).fit(y)
model_ns = fb.AutoARIMA(seasonal=False).fit(y)
print(f"AIC sazonal:     {model_s.aic_}")
print(f"AIC nao-sazonal: {model_ns.aic_}")
```

!!! warning "Minimo de dados"
    Para capturar sazonalidade de forma confiavel, voce precisa de pelo menos **3 ciclos completos**: 36 observacoes para dados mensais, 12 para dados trimestrais.
