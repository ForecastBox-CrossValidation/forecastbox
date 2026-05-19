---
title: Getting Started
description: Instale o forecastbox e faca sua primeira previsao em poucos minutos
---

# Getting Started

Bem-vindo ao **forecastbox**! Esta secao vai te levar de zero a previsoes
em poucos minutos.

<div class="grid cards" markdown>

- :material-download: **[Instalacao](installation.md)**

    Instale o forecastbox e verifique seu setup

- :material-rocket-launch: **[Quickstart](quickstart.md)**

    Faca sua primeira previsao em 5 minutos

- :material-book-open-variant: **[Conceitos Fundamentais](core-concepts.md)**

    Entenda a arquitetura do forecastbox e o fluxo de trabalho

- :material-map-marker-path: **[Escolhendo o Metodo](choosing-method.md)**

    Guia de decisao para selecionar a abordagem certa

</div>

---

## Roteiro Recomendado

| # | Pagina | Tempo | Descricao |
|:-:|--------|:-----:|-----------|
| 1 | [Instalacao](installation.md) | 2 min | Setup do ambiente e verificacao |
| 2 | [Quickstart](quickstart.md) | 5 min | Primeira previsao com auto-forecast, combinacao e avaliacao |
| 3 | [Conceitos Fundamentais](core-concepts.md) | 10 min | Entender a arquitetura e os modulos |
| 4 | [Escolhendo o Metodo](choosing-method.md) | 5 min | Qual abordagem usar para seu problema |

---

## Pre-requisitos

| Requisito | Versao | Notas |
|-----------|--------|-------|
| Python | >= 3.11 | Testado com 3.11 e 3.12 |
| numpy | >= 1.24 | Operacoes numericas |
| pandas | >= 2.0 | Manipulacao de dados |
| matplotlib | >= 3.7 | Visualizacao |

!!! note "Dependencias do ecossistema"

    O forecastbox faz parte do ecossistema **NodesEcon**. Para funcionalidades
    avancadas, voce pode precisar de pacotes adicionais:

    - **chronobox** -- series temporais e transformacoes (instalado automaticamente no futuro)
    - **kalmanbox** -- filtro de Kalman para nowcasting (extra `[nowcasting]`)
    - **archbox** -- modelos GARCH para volatilidade (extra `[full]`)
