* ============================================================
* Validacao Stata: MIDAS Regression
* Requer Stata 17+ com comando midas
* ============================================================
* Este script replica a estimacao MIDAS feita pelo forecastbox
* para fins de validacao cruzada, usando U-MIDAS como alternativa
* acessivel ao comando `midas` proprietario do Stata 17+.
*
* Dados: ../data/mixed_freq.csv (mesmo dataset dos notebooks Python)
*
* Limitacoes do Stata para nowcasting via MIDAS:
* - O comando `midas` so esta disponivel a partir do Stata 17+,
*   versoes anteriores nao possuem suporte nativo.
* - U-MIDAS (unrestricted MIDAS) via `regress` e uma alternativa,
*   mas nao impoe restricoes polinomiais nos pesos (Almon, beta,
*   exponential), resultando em mais parametros e potencial
*   sobreajuste com amostras pequenas.
* - Nao ha suporte nativo para mixed-frequency state space /
*   Dynamic Factor Models (DFM); o comando `factor` e estatico.
* - Nowcasting com ragged-edge data requer manipulacao manual.
* - Comparacao: forecastbox oferece MIDAS com pesos Almon/beta,
*   U-MIDAS, DFM via Kalman filter e news decomposition em um
*   unico framework integrado.
* ============================================================

clear all
set more off

import delimited using "../data/mixed_freq.csv", clear
gen date_stata = monthly(substr(date, 1, 7), "YM")
format date_stata %tm
tsset date_stata

* --- Preparar variaveis ---
* GDP trimestral (NaN nos meses nao-trimestrais)
* IP mensal como regressor de alta frequencia

* Nota: O comando midas do Stata requer setup especifico.
* Alternativa simplificada: U-MIDAS via regressao com lags mensais

* --- U-MIDAS: GDP_q ~ IP_m1 + IP_m2 + IP_m3 ---
gen quarter = qofd(dofm(date_stata))
gen month_in_quarter = mod(month(dofm(date_stata)) - 1, 3) + 1

* Criar lags mensais dentro do trimestre
gen ip_m1 = industrial_production if month_in_quarter == 1
gen ip_m2 = industrial_production if month_in_quarter == 2
gen ip_m3 = industrial_production if month_in_quarter == 3

* Colapsar para trimestral
preserve
collapse (mean) ip_m1 ip_m2 ip_m3 (lastnm) gdp_growth, by(quarter)
drop if gdp_growth == .

tsset quarter

* --- U-MIDAS regression ---
regress gdp_growth ip_m1 ip_m2 ip_m3
display "U-MIDAS R2: " e(r2)

estat ic
matrix ic = r(S)
local aic = ic[1,5]
display "AIC: `aic'"

* --- Exportar ---
local r2 = e(r2)
clear
input str15 method float r2 float aic
"u_midas" `r2' `aic'
end
export delimited using "../data/stata_midas_validation.csv", replace
restore
