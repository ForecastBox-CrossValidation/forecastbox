* ============================================================
* Validacao Stata: Cross-Validation Temporal
* ============================================================

clear all
set more off

import delimited using "../data/macro_brazil.csv", clear
gen date_stata = monthly(substr(date, 1, 7), "YM")
format date_stata %tm
tsset date_stata

local n = _N
local min_train = 60
local h = 12
local n_folds = `n' - `min_train' - `h' + 1

* --- Expanding window CV ---
* Nota: Stata nao tem CV temporal nativo, implementamos com loop
display "Expanding window CV com " `n_folds' " folds"
display "Horizonte: `h' meses"
display "Treino minimo: `min_train' obs"

* Para cada fold, calcular naive forecast error
gen fold_error = .
forvalues fold = 1/`n_folds' {
    local train_end = `min_train' + `fold' - 1
    local test_start = `train_end' + 1
    local test_end = `test_start' + `h' - 1

    if `test_end' > `n' continue

    local last_val = inflation[`train_end']

    * MAE para este fold
    local fold_mae = 0
    forvalues t = `test_start'/`test_end' {
        local err = abs(inflation[`t'] - `last_val')
        local fold_mae = `fold_mae' + `err'
    }
    local fold_mae = `fold_mae' / `h'
    quietly replace fold_error = `fold_mae' in `fold'
}

quietly summarize fold_error if fold_error != .
display "Mean MAE (expanding): " r(mean)
display "SD MAE (expanding): " r(sd)

export delimited fold_error using "../data/stata_cv_validation.csv" if fold_error != ., replace
