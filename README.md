# BLP Demand Estimation in MATLAB

This repository estimates demand for electric and gasoline vehicles using product-level sales and characteristics by geographic market and year. It includes a baseline logit estimated by OLS and 2SLS, a random-coefficients logit estimated by two-step GMM, and post-estimation calculations for elasticities, marginal costs, and markups.

## Requirements

- MATLAB with support for `parquetread`, `readtable`, and `readcell`.
- Statistics and Machine Learning Toolbox for functions such as `norminv` and `tcdf`.
- Optimization Toolbox for `fmincon` and `checkGradients`.
- Parallel Computing Toolbox for the local worker pool and market-level parallel computations. Set `n_workers` in [C2a_blp_estimation.m](C2a_blp_estimation.m) to a value supported by your local MATLAB profile. The script currently requests 10 workers.

## Data and directory layout

The input data are stored separately. [functions/set_data_path.m](functions/set_data_path.m) resolves the project directory from the location of the code. To use its default paths, clone the repository into `codes/demand_estimation` inside your project directory.

```text
<project>/
├── codes/
│   └── demand_estimation/              # This repository
└── data/
    ├── final_estimation_data/
    │   └── demand/
    │       └── df.parquet              # Product-market input
    └── outputs/
        └── demand/
            ├── OutData/                # Generated MATLAB files
            └── Log/                    # MATLAB diary logs
```

`df.parquet` contains vehicle sales, market and product identifiers, prices, vehicle characteristics, and variables used to construct the instruments. The `VarMap` and `Spec` sections in [C0_generate_data.m](C0_generate_data.m) define the expected source column names and the model specification.

The data preparation script also reads the following inputs from `<Dropbox root>/GS_Network/Data/final_estimation_data/demand/`.

| File | Purpose |
| --- | --- |
| `Household Size.xlsx` | Household size used to construct the potential market size. The script reads the `Household Size` sheet. |
| `IncDist_GeoMarket.xlsx` | Income distribution inputs used to simulate household incomes. The script reads the `urban` sheet. |
| `micro moments.xlsx` | Survey buyer shares by income group and year, with survey sample sizes. The main script reads the `hyuksoo split 5g` sheet. |
| `National_level_deflator.xlsx` | Annual national deflator series read during micro-moment preparation. The active income-group assignment uses the sheet's nominal cutoffs directly. |

The path helper searches common Dropbox locations. For a custom installation, set the environment variable in MATLAB before running the scripts.

```matlab
setenv('DROPBOX_ROOT', '/path/to/Dropbox')
```

The helper checks that both the local data directory and the Dropbox input directory exist, even for scripts that load saved estimates. Adjust the helper if your data are stored in a different layout.

## Estimation workflow

Open the checkout folder in MATLAB and run the following scripts in order.

```matlab
C0_generate_data
C1_simple_logit
C2a_blp_estimation
C2b_blp_results
```

| Script | Reads | What it does and writes |
| --- | --- | --- |
| [C0_generate_data.m](C0_generate_data.m) | Product data and auxiliary spreadsheets | Builds product and outside shares, prepares fixed effects and instruments, generates market-specific Halton draws for household incomes and tastes, and aligns survey moments with the estimation sample. Writes `Data.mat` and `C0_generate_data_log.txt`. |
| [C1_simple_logit.m](C1_simple_logit.m) | `Data.mat` | Runs OLS and 2SLS on the simple-logit mean utility, with market-clustered standard errors. Reports a price first-stage regression and writes `SimpleLogit.mat` and `C1_simple_logit_log.txt`. |
| [C2a_blp_estimation.m](C2a_blp_estimation.m) | `Data.mat` | Fits the random-coefficients model by two-step GMM. Writes the first-step estimates to `GMM_fst_stage.mat`, the second-step estimates to `Main.mat`, and the working mean-utility cache to `CL_Mvalold.mat`. |
| [C2b_blp_results.m](C2b_blp_results.m) | `Data.mat`, `GMM_fst_stage.mat`, `Main.mat` | Prints parameter estimates, second-step standard errors, and market summaries. Writes `Supply.mat` with model-year elasticities and supply-side results. |

The baseline logit is a benchmark. The BLP estimation script computes its own preliminary logit fit and reads its inputs from `Data.mat`. Rerun data preparation and estimation together after changing the sample, instruments, random coefficients, or survey moments so that post-estimation uses matching files.

### What the BLP estimator fits

For each candidate vector of nonlinear parameters, the estimator solves for mean utilities that reproduce observed shares using a market-by-market contraction mapping. It then concentrates out the linear parameters using the current GMM weighting matrix. The criterion combines instrument-based moments for the demand residuals with survey moments for the income distribution of vehicle buyers. The second GMM step uses a market-clustered covariance estimate for the aggregate moments and a multinomial covariance estimate for the survey shares based on first-step predictions and survey sample sizes.

In the active specification, the random coefficients apply to the constant, gasoline-vehicle indicator, log driving range, log horsepower, and real fuel cost. Price enters through a household-specific coefficient of the form `-exp(alpha1 - alpha2 * log(income) + sigma_p * draw)`. The main specification is set in [C0_generate_data.m](C0_generate_data.m), and the household-specific utility term is implemented in [functions/F3_mu_ij.m](functions/F3_mu_ij.m).

The optimizer uses analytical gradients and supports multiple starting points through `n_iter_1st` and `n_iter_2nd`. Both currently default to one. Each step saves the best solution found among the configured starts. The estimation script uses the same mean-utility cache throughout a run, so run one estimation at a time per output directory.

### Gradient check

Run [C2c_gradient_check.m](C2c_gradient_check.m) after generating `Data.mat` to compare the analytical GMM gradient with central finite differences. It evaluates both derivatives using a common set of initial mean utilities and prints the per-parameter differences. This check is useful after editing the utility specification, moment definitions, or derivative code.

## Output files

`Data.mat` contains five structures that are shared by estimation, diagnostics, and post-estimation.

| Structure | Contents |
| --- | --- |
| `Cst` | Dimensions, market indices, product and firm identifiers, sales, and contraction settings. |
| `YMat` | Observed product shares, market sales totals, and initial mean utilities. |
| `XMat` | Product characteristics, prices, regressors, and instruments. |
| `MuMat` | Simulated household taste and income draws and their income-group assignments. |
| `MicroMom` | Survey buyer shares, survey years, and sample sizes. |

In `Supply.mat`, `ElasMat` contains price elasticities aggregated to the national model-year level. `Supply` contains model-year quantities, prices, and own-price elasticities, along with marginal costs recovered under multiproduct Bertrand pricing using firm ownership and tax wedges. It also records markups and Lerner indices.

## Code reference

| Files | Role |
| --- | --- |
| `functions/F0_*`, `functions/F1_*` | GMM objective and gradient. `F0` uses fixed initial mean utilities for derivative checks. `F1` reads and updates the mean-utility cache during estimation. |
| `functions/F2_*`, `functions/F3_*`, `functions/F4*` | Mean-utility inversion, heterogeneous utility, and individual choice probabilities. |
| `functions/F5*`, `functions/F6*` | Predicted buyer income shares, micro-moment covariance, and analytical derivatives of the aggregate and micro moments. |
| `functions/F7*` | Fitted moments, GMM Jacobian, and parameter covariance matrix. |
| `functions/F8*` | Market-level price derivatives, national model-year elasticities, and implied marginal costs. |
| `econometrics_toolbox/` | OLS, IV and linear GMM routines, weighting, grouping, fixed-effect demeaning, rank checks, Halton draws, and supporting inference utilities. |
