# BLP Demand Estimation in MATLAB

**Authors:** Hyuk-soo Kwon and Alfred Yuan  
**Date:** November 17, 2026

**Description:** This repository contain the whole workflow for BLP demand estimation in MATLAB, using vehicle attribute data at market level and income relevant micro moment data. The code was built from Hyuk-soo Kwon's job market paper code, and revised by his predoc Alfred Yuan.

## Requirements

- Statistics and Machine Learning Toolbox for functions such as `norminv` and `tcdf`.
- Optimization Toolbox for `fmincon` and `checkGradients`.
- Parallel Computing Toolbox for the local worker pool and market-level parallel computations.
- **This code is built for our current project only, so you need to change all pathes setup according to your environment.**

## Directory layout

```text
demand_estimation/
├── C0_generate_data.m
├── C1_simple_logit.m
├── C2a_blp_estimation.m
├── C2b_blp_results.m
├── C2c_gradient_check.m
├── econometrics_toolbox/
│   ├── CC.m
│   ├── aggregate.m
│   ├── demean.m
│   ├── dummify.m
│   ├── gmm_regress.m
│   ├── gmm_weight.m
│   ├── halton_draws.m
│   ├── indep_columns.m
│   ├── iv_regress.m
│   ├── ols.m
│   └── simple_ols.m
└── functions/
    ├── F0_objective_fixed_mval.m
    ├── F1_gmm_objective.m
    ├── F2_mean_value.m
    ├── F3_mu_ij.m
    ├── F4a_IndShr.m
    ├── F4b_IndShr_all.m
    ├── F5a_IncShr.m
    ├── F5b_cov_micro.m
    ├── F6a_macro_grad.m
    ├── F6b_micro_grad.m
    ├── F7a_function_values.m
    ├── F7b_jacobian.m
    ├── F7c_variance.m
    ├── F8a_dSdP.m
    ├── F8b_elasticity.m
    ├── F8c_mc.m
    └── set_data_path.m
```

## How to use this code

1. Clone this repository and setup all the pathes based on your own environment.

2. Change/Rewrite C0 to fit with your own data structure and variable names.

3. Run the scripts in this order.

   ```matlab
   C0_generate_data
   C1_simple_logit
   C2a_blp_estimation
   C2b_blp_results
   ```

   `C0_generate_data` prepares the estimation dataset. `C1_simple_logit` runs the baseline logit and is optional for BLP estimation. `C2a_blp_estimation` runs the two-step GMM estimator, and `C2b_blp_results` reports the estimates and computes elasticities and implied marginal costs.

To compare the analytical GMM gradient with central finite differences, run `C2c_gradient_check` after generating `Data.mat`.


# Description of each scripts
## C-series scripts

C-series scipts are the main files responsible for data generation, estimation and post-estimation alalysis. Usually, you will need to revise these files to adjust to your data structure.

- [C0_generate_data.m](C0_generate_data.m):  
  C0 mereges different dataset and generates the data files for subsequent estimation. It includes data import, variable mapping, specificaiton setup, market share generation, differentiation IV (GH IV) generation, micro moment data import, and household income simulation.  
  Usually, you will need to completely rewrite this file to suit with your own dataset, you can treat this file as reference.
- [C1_simple_logit.m](C1_simple_logit.m):  
  C1 estimates the simple logit using OLS and 2SLS. Given that you can simply replicate these results using stata or python, I stongly suggest you compare its results with C1's result, their results should be identical.
- [C2a_blp_estimation.m](C2a_blp_estimation.m):  
  C2 is the main BLP estimation file, estimating the random-coefficients logit by two-step GMM using both macro and micro moments.   
  You can adjust the following things based on your needs: the number of starting points, the number of workers (for paralleling computation), the starting points' bound and dimension (depending on the specificaiton). The rest part has been setup already.

- [C2b_blp_results.m](C2b_blp_results.m):  
 C2b reads the saved estimates, reports first and second stage estimates, and calculate the elastisity, markup, supplier side cost, and learner index.
- [C2c_gradient_check.m](C2c_gradient_check.m):  
 C2c evaluates the analytical GMM gradient and a central finite-difference approximation at the same parameter vector based on the results of C0. This file can tell whether your analytical GMM gradient is correctly calculated. I strongly suggest running this before C2b to check the gradient calculation. 

## Functions

The `Functions` folder contains all the supporting fuctions for our main BLP estimation. Theoritically, all files, except F3 and F6a, do not need to be changed when you try different specifications, you can use those funtions directly.

### GMM objective

- [F0_objective_fixed_mval.m](functions/F0_objective_fixed_mval.m):  
F0 evaluates the GMM objective and its gradient using caller-supplied starting values for the mean-utility inversion. It serves for the gradient check in C2c.
- [F1_gmm_objective.m](functions/F1_gmm_objective.m):  
 F1 utilizes and combines all the other functions' results to calculate objective function value and analytical gradient.

### Utility, choice probabilities, and micro moments

- [F2_mean_value.m](functions/F2_mean_value.m):  
 F2 recovers mean utilities by contraction mapping, in parallel across markets. I use a "SQUAREM" method to accelerate the contraction mapping here.
- [F3_mu_ij.m](functions/F3_mu_ij.m):  
 F3 specifies price coefficient that you use and calculate the corresponding price term in utility. I have included mutiple different price coefficient specifications, you can choose / write the one works for you best.
- [F4a_IndShr.m](functions/F4a_IndShr.m):  
  F4a computes household-level product choice probabilities in one market, I used the log-sum method to avoid the overflow problem.
- [F4b_IndShr_all.m](functions/F4b_IndShr_all.m):  
  F4b applies that calculation across markets and stacks the probabilities in product-row order.
- [F5a_IncShr.m](functions/F5a_IncShr.m):  
 F5a aggregates purchase probabilities into predicted shares of vehicle buyers by income group and year. 
- [F5b_cov_micro.m](functions/F5b_cov_micro.m):  
  F5b constructs the multinomial covariance matrix for the survey shares using purchaser shares and annual survey sample sizes.

### Derivatives and inference

- [F6a_macro_grad.m](functions/F6a_macro_grad.m):  
 F6a uses market-level share derivatives to differentiate the inverted mean utilities and demand residuals with respect to the nonlinear parameters. Please notice that, when you change the price coefficient specification in F3, the F6a also needs to be changed accordingly.
- [F6b_micro_grad.m](functions/F6b_micro_grad.m):  
 F6b differentiates the buyer income-share moments.
- [F7a_function_values.m](functions/F7a_function_values.m) evaluates the fitted moments, residuals, GMM criterion, and choice probabilities at the supplied estimates. [F7b_jacobian.m](functions/F7b_jacobian.m) constructs the moment Jacobian, and [F7c_variance.m](functions/F7c_variance.m) uses it with the moment covariance and GMM weights to compute the parameter covariance matrix.

### Elasticities and marginal costs

- [F8a_dSdP.m](functions/F8a_dSdP.m):  
 F8a computes within-market derivatives of product shares with respect to own and rival product prices.
- [F8b_elasticity.m](functions/F8b_elasticity.m):  
 F8b aggregates market-level derivatives to national model-year quantities and calculates own- and cross-price elasticities.
- [F8c_mc.m](functions/F8c_mc.m):  
 F8c recovers model-year marginal costs from multiproduct Bertrand first-order conditions using sales, demand derivatives, firm ownership, and tax wedges.

### Path configuration

- [set_data_path.m](functions/set_data_path.m) resolves the local data, output, and log directories and locates auxiliary inputs in Dropbox. Set `DROPBOX_ROOT` to specify a custom Dropbox root.

## Econometrics toolbox
These functions are the econometrics relevant toolbox including demean, GMM, OLS, 2SLS, Halton draws and so on. You can use this toolbox directly without further adjustments.

| File | Purpose |
| --- | --- |
| [aggregate.m](econometrics_toolbox/aggregate.m) | Groups rows by key and optionally applies a function within each group. |
| [CC.m](econometrics_toolbox/CC.m) | Implements the Cox–Shi test for moment inequalities. |
| [demean.m](econometrics_toolbox/demean.m) | Subtracts group means and restores the original row order. |
| [dummify.m](econometrics_toolbox/dummify.m) | Converts category labels into indicator columns. |
| [gmm_regress.m](econometrics_toolbox/gmm_regress.m) | Fits a linear GMM regression with optional clustered standard errors. |
| [gmm_weight.m](econometrics_toolbox/gmm_weight.m) | Builds a GMM weighting matrix from moment contributions, optionally grouped by cluster. |
| [halton_draws.m](econometrics_toolbox/halton_draws.m) | Generates market-specific blocks of Halton draws with optional scrambling. |
| [indep_columns.m](econometrics_toolbox/indep_columns.m) | Uses pivoted QR to select numerically independent columns. |
| [iv_regress.m](econometrics_toolbox/iv_regress.m) | Fits a 2SLS regression with optional clustered standard errors. |
| [ols.m](econometrics_toolbox/ols.m) | Fits OLS with optional clustered standard errors and reports adjusted R-squared. |
| [simple_ols.m](econometrics_toolbox/simple_ols.m) | Fits OLS and reports conventional standard errors and test statistics. |
