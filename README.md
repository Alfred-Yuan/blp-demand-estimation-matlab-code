# BLP Demand Estimation in MATLAB

MATLAB code for estimating vehicle demand using a baseline logit model and a random-coefficients BLP model with micro moments.

## Running the MATLAB scripts

Run the scripts in this order:

1. `C0_generate_data.m` builds `Data.mat` from the product and auxiliary data.
2. `C1_simple_logit.m` estimates the baseline logit model.
3. `C2a_blp_estimation.m` runs the two-stage BLP estimation using MATLAB's Optimization Toolbox.
4. `C2b_blp_results.m` computes elasticities and marginal costs from the BLP estimates.

The project data are held separately. The existing MATLAB path helper, `functions/set_data_path.m`, assumes this checkout is placed at `<project>/codes/demand_estimation`. It reads `df.parquet` from `<project>/data/final_estimation_data/demand` and writes output and logs under `<project>/data/outputs/demand`.

The helper looks for the auxiliary spreadsheets in `GS_Network/Data/final_estimation_data/demand` under the Dropbox root. Set the `DROPBOX_ROOT` environment variable if Dropbox is installed outside the default locations. For other directory layouts, update the path helper before running MATLAB.
