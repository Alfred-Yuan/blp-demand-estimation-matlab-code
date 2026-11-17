function [f,g] = F1_gmm_objective(theta2, Cst, XMat, YMat, MuMat, MicroMom, mval_file)
%function f = F1_gmm_objective(theta2, Cst, XMat, YMat, MuMat, MicroMom, mval_file)
% This function computes the objective function for the optimal GMM
% There are two sets of moments:
% mom1: J by 1 vector, where J is the number of instruments
% mom2: fraction of vehicle purchasers in each included income group by year

% Check if individual utilities explode or not
[mu_ij, P_ij, ~] = F3_mu_ij(theta2, Cst, XMat, MuMat);

% Obtain constants, attributes, and hh draws
nprods = Cst.nprods;

% Demeaning variable
demean_var = Cst.brand_model;

% Demeaned attributes
Xfull = XMat.Xfull;    % demeaned
IVall = XMat.IVall;    % demeaned

% Projection matrix
Projection = XMat.Projection;

% Weighting matrices
invWt = MicroMom.invWt;
invWinc = MicroMom.invWinc;

% Micro moment data
IncMom = MicroMom.IncMom;

%% Part 0: update the mean utility level implied by theta2

% Load CL_mvalold (cell array of mvalold) from previous iteration
load(mval_file, 'CL_mvalold')
YMat.CL_mvalold = CL_mvalold;

% Update CL_mvalold at the current theta2
CL_mvalold = F2_mean_value(mu_ij, Cst, YMat);

mval_All = cell2mat(CL_mvalold);

% Store the updated mvalold
save(mval_file, 'CL_mvalold')

%% Part 1: Aggregate Moment

% Demean mean utility
mval_All_dm = demean(demean_var, mval_All);

% Project mval onto Xfull to obtain theta1
theta1 = Projection * mval_All_dm;

% Residuals (xi_j)
gmmresid = mval_All_dm - Xfull*theta1;

% The first set of moments
mom1 = IVall' * gmmresid;

%% Part 2: Micro Moment

% Individual purchase probability for all products
indshr_All = F4b_IndShr_all(Cst, mval_All, mu_ij);

% Model-predicted fraction of buyers in each included income bracket.
% Row 1 refers to year 1 and column 1 to the first income bracket
[incshr_AllMk, tmpMat] = F5a_IncShr(indshr_All, Cst, YMat, MuMat);

% Keep only the years and income groups covered by the income survey
nincmom = size(IncMom,2);
incshr_AllMk = incshr_AllMk(MicroMom.year_idx, 1:nincmom);
tempincresid = (IncMom - incshr_AllMk)';

% For gradient
InMat.incshr = tmpMat.incshr;
InMat.tot_buy = tmpMat.tot_buy;

% Moments are stacked by year, with income groups varying fastest
mom2 = tempincresid(:); % convert to a column vector

%% Part 3: GMM objective function

% Objective
tmp1 = mom1' * invWt * mom1 / nprods;
tmp2 = mom2' * invWinc * mom2;
f = tmp1 + tmp2;

%% Part 4: Gradients

% P_ij: non-linear utility of price (alpha_i * p_j)
if nargout > 1
    [dxi_dtheta2, ddelta_dtheta2, ds_dtheta2Mat] = F6a_macro_grad(Cst, XMat, MuMat, indshr_All, P_ij);

    dmom2 = F6b_micro_grad(Cst, YMat, MuMat, ds_dtheta2Mat, ddelta_dtheta2, indshr_All, InMat);

    tmp1 = 2 * dxi_dtheta2' * IVall * invWt * mom1 / nprods;
    tmp2 = 2 * dmom2' * invWinc * mom2;
    g = tmp1 + tmp2;
end

end
