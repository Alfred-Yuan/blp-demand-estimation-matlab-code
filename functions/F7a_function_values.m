function [FitMM, InMat]= F7a_function_values(theta2, Cst, XMat, YMat, MuMat, MicroMom)
% Calculate objective fn value, different moments, and some input gradients
% FitMM: gmmresid, mval_All, incshr (1st micro mom), mom1, mom2, TwoF
% InMat: P_ij, price_coef, indshr_All, inc_shr, tot_buy

% Obtain constants
nprods = Cst.nprods;
nrand = Cst.nrand;
demean_var = Cst.brand_model;

Xfull = XMat.Xfull;
IVall = XMat.IVall;
Projection = XMat.Projection;

invWt = MicroMom.invWt;
invWinc = MicroMom.invWinc;

IncMom = MicroMom.IncMom;

%% Part 0: Preparation

% Load the current mvalold
CL_mvalold = YMat.CL_mvalold;

% mval_All: mean value for all products
mval_All = cell2mat(CL_mvalold);
InMat.mval = mval_All;
FitMM.mval_All = mval_All;

% Demean mean utility
mval_All_dm = demean(demean_var, mval_All);
FitMM.mval_All_dm = mval_All_dm;

% Heterogenous utility part, mu_i
[mu_ij, P_ij, price_coef] = F3_mu_ij(theta2, Cst, XMat, MuMat);
InMat.mu = mu_ij;


%% Part 1: Aggregate Moment

% Project mval onto Xfull to obtain theta1
theta1 = Projection * mval_All_dm;
FitMM.theta1 = theta1;

% Residuals (xi_j)
gmmresid = mval_All_dm - Xfull*theta1;
FitMM.gmmresid = gmmresid;

% The first set of moments
mom1 = IVall' * gmmresid;


%% Part 2: Micro Moment

% Individual purchase probability for all products
indshr_All = F4b_IndShr_all(Cst, mval_All, mu_ij);
InMat.indshr_All = indshr_All;

% Model predicted fraction of buyers in each income bracket. Row 1 refers
% to yr1, col1 refers to 1st income bracket
% FitMM.incshr is the model fitted 1st micro moments
% InMat.incshr is nmk by nincgrp, an input for gradient calculation
[incshr_AllMk, tmpMat] = F5a_IncShr(indshr_All, Cst, YMat, MuMat);
FitMM.incshr = incshr_AllMk;
InMat.incshr = tmpMat.incshr;
InMat.tot_buy = tmpMat.tot_buy;

% Keep only the years and income groups covered by the income survey
nincmom = size(IncMom,2);
incshr_AllMk = incshr_AllMk(MicroMom.year_idx, 1:nincmom);
tempincresid = (IncMom - incshr_AllMk)';

% Moments are stacked by year, with income groups varying fastest
mom2 = tempincresid(:); % convert to a column vector


%% Part 3: GMM objective fn
tmp1 = mom1' * invWt * mom1/nprods;
tmp2 = mom2' * invWinc * mom2;

% Save the value of the two sets of moments
FitMM.mom1 = mom1;
FitMM.mom2 = mom2;
FitMM.TwoF = [tmp1; tmp2];


%% Part 5: Price coefficient

% - price_coeff * price
InMat.P_ij = P_ij;

% Add the linear price coefficient if price is included in mean utility
if Cst.price_in_mean_utility
    InMat.price_coef_wo_mean_price = price_coef;
    InMat.price_coef = price_coef - theta1(1);
else
    InMat.price_coef = price_coef;
end

end
