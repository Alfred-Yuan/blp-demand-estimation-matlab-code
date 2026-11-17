clc; clear
format short g

% Directories
project_dir = fileparts(mfilename('fullpath'));
repo_dir = fileparts(fileparts(project_dir));

addpath(fullfile(project_dir,'functions'));
[~, out_dir] = set_data_path();

% Output file names
data_file = fullfile(out_dir,'Data.mat');

% Functions
addpath(genpath(fullfile(project_dir,'econometrics_toolbox')));
addpath(genpath(fullfile(project_dir,'functions')));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 0: Preliminaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load variables, IVs, Weighting matrices for micro moments
load(data_file, 'Cst', 'YMat', 'XMat', 'MuMat', 'MicroMom')

IVall = XMat.IVall;    % Demeaned
Xfull = XMat.Xfull;    % Demeaned
chars_dm = XMat.chars_dm;

nprods = Cst.nprods;
nmkt = Cst.nmkt;
nrand = Cst.nrand;

index_begin = Cst.index_begin;
index_end = Cst.index_end;


% Initialize mvalold at the mean utility for simple logit regression
mean_utility = YMat.mean_utility;

CL_mvalold = cell(nmkt,1);
for t = 1:nmkt
    aa = index_begin(t);
    bb = index_end(t);
    CL_mvalold{t} = mean_utility(aa:bb);
end
YMat.CL_mvalold = CL_mvalold;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 1: Simple Logit Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mean_utility_dm = YMat.mean_utility_dm;

% 2SLS regression
[tsls_beta, ~, ~, ~] = iv_regress(mean_utility_dm, chars_dm, IVall);

% Residuals for the initial weighting matrix for the GMM estimation
resid = mean_utility_dm - chars_dm*tsls_beta;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 2: Initial parameter values and feasible boundaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% theta2 = [sigma_const; sigma_(Spec.random_chars) ...; alpha1; alpha2; sigma_p]
LowerBnd = [ zeros(nrand,1); 0.0; 0.0; 0.0];
UpperBnd = [ 2.5; 3.0*ones(nrand-1,1); 2.0; 2.5; 2.0];

theta2 = (LowerBnd + UpperBnd) / 2;



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 3: Construct fixed initial weighting matrices for the 1st stage
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initial weighting matrix
shat = (resid * ones(1,size(IVall,2))) .* IVall;
invWt = inv(shat'*shat/nprods);
%invWt = inv(IVall'*IVall);

MicroMom.invWt = invWt;

% Pre-calculate the projection matrix for theta1
XMat.Projection = (Xfull'*IVall*invWt*IVall'*Xfull)\Xfull'*IVall*invWt*IVall';

% Compute the initial market shares for the income-share moments
any_theta2 = zeros(Cst.nK,1);
[mu_ij, ~, ~] = F3_mu_ij(any_theta2, Cst, XMat, MuMat);

mval_All = mean_utility;
indshr_All = F4b_IndShr_all(Cst, mval_All, mu_ij);

% Model-predicted purchaser income shares
[incshr_AllMk, ~] = F5a_IncShr(indshr_All, Cst, YMat, MuMat);

% Keep only the years and income groups covered by the income survey
incshr_AllMk = incshr_AllMk(MicroMom.year_idx, 1:Cst.nincmom);

% Compute the initial covariance matrix for the income-share moments
Winc = F5b_cov_micro(incshr_AllMk, MicroMom.IncSampleSize);
invWinc = inv(Winc);

MicroMom.invWinc = invWinc;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 4: Compare analytical and numerical gradients
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define the objective function using the same fixed initial CL_mvalold
objfun = @(theta2) F0_objective_fixed_mval(theta2, Cst, XMat, YMat, MuMat, MicroMom, CL_mvalold);

% Evaluate the objective and analytical gradient at theta2
[objective_value, analytical_gradient] = objfun(theta2);

% Initialize the numerical gradient vector
numerical_gradient = zeros(Cst.nK,1);

% Set the central finite-difference step size
step_size = 1e-4;

% Compute the numerical derivative for each element of theta2
for k = 1:Cst.nK

    % Copy theta2 for the positive and negative perturbations
    theta2_plus = theta2;
    theta2_minus = theta2;

    % Perturb only parameter k
    theta2_plus(k) = theta2_plus(k) + step_size;
    theta2_minus(k) = theta2_minus(k) - step_size;

    % Evaluate the objective at the positive and negative perturbations
    f_plus = objfun(theta2_plus);
    f_minus = objfun(theta2_minus);

    % Calculate the central finite-difference derivative for parameter k
    numerical_gradient(k) = (f_plus - f_minus)/(2*step_size);
end

% Place the analytical gradient, numerical gradient, and difference side by side
gradient_comparison = [analytical_gradient numerical_gradient analytical_gradient - numerical_gradient];

% Display the objective value at theta2
disp('Objective value');
disp(objective_value);

% Display the analytical gradient, numerical gradient, and their difference
disp('Analytical gradient, numerical gradient, and difference');
disp(gradient_comparison);
