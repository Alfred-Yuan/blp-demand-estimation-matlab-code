clc; clear
format short g

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implement BLP estimation for multiple starting values
%
% Structure:
%   1. Run the 1st stage GMM from many starting points.
%   2. Keep only the best 1st stage estimate.
%   3. Construct the efficient macro and micro weighting matrices once
%      using that estimate.
%   4. Run the 2nd stage GMM from many starting points.
%   5. Keep only the best 2nd stage estimate and save it as Main.mat.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Number of starting points
n_iter_1st = 1;
n_iter_2nd = 1;

% Set number of parallel workers
n_workers = 10;

% Set the early stop parameters
early_stop_fval_ratio = 2;
early_stop_step_tol = 1e-3;

% Directories
project_dir = fileparts(mfilename('fullpath'));
repo_dir = fileparts(fileparts(project_dir));

addpath(fullfile(project_dir,'functions'));
[~, out_dir] = set_data_path();

% Output file names
data_file = fullfile(out_dir,'Data.mat');
mval_file = fullfile(out_dir,'CL_Mvalold.mat');
fst_stage_file = fullfile(out_dir,'GMM_fst_stage.mat');
snd_stage_file = fullfile(out_dir,'Main.mat');

% Functions
addpath(genpath(fullfile(project_dir,'econometrics_toolbox')));
addpath(genpath(fullfile(project_dir,'functions')));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 0: Preliminaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load variables, IVs, Weighting matrices for micro moments
% Fields of Cst: nyear, nmkt, ndraws, nprods, nrand,
%                index_begin, index_end, nprod_vec, maxiter
% Fields of YMat: salesMk, salesAllMk, CL_sjt, mean_utility
% Fields of XMat: randchars, chars_dm, Xfull, IVall, p_real
% Fields of MuMat: hh (household taste draws for random chars and price),
%                  inc (household income draws), incgrpSm
% Fields of MicroMom: IncMom, IncSampleSize, and stage-specific weights
load(data_file, 'Cst', 'YMat', 'XMat', 'MuMat', 'MicroMom')

IVall = XMat.IVall;    % Demeaned
Xfull = XMat.Xfull;    % Demeaned
chars_dm = XMat.chars_dm;

nprods = Cst.nprods;
nmkt = Cst.nmkt;
nrand = Cst.nrand;
demean_var = Cst.brand_model;

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
save(mval_file, 'CL_mvalold')


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 1: Simple Logit Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mean_utility_dm = YMat.mean_utility_dm;

% OLS
[ols_beta, ols_se, ols_tstat, ols_pval] = ols(mean_utility_dm, chars_dm);

disp('OLS results');
round([ols_beta(1:8), ols_se(1:8), ols_tstat(1:8), ols_pval(1:8)], 3)

% 2SLS regression
[tsls_beta, tsls_se, tsls_tstat, tsls_pval] = iv_regress(mean_utility_dm, chars_dm, IVall);

disp('2SLS results');
round([tsls_beta(1:8), tsls_se(1:8), tsls_tstat(1:8), tsls_pval(1:8)], 3)

% Residuals for the initial weighting matrix for the GMM estimation
resid = mean_utility_dm - chars_dm*tsls_beta;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 2: Initial parameter values and feasible boundaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Find feasible values of theta2
% If elements of theta2 are out of some bounds (too large or small),
% we have exp(mu_ij) = inf or 0 => log(0) = -inf
% If the absolute value of max or min is greater than 500, problematic

% Thus, using the below code, check proper bounds
%{
theta2 = zeros(Cst.nK,1);
[mu_ij, P_ij, ~] = F3_mu_ij(theta2, Cst, XMat, MuMat);
[max(max(mu_ij)) max(max(P_ij))]
[min(min(mu_ij)) min(min(P_ij))]
%}


% theta2 = [sigma_const; sigma_(Spec.random_chars) ...; alpha1; alpha2; sigma_p]
% nrand = 1 (constant) + number of variables in Spec.random_chars
LowerBnd = [-8.0*ones(nrand,1); -3.0; 0.0; -3.5];
UpperBnd = [ 8.0*ones(nrand,1);  6.0; 3.5;  3.5];

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 3: Construct fixed initial weighting matrices for the 1st stage
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initial weighting matrix (clustered by market-year)
shat = (resid * ones(1,size(IVall,2))) .* IVall;
% invWt = inv(shat'*shat/nprods); % Weighting matrix construction

% Clustered weighting matrix construction
shat_cl = zeros(Cst.nmkt, size(IVall,2));
for m = 1:Cst.nmkt
    shat_cl(m,:) = sum(shat(Cst.index_begin(m):Cst.index_end(m),:), 1);
end
invWt = inv(shat_cl'*shat_cl/nprods);

MicroMom.invWt = invWt;

% Pre-calculate the projection matrix for theta1
XMat.Projection = (Xfull'*IVall*invWt*IVall'*Xfull)\Xfull'*IVall*invWt*IVall';

% Compute the initial market shares for the income-share moments
Winc = F5b_cov_micro(MicroMom.IncMom, MicroMom.IncSampleSize);
invWinc = inv(Winc);

MicroMom.invWinc = invWinc;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 4: First round GMM estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Restart parallel pool only if needed
p = gcp('nocreate');

if isempty(p)
    parpool('local', n_workers);
elseif p.NumWorkers ~= n_workers
    delete(p);
    parpool('local', n_workers);
end

% Optimization options
options = optimoptions('fmincon','Algorithm','interior-point','Display','iter', ...
    'SpecifyObjectiveGradient',true,'FiniteDifferenceType','central');
    %'OptimalityTolerance', 1e-6,'StepTolerance', 1e-8);


% Set the seed for the random number generator
rng(12345,'twister');

% Collection of starting points (1/4 length of the total bound width)
mid = (LowerBnd + UpperBnd) / 2;
half_width = (UpperBnd - LowerBnd) / 8;
theta2_start_1st = mid + ...
    (2 * rand(Cst.nK, n_iter_1st) - 1) .* half_width;


% Store all 1st stage results
theta2_A_all = nan(Cst.nK, n_iter_1st);
fvalA_all = nan(n_iter_1st, 1); 
flagA_all = nan(n_iter_1st, 1);

% Prepare the iteration
fvalA_best = 1e+10;

for i = 1:n_iter_1st
    t_s1_total = tic;

    fprintf('\n============================================================\n');
    fprintf('1st stage GMM: starting point %d of %d\n', i, n_iter_1st);
    fprintf('Current best 1st stage fval: %.12g\n', fvalA_best);
    fprintf('============================================================\n');

    % Initial theta2
    theta2 = theta2_start_1st(:,i);

    % If want to check the gradient
    %{
    objfun = @(theta2) F1_gmm_objective(theta2, Cst, XMat, YMat, MuMat, MicroMom, mval_file);
    grad_options = optimoptions('fmincon','FiniteDifferenceType','central');
    [grad_valid, grad_error] = checkGradients(objfun, theta2, grad_options,'Display', 'on');
    %}

    % Use the best completed run as the benchmark for early stopping
    options_run = optimoptions(options, 'OutputFcn', ...
        @(x,optimValues,state) stop_bad_start(x, optimValues, state, ...
        fvalA_best, early_stop_fval_ratio, early_stop_step_tol));
    
    % Put the fixed objective-function inputs inside an anonymous function
    gmm_objective = @(theta) F1_gmm_objective(theta, Cst, XMat, YMat, MuMat, MicroMom, mval_file);
    
    % First round estimation
    [theta2_A, fvalA, flagA, outputA] = fmincon(gmm_objective, theta2, [], [], [], [], ...
        LowerBnd, UpperBnd, [], options_run);

    % Save all 1st stage outputs in memory
    theta2_A_all(:,i) = theta2_A;
    fvalA_all(i) = fvalA;
    flagA_all(i) = flagA;

    % Print the current theta2
    fprintf('Estimated theta2 for starting point %d of %d:\n', i, n_iter_1st)
    disp(theta2_A);

    % Update the best 1st stage estimate
    if fvalA < fvalA_best

        fvalA_best = fvalA;

        % Save output
        load(mval_file, 'CL_mvalold')
        save(fst_stage_file,'theta2_A','fvalA','flagA','outputA','invWt','invWinc','CL_mvalold')
    end
    fprintf('[Stage 1] One starting point total time: %.3f seconds.\n', toc(t_s1_total));
end

% Load the 1st stage results
load(fst_stage_file)

fprintf('\nBest 1st stage fval: %.12g\n', fvalA)
disp('Best 1st stage theta2_A:')
disp(theta2_A);

% Update the mean values
YMat.CL_mvalold = CL_mvalold;
[mu_ij, ~, ~] = F3_mu_ij(theta2_A, Cst, XMat, MuMat);
CL_mvalold = F2_mean_value(mu_ij, Cst, YMat);

mval_All = cell2mat(CL_mvalold);
save(mval_file, 'CL_mvalold')

% Obtain the linear parameters
mval_All_dm = demean(demean_var, mval_All);
theta1_A = XMat.Projection * mval_All_dm;

% Save additional information
save(fst_stage_file,'CL_mvalold','theta1_A','theta2_A_all','fvalA_all','flagA_all','theta2_start_1st','-append')


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 5: Construct efficient weighting matrices using the best 1st stage
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Obtain the residuals
gmmresid = mval_All_dm - Xfull*theta1_A;

% Efficient GMM weighting matrix for macro moments
shat = (gmmresid * ones(1,size(IVall,2))) .* IVall;
% invWt = inv(shat'*shat/nprods);

% Cluster weighting matrix construction
shat_cl = zeros(Cst.nmkt, size(IVall,2));
for m = 1:Cst.nmkt
    shat_cl(m,:) = sum(shat(Cst.index_begin(m):Cst.index_end(m),:), 1);
end
invWt = inv(shat_cl'*shat_cl/nprods);

MicroMom.invWt = invWt;

% Pre-calculate the inverse matrix of iv regression for theta1
XMat.Projection = (Xfull'*IVall*invWt*IVall'*Xfull)\Xfull'*IVall*invWt*IVall';


% Do not update the micro moment weighting matrix for now

% Model-predicted purchaser income shares
indshr_All = F4b_IndShr_all(Cst, mval_All, mu_ij);
[incshr_AllMk, ~] = F5a_IncShr(indshr_All, Cst, YMat, MuMat);

% Keep only the years and income groups covered by the income survey
incshr_AllMk = incshr_AllMk(MicroMom.year_idx, 1:Cst.nincmom);

% Efficient micro weighting matrix
Winc = F5b_cov_micro(incshr_AllMk, MicroMom.IncSampleSize);
invWinc = inv(Winc);

MicroMom.invWinc = invWinc;



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 6: Efficient GMM estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set the seed for the 2nd stage random starting points
rng(54321,'twister');

% Collection of 2nd stage starting points (1/4 length of the total band width)
mid = (LowerBnd + UpperBnd) / 2;
half_width = (UpperBnd - LowerBnd) / 8;
theta2_start_2nd = mid + ...
    (2 * rand(Cst.nK, n_iter_2nd) - 1) .* half_width;
% Make the best 1st stage estimate the first 2nd stage starting point
theta2_start_2nd(:,1) = theta2_A;

% Store all 2nd stage results
theta2_B_all = nan(Cst.nK, n_iter_2nd);
fvalB_all = nan(n_iter_2nd, 1);
flagB_all = nan(n_iter_2nd, 1);

% Prepare the iteration
fvalB_best = 1e+10;

for i = 1:n_iter_2nd
    t_s2_total = tic;

    fprintf('\n============================================================\n');
    fprintf('2nd stage GMM: starting point %d of %d\n', i, n_iter_2nd);
    fprintf('Current best 2nd stage fval: %.12g\n', fvalB_best);
    fprintf('============================================================\n');

    % Initial theta2
    theta2 = theta2_start_2nd(:,i);

    % Use the best completed run as the benchmark for early stopping
    options_run = optimoptions(options, 'OutputFcn', ...
        @(x,optimValues,state) stop_bad_start(x, optimValues, state, ...
        fvalB_best, early_stop_fval_ratio, early_stop_step_tol));
    
    % Put the fixed objective-function inputs inside an anonymous function
    gmm_objective = @(theta) F1_gmm_objective(theta, Cst, XMat, YMat, MuMat, MicroMom, mval_file);
    
    % Efficient GMM estimation
    [theta2_B, fvalB, flagB, outputB] = fmincon(gmm_objective, theta2, [], [], [], [], ...
        LowerBnd, UpperBnd, [], options_run);

    % Save all 2nd stage outputs in memory
    theta2_B_all(:,i) = theta2_B;
    fvalB_all(i) = fvalB;
    flagB_all(i) = flagB;

    % Update the best 2nd stage estimate
    if fvalB < fvalB_best

        fvalB_best = fvalB;

        % Save output
        load(mval_file, 'CL_mvalold')
        save(snd_stage_file,'theta2_B','fvalB','flagB','outputB','invWt','invWinc','CL_mvalold')
    end
    fprintf('[Stage 2] One starting point total time: %.3f seconds.\n', toc(t_s2_total));
end

% Load the 2nd stage results
load(snd_stage_file)

% Update the mean values
YMat.CL_mvalold = CL_mvalold;
[mu_ij, ~, ~] = F3_mu_ij(theta2_B, Cst, XMat, MuMat);
CL_mvalold = F2_mean_value(mu_ij, Cst, YMat);

mval_All = cell2mat(CL_mvalold);
save(mval_file, 'CL_mvalold')

% Obtain the linear parameters
mval_All_dm = demean(demean_var, mval_All);
theta1_B = XMat.Projection * mval_All_dm;

% Save additional information
save(snd_stage_file,'CL_mvalold','theta1_B','theta2_B_all','fvalB_all','flagB_all','theta2_start_2nd','-append')

% Display the results
fprintf('Best 2nd stage fval: %.12g\n', fvalB);
disp('Best 2nd stage theta2_B:');
disp(theta2_B);





%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Local function: stop a clearly inferior starting point
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function stop = stop_bad_start(~, optimValues, state, best_fval, fval_ratio, step_tol)

stop = false;

% The first completed starting point establishes the initial best fval.
% Also skip iteration 0 because its step size does not yet represent an
% actual optimization step.
if ~strcmp(state, 'iter') || optimValues.iteration == 0 || ...
        ~isfinite(best_fval)
    return
end

% Stop only when both conditions hold:
% 1. The current objective is much larger than the best completed fval.
% 2. fmincon is taking a negligible step and is therefore unlikely to escape this inferior local solution.
if optimValues.fval >= fval_ratio * best_fval && optimValues.stepsize <= step_tol

    fprintf('\nStopping early: current fval %.12g >= %.4g x best fval %.12g, and step size %.4g <= %.4g.\n', ...
        optimValues.fval, fval_ratio, best_fval, optimValues.stepsize, step_tol);

    stop = true;
end
end
