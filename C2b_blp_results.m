clc; clear
format short g

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Short Code to See the BLP Estimation Results and Standard Errors
% Authors: Hyuk-soo Kwon
% Adapted to the EV-CS project
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Directories
project_dir = fileparts(mfilename('fullpath'));
repo_dir = fileparts(fileparts(project_dir));

addpath(fullfile(project_dir,'functions'));
[~, out_dir] = set_data_path();

% Output file names
data_file = fullfile(out_dir,'Data.mat');
fst_stage_file = fullfile(out_dir,'GMM_fst_stage.mat');
main_file = fullfile(out_dir,'Main.mat');
supply_file = fullfile(out_dir,'Supply.mat');

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
%                  lninc (log of household income draws), incgrpSm
% Fields of MicroMom: IncMom, IncSampleSize, and stage-specific weights
load(data_file, 'Cst', 'YMat', 'XMat', 'MuMat', 'MicroMom')
load(fst_stage_file, 'theta2_A')
load(main_file, 'theta1_B', 'theta2_B', 'invWt', 'invWinc', 'CL_mvalold')

% Read information
IVall = XMat.IVall;    % Demeaned
Xfull = XMat.Xfull;    % Demeaned
theta1_names = XMat.Xfull_names;
theta2_names = XMat.theta2_names;

% Update fields
Cst.nK = length(theta2_B);

MicroMom.invWt = invWt;
MicroMom.invWinc = invWinc;

YMat.CL_mvalold = CL_mvalold;
XMat.Projection = (Xfull'*IVall*invWt*IVall'*Xfull)\Xfull'*IVall*invWt*IVall';


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 1: BLP Estimates
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute the variance matrix
[FitMM, InMat] = F7a_function_values(theta2_B, Cst, XMat, YMat, MuMat, MicroMom);
gMat = F7b_jacobian(Cst, XMat, YMat, MuMat, MicroMom, FitMM, InMat);
VarMat = F7c_variance(Cst, XMat, MicroMom, FitMM, gMat);

nK = length(theta2_B);

% VarMat is ordered as [theta2; theta1]
theta2_se = sqrt(diag(VarMat(1:nK, 1:nK)));
theta1_se = sqrt(diag(VarMat(nK+1:end, nK+1:end)));

% Estimation results
disp('1st-stage non-linear parameters');
theta2_A_table = table(theta2_names(:), round(theta2_A,3), ...
    'VariableNames', {'Variable','Estimate'});
disp(theta2_A_table)

disp('Linear parameters');
theta1_show = ~cellfun(@isempty, theta1_names);
theta1_table = table(theta1_names(theta1_show)', round(theta1_B(theta1_show),3), ...
    round(theta1_se(theta1_show),3), round(theta1_B(theta1_show)./theta1_se(theta1_show),3), ...
    'VariableNames', {'Variable','Estimate','StdErr','Tstat'});
disp(theta1_table)

disp('Non-linear parameters');
theta2_table = table(theta2_names(:), round(theta2_B,3), round(theta2_se,3), ...
    round(theta2_B./theta2_se,3), ...
    'VariableNames', {'Variable','Estimate','StdErr','Tstat'});
disp(theta2_table)


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 2: Elasticities and Marginal Costs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Construct model-year data
year = double(Cst.year);
model_id = double(Cst.model_id);
firm_id = double(Cst.firm_id);
GV = double(Cst.GV);
BEV = double(Cst.BEV);
PHEV = double(Cst.PHEV);
EV = double(Cst.EV);
REEV = double(Cst.REEV);
sales = double(Cst.sales);
subsidy = double(Cst.centralS_real + Cst.localS_real);

price = double(XMat.p_real);
firm_price = double(Cst.firm_price);

[G, Supply.year, Supply.model_id] = findgroups(year, model_id);

% Sales-weighted average within each model-year
wmean = @(x,w) sum(x.*w)/sum(w);

Supply.firm_id = splitapply(@(x) x(1), firm_id, G);
Supply.GV = splitapply(@(x) x(1), GV, G);
Supply.BEV = splitapply(@(x) x(1), BEV, G);
Supply.PHEV = splitapply(@(x) x(1), PHEV, G);
Supply.EV = splitapply(@(x) x(1), EV, G);
Supply.REEV = splitapply(@(x) x(1), REEV, G);
Supply.sales = splitapply(@sum, sales, G);
Supply.price = splitapply(wmean, price, sales, G);
Supply.firm_price = splitapply(wmean, firm_price, sales, G);
Supply.subsidy = splitapply(wmean, subsidy, sales, G);

% Tax wedges implied by price = t * msrp - subsidy and
% firm_price = tau * msrp
% Construct the total tax wedge T =  t / tau
Supply.T = (Supply.price+ Supply.subsidy) ./ (Supply.firm_price);

% Sort by year and model
[~, ord] = sortrows([Supply.year Supply.model_id]);
Supply.year = Supply.year(ord);
Supply.model_id = Supply.model_id(ord);
Supply.firm_id = Supply.firm_id(ord);
Supply.GV = Supply.GV(ord);
Supply.BEV = Supply.BEV(ord);
Supply.PHEV = Supply.PHEV(ord);
Supply.EV = Supply.EV(ord);
Supply.REEV = Supply.REEV(ord);
Supply.sales = Supply.sales(ord);
Supply.price = Supply.price(ord);
Supply.firm_price = Supply.firm_price(ord);
Supply.T = Supply.T(ord);

% Map national model-year own-price elasticities back to the model-year data
[ElasMat,~,~] = F8b_elasticity(Cst, XMat, YMat, InMat);
model_ids = ElasMat.model_ids;
years = sort(unique(Supply.year));

Supply.elasticity = zeros(length(Supply.year),1);
for i = 1:length(Supply.year)
    [~, tpos] = ismember(Supply.year(i), years);
    [~, mpos] = ismember(Supply.model_id(i), model_ids);
    Supply.elasticity(i) = ElasMat.OwnElas((tpos-1)*length(model_ids) + mpos);
end

% Recover MC from the multiproduct Bertrand first-order conditions
Supply.mc = F8c_mc(Cst, XMat, YMat, InMat, Supply);

% Markup and Lerner index
Supply.markup = Supply.firm_price - Supply.mc;
Supply.lerner = Supply.markup ./ Supply.firm_price;

save(supply_file, 'Supply', 'ElasMat')

% Check the outcomes
supply_stats = [Supply.firm_price Supply.mc Supply.markup Supply.lerner Supply.elasticity];
supply_summary = array2table([min(supply_stats); mean(supply_stats); ...
    max(supply_stats); std(supply_stats)]', ...
    'VariableNames', {'min','mean','max','std'}, ...
    'RowNames', {'firm_price','mc','markup','lerner','elasticity'});

disp('Supply-side summary');
disp(supply_summary);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 3: Main demand and market summaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sales = Supply.sales;
group_name = ["GV";"PHEV";"BEV";"EV";"REEV"];
group_ix = {Supply.GV==1, Supply.PHEV==1, Supply.BEV==1, Supply.EV==1, Supply.REEV==1};
summary_values = zeros(length(group_name),6);

% Report sales-weighted elasticities, markups, Lerner indices, and prices
for g = 1:length(group_name)
    ix = group_ix{g};

    % Sales-weighted average within each fuel-type group
    w = sales(ix)/sum(sales(ix));

    summary_values(g,:) = [sum(ix), sum(sales(ix)), w'*(-Supply.elasticity(ix)), ...
        w'*Supply.markup(ix), w'*Supply.lerner(ix), w'*Supply.firm_price(ix)];
end

demand_summary = table(group_name,summary_values(:,1),summary_values(:,2), ...
    summary_values(:,3),summary_values(:,4),summary_values(:,5), ...
    summary_values(:,6),'VariableNames', ...
    {'Fuel_type','Model_years','Sales','Mean_elasticity', ...
    'Mean_markup','Mean_lerner','Mean_firm_price'});

disp(demand_summary)

