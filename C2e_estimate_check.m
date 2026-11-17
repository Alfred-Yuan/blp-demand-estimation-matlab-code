clc; clear
format short g

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check of the saved estimate: macro vs. micro, and model vs. data purchase table
%
% Reads Data.mat and the saved estimate (Main.mat, 2nd stage, or
% GMM_fst_stage.mat, 1st stage), reproduces the GMM objective function of
% F1_gmm_objective piece by piece (without calling F1, which overwrites the
% mean-utility file), and prints
%   Part 1. the objective function: macro part vs. micro part, the share of
%           each in the total, the macro contribution of each instrument, and
%           the micro contribution of each year x income group;
%   Part 2. for each micro-moment year and income group: simulated population
%           share, buyer share in the data and in the model, the implied
%           purchase probability P (= buyer share * inside share of the year /
%           market-size weighted population share) for data and model, and
%           P / P1 (relative to income group 1 of the same year).
%
% Nothing is saved or changed. The same file works in the new and the old code
% folders (the folder's own set_data_path decides which outputs are read).
%
% NOTE: not yet run in MATLAB.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Settings ----------------------------------------------------------------

stage = 2;               % 2: Main.mat (2nd stage estimate and weights)
                         % 1: GMM_fst_stage.mat (1st stage estimate and weights)
n_top_iv = 8;            % number of instruments listed in the macro breakdown

%% Paths and data ----------------------------------------------------------

project_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(project_dir,'functions'));
[~, out_dir] = set_data_path();
addpath(genpath(fullfile(project_dir,'econometrics_toolbox')));
addpath(genpath(fullfile(project_dir,'functions')));

data_file = fullfile(out_dir,'Data.mat');
if stage == 2
    estimate_file = fullfile(out_dir,'Main.mat');
else
    estimate_file = fullfile(out_dir,'GMM_fst_stage.mat');
end

load(data_file, 'Cst', 'YMat', 'XMat', 'MuMat', 'MicroMom')
E = load(estimate_file);

if stage == 2
    theta2 = E.theta2_B(:);
    fval_saved = E.fvalB;
else
    theta2 = E.theta2_A(:);
    fval_saved = E.fvalA;
end
invWt = E.invWt;
invWinc = E.invWinc;
YMat.CL_mvalold = E.CL_mvalold;      % starting values for the contraction

nprods = Cst.nprods;
Xfull = XMat.Xfull;
IVall = XMat.IVall;
IncMom = MicroMom.IncMom;            % data buyer shares (survey years x income groups 1..nincmom)
year_idx = MicroMom.year_idx;
nincmom = Cst.nincmom;
nincgrp = Cst.nincgrp;
ndraws = Cst.ndraws;

disp(['Estimate file: ', estimate_file]);
disp('theta2:');
disp(theta2');

%% Objective function (copy of F1_gmm_objective, parts 0-3) ----------------

% Mean utility implied by theta2
[mu_ij, ~, ~] = F3_mu_ij(theta2, Cst, XMat, MuMat);
CL_mvalold = F2_mean_value(mu_ij, Cst, YMat);
mval_All = cell2mat(CL_mvalold);

% Macro moments (projection computed with the saved weighting matrix)
Projection = (Xfull'*IVall*invWt*IVall'*Xfull) \ (Xfull'*IVall*invWt*IVall');
mval_All_dm = demean(Cst.brand_model, mval_All);
theta1 = Projection * mval_All_dm;
gmmresid = mval_All_dm - Xfull*theta1;
mom1 = IVall' * gmmresid;

macro_total = mom1' * invWt * mom1 / nprods;
macro_by_iv = mom1 .* (invWt * mom1) / nprods;      % contribution of each instrument

% Micro moments
indshr_All = F4b_IndShr_all(Cst, mval_All, mu_ij);
[incshr_AllMk, tmpMat] = F5a_IncShr(indshr_All, Cst, YMat, MuMat);
incshr_model = incshr_AllMk(year_idx, 1:nincmom);   % model buyer shares

tempincresid = (IncMom - incshr_model)';
mom2 = tempincresid(:);                             % stacked by year, groups fastest

micro_total = mom2' * invWinc * mom2;
micro_by_cell = mom2 .* (invWinc * mom2);           % contribution of each year x group
micro_by_cell = reshape(micro_by_cell, nincmom, numel(year_idx))';   % years x groups

total = macro_total + micro_total;

%% Part 1. Macro vs. micro -------------------------------------------------

disp(' ');
disp('=== Part 1. Objective function at the saved estimate ===');
objective_table = table([macro_total; micro_total; total], ...
    [100*macro_total/total; 100*micro_total/total; 100], ...
    'VariableNames', {'value','percent_of_total'}, ...
    'RowNames', {'macro','micro','total'});
disp(objective_table);
fprintf('Stored fval in the estimate file: %.4f (recomputed total: %.4f)\n', fval_saved, total);

disp(' ');
disp(['Macro contribution by instrument (largest ', num2str(n_top_iv), ' by value):']);
[~, order] = sort(macro_by_iv, 'descend');
top = order(1:min(n_top_iv, numel(order)));
iv_names = XMat.IVall_names;
labels = cell(numel(top),1);
for k = 1:numel(top)
    nm = '';
    if top(k) <= numel(iv_names)
        nm = char(string(iv_names{top(k)}));
    end
    if isempty(nm)
        nm = ['FE/dummy ', num2str(top(k))];
    end
    labels{k} = nm;
end
labels = matlab.lang.makeUniqueStrings(matlab.lang.makeValidName(labels));
disp(table(top, macro_by_iv(top), 100*macro_by_iv(top)/macro_total, ...
    'RowNames', labels, 'VariableNames', {'IV_index','contribution','percent_of_macro'}));

model_years = sort(unique(double(Cst.year)));
survey_years = model_years(year_idx);

disp(' ');
disp('Micro contribution by year (rows) and income group (columns):');
disp(array2table(micro_by_cell, 'VariableNames', strcat('group', string(1:nincmom)), ...
    'RowNames', cellstr(string(survey_years))));
disp(['Micro contribution by group (sum over years): ', mat2str(round(sum(micro_by_cell,1)))]);

%% Part 2. Model vs. data: buyer shares, purchase probability, P/P1 --------

mkt_index = Cst.mkt_index;
idx_begin = Cst.index_begin;

market_year = double(Cst.year(idx_begin));
size_mkt = double(Cst.mkt_size(idx_begin));

% Inside share of each market (sum of observed shares; the model's equals it by construction)
inside_data = accumarray(mkt_index, YMat.s_jt);

incgrpSm = MuMat.incgrpSm;                    % nmkt x ndraws, groups 1..nincgrp

rows = [];
for i = 1:numel(year_idx)
    y = survey_years(i);
    ix = (market_year == y);
    w = size_mkt(ix) / sum(size_mkt(ix));

    S_year = sum(w .* inside_data(ix));

    % Buyer shares; the last group is the remainder
    data_share = [IncMom(i,:), 1 - sum(IncMom(i,:))];
    model_share = [incshr_model(i,:), 1 - sum(incshr_model(i,:))];

    for g = 1:nincgrp
        pop_w = sum(w .* mean(incgrpSm(ix,:) == g, 2));        % market-size weighted population share
        P_data = data_share(g) * S_year / pop_w;
        P_model = model_share(g) * S_year / pop_w;
        rows = [rows; y, g, pop_w, data_share(g), model_share(g), ...
                data_share(g) - model_share(g), P_data, P_model]; %#ok<AGROW>
    end
end

% P relative to income group 1 of the same year
P1_data = nan(size(rows,1),1);
P1_model = nan(size(rows,1),1);
for y = unique(rows(:,1))'
    in_y = rows(:,1) == y;
    P1_data(in_y) = rows(in_y & rows(:,2) == 1, 7);
    P1_model(in_y) = rows(in_y & rows(:,2) == 1, 8);
end
rows = [rows, rows(:,7)./P1_data, rows(:,8)./P1_model];

result = array2table(rows, 'VariableNames', ...
    {'year','group','pop_share','data_buyer','model_buyer','data_minus_model', ...
     'P_data','P_model','P_over_P1_data','P_over_P1_model'});

disp(' ');
disp('=== Part 2. Buyer shares, purchase probability and P/P1: data vs. model ===');
disp(result);
