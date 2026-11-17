clc; clear
format short g

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Quick check of the micro-moment table (a slim version of C0)
%
% Given a micro-moment sheet and the survey years to keep, this script
%   1. rebuilds the market sample exactly as C0 does (sales filters, market
%      size, s_0), and prints the descriptive statistics of s_0;
%   2. simulates household incomes exactly as C0 does (same seed, same
%      Halton draws), assigns them to the income groups of the chosen sheet;
%   3. prints the table "Income group shares: simulated vs. micro moments"
%      (simulated = unweighted population share; implied_buy_prob = purchaser
%      share * inside share of the year / market-size weighted population
%      share).
%
% Nothing is saved: no Data.mat, no log file. Instruments, fixed effects,
% differentiation IVs and taste draws are skipped because they do not affect
% the table.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Settings ----------------------------------------------------------------

micro_sheet = 'hyuksoo split';        % same sheet as in C0
keep_micro_years = [2015 2016 2017 2024];      % same years as C0 keeps (C0 drops 2020-2025;
                                          % [] = all years in both the sheet and the data)
deflate_cutoffs = false;                  % false: nominal cutoffs vs. IncomeMat as in C0
                                          % (the deflator division is commented out there);
                                          % true: divide the cutoffs by the national deflator

replacement_cycle = 5;                    % same as C0
ndraws = 200;                             % same as C0
nrand = 5;                                % constant + 4 random-coefficient attributes, as in C0
                                          % (Spec.random_chars = {'GV','ln_driving_range','ln_hp_per_weight','fuel_cost_real'});
                                          % only used for the Halton dimension so that the
                                          % income draws are identical to those in C0

%% Paths and data ----------------------------------------------------------

rng(12345,'twister');

project_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(project_dir,'functions'));
[data_dir, ~, ~, aux_dir] = set_data_path();
addpath(genpath(fullfile(project_dir,'econometrics_toolbox')));
addpath(genpath(fullfile(project_dir,'functions')));

data = parquetread(fullfile(data_dir,'df.parquet'));
household_file = fullfile(aux_dir,'Household Size.xlsx');
income_file = fullfile(aux_dir,'IncDist_GeoMarket.xlsx');
micro_file = fullfile(aux_dir,'micro moments.xlsx');
national_deflator_file = fullfile(aux_dir,'National_level_deflator.xlsx');

VarMap = {
    'year',             'Year';
    'sales',            'Quantity';
    'location_id',      'Geo_Market_id';
    'population',       'Population';
    'GV',               'gv';
};
data = renamevars(data, VarMap(:,2), VarMap(:,1));

%% Market sample and s_0 (as in C0, Step 1) ---------------------------------

% Drop models with little sales in a market
data = data(data.sales >= 0, :);

data = data(data.year < 2025, :);

% Drop models with little total sales in a year (GV: 1000, EV: 200)
g = findgroups(data.year, data.model_id);
model_year_sales = splitapply(@sum, data.sales, g);
keep = (data.GV == 1 & model_year_sales(g) >= 500) | ...
       (data.GV == 0 & model_year_sales(g) >= 0);
data = data(keep, :);

data = sortrows(data, {'year','location_id','model_id'});

% Household size
household_size = readtable(household_file, 'Sheet', 'Household Size');
household_size = renamevars(household_size, 'Year', 'year');
household_size = household_size(:, {'year','Geo_Market','total_hh_size'});
data = join(data, household_size, 'Keys', {'year','Geo_Market'});
data = sortrows(data, {'year','location_id','model_id'});

% Market size, shares and s_0
mkt_size = double(data.population) .* 10000 ./ ...
           (double(data.total_hh_size) .* replacement_cycle);
s_jt = double(data.sales) ./ mkt_size;

totshr = zeros(height(data), 1);
market_ids = unique(data.market_id);
for i = 1:length(market_ids)
    ix = find(data.market_id == market_ids(i));
    totshr(ix,:) = sum(s_jt(ix));
end
s_0 = 1 - totshr;

disp('Market share construction');
disp(['Replacement cycle used in market size: ', num2str(replacement_cycle), ' years']);

% Keep s_0 and the year before the filter below (as in C0); the statistics
% are printed by year once the survey years are known
s_0_all = s_0;
year_all = double(data.year);

keep = (s_0 > 1e-4) & (s_jt > 0);
data = data(keep,:);
mkt_size = mkt_size(keep);
s_0 = s_0(keep);

% Market index
nprods = height(data);
market_ids = sort(unique(data.market_id));
nmkt = length(market_ids);
index_begin = zeros(nmkt, 1);
index_end = zeros(nmkt, 1);
for m = 1:nmkt
    tt = find(data.market_id == market_ids(m));
    index_begin(m) = tt(1);
    index_end(m) = tt(end);
end
assert(issorted(index_begin), 'market order != row order');

fprintf('Products: %d, markets: %d\n', nprods, nmkt);

%% Simulated household incomes (as in C0, Step 3) ---------------------------

census = readtable(income_file, 'Sheet', 'urban');
census = renamevars(census, 'Year', 'year');

temp = unique(data(:, {'year','Geo_Market','market_id','deflator','total_hh_size'}));
census = unique(census(:, {'year','Geo_Market','inc_mu','inc_sigma'}));
census = innerjoin(temp, census, 'Keys', {'year','Geo_Market'});
census = sortrows(census, 'market_id');

LogNormalpara = double([census.inc_mu census.inc_sigma]);

n_halt_dim = nrand + 2;
halt = halton_draws(ndraws, n_halt_dim, nmkt, 12345, 100, 1);
halt = min(halt,0.99);
halt = max(halt,0.01);

halt_inc = halt(:,:,1);
halt_inc = min(halt_inc, 0.97);
halt_inc = max(halt_inc, 0.03);
HaltMat = norminv(halt_inc,0,1);

IncomeMat = zeros(nmkt,ndraws);
for m = 1:nmkt
    IncomeMat(m,:) = exp(LogNormalpara(m,1) + LogNormalpara(m,2) * HaltMat(m,:));
    IncomeMat(m,:) = IncomeMat(m,:) * double(census.total_hh_size(m));
    IncomeMat(m,:) = IncomeMat(m,:) / double(census.deflator(m));
end
IncomeMat = max(IncomeMat,24000);  % income floor of 24,000 yuan, as in C0
IncomeMat = IncomeMat / 10000;     % unit: 10k yuan
clear halt halt_inc HaltMat

%% Micro-moment sheet and income groups (as in C0, Step 4) ------------------

micro = readtable(micro_file, 'Sheet', micro_sheet);
micro_raw = readcell(micro_file, 'Sheet', micro_sheet);
micro(1,:) = [];

national_deflator = readtable(national_deflator_file);
national_deflator = renamevars(national_deflator, 'Year', 'year');

micro_labels = strip(string(micro{:,1}));
sample_row = find(strcmp(micro_labels,'Sample Size'), 1);
group_rows = (1:sample_row-1)';

lower_bound = zeros(length(group_rows),1);
upper_bound = zeros(length(group_rows),1);
for g = 1:length(group_rows)
    r = group_rows(g);
    nums = str2double(regexp(char(micro_labels(r)), '\d+(?:\.\d+)?', 'match'));
    if isscalar(nums) && contains(micro_labels(r), '>')
        lower_bound(g) = nums(1);
        upper_bound(g) = Inf;
    elseif isscalar(nums)
        lower_bound(g) = -Inf;
        upper_bound(g) = nums(1);
    else
        lower_bound(g) = nums(1);
        upper_bound(g) = nums(end);
    end
end
nincgrp = length(group_rows);

% Assign each simulated household to an income group
market_year = double(census.year);
incgrpSm = zeros(size(IncomeMat));
for m = 1:nmkt
    income_m = IncomeMat(m,:);
    if deflate_cutoffs
        d_m = double(national_deflator.deflator(national_deflator.year == market_year(m)));
    else
        d_m = 1;
    end
    for g = 1:nincgrp
        lower_m = lower_bound(g) / d_m;
        upper_m = upper_bound(g) / d_m;
        if isinf(lower_m)
            incgrpSm(m, income_m < upper_m) = g;
        elseif isinf(upper_m)
            incgrpSm(m, income_m >= lower_m) = g;
        else
            incgrpSm(m, income_m >= lower_m & income_m < upper_m) = g;
        end
    end
end

% Survey years
year_cols = micro.Properties.VariableNames(2:end);
year_headers = micro_raw(1,2:end);
micro_years = zeros(length(year_cols),1);
for i = 1:length(year_cols)
    micro_years(i) = str2double(regexp(string(year_headers{i}), '\d{4}', 'match', 'once'));
end

model_years = sort(unique(double(data.year)));
if isempty(keep_micro_years)
    year_keep = find(ismember(micro_years, model_years));
else
    year_keep = find(ismember(micro_years, model_years) & ismember(micro_years, keep_micro_years));
    missing_years = setdiff(keep_micro_years(:), micro_years(year_keep));
    if ~isempty(missing_years)
        warning('Requested years not available in the sheet or in the model data: %s', ...
            mat2str(missing_years(:)'));
    end
end
if isempty(year_keep)
    error('No survey year is kept. Check micro_sheet and keep_micro_years.');
end

disp(['Micro-moment sheet: ', micro_sheet]);
disp(['Micro moment years kept: ', strjoin(cellstr(string(micro_years(year_keep)')), ', ')]);
% s_0 in each micro-moment year
kept_years = micro_years(year_keep);
s0_by_year = zeros(numel(kept_years), 4);
for i = 1:numel(kept_years)
    s0_y = s_0_all(year_all == kept_years(i));
    s0_by_year(i,:) = [min(s0_y), mean(s0_y), max(s0_y), std(s0_y)];
end
disp('s_0 by micro-moment year:');
disp(array2table(s0_by_year, 'VariableNames', {'min','mean','max','std'}, ...
    'RowNames', cellstr(string(kept_years))));

disp(['Income groups: ', num2str(nincgrp), '; cutoffs deflated: ', mat2str(deflate_cutoffs)]);
disp(table(string(micro_labels(group_rows)), lower_bound, upper_bound, ...
    'VariableNames', {'label','lower','upper'}));
disp(table(micro_years(year_keep), double(micro{sample_row, year_cols(year_keep)})', ...
    'VariableNames', {'year','sample_size'}));

IncMom_all = double(micro{group_rows, year_cols(year_keep)})';

%% Income-group shares: simulated vs. micro moments -------------------------

inside_mkt = 1 - s_0(index_begin);
size_mkt = mkt_size(index_begin);

incgrp_compare = [];
for i = 1:length(year_keep)
    y = micro_years(year_keep(i));
    ix = (market_year == y);

    w = size_mkt(ix) / sum(size_mkt(ix));
    inside_share_y = sum(w .* inside_mkt(ix));

    for g = 1:nincgrp
        simulated = sum(sum(incgrpSm(ix,:) == g)) / (sum(ix) * ndraws);
        micro_moment = IncMom_all(i,g);
        pop_w = sum(w .* mean(incgrpSm(ix,:) == g, 2));
        implied_prob = micro_moment * inside_share_y / pop_w;

        incgrp_compare = [incgrp_compare; ...
            y, g, simulated, micro_moment, simulated - micro_moment, implied_prob]; %#ok<AGROW>
    end
end

% Purchase probability relative to the first income group in the same year
p_rel = nan(size(incgrp_compare,1),1);
for y = unique(incgrp_compare(:,1))'
    in_y = incgrp_compare(:,1) == y;
    p1 = incgrp_compare(in_y & incgrp_compare(:,2) == 1, 6);
    p_rel(in_y) = incgrp_compare(in_y, 6) / p1;
end
incgrp_compare = [incgrp_compare, p_rel];

incgrp_compare = array2table(incgrp_compare, ...
    'VariableNames', {'year','group','simulated','micro','difference','implied_buy_prob', ...
                      'P_over_P1'});

disp('Income group shares: simulated vs. micro moments:');
disp(incgrp_compare);
