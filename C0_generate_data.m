clc; clear
format short g

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create the estimation data set that includes:
% Product attributes, individual taste draws and income draws
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 0: Preliminaries
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set the seed for the random number generator
% 'rng' is Matlab's build-in function to generate random numbers
rng(12345,'twister');

% Directories
project_dir = fileparts(mfilename('fullpath'));
repo_dir = fileparts(fileparts(project_dir));

addpath(fullfile(project_dir,'functions'));
[data_dir, out_dir, log_dir, aux_dir] = set_data_path();

data_file = fullfile(data_dir,'df.parquet');
data = parquetread(data_file);
household_file = fullfile(aux_dir,'Household Size.xlsx');
income_file = fullfile(aux_dir,'IncDist_GeoMarket.xlsx');
micro_file = fullfile(aux_dir,'micro moments.xlsx');
national_deflator_file = fullfile(aux_dir,'National_level_deflator.xlsx');

if ~exist(out_dir,'dir')
    mkdir(out_dir);
end

if ~exist(log_dir,'dir')
    mkdir(log_dir);
end

% Log all printed output while keeping it visible in the terminal
log_file = fullfile(log_dir,'C0_generate_data_log.txt');
diary(log_file);
diary on

% Functions
addpath(genpath(fullfile(project_dir,'econometrics_toolbox')));
addpath(genpath(fullfile(project_dir,'functions')));

% Constants
ndraws = 200;             % number of pseudo households in each market
maxiter = 1500;           % maximum number of iterations for mean value inversion
mvaltol = 1e-10;          % tolerance for inversion of mean value
replacement_cycle = 5;    % households are assumed to replace cars every 5 years
lc_band = 0.75;           % local-count IV bandwidth in standardized units

% Variable names mapping
% First column: variable name used in the code
% Second column: variable name in the raw data set
VarMap = {
    'year',             'Year';
    'sales',            'Quantity';
    'location_id',      'Geo_Market_id';
    'firm_id',          'manufacturer_id';
    'price_real',       'price';
    'ln_price',         'log_price';
    'ln_volume',        'log_size';
    'ln_driving_range', 'log_ev_range';
    'ln_hp',            'log_power';
    'fuel_type_id',     'fueltype_num';
    'segment_id',       'sizesegment_num';
    'bodytype_id',      'bodytype_num';
    'population',       'Population';
    'duration_yr',      'duration';
    'centralS_real',    'CentralS';
    'localS_real',      'LocalS';
    'GV',               'gv';
    'EV',               'ev';
    'BEV',              'is_bev';
    'PHEV',             'is_phev';
    'REEV',             'is_reev';
};

cs_iv = {'z_ev_iv_charger_poi_cs_othermkt', ...
         'z_ev_iv_charger_poi_cs_cost'};

gs_iv = {'z_gv_iv_charger_carfix_poi_gs_othermkt', ...
         'z_gv_iv_charger_bus_terminal_poi_gs_othermkt', ...
         'z_gv_iv_charger_car_service_poi_gs_othermkt', ...
         'z_gv_iv_charger_road_poi_oil_price', ...
         'z_gv_iv_charger_carfix_poi_oil_price', ...
         'z_gv_iv_charger_bus_terminal_poi_oil_price', ...
         'z_gv_iv_charger_car_service_poi_oil_price', ...
         'z_gv_iv_charger_road_poi_gs_othermkt'};

diff_iv = {};

price_iv = {'st_rate', 'centralS_real'};

% Model specification (If you want to change the model specification, you can do it here)
Spec.price_in_mean_utility = false;      % whether mean utility includes price
Spec.price_var = 'price_real';           % price used in random price coefficient
Spec.included_chars = {'ln_volume', 'ln_driving_range', 'ln_hp', 'fuel_cost_real',};
Spec.endog_chars = {'price_real', 'ln_cs_density_ev', 'ln_gs_density_gv'};
% Random-coefficient attributes (in addition to the constant):
% GV, range, power, running cost
Spec.random_chars = {'GV', 'ln_driving_range', 'ln_hp', 'fuel_cost_real'};

Spec.excluded_iv = [price_iv, cs_iv, gs_iv];
Spec.fixed_effects = {'location_id', 'year', 'fuel_type_id', ...
                      'duration_yr', 'bev_brand_model_id', 'segment_id', ...
                      'bodytype_id'};
Spec.cluster = 'market_id';              % cluster for standard errors


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 1: Main Data (Vehicle sales and attributes by markets)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1.1 Load and prepare the data
data_file = fullfile(data_dir,'df.parquet');

% Map variable names
data = renamevars(data, VarMap(:,2), VarMap(:,1));

% Drop those models have very little sales (<30) in each market
data = data(data.sales >= 0, :);


% Nest for the fixed effects (same definition as the new data builder):
% model for BEVs, brand for all other fuel types
bev_key = string(data.Brand);
is_bev_row = data.BEV == 1;
bev_key(is_bev_row) = string(data.Model(is_bev_row));
[~, ~, data.bev_brand_model_id] = unique(bev_key);

% Drop those models have very little total sales (GV: 1000, EV: 200) each year
g = findgroups(data.year, data.model_id);
model_year_sales = splitapply(@sum, data.sales, g);
keep = (data.GV == 1 & model_year_sales(g) >= 0) | ...
       (data.GV == 0 & model_year_sales(g) >= 0);
data = data(keep, :);

% Order very important: sort by year, location id, and model id.
data = sortrows(data, {'year','location_id','model_id'});

% 1.2 Create differentiation iv

%% Estimate exogenous component of price
p_hat_fe_dum = [];
for i = 1:length(Spec.fixed_effects)
    if strcmp(Spec.fixed_effects{i}, 'bev_brand_model_id')
        continue
    end
    tmpdum = dummify(data.(Spec.fixed_effects{i}));
    tmpdum(:,1) = [];
    p_hat_fe_dum = [p_hat_fe_dum tmpdum];

end

% Construct first-stage variables
p_hat_X = [double(data{:,Spec.included_chars}), ...
           double(data{:,Spec.excluded_iv}), ...
           p_hat_fe_dum];

p_hat_y = double(data.(Spec.price_var));

% Absorb brand-model fixed effects
p_hat_X_dm = demean(data.bev_brand_model_id, p_hat_X);
p_hat_y_dm = demean(data.bev_brand_model_id, p_hat_y);

% Remove collinear columns
[p_hat_X_dm, keep_idx] = indep_columns(p_hat_X_dm, 1e-5);
p_hat_X = p_hat_X(:, keep_idx);

% Estimate first stage
[p_hat_beta, ~, ~, ~, ~] = ols(p_hat_y_dm, p_hat_X_dm);

% Recover absorbed FE and fitted price
g = findgroups(data.bev_brand_model_id);

data.p_hat = p_hat_X * p_hat_beta;
fe = accumarray(g, p_hat_y - data.p_hat, [], @mean);
data.p_hat = data.p_hat + fe(g);
market_ids = unique(data.market_id);

% Add differentiation style iv for random coefficient variables
iv_chars = Spec.random_chars;   % lc_iv_p_hat is no longer constructed or used
for k = 1:length(iv_chars)
    xname = iv_chars{k};
    if ismember(xname, {'GV','EV'})
        continue
    end

    lc_name = ['lc_iv_', xname];
    ed_name = ['ed_iv_', xname];
    lc_iv = zeros(height(data),1);
    ed_iv = zeros(height(data),1);

    x = double(data.(xname));
    if strcmp(xname, 'ln_driving_range')
        ix_ev = (data.GV == 0);
        z = zeros(height(data),1);
        z(ix_ev) = (x(ix_ev) - mean(x(ix_ev))) ./ std(x(ix_ev));

        for i = 1:length(market_ids)
            ix = find(data.market_id == market_ids(i) & data.GV == 0);
            zi = z(ix);
            lc_iv(ix) = sum(abs(zi - zi') <= lc_band, 2) - 1;
            ed_iv(ix) = sqrt(max(0, sum((zi - zi').^2, 2)));
        end
    else
        z = (x - mean(x)) ./ std(x);

        for i = 1:length(market_ids)
            ix = find(data.market_id == market_ids(i));
            zi = z(ix);
            lc_iv(ix) = sum(abs(zi - zi') <= lc_band, 2) - 1;
            ed_iv(ix) = sqrt(max(0, sum((zi - zi').^2, 2)));
        end
    end

    data.(lc_name) = lc_iv;
    data.(ed_name) = ed_iv;
    % diff_iv = [diff_iv, {lc_name}];
end

% Add differentiation style IV for the GV dummy (Hyuk-soo style).
% lc_iv_GV: for each GV product, count the other GV products in the same market
% whose standardized fitted price (p_hat) is within lc_band; non-GV rows = 0.
% (To use fuel_cost_real instead, replace the next line with fuel_cost_real.)
lc_gv_var = 'p_hat';
x_gv = double(data.(lc_gv_var));
z = (x_gv - mean(x_gv)) ./ std(x_gv);

is_gv = data.GV == 1;
lc_iv_GV = zeros(height(data),1);

for i = 1:length(market_ids)
    ix = find(data.market_id == market_ids(i) & is_gv);
    zi = z(ix);
    lc_iv_GV(ix) = sum(abs(zi - zi') <= lc_band, 2) - 1;
end

data.lc_iv_GV = lc_iv_GV;
clear lc_iv_GV is_gv

diff_iv = [diff_iv, {'lc_iv_GV'}]; % Choose the IVs we should use
% Drop lc_iv_fuel_cost_real from the instrument set (the random coefficient on
% fuel_cost_real is kept); the column stays in data but is not used as an IV.
diff_iv(strcmp(diff_iv, 'lc_iv_fuel_cost_real')) = [];
Spec.excluded_iv = [Spec.excluded_iv, diff_iv];

% 1.3 Load the HH size data used for calculating mean utility and HH income

% Merge household size by year and geographic market
household_size = readtable(household_file, 'Sheet', 'Household Size');
household_size = renamevars(household_size, 'Year', 'year');

household_size = household_size(:, {'year','Geo_Market','total_hh_size'});
data = join(data, household_size, 'Keys', {'year','Geo_Market'});
data = sortrows(data, {'year','location_id','model_id'});

% 1.4 Market share and mean utility

% Population is in 10k persons in the raw data.
% Market size is population divided by household size and replacement cycle.
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
s0_stats = table(min(s_0), mean(s_0), max(s_0), std(s_0), ...
    'VariableNames', {'min','mean','max','std'}, 'RowNames', {'s_0'});
disp(s0_stats);

keep = (s_0 > 1e-4) & (s_jt > 0);
data = data(keep,:);
mkt_size = mkt_size(keep);
s_jt = s_jt(keep);
s_0 = s_0(keep);
mean_utility = log(s_jt) - log(s_0);

% 1.5 Create index for the markets

% Number of observations and markets after filtering
nprods = height(data);
nmkt = length(unique(data.market_id));

% First and last indinces for each market
index_begin = zeros(nmkt, 1);  % start of observations for one market
index_end   = zeros(nmkt, 1);  % end of observations for one market
nprod_vec   = zeros(nmkt, 1);  % number of products in each market

market_ids = sort(unique(data.market_id));
for m = 1:nmkt
    tt = find(data.market_id == market_ids(m));
    index_begin(m) = tt(1);
    index_end(m) = tt(end);
    nprod_vec(m) = length(tt);
end

% Read data variables
for i = 1:size(data,2)
    eval([data.Properties.VariableNames{i}, ' = data.(data.Properties.VariableNames{i});']);
end

assert(issorted(index_begin), 'market order != row order');


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 2: Rescale, Fixed Effects and IVs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 2.1 Rescale attributes and create new variables

% Rescale attributes used in the random coefficients
data.ln_volume = double(data.ln_volume) - log(1000000);      % Change the unit to be m^3
data.fuel_cost_real = double(data.fuel_cost_real) ./ 100;    % Change the unit to be 100 yuan per 100 km

% Charging and gas station densities interacted with vehicle fuel type
data.cs_density = double(data.cs_count) ./ double(data.population) * 100;
data.gs_density = double(data.gs_count) ./ double(data.population) * 100;
data.ln_cs_density_ev = double(data.EV) .* log(data.cs_density + 1);
data.ln_gs_density_gv = double(data.GV) .* log(data.gs_density + 1);

Spec.linear_chars = [Spec.endog_chars,Spec.included_chars];

disp('Model specification');
disp(['Control variables: ', strjoin(Spec.included_chars, ', ')]);
disp(['Random coefficients: ', strjoin([{'constant'}, Spec.random_chars], ', ')]);

disp(['Fixed effects: ', strjoin(Spec.fixed_effects, ', ')]);
disp(['Cluster variable: ', Spec.cluster]);
disp(['Differentiation IVs: ', strjoin(diff_iv, ', ')]);
disp(' ');

data.lc_iv_GV = zscore(double(data.lc_iv_GV));

IV_exc = double(data{:,Spec.excluded_iv});
IV_exc_names = Spec.excluded_iv;


% 2.2. Create FE objects, independent variables and random chars

% Fixed effects included as dummy variables
FE_dum = [];
FE_names = {};  % Names are kept only for variables reported later
for i = 1:length(Spec.fixed_effects)
    if strcmp(Spec.fixed_effects{i}, 'bev_brand_model_id')
        continue
    end
    tmpdum = dummify(data.(Spec.fixed_effects{i}));
    tmpdum(:,1) = [];
    FE_dum = [FE_dum tmpdum];
    FE_names = [FE_names, repmat({''}, 1, size(tmpdum,2))];
end

% Independent variables
chars = [double(data{:,Spec.linear_chars}) FE_dum];
chars_names = [Spec.linear_chars FE_names];

% Attributes with a random coefficient
if isempty(Spec.random_chars)
    randchars = ones(nprods,1);
else
    % Convert column by column: data{:,cols} would cast the whole matrix to the
    % integer class of GV and round the continuous attributes to integers.
    randchars = ones(nprods,1);
    for k = 1:length(Spec.random_chars)
        randchars = [randchars, double(data.(Spec.random_chars{k}))];
    end
end
nrand = size(randchars,2);  % number of attributes with random coef


% 2.3. Demean to absorb model fixed effects

% Demean mean utility
mean_utility_dm = demean(data.bev_brand_model_id, mean_utility);

% Demean independent variables
chars_dm = demean(data.bev_brand_model_id, chars);

% Demean instruments
IV_exc_dm = demean(data.bev_brand_model_id, IV_exc);

% Remove dependent columns from demeaned attributes

% Demeaned attributes
[chars_dm, idx] = indep_columns(chars_dm, 1e-5);
chars = chars(:,idx);
chars_names = chars_names(idx);
% idx(1:min(10,length(idx)))  % check if all necessary variables remain

% Remove dummies that creates full rank problem as computing invWt
% (If a dummy varialbe has only a single '1', then drop it)
ix = 1:size(chars,2);
idx_to_drop = ix(sum(abs(chars),1) == 1);
chars(:, idx_to_drop) = [];
chars_dm(:, idx_to_drop) = [];
chars_names(idx_to_drop) = [];

% If mean values include the price variable
if Spec.price_in_mean_utility
    Xfull = chars_dm;
    Xfull_names = chars_names;
else
    Xfull = chars_dm(:,2:end);
    Xfull_names = chars_names(2:end);
end

% The demeaned price
price_real_dm = chars_dm(:,1);

% Demeaned instruments including exogenous attributes
is_exog = ~ismember(chars_names, Spec.endog_chars);
IVall = [IV_exc_dm, chars_dm(:, is_exog)];
IVall_names = [IV_exc_names, chars_names(is_exog)];

[IVall, idx] = indep_columns(IVall, 1e-5);
IVall_names = IVall_names(idx);
% idx(1:min(15,length(idx)))  % check if all necessary variables remain


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 3: Draw of incomes and random tastes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 3.1. Draw Pseudo household from simulated annual incomes
census = readtable(income_file, 'Sheet', 'urban');

% Map variable names
census = renamevars(census, 'Year', 'year');

% Re-order the IncDist (because now we have a different market order)
temp = unique(data(:, {'year','Geo_Market','market_id','deflator','total_hh_size'}));
census = unique(census(:, {'year','Geo_Market','inc_mu','inc_sigma'}));
census = innerjoin(temp, census, 'Keys', {'year','Geo_Market'});
census = sortrows(census, 'market_id');

% Parameters of the log normal distribution
LogNormalpara = double([census.inc_mu census.inc_sigma]);


% 3.2. Generate market-specific Halton draws

% Halton sequence for income (1), attributes (nrand), and price (1)
% Each market receives a different, non-overlapping block of Halton draws
n_halt_dim = nrand + 2;
halt = halton_draws(ndraws, n_halt_dim, nmkt, 12345, 100, 1);

% Truncate extreme draws
halt = min(halt,0.99);
halt = max(halt,0.01);


% 3.3. Simulate market-specific household incomes

% The first column of halt is for income draws
halt_inc = halt(:,:,1);

% Truncate income draws to avoid too large or small income
halt_inc = min(halt_inc, 0.97);
halt_inc = max(halt_inc, 0.03);

% Draw from the inverse of standard normal distribution
HaltMat = norminv(halt_inc,0,1);

% Simulate household incomes in each market
IncomeMat = zeros(nmkt,ndraws);
for m = 1:nmkt
    IncomeMat(m,:) = exp(LogNormalpara(m,1) + LogNormalpara(m,2) * HaltMat(m,:));
    IncomeMat(m,:) = IncomeMat(m,:) * double(census.total_hh_size(m));
end

% Adjust for inflation
for m = 1:nmkt
    IncomeMat(m,:) = IncomeMat(m,:) / double(census.deflator(m));
end

% Truncate household income draws to [24,000, 1,000,000] yuan
% IncomeMat = min(IncomeMat,1000000);
IncomeMat = max(IncomeMat,24000);

IncomeMat = IncomeMat/10000;   % Change the unit to be 10k yuan

% Expand IncomeMat from nmkt by ndraws to nprods by ndraws
% (in each market, each household has the same draw for all products)
IncomeMat_all = zeros(nprods,ndraws);
for m = 1:nmkt
    IncomeMat_all(index_begin(m):index_end(m),:) = ...
        ones(nprod_vec(m),1) * IncomeMat(m,:);
end


% 3.4. Draw random tastes for vehicle attributes and price

% Draw from the inverse of standard normal distribution
% Columns 2 to nrand+1 are for attributes with random coefficients;
% the last column is for the price random coefficient
house_draws = norminv(halt(:,:,2:end),0,1);

% Default \nu_p comes from a normal distribution; this line can transform it
% to a log-normal distribution
% house_draws(:,:,nrand+1) = exp(house_draws(:,:,nrand+1));

% Reformat the household draws
house_draws_all = zeros(nprods,ndraws,nrand+1);

% In each market, each household has the same draw for all products
for m = 1:nmkt
    for k = 1:nrand+1
        house_draws_all(index_begin(m):index_end(m),:,k) = ...
            ones(nprod_vec(m),1) * house_draws(m,:,k);
    end
end

clear HaltMat halt halt_inc house_draws n_halt_dim


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 4: Micro moment data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Income groups for micro moments
micro_sheet = 'hyuksoo split 5g';
micro = readtable(micro_file, 'Sheet', micro_sheet);
micro_raw = readcell(micro_file, 'Sheet', micro_sheet);
micro(1,:) = [];

% National deflator used for micro-moment income thresholds
national_deflator = readtable(national_deflator_file);
national_deflator = renamevars(national_deflator, 'Year', 'year');

% The first column stores income-range labels; the last row stores sample size
micro_labels = string(micro{:,1});
micro_labels = strip(micro_labels);
sample_row = find(strcmp(micro_labels,'Sample Size'), 1);
group_rows = (1:sample_row-1)';

% Extract nominal income cutoffs from labels such as "<4.8", "4.8-7.2", and ">36"
% The raw labels and IncomeMat are both in 10k yuan.
lower_bound = zeros(length(group_rows),1);
upper_bound = zeros(length(group_rows),1);
for g = 1:length(group_rows)
    r = group_rows(g);
    nums = regexp(char(micro_labels(r)), '\d+(?:\.\d+)?', 'match');
    nums = str2double(nums);

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

% Assign each simulated household income draw to one income group.
% The micro-moment cutoffs are nominal income cutoffs, so each year's cutoff
% is divided by the national deflator before matching to real income draws.
market_year = double(census.year);
incgrpSm = zeros(size(IncomeMat));
for m = 1:nmkt
    deflator_m = double(national_deflator.deflator(national_deflator.year == market_year(m)));
    income_m = IncomeMat(m,:);

    for g = 1:length(group_rows)
        lower_m = lower_bound(g) / 1; % deflator_m;
        upper_m = upper_bound(g) / 1; % deflator_m;

        if isinf(lower_m)
            incgrpSm(m, income_m < upper_m) = g;
        elseif isinf(upper_m)
            incgrpSm(m, income_m >= lower_m) = g;
        else
            incgrpSm(m, income_m >= lower_m & income_m < upper_m) = g;
        end
    end
end

% Income-share micromoments and annual survey sample sizes
year_cols = micro.Properties.VariableNames(2:end);
year_headers = micro_raw(1,2:end);
micro_years = zeros(length(year_cols),1);
for i = 1:length(year_cols)
    tmp = regexp(string(year_headers{i}), '\d{4}', 'match', 'once');
    micro_years(i) = str2double(tmp);
end

model_years = sort(unique(double(data.year)));

% Drop selected micro moment years, if needed
drop_micro_years = [2020; 2021; 2022; 2023; 2024; 2025];   % keeps 2015, 2016, 2017
% Keep only years that appear both in the model data and the income survey
year_keep = find(ismember(micro_years, model_years) & ...
                 ~ismember(micro_years, drop_micro_years));

year_idx = zeros(length(year_keep),1);
for i = 1:length(year_keep)
    year_idx(i) = find(model_years == micro_years(year_keep(i)), 1);
end
year_idx = year_idx';

disp(['Micro moment years used in estimation data: ', ...
      strjoin(cellstr(string(micro_years(year_keep)')), ', ')]);

nincgrp = length(group_rows);       % number of income groups
nincmom = nincgrp - 1;              % number of income moments
nyr_micromom = length(year_idx);    % number of years with income moments

IncMom = micro{group_rows(1:nincmom), year_cols(year_keep)};
IncMom = double(IncMom)';

IncSampleSize = micro{sample_row, year_cols(year_keep)};
IncSampleSize = double(IncSampleSize(:));

% Check the number of each income group
incgrp_share = zeros(length(model_years), nincgrp);
for t = 1:length(model_years)
    ix = (market_year == model_years(t));
    for g = 1:nincgrp
        incgrp_share(t,g) = sum(sum(incgrpSm(ix,:) == g)) / ...
            (sum(ix) * ndraws);
    end
end

% Compare simulated income-group shares with the micro moment data
IncMom_all = micro{group_rows, year_cols(year_keep)};
IncMom_all = double(IncMom_all)';

% Market-level inside share (1 - s_0) and market size, in market order
inside_mkt = 1 - s_0(index_begin);
size_mkt = mkt_size(index_begin);

incgrp_compare = [];
for i = 1:length(year_keep)
    y = micro_years(year_keep(i));
    ix = (market_year == y);

    % Market-size weights within the year
    w = size_mkt(ix) / sum(size_mkt(ix));

    % Average inside share of the year (= total sales / total market size)
    inside_share_y = sum(w .* inside_mkt(ix));

    for g = 1:nincgrp
        simulated = sum(sum(incgrpSm(ix,:) == g)) / (sum(ix) * ndraws);
        micro_moment = IncMom_all(i,g);

        % Simulated population share of the group, market-size weighted
        pop_w = sum(w .* mean(incgrpSm(ix,:) == g, 2));

        % Purchase probability of the group implied by the survey:
        %   purchaser share = pop share * P(buy | group) / P(buy)
        %   => P(buy | group) = purchaser share * P(buy) / pop share
        % A value above 1 is impossible in the model.
        implied_prob = micro_moment * inside_share_y / pop_w;

        incgrp_compare = [incgrp_compare; ...
            y, g, simulated, micro_moment, simulated - micro_moment, ...
            implied_prob];
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
    'VariableNames', {'year','group','simulated','micro','difference', ...
                      'implied_buy_prob','P_over_P1'});

disp('Income group shares: simulated vs. micro moments:');
disp(incgrp_compare);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 5: Collect constants, variables, and matrices relevant for estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 5.1. Constants
nyear = length(model_years);                  % number of years
nmarket_per_year = nmkt / nyear;             % number of markets in each year
nK = nrand + 3;                              % number of non-linear parameters (sigmas, alpha1, alpha2, sigma_p)

Cst.nyear = nyear;
Cst.nmkt = nmkt;
Cst.ndraws = ndraws;
Cst.nprods = nprods;
Cst.nrand = nrand;
Cst.nK = nK;
Cst.nmarket_per_year = nmarket_per_year;
Cst.nincgrp = nincgrp;
Cst.nincmom = nincmom;
Cst.year_idx = year_idx;

Cst.index_begin = index_begin;
Cst.index_end = index_end;
Cst.nprod_vec = nprod_vec;

% Whether price enters mean utility or not
Cst.price_in_mean_utility = Spec.price_in_mean_utility;

% mkt_index (nprods by 1): the market that product j belongs to
mkt_index = zeros(nprods,1);
for m = 1:nmkt
    aa = index_begin(m);
    bb = index_end(m);
    mkt_index(aa:bb) = m;
end
clear aa bb

Cst.mkt_index = mkt_index;
Cst.model_id = double(data.model_id);
Cst.firm_id = data.firm_id;
Cst.brand_model = data.bev_brand_model_id;

Cst.mkt_size = double(mkt_size);
Cst.sales = double(data.sales);
Cst.year = data.year;
Cst.EV = data.EV;
Cst.GV = data.GV;
Cst.BEV = data.BEV;
Cst.PHEV = data.PHEV;
Cst.REEV = data.REEV;

Cst.centralS_real = data.centralS_real;
Cst.localS_real = data.localS_real;
Cst.total_tax = data.total_tax;
Cst.firm_price = data.firm_price;
Cst.ct = data.ct_rate;
Cst.st = data.st_rate;

Cst.maxiter = maxiter;
Cst.mvaltol = mvaltol;  % tolerance for inversion of mean value 'delta'


% 5.2. Characteristics
p_real = double(data.(Spec.price_var));
% Slice price and randchars market by market
CL_randchars = cell(nmkt,1);
CL_p_real = cell(nmkt,1);
for t = 1:nmkt
    aa = index_begin(t);
    bb = index_end(t);
    CL_randchars{t} = randchars(aa:bb,:);
    CL_p_real{t} = p_real(aa:bb);
end
clear aa bb

XMat.chars = chars;          % All characteristics including price
XMat.chars_dm = chars_dm;    % All demeaned characteristics including price
XMat.chars_names = chars_names;
XMat.p_real = p_real; % Real consumer price
XMat.CL_p_real = CL_p_real;
XMat.p_real_dm = price_real_dm;
XMat.Xfull = Xfull;          % Demeaned characteristics
XMat.Xfull_names = Xfull_names;
XMat.IVall = IVall;          % Demeaned IVs
XMat.IVall_names = IVall_names;
XMat.IV_exc_dm = IV_exc_dm;
XMat.IV_exc = IV_exc;
XMat.IV_exc_names = IV_exc_names;
XMat.randchars = randchars;  % Characteristics with random coeff
XMat.CL_randchars = CL_randchars;
XMat.theta2_names = [{'constant'}, Spec.random_chars, {'alpha1','alpha2','sigma_p'}];
XMat.ln_volume = data.ln_volume;
XMat.ln_driving_range = data.ln_driving_range;
XMat.ln_hp = data.ln_hp;
XMat.fuel_cost_real = data.fuel_cost_real;
XMat.centralS_real = data.centralS_real;
XMat.localS_real = data.localS_real;
XMat.total_tax = data.total_tax;
XMat.firm_price = data.firm_price;
XMat.ct_rate = data.ct_rate;
XMat.st_rate = data.st_rate;
XMat.cs_density = data.cs_density;
XMat.gs_density = data.gs_density;
XMat.duration_yr = data.duration_yr;


% Slice the household size and income
CL_hh = cell(nmkt,1);
CL_inc = cell(nmkt,1);
for t = 1:nmkt
    aa = index_begin(t);
    bb = index_end(t);
    CL_hh{t} = house_draws_all(aa:bb,:,:);
    CL_inc{t} = IncomeMat_all(aa:bb,:);
end
clear aa bb


MuMat.hh = house_draws_all;                  % nprods by ndraws by nrands+1
MuMat.inc = IncomeMat_all;
MuMat.CL_hh = CL_hh;
MuMat.CL_inc = CL_inc;

inc_grp_all = zeros(nprods,ndraws);
for m = 1:nmkt
    inc_grp_all(index_begin(m):index_end(m),:) = ...
        ones(nprod_vec(m),1) * incgrpSm(m,:);
end


MuMat.incgrp = inc_grp_all;                  % nprods by ndraws
MuMat.incgrpSm = incgrpSm;                   % nmkt by ndraws

MicroMom.IncMom = IncMom;
MicroMom.IncSampleSize = IncSampleSize;
MicroMom.year_idx = year_idx;
MicroMom.invWinc = eye(numel(IncMom));


% 5.3. Sales

% salesMK: total sales in each market (nmk by 1)
salesMk = zeros(nmkt,1);
for m = 1:nmkt
    salesMk(m) = sum(data.sales(index_begin(m):index_end(m)));
end
YMat.salesMk = salesMk;

% salesAllMK: total national sales in each year (nyr by 1)
salesAllMk = zeros(nyear,1);
for t = 1:nyear
    aa = (t-1)*nmarket_per_year + 1;
    bb = t*nmarket_per_year;
    salesAllMk(t) = sum(salesMk(aa:bb));
end
clear aa bb
YMat.salesAllMk = salesAllMk;

% Slice s_jt market by market
CL_sjt = cell(nmkt,1);
CL_mvalold = cell(nmkt,1);
for t = 1:nmkt
    aa = index_begin(t);
    bb = index_end(t);
    CL_sjt{t} = s_jt(aa:bb);
    CL_mvalold{t} = mean_utility(aa:bb);
end
clear aa bb

YMat.s_jt = s_jt;
YMat.CL_sjt = CL_sjt;
YMat.mean_utility = mean_utility;       % predicted mean_value for simple logit
YMat.mean_utility_dm = mean_utility_dm; % predicted mean_value for simple logit
YMat.CL_mvalold = CL_mvalold;

% Save Cst/YMat/XMat/MuMat
save(fullfile(out_dir,'Data.mat'),'MuMat','-v7.3')
save(fullfile(out_dir,'Data.mat'),'Cst','-append')
save(fullfile(out_dir,'Data.mat'),'YMat','-append')
save(fullfile(out_dir,'Data.mat'),'XMat','-append')
save(fullfile(out_dir,'Data.mat'),'MicroMom','-append')

diary off
