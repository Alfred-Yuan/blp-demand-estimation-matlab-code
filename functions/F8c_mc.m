function MC_All = F8c_mc(Cst, XMat, YMat, InMat, Supply)
% Calculate MC by year and model
% Elas = (dQ_j/dP_k)*(P_k/Q_j), where Q_j is model j's national sales

% Obtain constants
model_id = Cst.model_id;     % model id at market-product level
[model_ids, ~, ~] = unique(model_id);
nmod = length(model_ids);    % num of unique models

% Model level firm id, tax
period = double(Supply.year);
firm_id = double(Supply.firm_id);
agg_model_id = double(Supply.model_id);
T = double(Supply.T);
firm_price = double(Supply.firm_price);

sales = double(Supply.sales);

[~, dQdP, ~] = F8b_elasticity(Cst, XMat, YMat, InMat);

years = sort(unique(period));
MC_All = zeros(length(period),1);

for i = 1:length(years)
    tmpdQdP = dQdP((i-1)*nmod+1:i*nmod,:);

    % ttt substract the rows corrspounding to year i
    ttt = find(period == years(i));

    % Acquire the location for year i's model from dQdp matrix
    [~, tid] = ismember(agg_model_id(ttt), model_ids);

    tmpdQdP = tmpdQdP(tid,tid);   % dQ/dp^c at the model-year level

    % Construct ownership matrix using firm id
    tmpFirmID = firm_id(ttt);
    Ownership = (tmpFirmID == tmpFirmID');

    tmpSales = sales(ttt);
    tmpT = T(ttt);
    tmpfirm_price = firm_price(ttt);

    % FOC: Q + T .* (Omega .* dQdp^c)'(firm price -mc) = 0

    % Define: Delta = T .* (Omega .* dQdp^c)'
    Delta = diag(tmpT) * (tmpdQdP .* Ownership)';

    % Recover marginal cost: mc= inv(Delta) * Sales + firm price
    MC_All(ttt) = tmpfirm_price + pinv(Delta) * tmpSales; % sparse(Delta) \ tmpSales = inv(Delta) * tmpSales

    if any(isnan(tmpdQdP(:))) || any(isinf(tmpdQdP(:))) || ...
       any(isnan(tmpT(:))) || any(isinf(tmpT(:))) || ...
       any(isnan(tmpfirm_price(:))) || any(isinf(tmpfirm_price(:))) || ...
       any(isnan(tmpSales(:))) || any(isinf(tmpSales(:)))
    
        fprintf('[F8c] Bad input in year %.0f\n', years(i));
        fprintf('NaN tmpdQdP: %d, NaN tmpT: %d, NaN firm_price: %d, NaN sales: %d\n', ...
            sum(isnan(tmpdQdP(:))), sum(isnan(tmpT(:))), ...
            sum(isnan(tmpfirm_price(:))), sum(isnan(tmpSales(:))));
    end
end

end
