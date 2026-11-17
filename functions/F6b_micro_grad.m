function dmom2 = F6b_micro_grad(Cst, YMat, MuMat, ds_dtheta2Mat, ddelta_dtheta2, indshr_All, InMat)
% Gradient of mom2 (fst set of micro moments) w.r.t. theta2

% dmom2: dmom2/d_theta, sorted by yr & inc-bracket

% InMat: 5 fields, incshr, nmk by nincgrp, shares of each income group by mk;

% Read constants
nyear = Cst.nyear;
nmkt = Cst.nmkt;
ndraws = Cst.ndraws;
nK = Cst.nK;
nmarket_per_year = Cst.nmarket_per_year;
year_idx = Cst.year_idx;
nincmom = Cst.nincmom;

index_begin = Cst.index_begin;
index_end = Cst.index_end;

salesMk = YMat.salesMk;       % Sales in each market
salesAllMk = YMat.salesAllMk; % Annual domestic sales

% tot_buy, nmk by 1, model predicted buyers in each income bracket by mkt
incgrpSm = MuMat.incgrpSm; % Income bracket (nmkt by ndraws)
incshr = InMat.incshr;     % Fraction of buyers in each income bracket by market
                           % (nmkt by nincgrp)
tot_buy = InMat.tot_buy;   % Number of cars bought by ndraws HH, by market
                           % (nmkt by 1)

nincgrp = size(incshr,2);  % Number of income groups

%% Step2: iterate over nK non-linear parameters

% df_dtheta2: f is the predicted fraction of buyers in each income group
% df_dtheta2A is the direct derivative of f wrt theta2
% The 1st nincgrp columns of df_dtheta2A are derivatives wrt 1st element of theta2, by inc bracket
df_dtheta2A = zeros(nmkt, nincgrp*nK);
for j = 1:nincgrp % nincgrp income groups
    tmpincgrp = (incgrpSm==j); % nmk by ndraws
    tmpincresid = tmpincgrp - incshr(:,j)*ones(1,ndraws);

    for k = 1:nK
        % Matrix of individual derivatives (nmk by ndraws): sum_j [ds_{mij}/d_theta2_k]
        tmpdtheta2 = ds_dtheta2Mat(:,:,k); % nmk by ndraws by 1
        temp1 = tmpincresid .* tmpdtheta2;
        df_dtheta2A(:,(k-1)*nincgrp+j) = sum(temp1,2) ./ tot_buy;  % tot_buy: nmk by 1
    end
end

% Split the vector by market before doing the parallel calculation
CL_indshr_All = cell(nmkt,1);
CL_ddelta_dtheta2 = cell(nmkt,1);

for t = 1:nmkt
    aa = index_begin(t);
    bb = index_end(t);
    CL_indshr_All{t} = indshr_All(aa:bb,:);
    CL_ddelta_dtheta2{t} = ddelta_dtheta2(aa:bb,:);
end

% df_dtheta2B is the indirect derivative through mean utility:
% df/d_delta * d_delta/d_theta
% The fst nincgrp columns of df_dtheta2B are derivatives wrt 1st element of theta2, by inc bracket
df_dtheta2B = zeros(nmkt, nincgrp*nK);
parfor t = 1:nmkt
    tmpshare = CL_indshr_All{t};
    tmpddelta = CL_ddelta_dtheta2{t}; % J by nK

    df_dtheta2B_t = zeros(1, nincgrp*nK);
    for j = 1:nincgrp  % nincgrp income brackets
        % Individual income-group residual multiplied by individual shares
        tmpincgrp_t = (incgrpSm(t,:) == j); % 1 by ndraws
        tmpX = tmpshare .* (tmpincgrp_t - incshr(t,j));
     
        % tmpshare is J by ndraws
        % tmpincgrp_t is 1 by ndraws; it indicates whether each draw in
        % market t belongs to income group j
        % incshr(t,j) is a scalar

        % Off-diagonal terms of df/ddelta
        tmpXS = - tmpX * tmpshare';

        % Correct the diagonal terms of df/ddelta
        tmp = sum(tmpX,2);
        tmpXS = tmpXS + diag(tmp);

        % tmpXS is symmetric. sum(tmpXS,1) is 1 by J, tmpddelta is J by nK
        df_dtheta2B_t(1,j:nincgrp:end) = sum(tmpXS,1) * tmpddelta / tot_buy(t);
    end
    df_dtheta2B(t,:) = df_dtheta2B_t;
end

% df_dtheta2 = df/d_theta2 + df/d_delta * d_delta/d_theta2
df_dtheta2 = df_dtheta2A + df_dtheta2B; % nmk by nincgrp*nK


%% Step3: calculate dmom2
% dmom2 (nyr_micromom*nincmom by nK): Jacobian of mom2, each column refers to an element of theta2
% Rows are sorted by inc and yr
% 1st row: yr1, inc1; 2nd row: yr2, inc1; 3rd row: yr3, inc1, ...

% temp2 is ds/d_theta2 where s is the predicted market share
temp = df_dtheta2 .* repmat(salesMk,1,nincgrp*nK); % nmkt by nincgrp*nK
temp2 = zeros(nyear, nincgrp*nK);                  % nyear by nincgrp*nK

% Aggregate market-level derivatives to the annual national level
for t = 1:nyear
    aa = (t-1)*nmarket_per_year + 1;
    bb = t*nmarket_per_year;
    temp2(t,:) = sum(temp(aa:bb, :), 1) / salesAllMk(t);
end

% dmom2 is de/d_theta2 where e is (actual s - predicted s)
% So, de/d_theta2 = - ds/d_theta2
% Also, change the arrangement from (nyear by nincgrp*nK) to
% (nyr_micromom*nincmom by nK)
% Keep only the years and income groups covered by the micro moments
dmom2 = zeros(length(year_idx)*nincmom, nK);
row = 1;
for t = year_idx
    for j = 1:nincmom
        dmom2(row,:) = - temp2(t,j:nincgrp:end);
        row = row + 1;
    end
end

end
