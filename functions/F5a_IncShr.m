function [incshr_AllMk, InMat] = F5a_IncShr(indshr_All, Cst, YMat, MuMat)
% Use ACCUMARRAY instead of CUMSUM-DIFF (accumarray is more accurate when
% the matrix contains both very large and very small numbers)
% This function computes the model predicted share of consumers in each
% income group by year

% Input
% indshr_All: nprods by ndraws, individual purchase probability across entire markets.

% Output
% 1) incshr_AllMk: the model predicted share of consumers in each income group by year
% 2) InMat: input of Jacobian of mom2

% Read constants
nmkt = Cst.nmkt;
nyear = Cst.nyear;
nmarket_per_year = Cst.nmarket_per_year;
ndraws = Cst.ndraws;
mkt_index = Cst.mkt_index;    % the market that product j belongs to (nprods by 1)
nincgrp = Cst.nincgrp;        % number of income groups

salesMk = YMat.salesMk;       % total sales of all vehicles in each market
salesAllMk = YMat.salesAllMk; % total sales at the annual national level

incgrpSm = MuMat.incgrpSm;    % income bracket for each hh by mk (nmk by ndraws)

%% HH purchase probabilities

% xx is repmat(mkt_index, 1, ndraws); yy is repmat(1:ndraws, nprods, 1)
[xx, yy] = ndgrid(mkt_index, 1:ndraws);

% accumarray calculates purchase probability of inside products (purchasing a car) for each household by market
indshr_in = accumarray([xx(:) yy(:)], indshr_All(:), [nmkt ndraws]); % sum of the rows within a market

%% Micro moments, fraction of car buyers in each income bracket

% nmkt by 1: total number of cars bought by ndraws HH, by market
tot_buy = sum(indshr_in,2);

% nmkt by nincgrp: total number of cars bought by ndraws HH in each income bracket
tot_indbuy_inc = zeros(nmkt,nincgrp);
for j = 1:nincgrp
    temp = indshr_in .* (incgrpSm==j);
    tot_indbuy_inc(:,j) = sum(temp,2);
end

% nmkt by nincgrp: fraction of buyers in each income bracket, by market
incshr = tot_indbuy_inc./repmat(tot_buy,1,nincgrp); % estimated share

% incshr_AllMk: nyear by nincgrp, frc of national buyers in each income bracket, by yr
incshr_AllMk = zeros(nyear,nincgrp);
temp = incshr .* repmat(salesMk,1,nincgrp);

for t = 1:nyear
    aa = (t-1)*nmarket_per_year + 1;
    bb = t*nmarket_per_year;
    incshr_AllMk(t,:) = sum(temp(aa:bb, :), 1) / salesAllMk(t);
end

% Save incshr and tot_buy (input of Jacobian of mom2)
InMat.incshr = incshr;             % nmkt by nincgrp
InMat.tot_buy = tot_buy;           % nmkt by 1

end
