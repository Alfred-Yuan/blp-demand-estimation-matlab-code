function indshr_All = F4b_IndShr_all(Cst, mval_All, mu_ij)
% This function computes the individual purchase probability for all products across entire markets.
% Authors: Panle Barwick, Shengmao Cao, Shanjun Li
% Date: Jan 14, 2017

% Numerator is the exponent of utilities
% Denominator is the sum of the numerators of all products in the market
% Use log-sum-exp to avoid overflow in exp(delta + mu)

% Read constants
ndraws = Cst.ndraws;
nmkt = Cst.nmkt;
index_begin = Cst.index_begin;
index_end = Cst.index_end;

indshr_All = zeros(size(mu_ij));

for t = 1:nmkt

    aa = index_begin(t);
    bb = index_end(t);

    mval = mval_All(aa:bb);
    mu = mu_ij(aa:bb,:);

    indshr_All(aa:bb,:) = F4a_IndShr(ndraws, mval, mu);
end

end
