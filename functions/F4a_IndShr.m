function f = F4a_IndShr(ndraws, mval, mu_ij)
% This function computes the individual purchase probability in a specific market.
% nprod is the number of products in that market
% The function is made as lean as possible to reduce computation time.

% Numerator is the exponent of utilities
% Denominator is the sum of the numerators of all products in the market
% Use log-sum-exp to avoid overflow in exp(delta + mu)

util = mval + mu_ij; % This line uses implict expansion, if not work, use mval * ones(1, ndraws) + mu_ij;

max_log = max(util, [], 1);
max_log = max(max_log, 0);

Numerator = exp(util - max_log);
Denominator = exp(-max_log) + sum(Numerator, 1);

f = Numerator ./ Denominator;

end
