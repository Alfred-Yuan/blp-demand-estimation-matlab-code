function [mu_ij, P_ij, price_coef] = F3_mu_ij(theta2, Cst, XMat, MuMat)
% This function computes the non-linear part of the utility, mu_ij
% Input parameters:
% 1) theta2: non-linear paramters
% 2) p_real: price faced by consumers
% 3) chars: attributes with heteregeneous tastes
% 4) house_draws (hh taste for attributes and price), inc_draws (hh income)

% Output
% 1) mu: the non-linear part of the utility, mu_ij
% 2) price_coef: the value multiplied in front of price
% 3) P_ij: - price_coef * price

% ========================

% Step 0: constants and local dummies
nprods = Cst.nprods;
ndraws = Cst.ndraws;
nrand = Cst.nrand;
randchars = XMat.randchars;

% Obtain individual draws
house_draws = MuMat.hh; % nprods by ndraws by nrands+1
inc = MuMat.inc;
lninc = log(inc);

% Price
p_real = XMat.p_real;

% Initialize nonlinear utility
mu_ij = zeros(nprods,ndraws);

% Random tastes for the attributes
for i = 1:nrand
    mu_ij = mu_ij + (theta2(i)*randchars(:,i)*ones(1,ndraws)) .* house_draws(:,:,i);
end

% Matrix of alpha for all pseudo-households
alpha1 = theta2(nrand+1);

% Price coefficient
% IMPORTANT: no minus sign in front of exp(.)
% Specification 1: - exp(alpha1 - ln(income_i) + sigma_p*nu_ip) * price_j
% price_coef = exp(alpha1 - lninc + sigma_p*house_draws(:,:,nrand+1));
% P_ij = - price_coef .* (p_real*ones(1,ndraws));

% Specification 2 (ACTIVE): - exp(alpha1 - alpha2*ln(income_i) + sigma_p*nu_ip) * price_j
% theta2 = [sigma_1..sigma_nrand, alpha1, alpha2, sigma_p]
alpha2 = theta2(nrand+2);
sigma_p = theta2(nrand+3);
price_coef = exp(alpha1 - alpha2*lninc + sigma_p*house_draws(:,:,nrand+1));
P_ij = - price_coef .* (p_real*ones(1,ndraws));

% Specification 3: - [(alpha1 + sigma_p*nu_ip) / income_i] * price_j
%{
sigma_p = theta2(nrand+2);
price_coef = (alpha1 + sigma_p*house_draws(:,:,nrand+1)) ./ inc;
P_ij = - price_coef .* (p_real*ones(1,ndraws));
%}

% Specification 4: - (alpha1 / income_i + sigma_pi * nu_pi) * price_j
%{
price_coef = alpha1 ./ inc + sigma_p*house_draws(:,:,nrand+1);
P_ij = - price_coef .* (p_real*ones(1,ndraws));
%}

% Specification 5: - alpha1 / income_i * price_j (must include linear price)
%{
price_coef = alpha1 ./ inc;
P_ij = - price_coef .* (p_real*ones(1,ndraws));
%}


% Specification 6: - exp(alpha1 - alpha2*ln(income_i)) * price_j (no sigma_p)
% theta2 = [sigma_1..sigma_nrand, alpha1, alpha2]
%{
alpha2 = theta2(nrand+2);
price_coef = exp(alpha1 - alpha2*lninc);
P_ij = - price_coef .* (p_real*ones(1,ndraws));
%}

% Specification 7: - alpha1 / log(income_i) * price_j (must include linear price)
%{
lninc_safe = max(lninc, 0.1);
price_coef = alpha1 ./ lninc_safe;
P_ij = - price_coef .* (p_real*ones(1,ndraws));
%}

mu_ij = mu_ij + P_ij;

end
