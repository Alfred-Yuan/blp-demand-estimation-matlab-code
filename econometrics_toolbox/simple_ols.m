function [beta,se,t_stat,p_val] = simple_ols(y,X)

beta = (X'*X)\(X'*y);                                      % coefficients
e = y - X*beta;                                            % residuals

n = size(X,1);                                             % number of observation
k = size(X,2);                                             % number of covariates
df = n - k;                                                % degree of freedom

sigma_hat = e'*e./df;
var_hat = sigma_hat * inv(X'*X);

se = sqrt(diag(var_hat));                                  % standard errors
t_stat = beta./se;                                         % t-statistics
p_val = betainc(df./(df+(1.*t_stat.^2)),(df./2),(1./2));   % p-values
 