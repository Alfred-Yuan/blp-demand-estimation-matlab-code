function [beta,se,t_stat,p_val,R2] = ols(y,X,cluster_id)
% OLS regression with optional clustered standard errors
%
% If cluster_id is not supplied, this function is the same as simple_ols.m
% except that it also reports adjusted R-squared.

beta = (X'*X)\(X'*y);                                      % coefficients
e = y - X*beta;                                            % residuals

n = size(X,1);                                             % number of observation
k = size(X,2);                                             % number of covariates
df = n - k;                                                % degree of freedom

if nargin < 3 || isempty(cluster_id)

    sigma_hat = e'*e./df;
    var_hat = (sigma_hat * eye(k)) / (X'*X);               % Same as sigma_hat * inv(X'*X)

    se = sqrt(diag(var_hat));                              % standard errors
    t_stat = beta./se;                                     % t-statistics
    p_val = betainc(df./(df+(1.*t_stat.^2)),(df./2),(1./2)); % p-values

else

    clusters = unique(cluster_id);
    G = length(clusters);

    cluster_var_hat = zeros(k,k);
    for i = 1:G
        idx = (cluster_id == clusters(i));
        X_i = X(idx,:);
        e_i = e(idx);
        cluster_var_hat = cluster_var_hat + (X_i'*e_i) * (e_i'*X_i);
    end

    if n > k && G > 1
        cluster_var_hat = cluster_var_hat * (G/(G-1)) * ((n-1)/(n-k));
    end

    var_hat = (X'*X) \ cluster_var_hat / (X'*X);
    var_hat = (var_hat + var_hat') / 2;

    se = sqrt(diag(var_hat));                              % standard errors
    t_stat = beta./se;                                     % t-statistics
    p_val = 2 * (1 - tcdf(abs(t_stat), G - 1));             % p-values
end

% Adjusted R-squared
hasIntercept = any(all(abs(X - 1) < 1e-12, 1));
SSE = sum(e.^2);

if hasIntercept
    TSS = sum((y - mean(y)).^2);
    R2 = 1 - SSE / TSS;
    R2 = 1 - (1 - R2) * (n - 1) / (n - k);
else
    TSS = sum(y.^2);
    R2 = 1 - SSE / TSS;
    R2 = 1 - (1 - R2) * n / (n - k);
end

end
