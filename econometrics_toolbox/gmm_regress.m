function [theta,se,t_stat,p_val,R2] = gmm_regress(y,X,Z,W,cluster_id)
% GMM regression with optional clustered standard errors
%
% y: dependent variable
% X: regressors in the structural equation
% Z: instruments
% W: GMM weighting matrix

n = size(X,1);                                             % number of observations
k = size(X,2);                                             % number of regressors
l = size(Z,2);                                             % number of instruments
df = n - k;                                                % degree of freedom

ZtX = Z' * X;
Zty = Z' * y;

bread = ZtX' * W * ZtX;
theta = bread \ (ZtX' * W * Zty);                          % coefficients
u = y - X*theta;                                           % residuals

moment_mat = Z .* (u * ones(1,l));                          % moment conditions

if nargin < 5 || isempty(cluster_id)

    S_hat = moment_mat' * moment_mat;
    middle = ZtX' * W * S_hat * W * ZtX;
    var_hat = bread \ middle / bread;
    var_hat = (var_hat + var_hat') / 2;

    se = sqrt(diag(var_hat));                              % standard errors
    t_stat = theta./se;                                    % t-statistics
    p_val = betainc(df./(df+(1.*t_stat.^2)),(df./2),(1./2)); % p-values

else

    clusters = unique(cluster_id);
    G = length(clusters);

    S_cluster = zeros(l,l);
    for i = 1:G
        idx = (cluster_id == clusters(i));
        g_i = sum(moment_mat(idx,:),1);
        S_cluster = S_cluster + g_i' * g_i;
    end

    if n > k && G > 1
        S_cluster = S_cluster * (G/(G-1)) * ((n-1)/(n-k));
    end

    middle = ZtX' * W * S_cluster * W * ZtX;
    var_hat = bread \ middle / bread;
    var_hat = (var_hat + var_hat') / 2;

    se = sqrt(diag(var_hat));                              % standard errors
    t_stat = theta./se;                                    % t-statistics
    p_val = 2 * (1 - tcdf(abs(t_stat), G - 1));             % p-values
end

% Adjusted R-squared for the structural equation
hasIntercept = any(all(abs(X - 1) < 1e-12, 1));
SSE = sum(u.^2);

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
