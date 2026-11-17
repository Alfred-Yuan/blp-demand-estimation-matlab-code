function [coef,se,t_stat,p_val,R2] = iv_regress(y,X,IV,cluster_id)
% IV regression with optional clustered standard errors
%   y  : dependent variable
%   X  : regressors in the structural equation
%   IV : instruments, including all exogenous regressors
%   cluster_id(if any): cluster level

n = length(y);                                             % number of observations
k = size(X,2);                                             % number of the regressors in the original model
df = n - k;                                                % degree of freedom

% Stage 1 of 2SLS: regress X on IV and obtain the fitted regressors.
% P denotes P_Z*X, not the projection matrix P_Z itself.
P = IV * ( (IV'*IV)\IV'*X );

% Stage 2 of 2SLS: regress y on the fitted regressors.
coef = P'*P\P'*y;

u_iv = y - X*coef;                                         % structural residuals

if nargin < 4 || isempty(cluster_id)

    s_hat = u_iv'*u_iv/df;                                 % residual variance
    var_hat = (s_hat * eye(k)) / (P'*P);                   % same as s_hat * inv(P'*P)
    se  = sqrt(diag(var_hat));                             % standard errors
    t_stat  = coef./se;                                    % t-statistics
    p_val = betainc(df./(df+(1.*t_stat.^2)),(df./2),(1./2)); % p-values

else

    clusters = unique(cluster_id);
    G = length(clusters);

    cluster_var_hat = zeros(k,k);
    for i = 1:G
        idx = (cluster_id == clusters(i));
        P_i = P(idx,:);
        u_i = u_iv(idx);
        cluster_var_hat = cluster_var_hat + (P_i'*u_i) * (u_i'*P_i);
    end

    if n > k && G > 1
        cluster_var_hat = cluster_var_hat * (G/(G-1)) * ((n-1)/(n-k));
    end

    var_hat = (P'*P) \ cluster_var_hat / (P'*P);
    var_hat = (var_hat + var_hat') / 2;

    se  = sqrt(diag(var_hat));                             % standard errors
    t_stat  = coef./se;                                    % t-statistics
    p_val = 2 * (1 - tcdf(abs(t_stat), G - 1));             % p-values
end

% Adjusted R-squared for the structural equation
hasIntercept = any(all(abs(X - 1) < 1e-12, 1));
SSE = sum((y - X*coef).^2);

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
