function W = gmm_weight(moment_mat, cluster_id)
% Construct GMM weighting matrix with optional clustered moments
%
% moment_mat is an N by L matrix of moment contributions.
% If cluster_id is supplied, moment contributions are summed within clusters.

n = size(moment_mat,1);                                    % number of observations
l = size(moment_mat,2);                                    % number of moments

if nargin < 2 || isempty(cluster_id)  % Check whether cluster id exits                   
    S_hat = moment_mat' * moment_mat / n;
else

    % Input validation for clustered moments
    if n ~= length(cluster_id)
        error('Number of rows in moment_mat must match length of cluster_id');
    end

    clusters = unique(cluster_id);
    G = length(clusters);

    S_hat = zeros(l,l);
    for i = 1:G
        idx = (cluster_id == clusters(i));
        g_i = sum(moment_mat(idx,:),1);
        S_hat = S_hat + g_i' * g_i;
    end

    S_hat = S_hat / n;
end

S_hat = (S_hat + S_hat') / 2;

% Inverse to get efficient weighting matrix
rc = rcond(S_hat);
if rc < 1e-12
    warning('Clustered moment covariance is nearly singular.');
    W = pinv(S_hat);
else
    W = S_hat \ eye(size(S_hat));
end

end
