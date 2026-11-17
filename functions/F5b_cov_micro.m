function Winc = F5b_cov_micro(IncShr, IncSampleSize)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the variance-covariance matrix of the income-share micromoments
%
% Each micromoment is the fraction of vehicle purchasers in an income group
% in a given survey year. The final income group is omitted because the
% income-group shares sum to one.
%
% Since the micro values are income-group indicators in a purchaser sample,
% their covariance is multinomial:
%
%     Var(p_hat_t) = [diag(p_t) - p_t*p_t'] / N_t.
%
% Here, p_t is the model-predicted vector of purchaser income shares from
% F5a_IncShr and N_t is the number of purchaser observations in the survey.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Read # of years with micro moments and the number of income group
[nyr_micromom, nincmom] = size(IncShr);

% Annual surveys are treated as independent, so the covariance matrix is block diagonal
Winc = zeros(nyr_micromom*nincmom);

for t = 1:nyr_micromom

    % Model-predicted purchaser income shares in survey year t
    p_t = IncShr(t,:)';

    % Multinomial covariance of the included income-group shares
    Winc_t = (diag(p_t) - p_t*p_t') / IncSampleSize(t);

    aa = (t-1)*nincmom + 1;
    bb = t*nincmom;
    Winc(aa:bb, aa:bb) = Winc_t;
end

% Remove numerical asymmetry
Winc = (Winc + Winc') / 2;

end
