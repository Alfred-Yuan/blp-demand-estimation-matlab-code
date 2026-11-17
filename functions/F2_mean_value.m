function CL_mvalold = F2_mean_value(mu_ij, Cst, YMat)
% This function computes the mean utility level delta implied by each theta2.
% It uses parallel processing and implements contraction mapping market by market.
% Note: mvalold, mval, s_jt, mu are sliced by market for parfor
% Note: YMat.CL_mvalold stores mvalold from the previous iteration

% Output
% 1) CL_mvalold: the mean utility level implied by theta2
% 2) mu_All: the non-linear part of the utility, mu_ij
% 3) price_coef: the value multiplied in front of price
% 4) P_ij: - price_coef * price

% Obtain constants
nmkt = Cst.nmkt;
ndraws = Cst.ndraws;
maxiter = Cst.maxiter;
mvaltol = Cst.mvaltol;
if isfield(Cst, 'sqmaxiter')
    sqmaxiter = Cst.sqmaxiter;
else
    sqmaxiter = 60; % The max iter for squarem method, if not converge, back to original method
end
min_share = 1e-200;

index_begin = Cst.index_begin;
index_end = Cst.index_end;

% Sales
CL_sjt = YMat.CL_sjt;

% Previous mean value
CL_mvalold = YMat.CL_mvalold;

% Slice mu_ij market by market
CL_mu = cell(nmkt,1);
for t = 1:nmkt
    CL_mu{t} = mu_ij(index_begin(t):index_end(t),:);
end

% mval inversion

parfor t = 1:nmkt
    mvalold = CL_mvalold{t};
    s_jt = CL_sjt{t};
    mu = CL_mu{t};
    log_s_jt = log(s_jt);

    sqiter = 0;
    converged_sq = false;
    sq_step_min = 1.0;
    sq_step_max = 1.0;
    sq_step_factor = 4.0;

    % SQUAREM accelerates the fixed-point map delta -> G(delta).
    % Each cycle evaluates the BLP map twice to estimate an extrapolation
    % direction, then evaluates once more to check whether the extrapolated
    % point improves the fixed-point residual.
    while sqiter < sqmaxiter

        % First fixed-point map
        indshr = F4a_IndShr(ndraws, mvalold, mu);
        mktshr = max(mean(indshr,2), min_share);
        G0 = mvalold + (log_s_jt - log(mktshr));
        r0 = G0 - mvalold;
        sqiter = sqiter + 1;

        avgnorm = max(abs(r0));
        if avgnorm < mvaltol
            mvalold = G0;
            converged_sq = true;
            break
        end

        if sqiter >= sqmaxiter
            mvalold = G0;
            break
        end

        % Second fixed-point map
        indshr = F4a_IndShr(ndraws, G0, mu);
        mktshr = max(mean(indshr,2), min_share);
        G1 = G0 + (log_s_jt - log(mktshr));
        g1 = G1 - G0;
        sqiter = sqiter + 1;

        avgnorm = max(abs(g1));
        if avgnorm < mvaltol
            mvalold = G1;
            converged_sq = true;
            break
        end

        if sqiter >= sqmaxiter
            mvalold = G1;
            break
        end

        % SQUAREM acceleration with safeguard
        % r is the first map displacement. v approximates the change in
        % displacement between two consecutive fixed-point maps.
        r = r0;
        v = g1 - r0;
        rr = sum(r.^2);
        vv = sum(v.^2);

        % alpha controls how far to extrapolate. Bounding alpha keeps the
        % acceleration step from becoming too aggressive.
        if ~isfinite(rr) || ~isfinite(vv) || vv <= 0
            alpha = -1.0;
        else
            alpha = -sqrt(rr / vv);
        end
        alpha = -max(sq_step_min, min(sq_step_max, -alpha));
        if -alpha == sq_step_max
            sq_step_max = sq_step_max * sq_step_factor;
        end

        % Try the extrapolated delta and apply one BLP map from that point.
        mval_try = mvalold - 2 * alpha * r + alpha^2 * v;
        indshr = F4a_IndShr(ndraws, mval_try, mu);
        mktshr = max(mean(indshr,2), min_share);
        G_try = mval_try + (log_s_jt - log(mktshr));
        r_try = G_try - mval_try;
        sqiter = sqiter + 1;

        % Accept the extrapolated update only if it improves the residual;
        % otherwise fall back to the second ordinary fixed-point update.
        try_norm = max(abs(r_try));
        if all(isfinite(r_try)) && try_norm <= avgnorm
            mvalold = G_try;
            avgnorm = try_norm;
            if avgnorm < mvaltol
                converged_sq = true;
                break
            end
        else
            mvalold = G1;
        end
    end

    % If acceleration stalls, use the plain BLP contraction
    i = 0;
    if ~converged_sq
        avgnorm = 1;

        while i < maxiter && avgnorm > mvaltol

            i = i + 1;

            indshr = F4a_IndShr(ndraws, mvalold, mu);
            mktshr = max(mean(indshr,2), min_share);

            diff = log_s_jt - log(mktshr);
            avgnorm = max(abs(diff));

            if i < 20
                % For the first few iternations, reduce the step size
                mvalold = mvalold + diff*0.5;
            else
                mvalold = mvalold + diff;
            end
        end
    end

    %{
    % Display a message when max num of iteration is exceeded
    if i >= maxiter && avgnorm > mvaltol
        disp(['contraction mapping over limit in Mk ' num2str(t) ' and avgnorm*10^6']);
        num2str(avgnorm*1000000)
    end
    %}
    
    CL_mvalold{t} = mvalold;
end

end
