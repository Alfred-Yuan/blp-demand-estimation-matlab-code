function [dxi_dtheta2, ddelta_dtheta2, ds_dtheta2Mat] = F6a_macro_grad(Cst, XMat, MuMat, indshr_All, P_ij)
%function [dxi_dtheta2, ddelta_dtheta2, ds_dtheta2Mat, ds_dtheta2Seg] = F6a_macro_grad(Cst, XMat, YMat, MuMat, indshr_All, P_ij)

% ds_dtheta2Mat: \sum_j ds_{ij}/d_theta2, nmk-by-ndraws-by-K
% ds_dtheta2Seg: \sum_j for j in seg1, ds_{ij}/d_theta2, nseg*nmk-by-ndraws-by-K
% These are necessary for gradients of micromoments

% Read constants

nmkt = Cst.nmkt;
ndraws = Cst.ndraws;
nprods = Cst.nprods;
nrand = Cst.nrand;
nK = Cst.nK;
index_begin = Cst.index_begin;
index_end = Cst.index_end;
nprod_vec = Cst.nprod_vec;
CL_randchars = XMat.CL_randchars;
CL_p_real = XMat.CL_p_real;

CL_inc = MuMat.CL_inc;
CL_house_draws = MuMat.CL_hh;

demean_var = Cst.brand_model;
Xfull = XMat.Xfull;
Projection = XMat.Projection;

%% Step 1: Compute ddelta_dtheta2
% Market-by-market derivatives stacked together.
% For example, the first J rows are a J-by-J marix for market 1, with the rest of columns filled by zeros.

CL_ddelta_dtheta2 = cell(nmkt,1);
CL_ds_dtheta2Mat = cell(nmkt,1);

% Slice the indshr_All and P_ij in advance
CL_indshr_All = cell(nmkt,1);
CL_P_ij = cell(nmkt,1);
for t = 1:nmkt
    CL_indshr_All{t} = indshr_All(index_begin(t):index_end(t),:);
    CL_P_ij{t} = P_ij(index_begin(t):index_end(t),:);
end

parfor m = 1:nmkt
    nprods_m = nprod_vec(m);
    tmpshare = CL_indshr_All{m};

    ds_dtheta2Mat_m = zeros(1, ndraws, nK);


    % 1.1. Compute ds_ddelta for market m

    % ds_j / ddelta_q when j =/= q
    ds_ddelta = - tmpshare * tmpshare'/ ndraws;

    % ds_j / ddelta_q when j == q
    ds_ddelta = ds_ddelta + diag(mean(tmpshare,2));


    % 1.2. Compute ds_dtheta2 for market m

    % Initialize ds_dtheta2 for market m
    ds_dtheta2 = zeros(nprods_m, nK);


    % 1.2.1. ds_j / dsigma_k
    for k = 1:nrand
        tmpX = CL_randchars{m}(:,k); % Here use implicit expansion to replace CL_randchars{m}(:,k) * ones(1,ndraws);

        % tmpdraws: nprod_m by ndraws, not a 3-dim array
        tmpdraws = CL_house_draws{m}(:,:,k);

        % tmpdtheta is ds_ij / dtheta2_k for products and draws in market m
        tmpdtheta = tmpshare.*tmpdraws.*( tmpX - sum(tmpX.*tmpshare,1) ); % Here also use implicit expansio

        ds_dtheta2(:,k) = mean(tmpdtheta,2);
        ds_dtheta2Mat_m(1,:,k) = sum(tmpdtheta,1);
    end


    % 1.2.2. ds_j / dalpha and dsigma_p

    %{
    % Specification 1: - exp(alpha1 - ln(income_i) + sigma_p*nu_ip) * price_j
    tmpP = CL_P_ij{m};
    tmpPSh = tmpP - ones(nprods_m,1) * sum(tmpP .* tmpshare, 1);

    % ds_j / da1
    tmpdtheta = tmpshare .* tmpPSh;
    ds_dtheta2(:,nrand+1) = mean(tmpdtheta,2);
    ds_dtheta2Mat_m(1,:,nrand+1) = sum(tmpdtheta,1);

    % ds_j / d_sigma_p
    tmpdraws = CL_house_draws{m}(:,:,nrand+1);
    tmpdtheta = tmpshare .* tmpdraws .* tmpPSh;
    ds_dtheta2(:,nK) = mean(tmpdtheta,2);
    ds_dtheta2Mat_m(1,:,nK) = sum(tmpdtheta,1);
    %}



    % Specification 2 (ACTIVE): - exp(alpha1 - alpha2*ln(income_i) + sigma_p*nu_ip) * price_j
    % theta2 = [sigma_1..sigma_nrand, alpha1, alpha2, sigma_p]
    tmpP = CL_P_ij{m};
    tmpPSh = tmpP - ones(nprods_m,1) * sum(tmpP .* tmpshare, 1);

    % ds_j / d_alpha1
    tmpdtheta = tmpshare .* tmpPSh;
    ds_dtheta2(:,nrand+1) = mean(tmpdtheta,2);
    ds_dtheta2Mat_m(1,:,nrand+1) = sum(tmpdtheta,1);

    % ds_j / d_alpha2
    tmpinc = log(CL_inc{m});
    tmpdtheta = - tmpshare .* tmpinc .* tmpPSh;
    ds_dtheta2(:,nrand+2) = mean(tmpdtheta,2);
    ds_dtheta2Mat_m(1,:,nrand+2) = sum(tmpdtheta,1);

    % ds_j / d_sigma_p
    tmpdraws = CL_house_draws{m}(:,:,nrand+1);
    tmpdtheta = tmpshare .* tmpdraws .* tmpPSh;
    ds_dtheta2(:,nrand+3) = mean(tmpdtheta,2);
    ds_dtheta2Mat_m(1,:,nrand+3) = sum(tmpdtheta,1);

    %{
    % Specification 3: - [(alpha1 + sigma_p*nu_ip) / income_i] * price_j
    tmpinc = CL_inc{m};
    % The two lines below also use implict expansion
    tmpP = - CL_p_real{m} ./ tmpinc;
    tmpPSh = tmpP - sum(tmpP .* tmpshare, 1);

    % ds_j / da1
    tmpdtheta = tmpshare .* tmpPSh;
    ds_dtheta2(:,nrand+1) = mean(tmpdtheta,2);
    ds_dtheta2Mat_m(1,:,nrand+1) = sum(tmpdtheta,1);

    % ds_j / d_sigma_p
    tmpdraws = CL_house_draws{m}(:,:,nrand+1);
    tmpdtheta = tmpshare .* tmpdraws .* tmpPSh;
    ds_dtheta2(:,nK) = mean(tmpdtheta,2);
    ds_dtheta2Mat_m(1,:,nK) = sum(tmpdtheta,1);
    %}


    %{
    % Specification 4: - (alpha1 / income_i + sigma_p * nu_ip) * price_j
    tmpinc = CL_inc{m};
    
    % ds_j / d_alpha1
    tmpP = - CL_p_real{m} ./ tmpinc;
    tmpPSh = tmpP - sum(tmpP .* tmpshare, 1);
    
    tmpdtheta = tmpshare .* tmpPSh;
    ds_dtheta2(:,nrand+1) = mean(tmpdtheta,2);
    ds_dtheta2Mat_m(1,:,nrand+1) = sum(tmpdtheta,1);
    
    % ds_j / d_sigma_p
    tmpdraws = CL_house_draws{m}(:,:,nrand+1);
    tmpP = - CL_p_real{m} .* tmpdraws;
    tmpPSh = tmpP - sum(tmpP .* tmpshare, 1);
    
    tmpdtheta = tmpshare .* tmpPSh;
    ds_dtheta2(:,nK) = mean(tmpdtheta,2);
    ds_dtheta2Mat_m(1,:,nK) = sum(tmpdtheta,1);
    %}

    %{
    % Specification 5: - alpha1 / income_i * price_j
    tmpinc = CL_inc{m};
    
    % ds_j / d_alpha1
    tmpP = - CL_p_real{m} ./ tmpinc;
    tmpPSh = tmpP - sum(tmpP .* tmpshare, 1);
    
    tmpdtheta = tmpshare .* tmpPSh;
    ds_dtheta2(:,nrand+1) = mean(tmpdtheta,2);
    ds_dtheta2Mat_m(1,:,nrand+1) = sum(tmpdtheta,1);
    %}

    %{
    % Specification 7: - alpha1 / log(income_i) * price_j
    tmpinc = max(log(CL_inc{m}), 0.1);

    % ds_j / d_alpha1
    tmpP = - CL_p_real{m} ./ tmpinc;
    tmpPSh = tmpP - sum(tmpP .* tmpshare, 1);

    tmpdtheta = tmpshare .* tmpPSh;
    ds_dtheta2(:,nrand+1) = mean(tmpdtheta,2);
    ds_dtheta2Mat_m(1,:,nrand+1) = sum(tmpdtheta,1);
    %}

    %{
    % Specification 6: - exp(alpha1 - alpha2*ln(income_i)) * price_j
    tmpP = CL_P_ij{m};
    tmpPSh = tmpP - ones(nprods_m,1) * sum(tmpP .* tmpshare, 1);

    % ds_j / d_alpha1
    tmpdtheta = tmpshare .* tmpPSh;
    ds_dtheta2(:,nrand+1) = mean(tmpdtheta,2);
    ds_dtheta2Mat_m(1,:,nrand+1) = sum(tmpdtheta,1);

    % ds_j / d_alpha2
    tmpinc = log(CL_inc{m});
    tmpdtheta = - tmpshare .* tmpinc .* tmpPSh;
    ds_dtheta2(:,nrand+2) = mean(tmpdtheta,2);
    ds_dtheta2Mat_m(1,:,nrand+2) = sum(tmpdtheta,1);
    %}

    % 1.3. Compute jacobian matrix ddelta/dtheta2 (implicit fcn theorem)
    if rcond(ds_ddelta) > 1e-12
        % this formula is much faster than - pinv(ds_ddelta) * ds_dtheta2
        ddelta_m = - ds_ddelta \ ds_dtheta2;
    else
        % this formula is more stable when ds_ddelta is irregular
        % warning('ds_ddelta is nearly singular.');
        ddelta_m = - pinv(ds_ddelta) * ds_dtheta2;
    end

    % Store the value for each market
    CL_ddelta_dtheta2{m} = ddelta_m;
    CL_ds_dtheta2Mat{m} = ds_dtheta2Mat_m;

end


% Initialize ddelta_dtheta2 and ds_dtheta2Mat
ddelta_dtheta2 = zeros(nprods, nK);
ds_dtheta2Mat = zeros(nmkt, ndraws, nK);
% Merge the vector of different markets together.
for m = 1:nmkt
    aa = index_begin(m);
    bb = index_end(m);

    ddelta_dtheta2(aa:bb,:) = CL_ddelta_dtheta2{m};
    ds_dtheta2Mat(m,:,:) = CL_ds_dtheta2Mat{m};
end


%% Step 2: Compute dxi_dtheta2 and ds_dtheta2Mat
% When we partial out theta1, theta1 becomes a function of theta2

% 2.1. Compute dxi_dtheta2

% Demean ddelta_dtheta2
ddelta_dtheta2_dm = demean(demean_var, ddelta_dtheta2);
dxi_dtheta2 = ddelta_dtheta2_dm - Xfull*(Projection*ddelta_dtheta2_dm);

end
