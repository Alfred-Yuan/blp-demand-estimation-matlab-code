function VarMat = F7c_variance(Cst, XMat, MicroMom, FitMM, gMat)
% This function calculates the Variance-Covariance Matrix of theta
% estimates

% Obtain constants
nprods = Cst.nprods;
nmkt = Cst.nmkt;
index_begin = Cst.index_begin;
index_end = Cst.index_end;

gmmresid = FitMM.gmmresid;
incshr_AllMk = FitMM.incshr;
IVall = XMat.IVall;
Xfull = XMat.Xfull;

invWt = MicroMom.invWt;
invWinc = MicroMom.invWinc;

% Calculate the variance of macro moments
shat = (gmmresid * ones(1,size(IVall,2))) .* IVall;

% Cluster the moment contributions by market-year (location-year)
shat_cluster = zeros(nmkt,size(IVall,2));
for m = 1:nmkt
    shat_cluster(m,:) = sum(shat(index_begin(m):index_end(m),:),1);
end
S_macro = shat_cluster'*shat_cluster/nprods;

% Calculate the variance of micro moments at the final estimates
IncShr = incshr_AllMk(MicroMom.year_idx, 1:Cst.nincmom);
S_micro = F5b_cov_micro(IncShr, MicroMom.IncSampleSize);

% Jacobian of delta with respect to theta2
ddelta_dtheta2_dm = demean(Cst.brand_model, gMat.ddelta_dtheta2);

% Jacobian of micro moments with respect to theta2
dmom2 = gMat.dmom2;

% Full Jacobian for parameters ordered as [theta2; theta1]
D_macro = [IVall' * ddelta_dtheta2_dm, -IVall' * Xfull];
D_micro = [dmom2, zeros(size(dmom2,1), size(Xfull,2))];

% Joint variance-covariance matrix of [theta2; theta1]
A = D_macro' * invWt * D_macro / nprods + D_micro' * invWinc * D_micro;

B = D_macro' * invWt * S_macro * invWt * D_macro / nprods + ...
    D_micro' * invWinc * S_micro * invWinc * D_micro;

% Variance of theta
VarMat = A \ B / A';

% Remove numerical asymmetry
VarMat = (VarMat + VarMat') / 2;

end
