function gMat = F7b_jacobian(Cst, XMat, YMat, MuMat, MicroMom, FitMM, InMat)
% Calculate the Jacobian matrix (or gradient of GMM objective function)
% gMat has fields g, dmom1, dmom2

% Obtain constants
nprods = Cst.nprods;
IVall = XMat.IVall;

invWt = MicroMom.invWt;
invWinc = MicroMom.invWinc;

mom1 = FitMM.mom1;
mom2 = FitMM.mom2;

indshr_All = InMat.indshr_All;
P_ij = InMat.P_ij;

% Jacobian for the macro moments
[dxi_dtheta2, ddelta_dtheta2, ds_dtheta2Mat] = F6a_macro_grad(Cst, XMat, MuMat, indshr_All, P_ij);

% Jacobian for the first set of micro moments
InMat.ddelta_dtheta2 = ddelta_dtheta2;
InMat.ds_dtheta2Mat = ds_dtheta2Mat;
dmom2 = F6b_micro_grad(Cst, YMat, MuMat, ds_dtheta2Mat, ddelta_dtheta2, indshr_All, InMat);

tmp1 = 2 * dxi_dtheta2' * IVall * invWt * mom1 / nprods;
tmp2 = 2 * dmom2' * invWinc * mom2;
gMat.g = tmp1 + tmp2;

gMat.dmom1 = IVall' * dxi_dtheta2;
gMat.dmom2 = dmom2;
gMat.ddelta_dtheta2 = ddelta_dtheta2;