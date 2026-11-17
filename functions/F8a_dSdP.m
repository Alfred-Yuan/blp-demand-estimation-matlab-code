function dSdPMat = F8a_dSdP(Cst, InMat)
% Calculate own and cross price elasticities market by market

% Obtain constants
nprods = Cst.nprods;
nmkt = Cst.nmkt;
ndraws = Cst.ndraws;
index_begin = Cst.index_begin;
index_end = Cst.index_end;

% Inputs for elasticity calculation
price_coef = InMat.price_coef;
indshr_All = InMat.indshr_All;


%% Calculating own and cross price elasticity

% alpha_{ji} matrix
AlphaMat = - price_coef;

% Calculate Price Derivatives
dSdP = zeros(nprods, max(Cst.nprod_vec));
OwndSdP = zeros(nprods,1);
for t = 1:nmkt
    aa = index_begin(t);
    bb = index_end(t);
	tnobs = bb-aa+1;
    
    % Individual choice prob: K by Ndraws
    tmpshare = indshr_All(aa:bb, :);

    % Step 1 Create matrix S_m .* Alpha_m
    tmpX = (tmpshare .* AlphaMat(aa:bb,:))';

    % Step 2): create -SX
    tmpSX = -tmpshare * tmpX;

    % Step 3): correct the diagonal terms
    tmp = sum(tmpX);
    tmpdSdP = (tmpSX + diag(tmp))/ndraws;
    
    % Store results
    dSdP(aa:bb,1:tnobs) = tmpdSdP;
    OwndSdP(aa:bb) = diag(tmpdSdP);
    
end

dSdPMat.dSdP = dSdP; 
dSdPMat.OwndSdP = OwndSdP;

