function [ElasMat, dQdP_All, dTSdP_All] = F8b_elasticity(Cst, XMat, YMat, InMat)
% Calculate own and cross price elasticities
% National elasticity: sales-weighted average of market-level elasticities

% Obtain constants
nyear = Cst.nyear;
nmarket_per_year = Cst.nmarket_per_year;

index_begin = Cst.index_begin;
index_end = Cst.index_end;

model_id = Cst.model_id;
[model_ids, ~, model_pos] = unique(model_id);
nmod = length(model_ids);

mkt_size = Cst.mkt_size;
sales = Cst.sales;

subsidy = Cst.centralS_real + Cst.localS_real;

p_real = XMat.p_real;
s_jt = YMat.s_jt;

% Calculate dSdP matrix: dShares_j/dP_k mk by mk, stacked together
dSdPMat = F8a_dSdP(Cst, InMat);
dSdP = dSdPMat.dSdP;

% Not all models are sold in all markets.
% We use model_id to sum dSdP properly across markets
Elas_All = zeros(nyear*nmod, nmod); % stacked by yr, model
OwnElas = zeros(nyear*nmod, 1);

for t = 1:nyear
    Elas = zeros(nmod);
    TotalSales = zeros(nmod,1);

    aa_mkt = (t-1)*nmarket_per_year + 1;
    bb_mkt = t*nmarket_per_year;

    for m = aa_mkt:bb_mkt
        aa = index_begin(m);
        bb = index_end(m);
        tid = model_pos(aa:bb);
        TotalSales(tid) = TotalSales(tid) + sales(aa:bb);
    end

    for m = aa_mkt:bb_mkt
        aa = index_begin(m);
        bb = index_end(m);
        tnobs = bb-aa+1;
        tid = model_pos(aa:bb);

        % delta s/ delta p
        tmpdSdP = dSdP(aa:bb,1:tnobs);
        % p / s
        tmpP_S = (ones(tnobs,1)*(p_real(aa:bb))')./s_jt(aa:bb);
        % Sales weight
        mktsalesshr = repmat(sales(aa:bb)./TotalSales(tid),[1,tnobs]);
        % Sales weighted average elasticity 
        Elas(tid,tid) = Elas(tid,tid) + tmpdSdP.*tmpP_S.*mktsalesshr;
    end

    Elas_All((t-1)*nmod+1:t*nmod,:) = Elas;
    OwnElas((t-1)*nmod+1:t*nmod) = diag(Elas);
end

ElasMat.Elas = Elas_All;
ElasMat.OwnElas = OwnElas;
ElasMat.model_ids = model_ids;

dQdP_All = zeros(nyear*nmod,nmod);
dTSdP_All = zeros(nyear*nmod,nmod);

for t = 1:nyear
    dQdP = zeros(nmod);
    dTSdP = zeros(nmod); % TS: total subsidy (central+loc) of prod j in market m

    aa_mkt = (t-1)*nmarket_per_year + 1;
    bb_mkt = t*nmarket_per_year;

    for m = aa_mkt:bb_mkt
        aa = index_begin(m);
        bb = index_end(m);
        tnobs = bb-aa+1;
        tid = model_pos(aa:bb);

        tmpdSdP = dSdP(aa:bb,1:tnobs);  % tmpdSdP: jth row is derivative of S_mj w.r.t all prices: dS_{mj}/dP.
        tmk = mkt_size(aa);
        tmpsubsidy = subsidy(aa:bb);
        dQdP(tid,tid) = dQdP(tid,tid) + tmpdSdP*tmk;
        dTSdP(tid,tid) = dTSdP(tid,tid) + tmpdSdP*tmk .* (tmpsubsidy*ones(1,tnobs));
    end

    dQdP_All((t-1)*nmod+1:t*nmod,:) = dQdP;
    dTSdP_All((t-1)*nmod+1:t*nmod,:) = dTSdP;
end

end
