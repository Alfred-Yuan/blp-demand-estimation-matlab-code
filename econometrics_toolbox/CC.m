% Simple Adaptive Size-Exact Testing for Full-Vector and Subvector
% Inference in Moment Inequality Models
% Gregory Cox (National University of Singapore) and 
% Xiaoxia Shi (University of Wisconsin at Madison)
% May 17, 2021
% 
% This function implements the CC test. 

function [reject, value, rk, cv] = CC(moments,variance,n,alpha,A_ineq,b_ineq,A_eq,b_eq)
    %CC Implements the CC test
    %   moments is the sample moments
    %   variance is the estimated covariance matrix of the moments
    %   n is the sample size
    %   alpha is the nominal size
    %   A_ineq and b_ineq are the inequalities to be tested: Am<=b
    %   A_eq and b_eq are optional arguments for equalities to be tested 
    %   reject is one if the likelihood ratio statistic (a quadratic form in 
    %              the moments) exceeds a chi-squared critical value, where the
    %              degrees of freedom is the rank of the active inequalities
    %   value is the value of the likelihood ratio statistic
    %   rk is the rank of the active inequalities
    %   cv is the critical value used
    
    % the equality restrictions are optional
    if nargin==6
        A_eq=[];
        b_eq=[];
    end

    % if all inequalities are slack, the test does not reject and no active
    % inequalities
    if (sum(~(A_ineq*moments<b_ineq))==0) && nargin==6
        reject=0;
        value=0;
        rk=0;
        cv=0;
    else
        options = optimset('display','off');
        variance_inverse=inv(variance);
        variance_inverse = (variance_inverse+variance_inverse')/2;
        [x,fval,~,~,lambda] = quadprog(2*variance_inverse,...
            -variance_inverse*moments*2*sqrt(n),A_ineq,b_ineq,A_eq,b_eq,[],[],[],options);
        value = fval+n*moments'*variance_inverse*moments;
        active_eq = (lambda.eqlin~=0);
        active_ineq = A_ineq*x-b_ineq>=-10^(-5);
        A_active=[A_eq(active_eq,:);A_ineq(active_ineq,:)];
        rk=rank(A_active);
        cv=chi2inv(1-alpha,rk);
        reject = (value>cv);
    end
end




