function [Xsub, idx] = indep_columns(X, tol)
% Extract a linearly independent set of columns of a given matrix X
%
% [Xsub,idx]=licols(X)
%
% in:
%
% X: The given input matrix
% tol: A rank estimation tolerance. Default = 1e-10
%
% out:
%
% Xsub: The extracted columns of X
% idx:  The indices (into X) of the extracted columns

if ~nnz(X)    % X has no non-zeros and hence no independent columns
    Xsub=[]; idx=[];
    return
end

if nargin<2
    tol=1e-10;
end

% Pivoted economy-size QR orders the most informative columns first.
% With the third output, E is the permutation vector for the columns of X.
[Q, R, E] = qr(X,0);

if ~isvector(R)
    diagr = abs(diag(R));
else
    % R can be a scalar/vector when X has only one row or one column.
    % Use its magnitude, consistently with the matrix case above.
    diagr = abs(R(1));
end

% Estimate numerical rank from the decreasing diagonal of pivoted R.
r = find(diagr >= tol*diagr(1), 1, 'last');
% Convert the pivot order back to the original column order.
idx = sort(E(1:r));
Xsub = X(:,idx);