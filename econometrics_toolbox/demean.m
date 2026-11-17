function f = demean(id, X)

% Demean each column of the matrix, X, according to the index, id,
% Subtract the mean of values having the same id

% Demean each group separately; X_agg is a cell array of group blocks.
[~, X_agg] = aggregate(id, X, @(x) bsxfun(@minus, x, mean(x,1)));

% Collect the original row indices for each group.
[~, srt] = aggregate(id, (1:length(id))');

% isrt maps the grouped rows back to their original row order.
[~, isrt] = sort(cat(1, srt{:}));

% Stack the demeaned group blocks and then restore the original order.
X_agg = cat(1, X_agg{:});

f = X_agg(isrt, :);