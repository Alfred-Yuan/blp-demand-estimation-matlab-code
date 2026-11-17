function f = dummify(var_to_dum)

% Generate dummy variables for var_to_dum

% Convert text categories to upper case before identifying unique values.
% This keeps labels that differ only by capitalization in the same category.
if iscell(var_to_dum) == 1  % cell array
    var_to_dum = upper(var_to_dum);
end

% Identify the unique categories after the text normalization above.
temp = unique(var_to_dum);

% Initialize one dummy-variable column for each unique category.
dum=zeros(length(var_to_dum), length(temp));

if iscell(var_to_dum) == 1  % cell array
    
    % Find observations that belong to text category i
    for i=1:length(temp)
        temp1=find(strcmp(var_to_dum, temp(i)));
        dum(temp1,i)=1;
    end
else

    % Find observations that belong to numeric category i
    for i=1:length(temp)
        temp1=find(var_to_dum==temp(i));
        dum(temp1,i)=1;
    end
end

% Return the complete dummy-variable matrix.
f = dum(:,1:length(temp));
