function halt = halton_draws(ndraws, ndim, nmkt, seed, nskip, scramble)
% Generate market-specific Halton draws
%
% Output halt is nmkt by ndraws by ndim and contains uniform draws.
% Each market receives a different, non-overlapping block of Halton draws.

if nargin < 4 || isempty(seed)
    seed = 12345;
end

if nargin < 5 || isempty(nskip)
    nskip = 100;
end

if nargin < 6 || isempty(scramble)
    scramble = 1;
end

% Fix the scrambling across runs
rng(seed,'twister');

% Generate all prime numbers less than or equal to 100
prime_list = primes(100);

% Keep only the first ndim prime numbers
prime_list = prime_list(1:ndim);

% Initialize the matrix that will store all Halton draws
halt = zeros(nmkt, ndraws, ndim);

for m = 1:nmkt

    % First Halton sequence index used by the current market
    halton_start = nskip + (m-1)*ndraws;

    % Halton sequence indices assigned to the current market
    halton_index = (halton_start:halton_start+ndraws-1)';

    % Generate the Halton draws separately for each dimension
    for d = 1:ndim

        % Prime-number base used for the current dimension
        base = prime_list(d);

        % Weight placed on the first digit after the decimal point
        factor = 1/base;

        % Make a copy of the Halton indices that can be successively divided
        % by the current base to extract their base-specific digits
        index_d = halton_index;

        % Continue until factor becomes too small to affect a floating-point number
        while 1-factor < 1

            % Extract the rightmost digit of each index in the current base
            remainder = mod(index_d, base);

            % Remove the rightmost digit that was just extracted
            index_d = floor(index_d/base);

            if scramble == 1
                % Randomly permute the possible digits 0, ..., base-1
                digit_perm = randperm(base)-1;

                % Replace each original digit with its randomly permuted digit
                % Add 1 because MATLAB array indices start from 1
                remainder = digit_perm(remainder+1);
            end

            % Convert the scrambled digits into a column vector
            remainder = remainder(:);

            % Add the contribution of the current digit to the Halton draw
            % factor determines the digit's position after the decimal point
            halt(m,:,d) = halt(m,:,d) + (factor*remainder)';

            % Move to the next digit after the decimal point
            factor = factor/base;
        end
    end
end

end
