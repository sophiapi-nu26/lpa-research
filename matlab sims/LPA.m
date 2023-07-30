function [F, iteration, numEndLabels] = LPA(N, v, cap)
% LPA
%
% Inputs:
% N = number of vertices
% v = exponent of probability of being connected (p = n^v)
% cap = max number of iterations allowed before termination of algorithm
% 
% Outputs:
% F = N by cap matrix with the state history of the simulation
% iteration = Last completed iteration
% endLabels = vector containing the labels still present at the end

% construct graph
G = ER(N, N^v);

% initial states of vertices
X = 1:N;

% initialize state history matrix
F = zeros(N, cap);
F(:, 1) = X; % set first column equal to initial states

% initialize dummy vector to store labels of neighbors
nLabs = zeros(N, 1);

% initialize dummy vector to store majority labels
maj = zeros(N, 1);

iteration = 1;

% round 1 iteration
for i = 1:N % find majority labels...
    nLabs = F(neighbors(G, i), iteration); % get labels of all neighbors
    nLabs(end + 1) = F(i, iteration); % append label of self
    maj(i) = mode(nLabs); % get majority label (ties broken by choosing smallest label) 
end
F(:,iteration + 1) = maj; % update labels
iteration = iteration + 1; % update iteration number


% while we have not reached the cap...
while iteration < cap
    for i = 1:N % find majority labels...
        nLabs = F(neighbors(G, i), iteration); % get labels of all neighbors
        nLabs(end + 1) = F(i, iteration); % append label of self
        maj(i) = randMode(nLabs); % get majority label (ties broken by choosing a label uniformly at random) 
    end
    F(:,iteration + 1) = maj; % update labels
    iteration = iteration + 1; % update iteration number
    % if there is no change from the previous iteration, break
    if F(:, iteration) - F(:, iteration - 1) == zeros(N, 1)
        break
    end
end

numEndLabels = numel(unique(F(:, iteration)));

% helper function to randomly select mode from vector (tiebreaking)
function [m] = randMode(vector)
%     uniqueValues = unique(vector); % Get the unique values in the vector
%     counts = histcounts(vector); % Calculate the frequencies
%     counts = counts(counts > 0); % Remove the 0s
%     maxCount = max(counts); % Find the maximum frequency
%     modes = uniqueValues(counts == maxCount); % Get the values with maximum frequency
%     m = randsample(modes, 1); % Randomly select one mode from the set
    % reshape vector to ensure that it is a column vector
    vector = reshape(vector, numel(vector), 1);
    % gets frequencies and corresponding labels
    [frequencies, labels] = groupcounts(vector);
    % get vector of mode candidates
    modes = labels(frequencies == max(frequencies));
    % duplicate modes just in case there is only one mode, because MATLAB
    % will default to using randsample incorrectly otherwise
    modes(numel(modes)+1 : 2*numel(modes)) = modes;
    % select a mode at random
    m = randsample(modes, 1);

end

end