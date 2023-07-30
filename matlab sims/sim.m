function [F] = LPA(G, N, cap)
% LPA
%
% Inputs:
% G = simple graph object representing the contact network
% cap = max number of iterations allowed before termination of algorithm
%
% Outputs:
% F = N by cap matrix with the state history of the simulation

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

figure;
p = plot(G,'Layout','force','NodeLabel',[], ...
'MarkerSize',5,'NodeCData',X);
title(sprintf('Iteration %3u', iteration));

% round 1 iteration
for i = 1:N % find majority labels...
    nLabs = F(neighbors(G, i), iteration); % get labels of all neighbors
    nLabs(end + 1) = F(i, iteration); % append label of self
    maj(i) = mode(nLabs); % get majority label (ties broken by choosing smallest label) 
end
F(:,iteration + 1) = maj; % update labels
iteration = iteration + 1; % update iteration number


% while we have not reached the cap...
while iteration <= cap
    for i = 1:N % find majority labels...
        nLabs = F(neighbors(G, i), iteration); % get labels of all neighbors
        nLabs(end + 1) = F(i, iteration); % append label of self
        maj(i) = mode(nLabs); % get majority label (ties broken by choosing smallest label) 
    end
    F(:,iteration + 1) = maj; % update labels
    iteration = iteration + 1; % update iteration number
    % update graph with new node colors
    figure(1);
    p.NodeCData = F(:, iteration);
    title(sprintf('Iteration %3u', iteration));
    drawnow;
end

% % update graph with new node colors
% figure(1);
% p.NodeCData = X(:, iteration);
% title(sprintf('Iteration %3u', iteration));
% drawnow;