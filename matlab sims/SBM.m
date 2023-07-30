function [G] = SBM(N, communities, p, q)
% SBM
%
% Inputs:
% N = number of vertices
% communities = vector of length N containing the community that each node
% is in
% p = probability that two vertices in the same community are adjacent
% q = probability that two vertices in different communities are adjacent
%
% Outputs:
% G = stochastic block model with parameters above

arguments
    N (1,1) {mustBeInteger, mustBePositive}
    communities {mustBeInteger, mustBePositive}
    p (1,1) {mustBeNonnegative}
    q (1,1) {mustBeNonnegative}
end

% construct adjacency matrix
A = zeros(N, N);
for i = 1:N
    for j = 1:N
        if communities(i) == communities(j)
            A(i, j) = rand() <= p;
        else
            A(i, j) = rand() <= q;
        end
    end
end

% make adjacency matrix symmetric
A = tril(A, -1) + tril(A, -1)';

% generate graph
G = graph(A, 'omitselfloops');
