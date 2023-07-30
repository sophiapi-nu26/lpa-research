function [G] = ER(N, p)
% ER
%
% Inputs:
% N = number of vertices
% p = probability that any pair of vertices is adjacent
%
% Outputs:
% G = Erdos-Renyi graph with N vertices connected with probability p

arguments
    N (1,1) {mustBeInteger, mustBePositive}
    p (1,1) {mustBeNonnegative}
end

% construct symmetric adjacency matrix
A = rand(N, N);
A = (A <= p); % now A is a logical matrix where each entry is 1 w/ probability p
A = tril(A, -1) + tril(A, -1)';

% generate graph
G = graph(A, 'omitselfloops');
