function [G1,G2,G3,G4] = random_graphs(N,mu)
% RANDOM_GRAPHS  Generate random simple graphs using four methods.
%
% function [G1,G2,G3,G4] = random_graphs(N,mu)
%
% Inputs:
%   N  = the number of nodes in the generated graphs
%   mu = the desired average degree
%
% Outputs:
%   G1 = Erdős-Rényi random graph
%   G2 = random geometric graph on the unit disk
%   G3 = Watts-Strogatz random graph
%   G4 = random graph using preferential attachment (Barabási-Albert)
%

arguments
    N  (1,1) {mustBeInteger,mustBeGreaterThan(N,1)}
    mu (1,1) {mustBeInteger,mustBePositive,mustBeLessThan(mu,N)}
end

% Method 1: Erdős-Rényi
p = mu/(N-1);
A = rand(N) <= p;
G1 = graph(A,'lower','OmitSelfLoops');

% Method 2: geometric on unit disk
% https://mathworld.wolfram.com/DiskLinePicking.html
fa = @(s)((4/pi)*s.*acos(s/2)-(2/pi)*(s.^2).*sqrt(1-(s/2).^2));
fb = @(r)((N-1)*integral(fa,0,r)-mu);
R2 = fzero(fb,[0 2]);
r = sqrt(rand(1,N));
theta = 2*pi*rand(1,N);
D = r.^2 + r'.^2 - 2*r.*r'.*cos(theta-theta');
A = D <= R2^2;
G2 = graph(A,'OmitSelfLoops');

% Method 3: Watts-Strogatz
nu = ceil(mu/2);
G3 = graph(logical(sparse(N,N)));
for k = 1:N
    G3 = addedge(G3,k*ones(1,nu),mod(k:k-1+nu,N)+1);
end
% rewire
beta = 0.01;
for k = 1:N
    for t = mod(k:k-1+nu,N)+1
        if rand <= beta
            n = randi(N);
            while k == n || findedge(G3,k,n)
                n = randi(N);
            end
            G3 = rmedge(G3,k,t);
            G3 = addedge(G3,k,n);
        end
    end
end

% Method 4: preferential attachment (Barabási-Albert)
m = round(N*mu/(2*(N-1)));
% create "seed" path graph of m+1 nodes
G4 = graph(1:m,2:m+1);
% preferential attachment, add nodes one by one
for node = m+2:N
    d = degree(G4);
    % targets = existing nodes to attach to (m of them)
    targets = datasample(1:length(d),m,'Replace',false,'Weights',d);
    G4 = addedge(G4,node,targets);
end

end

