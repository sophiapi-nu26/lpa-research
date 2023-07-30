function [X,S,I,R,D] = sirs(G,x,dfinal,beta,gamma,delta,xi)
% SIRS
%
% Inputs:
% G = simple graph object representing the contact network
% x = vector of length 𝑁 whose elements are the initial states of each node in G
% dfinal = final day of the simulation (the initial day will be day 1)
% beta = model parameter for transition from 0 (susceptible) to 1 (infected)
% gamma = model parameter for transition from 1 (infected) to 2 (recovered)
% delta = model parameter for transition from 1 (infected) to 3 (deceased)
% xi = model parameter for transition from 2 (recovered) to 0 (susceptible)
%
% Outputs:
% X = N by dfinal matrix with the state history of the simulation
% S = vector of length dfinal containing the aggregate number of susceptible people for each day
% I = vector of length dfinal containing the aggregate number of infected people for each day
% R = vector of length dfinal containing the aggregate number of recovered people for each day
% D = vector of length dfinal containing the aggregate number of deceased people for each day

% Honors EA1
% Homework Program 6
%
% Name: Pi, Sophia
% Date: 11/03/2022

arguments
    G (1, 1)
    x (:, 1) {mustBeInteger}
    dfinal (1, 1) {mustBeInteger}
    beta (1, 1) {mustBeNumeric, mustBeNonnegative}
    gamma (1, 1) {mustBeNumeric, mustBeInRange(gamma, 0, 1)}
    delta (1, 1) {mustBeNumeric, mustBeInRange(delta, 0, 1)}
    xi (1, 1) {mustBeNumeric, mustBeInRange(xi, 0, 1)}
end

N = numnodes(G);
X = zeros(N, dfinal, 'uint8');
X(:,1) = x;

figure;
p = plot(G,'Layout','force','NodeLabel',[], ...
'MarkerSize',5,'NodeCData',x);
title(sprintf('Day %3u',1));

map = [0 0 1;
       1 0 0;
       0 1 0;
       0 0 0];
colormap(map);
clim([0 3]);
clim manual;

A = adjacency(G); % set adjacency matrix of G
proxy_X = repmat(X(:,1), 1, N);
infected_neighbors = full(sum(A .* (proxy_X == 1)));

% this was for debugging - please ignore
% testinfect = sum(X == 1);
% fprintf('infected on day %d: %d \n', 1, testinfect(1))

for day = 2:dfinal

    % brute force
    for n = 1:N
        r = rand();
        if X(n, day-1) == 0
            X(n, day) = (r <= sig(beta * infected_neighbors(n)));
        elseif X(n, day-1) == 1
            X(n, day) = 1 + (r <= gamma + delta) + (r <= delta);
        elseif X(n, day-1) == 2
            X(n, day) = 2 - 2 * (r <= xi);
        else
            X(n, day) = 3;
        end
    end
% this was for debugging - please ignore
%     testinfect = sum(X == 1);
%     fprintf('infected on day %d: %d \n', day, testinfect(day))

    proxy_X = repmat(X(:,day), 1, N);
    infected_neighbors = full(sum(A .* (proxy_X == 1)));

    % update graph with new node colors
    % (I comment out this block to save time when I want to generate graphs)
    figure(1);
    p.NodeCData = X(:,day);
    title(sprintf('Day %3u',day));
    drawnow;

    % if no one is infected, reinfect one alive person at random
    if sum(X(:, day) == 1) == 0
        aliveIndices = find(X(:, day) < 3);
        X(aliveIndices(randi(size(aliveIndices, 1))), day) = 1;
    end

end

% calculate aggregate output vectors S, I, R, and D
S = sum(X == 0, 1);
I = sum(X == 1, 1);
R = sum(X == 2, 1);
D = sum(X == 3, 1);

% set x-axis
xa = 1:dfinal;

% plot aggregate SIRD graph in new figure window
figure(2);
plot(xa, S, '-b', xa, I, '-r', xa, R, '-g', xa, D, '-k', 'LineWidth', 2);
title('Aggregate states versus time for the SIRS model');
xlabel('day');
ylabel('number of nodes');
legend({'susceptible', 'infected', 'recovered', 'deceased'});

end

% sigmoid function
function [s] = sig(num)
    s = num / (1 + abs(num));
end


% ANSWERS TO QUESTIONS
%
% 1. Using the model parameters beta = 0.1, gamma = 0.2, delta = 0.01, and 
% xi = 0, with 2000 nodes and an average degree of 20, the peak daily
% infection rate for each of the 4 different types of graphs were:
% G1: about 1100
% G2: between 300 and 400
% G3: between 500 and 600
% G4: about 1100
% Hence, it seems like graph G2 had the lowest peak daily infection rate
%
% 2. Using the model parameters beta = 0.1, gamma = 0.2, delta = 0.01, and 
% xi = 0.01, with 2000 nodes and an average degree of 20, over 1000 days it
% seemed that in general, graphs G2 and G3 showed more likelihood of
% generating spotaneous spikes in the infection rate. Graphs G1 and G4 also
% had small spikes, but they were shorter and less pronounced as the spikes
% for G2 and G3.