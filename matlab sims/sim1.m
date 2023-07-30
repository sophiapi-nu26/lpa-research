N = 200;
v = -9/15;
p = N^v;
cap = 10;
LPAwait(N, v, cap);

% G = ER(N, p); % N*p is the desired average degree of the graph (aas?)
% F = LPA(G, N, 10);

[N_arr, v_arr] = meshgrid(900:10:1000, -1:0.01:-0.1);

% count number of iterations of LPA to converge

iters = arrayfun(@countIters, N_arr, v_arr); % cap = 100
figure;
surf(N_arr, v_arr, iters)
xlabel('N')
ylabel('v (p = N^v)')
zlabel('iterations to converge')

% count number of distinct labels left after 5 iterations of LPA

vals = arrayfun(@countVals, N_arr, v_arr); % cap = 5
figure;
surf(N_arr, v_arr, vals)
xlabel('N')
ylabel('v (p = N^v)')
zlabel('unique vals after 5 iterations')