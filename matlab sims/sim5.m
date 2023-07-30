% # of labels - # of connected components

[N_arr, v_arr] = meshgrid(900:10:1000, -1:0.01:0);

comps = arrayfun(@countComps, N_arr, v_arr);
iters = arrayfun(@countIters, N_arr, v_arr); % cap = 100
vals = arrayfun(@countVals, N_arr, v_arr); % cap = 5

iterCompDiff = iters - comps;
valCompDiff = vals - comps;

figure;
surf(N_arr, v_arr, iterCompDiff)
xlabel('N')
ylabel('v (p = N^v)')
zlabel('number of iterations to converge (cap 100) - connected components')

figure;
surf(N_arr, v_arr, valCompDiff)
xlabel('N')
ylabel('v (p = N^v)')
zlabel('unique vals - connected components after 5 iterations')