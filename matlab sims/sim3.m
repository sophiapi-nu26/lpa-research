% when LPA converges (np > n^(⅝)), what is the distribution of the surviving label?
% plot over -3/8 < v < 0) for N = 1000

% one axis is v, the other axis is vertices 1 through N


% MinLPA:

% initialize the values
N = 50000;
v_arr = -1:0.015625:0;

% run LPA for each value of v
results = zeros(N, numel(v_arr));  % Preallocate a matrix to store the results
for i = 1:numel(v_arr)
    results(:, i) = findLabels(1, N, v_arr(i));
end

% plot the surface
figure;
surf(v_arr, 1:N, results);  % Plot the surface
xlabel('v (p = N^v)')
ylabel('vertex label')
zlabel('number of nodes with label')
title('Distribution of labels for N = %d with LPA(min, min)', N)


% RandLPA:

% initialize the values
N = 50000;
v_arr = -1:0.015625:0;

% run LPA for each value of v
results = zeros(N, numel(v_arr));  % Preallocate a matrix to store the results
for i = 1:numel(v_arr)
    results(:, i) = findLabels(2, N, v_arr(i));
end

% % AVERAGED PLOTS
% % run LPA for each value of v, N times (and take average)
% results = zeros(N, numel(v_arr));  % Preallocate a matrix to store the results
% for i = 1:numel(v_arr)
%     for j = 1:N
%         results(:, i) = results(:, i) + findLabels(2, N, v_arr(i));
%     end
%     results(:, i) = results(:, i) / N;
% end

% plot the surface
figure;
surf(v_arr, 1:N, results);  % Plot the surface
xlabel('v (p = N^v)')
ylabel('vertex label')
zlabel('number of nodes with label')
title('Distribution of labels for N = %d with LPA(rand, rand)', N)


% MinRandLPA (default, in paper):

% initialize the values
N = 50000;
v_arr = -1:0.015625:0;

% run LPA for each value of v
results = zeros(N, numel(v_arr));  % Preallocate a matrix to store the results
for i = 1:numel(v_arr)
    results(:, i) = findLabels(3, N, v_arr(i));
end

% plot the surface
figure;
surf(v_arr, 1:N, results);  % Plot the surface
xlabel('v (p = N^v)')
ylabel('vertex label')
zlabel('number of nodes with label')
title('Distribution of labels for N = %d with LPA(min, rand)', N)
