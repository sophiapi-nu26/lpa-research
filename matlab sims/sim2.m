[N_arr, v_arr] = meshgrid(900:1000, -1:0.01:-0.1);

% count number of connected components in graph

comps = arrayfun(@countComps, N_arr, v_arr);
figure;
surf(N_arr, v_arr, comps)
xlabel('N')
ylabel('v (p = N^v)')
zlabel('number of connected components')

% count number of vertices in largest connected component in graph

maxSize = arrayfun(@findMaxCompSize, N_arr, v_arr);
figure;
surf(N_arr, v_arr, maxSize)
xlabel('N')
ylabel('v (p = N^v)')
zlabel('number of vertices in largest connected component')

