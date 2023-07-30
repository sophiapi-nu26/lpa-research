% what is the distribution of the surviving label(s) on the SBM?

% p = N^v_p
% q = N^v_q

% one axis is v_p from -1 to 0, the other axis is vertices 1 through N
% animates as v_q varies from -1 to 0



% initialize the values
N = 10000;
numcommunities = 2;
communities = randi([1, numcommunities], 1, N);
vp_arr = -1:0.0625:0;
vq_arr = -1:0.0625:0;


% MinRandLPA:

figure;

pause(10)

% for each value of vq...
for vq = vq_arr

    % run LPA for each value of vq
    results = zeros(N, numel(vp_arr));  % Preallocate a matrix to store the results
    for i = 1:numel(vp_arr)
        results(:, i) = findLabelsSBM(3, N, communities, vp_arr(i), vq);
    end
    
    % plot the surface
    figure(1);
    surf(vp_arr, 1:N, results);  % Plot the surface
    xlabel('vp (p = N^vp)')
    ylabel('vertex label')
    zlabel('number of nodes with label')
    zlim([0, N])
    title(sprintf('Distribution of labels for N = %d with LPA(min, rand), numcommunities = %d, vq = %f', N, numcommunities, vq))
    drawnow;
    %pause(1);

end





% % MinLPA:
% 
% figure;
% 
% pause(10)
% 
% % for each value of vq...
% for vq = vq_arr
% 
%     % run LPA for each value of vq
%     results = zeros(N, numel(vp_arr));  % Preallocate a matrix to store the results
%     for i = 1:numel(vp_arr)
%         results(:, i) = findLabelsSBM(1, N, communities, vp_arr(i), vq);
%     end
%     
%     % plot the surface
%     figure(1);
%     surf(vp_arr, 1:N, results);  % Plot the surface
%     xlabel('vp (p = N^vp)')
%     ylabel('vertex label')
%     zlabel('number of nodes with label')
%     zlim([0, 1000])
%     title(sprintf('Distribution of labels for N = %d with LPA(min, min), numcommunities = %d, vq = %f', N, numcommunities, vq))
%     drawnow;
%     %pause(1);
% 
% end




% % RandRandLPA:
% 
% figure;
% 
% pause(10)
% 
% % for each value of vq...
% for vq = vq_arr
% 
%     % run LPA for each value of vq
%     results = zeros(N, numel(vp_arr));  % Preallocate a matrix to store the results
%     for i = 1:numel(vp_arr)
%         results(:, i) = findLabelsSBM(2, N, communities, vp_arr(i), vq);
%     end
%     
%     % plot the surface
%     figure(1);
%     surf(vp_arr, 1:N, results);  % Plot the surface
%     xlabel('vp (p = N^vp)')
%     ylabel('vertex label')
%     zlabel('number of nodes with label')
%     zlim([0, 1000])
%     title(sprintf('Distribution of labels for N = %d with LPA(rand, rand), numcommunities = %d, vq = %f', N, numcommunities, vq))
%     drawnow;
%     %pause(1);
% 
% end