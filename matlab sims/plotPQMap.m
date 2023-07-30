% customColors = [1.0 1.0 0;0.0 1.0 0.0;1.0 0.0 0.0];
% 
% figure;
% img = imagesc(pqMap);
% colormap(customColors);
% colorbar('Ticks', 1:3, 'TickLabels', {'Converge to 1', 'Converge to 2', 'Did not converge'});
% xlabel('p');
% ylabel('q');
% title('State Map for MinRandLPA on SBM(p,q) with N=10000')
% 
% % scaledXLabels = ((1:17) - 1) / 16;
% % scaledYLabels = ((1:17) - 1) / 16;
% % xticklabels(sprintfc('%.2f', scaledXLabels));
% % yticklabels(sprintfc('%.2f', scaledYLabels));
% 
% img.YDisplayData = -1:1/16:0;
% 
% grid on;

M=pqMap;

% Step 1: Define the custom colormap
customColors = [
    1.0 1.0 0.0;  % Yellow (for value 1 in M)
    0.0 1.0 0.0;  % Green (for value 2 in M)
    1.0 0.0 0.0   % Red (for value 3 in M)
];

% Step 2: Plot the colored matrix with gridlines
figure;
imagesc(M);
colormap(customColors);

% Step 3: Add gridlines
grid on;

% Step 4: Scale the axes with labels
nRows = size(M, 1);
nCols = size(M, 2);

xticks(1:nCols);
yticks(1:nRows);

% Calculate scaled tick labels
scaledXLabels = ((1:nCols) - 1) / 16;
scaledYLabels = ((1:nRows) - 1) / 16;

xticklabels(sprintfc('%.2f', scaledXLabels));
yticklabels(sprintfc('%.2f', scaledYLabels));

% Step 5: Add labels for each value in the matrix
for i = 1:nRows
    for j = 1:nCols
        text(j, i, num2str(M(i, j)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    end
end

% Optional: Add colorbar and labels for the colormap
colorbar('Ticks', [1, 2, 3], 'TickLabels', {'Converged to 1', 'Converged to 2', 'Did not converge'});

% Optional: Add axis labels and title
xlabel('a (where p = N^a)');
ylabel('b (where q = N^b');
title('State Map for MinRandLPA on 2-community SBM(p,q) with N=10000');
