% heatmap of unlogged maxNonzeroLabel
M = maxNonzeroLabelFiltered;

figure;
h = heatmap(M);

xlabel('a (where p = N^[(a-1)/16])');
ylabel('b (where q = N^[(b-1)/16]');
title('Max Nonzero Label Map for Convergent MinRandLPA on SBM(p,q) with N=10000');

% heatmap of logged maxNonzeroLabel
M = maxNonzeroLabelFilteredLogged;

figure;
h = heatmap(M);

xlabel('a (where p = N^[(a-1)/16])');
ylabel('b (where q = N^[(b-1)/16]');
title('log(m+1) Where m is Max Nonzero Label Map for Convergent MinRandLPA on SBM(p,q) with N=10000');