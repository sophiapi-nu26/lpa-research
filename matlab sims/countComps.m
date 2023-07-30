function [numComps] = countComps(N, v)
    p = N^v;
    G = ER(N, p);
    [~, binsizes] = conncomp(G);
    numComps = numel(binsizes);
end