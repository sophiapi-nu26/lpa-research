function [maxCompSize] = findMaxCompSize(N, v)
    p = N^v;
    G = ER(N, p);
    [~, binsizes] = conncomp(G);
    maxCompSize = max(binsizes);
end