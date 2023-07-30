function [numVals] = countVals(N, v)
    [~, ~, numVals] = LPA(N, v, 5);
end