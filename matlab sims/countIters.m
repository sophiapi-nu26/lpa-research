function [numIters] = countIters(N, v)
    [~, numIters, ~] = LPA(N, v, 100);
end