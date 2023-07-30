function [labelFrequencies] = findLabelsSBM(type, N, communities, vp, vq)
    if type == 1
        [F, iteration, ~] = MinLPAonSBM(N, communities, vp, vq, 100); % set high cap to allow for convergence
    elseif type == 2
        [F, iteration, ~] = RandLPAonSBM(N, communities, vp, vq, 100); % set high cap to allow for convergence
    else
        [F, iteration, ~] = LPAonSBM(N, communities, vp, vq, 100); % set high cap to allow for convergence
    end
    labelVector = F(:, iteration);
    [frequencies, labels] = groupcounts(labelVector);
    labelFrequencies = zeros(N, 1);
    for i = 1:numel(labels)
        labelFrequencies(labels(i)) = frequencies(i);
    end
end