function [labelFrequencies] = findLabels(type, N, v)
    if type == 1
        [F, iteration, ~] = MinLPA(N, v, 100); % set high cap to allow for convergence
    elseif type == 2
        [F, iteration, ~] = RandLPA(N, v, 100); % set high cap to allow for convergence
    else
        [F, iteration, ~] = LPA(N, v, 100); % set high cap to allow for convergence
    end
    labelVector = F(:, iteration);
    [frequencies, labels] = groupcounts(labelVector);
    labelFrequencies = zeros(N, 1);
    for i = 1:numel(labels)
        labelFrequencies(labels(i)) = frequencies(i);
    end
end