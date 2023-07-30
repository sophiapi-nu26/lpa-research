function [v] = BinSearchV(type, N, threshold)
% BinSearchV
% uses binary search to look for the value of v where MinLPA converged on N
% nodes about 50% of the time (+/- threshold)
%
% Inputs:
% N = number of nodes
% threshold = tolerance (in (0, 1)); will cease binary search when the
% number of converging trials is within [0.5 - threshold, 0.5 + threshold]
%
% Outputs:
% v = value of v that binary search produces

% stop when we get within [threshold] of 50% convergence

currentHalf = 0.5;
v = -1;
diffFrom50 = 0.5; % percent difference from 50% convergence

% initialize variable to count number of converged trials
converged = 0;

% set the number of trials for each v
numTrials = 32;

while true
    converged = 0;
    for i = 1:numTrials % run for each trial
        result = findLabels(type, N, v);
        % if there is only one surviving label...
        if numel(result(result > 0)) == 1
            % ...add
            converged = converged + 1;
        end
    end
    percentConverged = converged / 32;
    diffFrom50 = abs(percentConverged - 0.5);
    fprintf("v = %f, percentConverged = %f, currentHalf = %f\n", v, percentConverged, currentHalf)
    % if within threshold of 0.5, end loop
    if diffFrom50 < threshold
        break
    end
    % if most trials are not converging, then increase v
    if percentConverged < 0.5
        v = v + currentHalf;
        currentHalf = currentHalf / 2;
    % otherwise if most trials are converging, then decrease v
    else
        v = v - currentHalf;
        currentHalf = currentHalf/2;
    end
end

end