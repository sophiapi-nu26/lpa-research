function [diameter] = findDiameter(G)
% findDiameter
% finds the diameter of a graph; if graph is disconnected, it returns 0

D = distances(G); % N by N matrix storing distances from each node to every other node
diameter = max(max(distances));
if diameter > N % if it's Inf, i.e. if the graph is disconnected
    diameter = 0
end

end
