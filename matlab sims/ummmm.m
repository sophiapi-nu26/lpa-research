v = zeros(1, size(results, 2));

for col = 1:size(results, 2)
    nonZeroRows = find(results(:,col));
    if ~isempty(nonZeroRows)
        v(col) = max(nonZeroRows);
    end
end

v=flip(v);