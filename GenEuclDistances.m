function D = GenEuclDistances(xp, xq)
% function D = GenEuclDistances(xp, xq)
% Create distance matrix D with each entry D(i,j) the Euclidean
% distance between row xp(i, :) and row xq(j, :).

m = size(xp, 1);
n = size(xq, 1);
d = size(xp, 2);
assert(size(xq, 2) == d, 'xp and xq must have same number of columns');

D = zeros(m, n);

for i=1:m
  xi = xp(i, :);
  for j=1:n
    D(i, j) = sqrt(sum((xq(j, :)-xi).^2));
  end
end

end
