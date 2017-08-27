function [C, d] = GenFullMatricesTP(rsums, csums, warningOn)
% function [C, d] = GenFullMatricesTP(rsums, csums, warningOn)
% Return matrix C and column d so that C*X(:)=d corresponds
% to row sum and column sum constraints for the matrix X.
% If rsums and csums have the same number of elements and the
% elements are all ones; the C*z=d relation is the 
% linear assignment problem constraint.
%
% If numel(rsums)=m and numel(csums)=n then size(C) = [m+n, m*n] and
% rank(C) = m+n-1 and d = [rsums(:); csums(:)];
%

if nargin < 3
  warningOn = true;
end

rsums = rsums(:);
csums = csums(:);

m = numel(rsums);
n = numel(csums);

if warningOn
  err = sum(rsums) - sum(csums);
  sum_eptol = 1e-12;
  if abs(err) / norm([rsums; csums], 'inf') > sum_eptol
    fprintf(1, 'WARNING: sum of rowsums (%e) != sum of column sums (%e) @ eps = %e\n', ...
      sum(rsums), sum(csums), sum_eptol);
  end
end

C1 = zeros(m, m*n); % row sums
d1 = rsums;
for ii=1:m
  dmy = zeros(m, n);
  dmy(ii, :) = 1;
  C1(ii, :) = dmy(:)';
end
C2 = zeros(n, m*n); % column sums
d2 = csums;
for ii=1:n
  dmy = zeros(m, n);
  dmy(:, ii) = 1;
  C2(ii, :) = dmy(:)';
end
C = [C1; C2]; % NOTE: rank = m+n-1 < m+n
d = [d1; d2];

end
