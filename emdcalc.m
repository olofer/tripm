function [d, x, rep] = emdcalc(P, Q, Grnd, kmax, epstop, eta, typ)
% function [d, x, rep] = emdcalc(P, Q, Grnd, kmax, epstop, eta, typ)
%
% Solve the Earth Mover's Distance program (EMD) by trying to pack it into to
% the Hitchcock Transportation Problem and solving it with pdipmltp().
%
% EMD is defined as follows.
%
% Let i = 1..m, j = 1..n.
% With the solution f_{i, j} to the LP
%
% f* = min { sum_{i, j} f(i, j) * Grnd(i, j) }
%
%   s.t. sum_j f_{i, j} <= P_i
%        sum_i f_{i, j} <= Q_j
%        sum_{i, j} f_{i, j} = min{ sum_i P_i, sum_j Q_j }
%        f_{i, j} >= 0
%
% Then 
%
% d = EMD(P, Q) = f* / sum_{i, j} f_{i, j}
%

P = P(:);
Q = Q(:);
m = length(P);
n = length(Q);
sump = sum(P);
sumq = sum(Q);

assert(size(Grnd, 1) == m && size(Grnd, 2) == n, 'Grnd has incorrect dimensions.');

d = NaN;
x = [];

if sump > sumq
  % Over-supply case; add free dump market with apetite r
  r = sump - sumq;
  rep = pdipmltp([Grnd, zeros(m, 1)], P, [Q; r], kmax, epstop, eta, typ);
  if rep.isconverged
    d = rep.fx / sumq;
    if nargout > 1, x = rep.X(:, 1:n); end
  end
elseif sumq > sump
  % Under-supply case; add free source with produce r
  r = sumq - sump;
  rep = pdipmltp([Grnd; zeros(1, n)], [P; r], Q, kmax, epstop, eta, typ);
  if rep.isconverged
    d = rep.fx / sump;
    if nargout > 1, x = rep.X(1:m, :); end
  end
else
  % No need to augment the problem
  rep = pdipmltp(Grnd, P, Q, kmax, epstop, eta, typ);
  if rep.isconverged
    d = rep.fx / sump; % sump == sumq
    if nargout > 1, x = rep.X; end
  end
end

end
