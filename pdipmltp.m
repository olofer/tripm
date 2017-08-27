function rep = pdipmltp(Cost, Sup, Dem, kmax, epstop, eta, typ)
% function rep = pdipmltp(Cost, Sup, Dem, kmax, epstop, eta, typ)
%
% Standalone primal-dual interior point method solver for the 
% rectangular linear transportation problem (LTP); also
% known as the Hitchcock Transportation Problem (HTP).
%
% LTP/HTP is the linear program:
%
%   min_x sum_{i, j} Cost(i, j) * x(i, j)
%
%     s.t. x(i, j) >= 0
%          sum_{j} x_{i, j} = Sup(i)    "Supply", row-sums
%          sum_{i} x_{i, j} = Dem(j)    "Demand", column-sums
%
% Cost has m*n elements, Sup has m elements, Dem has n elements.
% It is required (and assumed) that sum(Sup) == sum(Dem).
%
% Memory is ~O((m+n)^2) and flops are ~O((m+n)^3) for typ=0.
%
% The linear assignment problem (LAP) is a special case.
% Set Sup=Dem=ones(n, 1) and Cost=n-by-n matrix for size-n LAP.
%

% Run a self-test of key solver components
if nargin==0 && nargout==0
  K = 10;
  fprintf(1, '*** RANDOM SELF-TEST (%iX)\n', K);
  for kk = 1:K
    m = randi(500);
    n = randi(500);
    L = rand(m*n, 1);
    fprintf(1, 'dims: m=%i, n=%i\n', m, n);
    err = testMultiply(m, n, L);
    fprintf(1, 'mult: err=%e\n', err);
    for ll = 0:2
      [err, res] = testSolve(m, n, ll, L);
      fprintf(1, 'solve[typ=%i]: err=%e, res=%e\n', ll, err, res);
    end
  end
  fprintf(1, '*** DONE.\n');
  return; % Exit
end

% Start actual initialization here; first input check
rep = struct;

nd = numel(Cost);
m = numel(Sup);
n = numel(Dem);

assert(nd == m*n, 'Dimensions do not match.');
assert(size(Cost, 1) == m && size(Cost, 2) == n, 'Shape of Cost matrix is incorrect.');

if nargin < 4 || (nargin >= 4 && isempty(kmax))
    kmax = 50;
    fprintf(1, '[%s]: warning; defaulted kmax = %i\n', mfilename, kmax);
end

if nargin < 5 || (nargin >= 5 && isempty(epstop))
    epstop = 1e-8;
    fprintf(1, '[%s]: warning; defaulted epstop = %e\n', mfilename, epstop);
end

if nargin < 6 || (nargin >= 6 && isempty(eta))
    eta = 0.95;
    fprintf(1, '[%s]: warning; defaulted eta = %e\n', mfilename, eta);
end

if nargin < 7 || (nargin >= 7 && isempty(typ))
    typ = 0;
    fprintf(1, '[%s]: warning; defaulted factorization typ = %i\n', mfilename, typ);
end

assert(kmax >= 1, 'iteration maximum incorrect.');
assert(epstop > 0 && epstop <= 1e-2, 'stopping epsilon not allowed.');
assert(eta > 0 && eta < 1, 'eta in (0,1) required.');
assert(typ == 0 || typ == 1 || typ == 2, 'incorrect factorization type specified.');

% Safeguards
if (n == 1 && typ == 1) || (m == 1 && typ == 2)
  typ = 0;
end

h = Cost(:);
nx = length(h);

sum_sup = sum(Sup(:));
sum_err = sum_sup - sum(Dem(:));
if abs(sum_err)/(1 + abs(sum_sup)) > 1e-12
  fprintf(1, '[%s]: warning; sum(Sup) != sum(Dem); error = %e\n', ...
    mfilename, abs(sum_err));
  % Continue as if the last element of Dem is adjusted so that it fixes the imbalance.
  % But be very explicit about this "fix"
  fprintf(1, '[%s]: warning: continuing as if Dem(n) = %e (its value is %e)\n', ...
    mfilename, Dem(n) + sum_err, Dem(n));
end

ny = m + n - 1; % number of rows in implied full-rank C matrix (# equalities)
d = [Sup(:); Dem(:)];
d = d(1:ny); % remove last element; fixed or not

nz = nx;

inff = 0;
infh = norm(h, 'inf');
infd = norm(d, 'inf');
infdat = max(infd, infh);

assert(infdat > 0, 'All input data are zeros?');

beta = sqrt(infdat);

x = zeros(nx, 1);
y = zeros(ny, 1);
z = ones(nz, 1) * beta;
s = ones(nz, 1) * beta;

e1 = ones(nz, 1);

% Simplified initial residuals (assuming x=0, y=0)

rC = h - z;         % rC = h + C'*y + E'*z; where E=-I
rE = -d;            % rE = C*x - d;
rI = s;             % rI = E*x + s - f; where f=0
rsz = ones(nz, 1);  % rsz = s.*z;
mu = sum(rsz) / nz;

thrC = epstop * (1 + infh);
thrE = epstop * (1 + infd);
thrI = epstop * (1 + inff);
thrmu = epstop;
k = 0;

tt0 = tic;
tt1 = 0;

infC = norm(rC, 'inf');
infE = norm(rE, 'inf');
infI = norm(rI, 'inf');
oktostop = (k>=kmax || (infC<thrC && infE<thrE && infI<thrI && mu<thrmu));
while ~oktostop
  invPhi = s./z;
  ell = z./s;
  
  ttmp = tic;
  [M11, M21, M22, Llo, pp] = factorizeImpliedCLCt(m, n, invPhi, typ);
  if pp>0
    rep.cholerr = pp;
    break;
  end
  tt1 = tt1 + toc(ttmp);
  
  tmp = invPhi.*(-((-rI+s).*ell)-rC); % inv(Phi)*(E'*diag(z./s)*r3+r1)
  dy_a = solveEqForRHS(m, n, M11, M21, M22, Llo, rE + impliedCmult(m, n, tmp), typ);
  dx_a = tmp-invPhi.*impliedCtmult(m, n, dy_a);
  dz_a = -ell.*(-rI+s+dx_a); % v3
  ds_a = -((rsz+s.*dz_a)./z);      
  alpha_a = 1;
  idx_z = find(dz_a<0);
  if (~isempty(idx_z))
    alpha_a = min(alpha_a, min(-z(idx_z)./dz_a(idx_z)));
  end
  idx_s = find(ds_a<0);
  if (~isempty(idx_s))
    alpha_a = min(alpha_a, min(-s(idx_s)./ds_a(idx_s)));
  end
  % Compute the affine duality gap
  mu_a = ((z+alpha_a*dz_a)'*(s+alpha_a*ds_a))/nz;
  % Compute the centering parameter
  sigma = (mu_a/mu)^3;
  % Solve system again (perturbed rhs)
  rsz = rsz + ds_a.*dz_a-sigma*mu*e1;
  % Normal equations
  tmp = invPhi.*(-((-rI+rsz./z).*ell)-rC);
  dy = solveEqForRHS(m, n, M11, M21, M22, Llo, rE + impliedCmult(m, n, tmp), typ);
  dx = tmp-invPhi.*impliedCtmult(m, n, dy);
  dz = -ell.*(-rI+rsz./z+dx);
  ds = -((rsz+s.*dz)./z);
  % Compute alpha
  alpha = 1;
  idx_z = find(dz<0);
  if (~isempty(idx_z))
    alpha = min(alpha,min(-z(idx_z)./dz(idx_z)));
  end
  idx_s = find(ds<0);
  if (~isempty(idx_s))
    alpha = min(alpha,min(-s(idx_s)./ds(idx_s)));
  end
  ea = eta*alpha;
  % Update x, y, z, s
  x = x + ea*dx;
  y = y + ea*dy;
  z = z + ea*dz;
  s = s + ea*ds;
  k = k + 1;
  % Update rhs
  rC = h + impliedCtmult(m, n, y) - z; % h + C'*y - z
  rE = impliedCmult(m, n, x) - d; % C*x - d
  rI = -x + s;
  rsz = s.*z;
  mu = sum(rsz)/nz;
  assert(all(rsz>0));
  infC = norm(rC, 'inf');
  infE = norm(rE, 'inf');
  infI = norm(rI, 'inf');
  oktostop = (k>=kmax || (infC<thrC && infE<thrE && infI<thrI && mu<thrmu));
end

% Output
rep.eta = eta;
rep.X = reshape(x, [m, n]);
rep.fx = h'*x;
rep.iters = k;
rep.epstop = epstop;
rep.maxiters = kmax;
rep.infdat = infdat;
rep.inftuple = [infC/(1+infh), infE/(1+infd), infI/(1+inff), mu];
rep.isconverged = (infC<thrC && infE<thrE && infI<thrI && mu<thrmu);
rep.t01 = [toc(tt0), tt1]; % tt1 = time spent on factorization, tt0 = total time incl. factorization

end

%
% The "inner equation" in the solver is M * u = v
% Where M = [M11, M21'; M21, M22] is symmetric and pos. def.
%
% M11 and M22 are diagonal matrices.
% The default is to instead factorize the Schur complement either of size n-1 or m 
% (selecting the smaller of the two).
% But this can be changed with the parameter "typ".
%

% Sub-programs for the core symmetric equation
function [M11, M21, M22, Llo, pp] = factorizeImpliedCLCt(m, n, L, typ)
  assert(m*n == length(L));
  M21 = reshape(L(1:(m*n-m)), [m, n-1]).';
  M11 = NaN(m, 1);
  for ii=1:m
    M11(ii) = sum(L(ii:m:(m*n)));
  end
  M22 = NaN(n-1, 1);
  for ii=1:(n-1)
    M22(ii) = sum(L((1:m)+m*(ii-1)));
  end
  if nargout > 3
    if typ == 1 % eliminate block 11
      [Llo, pp] = chol(diag(M22)-M21*diag(1./M11)*M21.', 'lower');
    elseif typ == 2 % eliminate block 22
      [Llo, pp] = chol(diag(M11)-M21.'*diag(1./M22)*M21, 'lower');
    else % factorize full equation (kind-of wasteful)
      [Llo, pp] = chol([diag(M11), M21.'; M21, diag(M22)], 'lower');
    end
  end
end

function x = solveEqForRHS(m, n, M11, M21, M22, Llo, b, typ)
  d = m+n-1;
  b1 = b(1:m);
  b2 = b((m+1):d);
  if typ == 1
    x2 = Llo'\(Llo\(b2-M21*(b1./M11)));
    x1 = (b1-M21.'*x2)./M11;
    x = [x1; x2];
  elseif typ == 2
    x1 = Llo'\(Llo\(b1-M21.'*(b2./M22)));
    x2 = (b2-M21*x1)./M22;
    x = [x1; x2];
  else
    x = Llo'\(Llo\b);
  end
end

% Sub-programs for implied multiplication by (huge) super-sparse C matrix
function y = impliedCmult(m, n, x)
  assert(length(x) == m*n);
  y = zeros(m+n-1, 1);
  for ii=1:m
    y(ii) = sum(x(ii:m:(m*n)));
  end
  for ii=1:(n-1)
    y(m+ii) = sum(x((1:m)+m*(ii-1)));
  end
end

function y = impliedCtmult(m, n, x)
  assert(length(x) == m+n-1);
  y = repmat(x(1:m), [n, 1]);
  for ii=1:(n-1)
    idx = (1:m)+(ii-1)*m;
    y(idx) = y(idx) + x(m+ii);
  end
end

% Test code; verify that implicit C, C' multiplications and that
% the matrix formations are "self-consistent"
function err = testMultiply(m, n, L)
  if isempty(L)
    L = rand(m*n, 1);
  end
  [M11, M21, M22] = factorizeImpliedCLCt(m, n, L, []);
  u = randn(m+n-1, 1);
  v1 = [diag(M11), M21'; M21, diag(M22)] * u;
  v2 = impliedCmult(m, n, L.*impliedCtmult(m, n, u));
  err = norm(v2-v1, 'inf') / max(1, norm(v1, 'inf'));
end

% Test code to solve M*u = v with a specific elimination typ
function [err, res] = testSolve(m, n, typ, L)
  if isempty(L)
    L = rand(m*n, 1);
  end
  [M11, M21, M22, Llo, pp] = factorizeImpliedCLCt(m, n, L, typ);
  assert(pp == 0, 'Cholesky failed.');
  v = randn(m + n - 1, 1); % random RHS
  u1 = [diag(M11), M21'; M21, diag(M22)] \ v; % reference solution
  u2 = solveEqForRHS(m, n, M11, M21, M22, Llo, v, typ); % test solution
  err = norm(u2-u1, 'inf') / max(1, norm(u1, 'inf'));
  if nargout > 1
    % If requested return the equation residual error for the test solution
    resvec = [diag(M11), M21'; M21, diag(M22)] * u2 - v;
    res = norm(resvec, 'inf') / max(1, norm(v, 'inf'));
  end
end
