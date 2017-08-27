%
% Script to test pdipmltp() against linprog(), or glpk(),
% in either MATLAB or OCTAVE environments.
% The MATLAB execution requires the Optimization Toolbox.
% The OCTAVE execution requires the optim package, for linprog().
%
% Generates random Linear Transportation Problems which are either
% rectangular (m != n) or quadratic (m == n).
% Can also restrict the problem class to Linear Assignments
% by setting isLSAP = true.
%

if exist('use_linprog', 'var') ~= 1
  use_linprog = true;
end

isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;

if isOctave
  if use_linprog
    pkg load optim
    fprintf(1, '[%s]: will try to use linprog() as reference solver\n', mfilename);
  else
    fprintf(1, '[%s]: will try to use glpk() as reference solver\n', mfilename);
  end
else
  assert(license('test', 'optimization_toolbox') == 1, ...
    'Failed to check Optimization Toolbox license');
end

if exist('isLSAP', 'var') ~= 1
  isLSAP = false;
end

if exist('make_timing_scatter_plot', 'var') ~= 1
  make_timing_scatter_plot = false;
end

num_skipped = 0;
if exist('nmax', 'var') ~= 1
  nmax = 100;
end
if exist('numtests', 'var') ~= 1
  numtests = 50;
end
dims = NaN(numtests, 3);
errs = NaN(numtests, 2);
vals = NaN(numtests, 2); % objective value for linprog, pdipmltp; when converged.
ltpi = NaN(numtests, 1);
ltpe = NaN(numtests, 1);
tims = NaN(numtests, 2); % solver execution times [linprog, pdipmltp]

if isLSAP
  fprintf(1, '*** will do LAP\n');
else
  fprintf(1, '*** will do LTP\n');
end

if exist('typ', 'var') ~= 1
  typ = 0; % can be set to 0, 1, 2
end

fprintf(1, '*** using factorization type = %i\n', typ);
fprintf(1, '*** doing %i tests (with nmax=%i)\n', numtests, nmax);

for t = 1:numtests
  % Generate random problems: min h'*x, s.t. C*x=d, x >= 0
  if isLSAP
    m = randi(nmax);
    n = m;
    rsums = ones(m, 1);
    csums = ones(n, 1);
  else
    m = randi(nmax);
    n = randi(nmax);
    rsums = rand(m, 1);
    csums = rand(n, 1);
    csums = csums * (sum(rsums)/sum(csums));
  end
  h = rand(m, n);
  h = h(:);
  [C, d] = GenFullMatricesTP(rsums, csums);
  assert(size(C, 1) == m+n && size(C, 2) == m*n);
  assert(size(d, 1) == m+n && size(d, 2) == 1);
  k = size(C, 1);
  C = C(1:(k-1), :); % remove last row from constraints
  d = d(1:(k-1));
  nd = numel(h);
  lb = zeros(nd, 1);
  ub = Inf(nd, 1);

  dims(t, :) = [m, n, nd];
  
  ttt = tic;
  % "reference" solution
  if isOctave
    if use_linprog
      [x, fx] = linprog(h, [], [], C, d, lb, ub);
      linprog_converged = (~isna(fx) && (numel(x)==nd));
    else
      [x, fx, xflag] = glpk(h, C, d);
      linprog_converged = (xflag == 0);
    end
  else
    [x, fx, xflag] = linprog(h, [], [], C, d, lb, ub);
    linprog_converged = (xflag == 1);
  end
  ttt = toc(ttt);
  
  if linprog_converged
    vals(t, 1) = fx;
    tims(t, 1) = ttt;
  end
  
  % pdipmltp(.) trial solution
  if ~isLSAP, epstop = 1e-10; else epstop = 1e-8; end % LAP solve happier with 1e-8 tolerance 
  eta = 0.96;
  kmax = 50;
  
  ttt = tic; % should run fine in both OCTAVE and MATLAB
  rep = pdipmltp(reshape(h, [m, n]), rsums, csums, kmax, epstop, eta, typ);
  ttt = toc(ttt);
  
  if rep.isconverged
    rep.x = rep.X(:); % add new field for vectorized solution
    vals(t, 2) = rep.fx;
    ltpi(t) = rep.iters;
    cerr = C*rep.x - d;
    ltpe(t) = norm(cerr, 'inf') / max(1, norm(d, 'inf'));
    tims(t, 2) = ttt;
  end
  
  if rep.isconverged && linprog_converged
    fxdiff = fx - rep.fx;
    ferr = abs(fxdiff) / max(1, abs(fx));
    xdiff = x(:) - rep.x(:);
    xerr = norm(xdiff, 'inf') / max(1, norm(x(:), 'inf'));
    errs(t, :) = [ferr, xerr];
  else
    num_skipped = num_skipped + 1;
  end
end

% Done; now summarize results
if num_skipped > 0
  fprintf(1, '[%s]: WARNING: %i test problems did not get solved by both codes.\n', ...
    mfilename, num_skipped);
end

fprintf('[%s]: avg. iterations = %f, range = [%i, %i]\n', ...
  mfilename, mean(ltpi), min(ltpi), max(ltpi));
fprintf('[%s]: avg. cnstrnt. violation = %e, range = [%e, %e]\n', ...
  mfilename, mean(ltpe), min(ltpe), max(ltpe));

reltol = 1e-6;

fprintf('[%s]: %i out of %i problems had fx-discrepancy above %e\n', ...
  mfilename, length(find(errs(:, 1) > reltol)), numtests, reltol);
fprintf('[%s]: %i out of %i problems had x-discrepancy above %e\n', ...
  mfilename, length(find(errs(:, 2) > reltol)), numtests, reltol);

disp(max(errs));
disp(mean(errs));
disp(median(errs));

if make_timing_scatter_plot
  figure;
  hold on;
  plot(dims(:, 3), tims(:, 1), 'ko');
  plot(dims(:, 3), tims(:, 2), 'bs');
  xlabel('decision vector size m * n');
  if isOctave, progstr = 'OCTAVE'; else progstr = 'MATLAB'; end
  if isOctave && ~use_linprog, refname = 'glpk'; else refname = 'linprog'; end
  ylabel(sprintf('%s solver time [sec]', progstr));
  legend(refname, 'pdipmltp');
  if isLSAP, lptype = 'LAP'; else lptype = 'LTP'; end
  title(sprintf('Solver execution speed for %i random %s problems', numtests, lptype));
end
