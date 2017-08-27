%
% Script to test the emdcalc code by comparing to linprog() "reference" solutions.
% Generate random problems corresponding to q-D Euclidean "ground distances" where 
% the "centroids" are scattered uniformly in the unit q-cube and the weights are 
% uniform on (0, 1).
%


isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;

if isOctave
  pkg load optim
else
  assert(license('test', 'optimization_toolbox') == 1);
end

if exist('nmax', 'var') ~= 1
  nmax = 100;
end
if exist('numtests', 'var') ~= 1
  numtests = 50;
end
if exist('make_timing_scatter_plot', 'var') ~= 1
  make_timing_scatter_plot = false;
end

fprintf('[%s]: sampling %i random test problems with m,n max = %i...\n', ...
  mfilename, numtests, nmax);

dims = NaN(numtests, 3);
vals = NaN(numtests, 2); % row = [linprog, emdcalc] solutions
errs = NaN(numtests, 2); % row = relative difference of opt. value and opt solutions
tims = NaN(numtests, 2); % row = [linprog time, emdcalc time]

num_skipped = 0;
numplarger = 0;

for t=1:numtests
  % Generate random EMD program data
  m = randi(nmax);
  n = randi(nmax);
  q = randi(9) + 1; % ~ U[2:10] 
  xp = rand(m, q);
  xq = rand(n, q);
  D = GenEuclDistances(xp, xq); % m-by-n matrix D with ground distances (table of costs)
  exp_weight_sum = (m+n)/2;
  wp = rand(m, 1)* exp_weight_sum/m; % random volumes m supplies
  wq = rand(n, 1) * exp_weight_sum/n; % random volumes n demands
  if sum(wp) > sum(wq)
    numplarger = numplarger + 1;
  end

  dims(t, :) = [m, n, q];

  % Create equality and inequality constrained standard LP for this random EMD instance
  [Ciq, diq] = GenFullMatricesTP(wp, wq, false);
  ceq = ones(1, m * n);
  deq = min([sum(wp), sum(wq)]);
  lb = zeros(m *n, 1);
  ub = Inf(m * n, 1);

  ttt = tic;
  if isOctave
    [x, fx] = linprog(D(:), Ciq, diq, ceq, deq, lb, ub);
    linprog_converged = (~isna(fx) && (numel(x) == m*n));
  else
    [x, fx, xflag] = linprog(D(:), Ciq, diq, ceq, deq, lb, ub);
    linprog_converged = (xflag == 1);
  end
  ttt = toc(ttt);

  if linprog_converged
    vals(t, 1) = fx / sum(x);
    tims(t, 1) = ttt;
  end

  % Here call emdcalc()
  kmax = 50;
  epstop = 1e-10;
  eta = 0.96;
  typ = 0;

  ttt = tic;
  [d, X, rep] = emdcalc(wp, wq, D, kmax, epstop, eta, typ);
  ttt = toc(ttt);

  if rep.isconverged
    vals(t, 2) = d;
    tims(t, 2) = ttt;
  end

  if rep.isconverged && linprog_converged
    errf = vals(t, 2) - vals(t, 1);
    errs(t, 1) = abs(errf) / max(1, abs(vals(t, 1)));
    errx = X(:) - x;
    errs(t, 2) = norm(errx, 'inf') / max(1, norm(x, 'inf'));
  else
    num_skipped = num_skipped + 1;
  end

end

% Done; now summarize results
if num_skipped > 0
  fprintf(1, '[%s]: WARNING: %i test problems did not get solved by both codes.\n', ...
    mfilename, num_skipped);
end

reltol = 1e-6;

fprintf('[%s]: %i out of %i problems had fx-discrepancy above %e\n', ...
  mfilename, length(find(errs(:, 1) > reltol)), numtests, reltol);
fprintf('[%s]: %i out of %i problems had x-discrepancy above %e\n', ...
  mfilename, length(find(errs(:, 2) > reltol)), numtests, reltol);
fprintf('[%s]: %i out of %i test problems were over-supplied\n', ...
  mfilename, numplarger, numtests);

% Show solver discrepancies; max, mean, median
disp(max(errs));
disp(mean(errs));
disp(median(errs));

if make_timing_scatter_plot
  figure;
  hold on;
  mnvec = dims(:, 1).*dims(:, 2);
  plot(mnvec, tims(:, 1), 'ko');
  plot(mnvec, tims(:, 2), 'bs');
  xlabel('decision vector size m * n');
  if isOctave, progstr = 'OCTAVE'; else progstr = 'MATLAB'; end
  ylabel(sprintf('%s solver time [sec]', progstr));
  legend('linprog', 'emdcalc / pdipmltp');
  title(sprintf('Solver execution speed scatter for %i random EMD problems', numtests));
end
