%
% Test script for the linear assignment problem (LAP) with random 
% costs uniform on (0, 1). It is conjectured that the optimal costs
% converges to pi*pi/6 as the LAP size n goes to infinity
%

if exist('n', 'var') ~= 1
  n = 500;
end

if exist('nsamp', 'var') ~= 1
  nsamp = 50;
end

kmax = 50;
epstop = 1e-8;
eta = 0.95;
typ = 2;

rsums = ones(n, 1);
csums = ones(n, 1);

fx = NaN(nsamp, 1);
itr = NaN(nsamp, 1);
tim = NaN(nsamp, 1);

fprintf(1, 'Will take %i samples of U(0,1)-LAP with n = %i\n', nsamp, n);

for pp = 1:nsamp
  costs = rand(n, n);
  ttt = tic;
  rep = pdipmltp(costs, rsums, csums, kmax, epstop, eta, typ);
  ttt = toc(ttt);
  
  if rep.isconverged
    fx(pp) = rep.fx;
    itr(pp) = rep.iters;
    tim(pp) = ttt;
  else
    fprintf(1, '[%s]: failed to converge instance %i\n', mfilename, pp);
  end
  
  if mod(pp, 10) == 0
    fprintf(1, '[%s]: sample %i done.\n', mfilename, pp);
  end
  
end

incl = ~isnan(fx);

fprintf(1, 'average iterations per instance = %.2f\n', mean(itr(incl)));
fprintf(1, 'average solve time per instance = %.3f secs.\n', mean(tim(incl)));

fprintf(1, 'conjecture = %.6f (for n -> Inf)\n', pi*pi/6);
fprintf(1, 'mean(fx)   = %.6f (nsamp = %i, n = %i)\n', mean(fx(incl)), nsamp, n);
fprintf(1, 'stdv(fx)   = %.6f ( ... )\n', std(fx(incl)));
