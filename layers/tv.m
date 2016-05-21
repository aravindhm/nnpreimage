function [e, dx] = tv(x,beta)
% --------------------------------------------------------------------

sz = size(x) ;
numPixels = prod(sz(1:2));

if(~exist('beta', 'var'))
  beta = 2; % the power to which the TV norm is raized
end
d1 = x(:,[2:end end],:,:) - x ;
d2 = x([2:end end],:,:,:) - x ;
v = (d1.*d1 + d2.*d2).^(beta/2) ;
e = sum(sum(sum(sum(v)))) / numPixels;
if nargout > 1
  % note that v is already raised to beta/2
  np = max(v, 1e-6).^(1 - 2/beta) ;
  d1_ = np .* d1;
  d2_ = np .* d2;
  d11 = d1_(:,[1 1:end-1],:,:) - d1_ ;
  d22 = d2_([1 1:end-1],:,:,:) - d2_ ;
  d11(:,1,:,:) = - d1_(:,1,:,:) ;
  d22(1,:,:,:) = - d2_(1,:,:,:) ;
  dx = (beta/numPixels) * (d11 + d22) ;
end
