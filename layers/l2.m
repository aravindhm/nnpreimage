function [e, dx] = l2(x, beta)
% --------------------------------------------------------------------

sz = size(x);
numPixels = prod(sz(1:2));

n = sum(x.^2, 3) ;
e = sum(n(:).^(beta/2)) / numPixels;

if (nargout > 1)
  dx = beta/numPixels * bsxfun(@times, x, n.^((beta/2)-1)) ;
end
