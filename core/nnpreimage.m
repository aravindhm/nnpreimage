function result = nnpreimage(net, ref, varargin)
%NNPREIMAGE  Compute the pre-image of a CNN

def = get_defaults() ;

opts.learningRate = def.fastLR ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.imageSize = [];
opts.initialImage = [] ;
opts.maxNumIterations = numel(opts.learningRate) ;
opts.mask = [] ;
opts.bound = def.x0_max ;
opts.objective = 'l2' ;
opts.objectiveWeight = 1 ;
opts.jitterAmount = 1 ;
opts.L.beta = def.alpha ;
opts.L.lambda = def.lambdaAlpha ;
opts.TV.beta = def.beta ;
opts.TV.lambda = def.lambdaBeta ;
opts.TG.beta = def.alpha ;
opts.TG.lambda = 0 ;
opts.momentum = 0.9 ;
opts.numRepeats = 1 ;
opts.normalize = @(x) x ;
opts.denormalize = @(x) x ;
opts.useAdagrad = true ;
opts.leak = 0;
opts.gpu = false;
opts = vl_argparse(opts, varargin) ;

if isinf(opts.maxNumIterations)
  opts.maxNumIterations = numel(opts.learningRate) ;
end

warning('off', 'MATLAB:Axes:NegativeDataInLogAxis') ;

% display config
disp('params:') ; disp(opts)
disp('L params:') ; disp(opts.L)
disp('TV params:') ; disp(opts.TV)


% --------------------------------------------------------------------
%                                                        Initial image
% --------------------------------------------------------------------

% initial inversion image of size x0_size
numPixels = prod(opts.imageSize(1:2)) ;
x0_size = cat(2, opts.imageSize(1:3), opts.numRepeats);

x0jit_size = x0_size ;
x0jit_size(1:2) = x0jit_size(1:2) + opts.jitterAmount - 1 ;
x = 2*rand(x0jit_size, 'single')-1 ;
for k = 1:size(x,4)
  t = x(:,:,:,k) ;
  %t = repmat(mean(t,3),[1 1 3]) ;
  %t = imgaussfilt(t, min(5, size(t,1)/5), 'padding','circular') ;
  %t = vl_imsmooth(t, min(20, size(t,1)/4), 'padding', 'zero') ;
  m = quantile(t(:),.95) ;
  %t = min(max(t,-m),m) ;
  x(:,:,:,k) = t / m * def.x0_sigma / sqrt(3) ;
end

if ~isempty(opts.initialImage)
  sx = (1:x0_size(2)) + floor((opts.jitterAmount-1)/2) ;
  sy = (1:x0_size(1)) + floor((opts.jitterAmount-1)/2) ;
  x = x * 0 ;
  x(sy,sx,:,:) = opts.normalize(opts.initialImage) ;
  x_init = x ;
end
x_momentum = zeros(x0jit_size, 'single') ;

% ---------------------------------------------------------------------
%                                                                Target
% ---------------------------------------------------------------------

if isempty(opts.mask), opts.mask = ones(size(ref),'single') ; end

y0 = repmat(ref, [1 1 1 opts.numRepeats]) ;
mask = repmat(opts.mask, [1 1 1 opts.numRepeats]) ;

if(opts.gpu)
  y0 = gpuArray(y0);
  mask = gpuArray(mask);
  opts.objectiveWeight = gpuArray(opts.objectiveWeight);
  x = gpuArray(x);
  x_momentum = gpuArray(x_momentum);
  net = vl_simplenn_move(net, 'gpu');
end

% --------------------------------------------------------------------
%                                                      Prepare network
% --------------------------------------------------------------------

if(opts.leak > 0)
  net = set_leak_factor(net, opts.leak);
end

layer_num = numel(net.layers) ;
switch opts.objective
  case 'l2'
    ly.type = 'custom' ;
    ly.w = y0 ;
    ly.mask = mask ;
    ly.forward = @nndistance_forward ;
    ly.backward = @nndistance_backward ;
    net.layers{end+1} = ly ;
  case 'inner'
    ly.type = 'custom' ;
    ly.w = - y0 .* mask ;
    ly.forward = @nninner_forward ;
    ly.backward = @nninner_backward ;
    net.layers{end+1} = ly ;
  otherwise
    error('unknown opts.objective') ;
end

% --------------------------------------------------------------------
%                                                         Optimisation
% --------------------------------------------------------------------

% recored results
output = {} ;
prevlr = 0 ;
adagradG = 0 ;
adadeltaG = 0 ;
adadeltaD = 0 ;

for t=1:opts.maxNumIterations
  % crop imagae with a random shift
  jx = randi(opts.jitterAmount) ;
  jy = randi(opts.jitterAmount) ;
  sx = (1:x0_size(2)) + jx - 1 ;
  sy = (1:x0_size(1)) + jy - 1 ;
  %  if randn > 0, sx = fliplr(sx) ; end
  xcrop = x(sy,sx,:,:) ;

  % backprop
  res = vl_simplenn(net, xcrop, opts.objectiveWeight) ;

  % current reconstruction
  y = res(end-1).x ;

  % loss and corresponding gradient
  E(1,t) = gather(res(end).x * opts.objectiveWeight) ;
  if(opts.gpu)
    dd = zeros(x0jit_size,'single', 'gpuArray') ;
  else
    dd = zeros(x0jit_size, 'single');
  end
  dd(sy,sx,:,:) = res(1).dzdx ;

  % regulariser and corresponding gradient
  if(opts.gpu)
    dr = zeros(size(x), 'single', 'gpuArray') ;
  else
    dr = zeros(size(x)) ;
  end

  if opts.TV.lambda > 0
    [r_,dr_] = tv(x,opts.TV.beta) ;
    E(2,t) = gather(opts.TV.lambda * r_) ;
    dr = dr + opts.TV.lambda * dr_ ;
  else
    E(2,t) = 0;
  end

  if opts.L.lambda > 0
    [r_, dr_] = l2(x, opts.L.beta);
    E(3,t) = gather(opts.L.lambda * r_) ;
    dr = dr + opts.L.lambda * dr_ ;
  else
    E(3,t) = 0;
  end

  if opts.TG.lambda > 0
    [r_, dr_] = l2(x - x_init, opts.L.beta);
    E(5,t) = gather(opts.TG.lambda * r_) ;
    dr = dr + opts.TG.lambda * dr_ ;
  else
    E(5,t) = 0;
  end

  % Total energy
  E(4,t) = sum(E(1:3,t)) ;
  fprintf('iter:%05d sq. loss:%7.4g; rtv:%7.4g; rlp:%7.4g; obj:%7.4g;\n', ...
    t, E(1,end), E(2,end), E(3,end), E(4,end)) ;

  % Get learning rate
  lr = numPixels * opts.learningRate(t) ;
  if lr ~= prevlr
    fprintf('switching learning rate (%f to %f) and resetting momentum\n', ...
      prevlr, lr) ;
    x_momentum = 0 * x_momentum ;
    adagraD = 0 ;
    prevlr = lr ;
  end

  % compute and clamp gradient
  gradient = dd + dr ;
  %  gradient = max(min(gradient,10/lr),-10/lr) ;

  % learning rates

  % momentum
  if opts.useAdagrad
    adagradG = opts.momentum * adagradG + gradient.^2 ;
    x_momentum = opts.momentum * x_momentum ...
        - 1 * gradient ./ (1/lr + sqrt(adagradG)) ;

    % rho = opts.momentum ;
    % eps = 1 ;
    % adadeltaG = rho * adadeltaG + (1-rho) * gradient.^2 ;
    % eta = sqrt(adadeltaD + eps) ./ sqrt(adadeltaG + eps/lr^2) ;
    % delta = - 2*eta .* gradient ;
    % adadeltaD = rho * adadeltaD + (1-rho) * delta.^2 ;
    % adadeltaG(1), eps/lr^2, adadeltaD(1), eta(1), delta(1)
    % x_momentum = delta ;

  else
    x_momentum = opts.momentum * x_momentum ...
      - lr * gradient  ;
  end

  % This is the main update step (we are updating the the variable
  % along the gradient
  x = x + x_momentum;

  % box constratint
  nx = sqrt(sum(x.^2,3)) ;
  x = bsxfun(@times, x, min(opts.bound ./ nx, 1)) ;

  % -----------------------------------------------------------------------
  % Plots - Generate several plots to keep track of our progress
  % -----------------------------------------------------------------------

  if mod(t-1,20)==0 || t == opts.maxNumIterations
    sx = (1:x0_size(2)) + floor((opts.jitterAmount-1)/2) ;
    sy = (1:x0_size(1)) + floor((opts.jitterAmount-1)/2) ;
    output{end+1} = opts.denormalize(gather(x(sy,sx,:,:))) ;

%    change_current_figure(1) ; clf ;
%
%    subplot(4,2,[1 3 5]) ;
%    if opts.numRepeats > 1
%      vl_imarraysc(output{end}) ;
%    else
%      imagesc(vl_imsc(output{end}) );
%    end
%    title(sprintf('jitter:%d', opts.jitterAmount)) ;
%    axis image ; colormap gray ;
%
%    subplot(4,2,2) ;
%    len = min(1000, numel(y0));
%    a = gather(squeeze(y0(1:len))) ;
%    b = gather(squeeze(y(1:len))) ;
%    plot(1:len,a,'b'); hold on ;
%    plot(len+1:2*len,b-a, 'r');
%    legend('\Phi_0', '|\Phi-\Phi_0|') ;
%    title(sprintf('reconstructed layer %d %s', ...
%      layer_num, ...
%      net.layers{layer_num}.type)) ;
%    legend('ref', 'delta') ;
%
%    subplot(4,2,4) ;
%    hist(gather(x(:)),100) ;
%    grid on ;
%    title('histogram of x') ;
%
%    subplot(4,2,6) ;
%    hist(gather(y(:)),100) ;
%    grid on ;
%    title('histogram of y') ;
%
%    subplot(4,2,7) ;
%    plot(1:size(E,2),E') ;
%    h = legend('recon', 'tv_reg', 'l2_reg', 'tot') ;
%    set(h,'color','none') ; grid on ;
%    title(sprintf('iter:%d \\lambda_{tv}:%g rate:%g obj:%s', ...
%      t, opts.TV.lambda, lr, opts.objective)) ;
%
%    subplot(4,2,8) ;
%    semilogy(1:size(E,2),E') ;
%    title('log scale') ;
%    grid on ;
%
%    %figure(1000) ; clf;
%    %imagesc(vl_imsc(adagradG)) ; axis image ;
%    drawnow ;
 end
end % end loop over maxNumIterations

% final result
result.output = output;
result.energy = E ;
result.y0 = gather(y0) ;
result.y = gather(y) ;
result.opts = opts ;
result.err = E(1,end) ;

% --------------------------------------------------------------------
function res_ = nndistance_forward(ly, res, res_)
% --------------------------------------------------------------------
res_.x = nndistance(res.x, ly.w, ly.mask) ;

% --------------------------------------------------------------------
function res = nndistance_backward(ly, res, res_)
% --------------------------------------------------------------------
res.dzdx = nndistance(res.x, ly.w, ly.mask, res_.dzdx) ;

% --------------------------------------------------------------------
function y = nndistance(x,w,mask,dzdy)
% --------------------------------------------------------------------
if nargin <= 3
  d = x - w ;
  y = sum(sum(sum(sum(d.*d.*mask)))) ;
else
  y = dzdy * 2 * (x - w) .* mask ;
end

% --------------------------------------------------------------------
function res_ = nninner_forward(ly, res, res_)
% --------------------------------------------------------------------
res_.x = nninner(res.x, ly.w) ;

% --------------------------------------------------------------------
function res = nninner_backward(ly, res, res_)
% --------------------------------------------------------------------
res.dzdx = nninner(res.x, ly.w, res_.dzdx) ;

% --------------------------------------------------------------------
function y = nninner(x,w,dzdy)
% --------------------------------------------------------------------
if nargin <= 2
  y = sum(sum(sum(sum(w.*x)))) ;
else
  y = dzdy * w ;
end

% --------------------------------------------------------------------
function change_current_figure(figid)
% --------------------------------------------------------------------
try
  set(0, 'CurrentFigure', figid);
catch
  figure(figid);
end

% -------------------------------------------------------------------------
function net = set_leak_factor(net, leak)
% -------------------------------------------------------------------------
for l = 1:numel(net.layers)
  if strcmp(net.layers{l}.type,'relu')
    net.layers{l}.leak = leak ;
  end
end

