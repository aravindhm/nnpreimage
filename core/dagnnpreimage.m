function result = dagnnpreimage(net, ref, varargin)
%DAGNNPREIMAGE  Compute the pre-image of a CNN in dagnn format

if(~isa(net, 'dagnn.DagNN'))
  net = dagnn.DagNN.fromSimpleNN(net);
end

def = get_defaults() ;

t = net.getInputs();
opts.inputVar = t{1};
t = net.getOutputs();
opts.outputVar = t{1};

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
opts.momentum = 0.9 ;
opts.numRepeats = 1 ;
opts.normalize = @(x) x ;
opts.denormalize = @(x) x ;
opts.useAdagrad = true ;
opts.leak = 0;
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
if(numel(opts.imageSize) == 3)
  x0_size = cat(2, opts.imageSize, opts.numRepeats) ;
elseif(numel(opts.imageSize) == 2)
  x0_size = cat(2, opts.imageSize, 3, opts.numRepeats) ;
elseif(numel(opts.imageSize) == 4)
  x0_size = opts.imageSize;
  x0_size(4) = opts.numRepeats;
end

x0jit_size = x0_size ;
x0jit_size(1:2) = x0jit_size(1:2) + opts.jitterAmount - 1 ;
x = 2*rand(x0jit_size, 'single')-1 ;
for k = 1:size(x,4)
  t = x(:,:,:,k) ;
  m = quantile(t(:),.95) ;
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

% --------------------------------------------------------------------
%                                                      Prepare network
% --------------------------------------------------------------------

if(opts.leak > 0)
  net = set_leak_factor(net, opts.leak);
end

% We'll be visualizing the features being inverted.
net.vars(net.getVarIndex(opts.outputVar)).precious = true;

% First add the appropriate loss layer

derOutputs = {};
netInputs{1} = opts.inputVar;
netInputs{2} = [];

layer_num = numel(net.layers) ;
switch opts.objective
  case 'l2'

    lossLayer = L2Loss();
    % y0 = y0;

    layer_name = 'loss_';
    net.addLayer(layer_name, lossLayer, ...
           {opts.outputVar, ['target_', opts.outputVar], ['mask_', opts.outputVar]},...
           layer_name, {});

    netInputs{end+1} = ['target_', opts.outputVar];
    netInputs{end+1} = y0;

    netInputs{end+1} = ['mask_', opts.outputVar];
    netInputs{end+1} = mask;

    derOutputs{end+1} = layer_name;
    derOutputs{end+1} = opts.objectiveWeight;

  case 'inner'

    lossLayer = InnerProductLoss();
    y0 = -y0 .* mask;

    layer_name = 'loss_';
    net.addLayer(layer_name, lossLayer, ...
       {opts.outputVar, ['target_', opts.outputVar]},...
       layer_name, {});

    netInputs{end+1} = ['target_', opts.outputVar];
    netInputs{end+1} = y0;

    derOutputs{end+1} = layer_name;
    derOutputs{end+1} = opts.objectiveWeight;

  otherwise
    error('unknown opts.objective') ;
end

net.vars(net.getVarIndex('loss_')).precious = 1;

% Now add the jitter layer right between the input and the first layer in the dag
if opts.jitterAmount > 1

  layer_name = ['jitter_', opts.inputVar];

  % First I find all the layers that depend on the input var and make them instead 
  % depend on jitter_ input var [same as layer_name above]
  for i = 1:numel(net.layers)
    layer = net.layers(i);
    layer_inputs = layer.inputs;
    for j = 1:numel(layer_inputs)
      if(strcmp(layer_inputs{j}, opts.inputVar))
        layer_inputs{j} = ['jitter_', opts.inputVar];
      end
    end
    net.setLayerInputs(layer.name, layer_inputs);
  end

  net.rebuild();

  jitterLayer = JitterLayer('jitterAmount', opts.jitterAmount);
  net.addLayer(layer_name, jitterLayer, opts.inputVar, layer_name, {});

end

% Next add a layer for TVnorm and another for L2Norm of the image
if opts.TV.lambda > 0
   tv_layer = TVNorm('beta', opts.TV.beta);
   layer_name = 'tvnorm_';
   net.addLayer(layer_name, tv_layer, opts.inputVar, layer_name, {});

   derOutputs{end+1} = layer_name;
   derOutputs{end+1} = opts.TV.lambda;

   net.vars(net.getVarIndex('tvnorm_')).precious = 1;
end

if opts.L.lambda > 0
   l2_layer = L2Norm('beta', opts.L.beta);
   layer_name = 'l2norm_';
   net.addLayer(layer_name, l2_layer, opts.inputVar, layer_name, {});

   derOutputs{end+1} = layer_name;
   derOutputs{end+1} = opts.L.lambda;

   net.vars(net.getVarIndex('l2norm_')).precious = 1;
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

  % backprop
  netInputs{2} = x;
  net.eval(netInputs, derOutputs);

  % current reconstruction
  y = net.vars(net.getVarIndex(opts.outputVar)).value;

  % loss and corresponding gradient
  E(1,t) = net.vars(net.getVarIndex('loss_')).value * opts.objectiveWeight ;
  dd = net.vars(net.getVarIndex(opts.inputVar)).der;

  % regulariser and corresponding gradient

  if opts.TV.lambda > 0
    E(2,t) = opts.TV.lambda * net.vars(net.getVarIndex('tvnorm_')).value ;
  else
    E(2,t) = 0;
  end

  if opts.L.lambda > 0
    E(3,t) = opts.L.lambda * net.vars(net.getVarIndex('l2norm_')).value ;
  else
    E(3,t) = 0;
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
  gradient = dd ;
  %  gradient = max(min(gradient,10/lr),-10/lr) ;

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
    output{end+1} = opts.denormalize(x(sy,sx,:,:)) ;

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
%    a = squeeze(y0(1:len)) ;
%    b = squeeze(y(1:len)) ;
%    plot(1:len,a,'b'); hold on ;
%    plot(len+1:2*len,b-a, 'r');
%    legend('\Phi_0', '|\Phi-\Phi_0|') ;
%    title(sprintf('reconstructed variable %s', ...
%      opts.outputVar)) ;
%    legend('ref', 'delta') ;
%
%    subplot(4,2,4) ;
%    hist(x(:),100) ;
%    grid on ;
%    title('histogram of x') ;
%
%    subplot(4,2,6) ;
%    hist(y(:),100) ;
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
result.y0 = y0 ;
result.y = y ;
result.opts = opts ;
result.err = E(1,end) ;

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
  if isa(net.layers{l}.block, 'dagnn.ReLU')
    net.layers{l}.block.leak = leak ;
  end
end

