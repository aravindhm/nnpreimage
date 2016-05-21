function opts_all = get_preimage_opts(mode, net, exp)
% Obtain the options prescribed in the paper. 
% Use the to do this computation
% Override them with specifications in exp if present

switch mode
case 'maximization'
  
  def = get_defaults();

  opts.L.beta = expOrDef(exp, def, 'alpha') ;
  opts.TV.beta = expOrDef(exp, def, 'beta') ;
  
  def = get_defaults('alpha', opts.L.beta, 'beta', opts.TV.beta);

  opts.objective = 'inner' ;

  if isfield(exp, 'numRepeats')
    opts.numRepeats = exp.numRepeats ;
  else
    opts.numRepeats = 1;
  end

  if ~isfield(exp, 'useJitter') || exp.useJitter

    info = vl_simplenn_display(net) ;
    rf = info.receptiveFieldSize(1,end) ;
    stride = info.receptiveFieldStride(1,end) ;
    
    opts.jitterAmount = max(round(stride / 4),1) ;

  else

    opts.jitterAmount = 1;

  end

  opts.normalize = @(im) bsxfun(@minus, im, net.meta.normalization.averageImage) ;
  opts.denormalize =  @(im) bsxfun(@plus, im, net.meta.normalization.averageImage) ;

  opts.bound = expOrDef(exp, def, 'x0_max') ;

  opts.L.lambda = expOrDef(exp, def, 'lambdaAlpha') ;

  opts.TV.lambda = expOrDef(exp, def, 'lambdaBeta') ;

  opts.learningRate = def.fastLR ;

  if ~isfield(exp, 'leak')
    opts.leak = 0.05;
  else
    opts.leak = exp.leak;
  end

  %opts2 = opts;
  %opts2.leak = 0;
  %opts2.learningRate = def.slowLR;

  %opts_all = {opts, opts2};


  opts_all = {opts};











case 'inversion'

  def = get_defaults();

  opts.L.beta = expOrDef(exp, def, 'alpha') ;
  opts.TV.beta = expOrDef(exp, def, 'beta') ;
  
  def = get_defaults('alpha', opts.L.beta, 'beta', opts.TV.beta);
  
  opts.objective = 'l2' ;

  opts.objectiveWeight = expOrDef(exp, def, 'objectiveWeight') ;

  opts.bound = expOrDef(exp, def, 'x0_max') ;

  opts.L.lambda = expOrDef(exp, def, 'lambdaAlpha') ;

  opts.TV.lambda = expOrDef(exp, def, 'lambdaBeta') ;

  opts.learningRate = def.fastLR ;

  opts.imageSize = net.meta.normalization.imageSize;

  if isfield(exp, 'numRepeats')
    opts.numRepeats = exp.numRepeats ;
  else
    opts.numRepeats = 1;
  end

  info = vl_simplenn_display(net, 'inputSize', [opts.imageSize(1:2), 3, opts.numRepeats]) ;
  if ~isfield(exp, 'useJitter') || exp.useJitter

    stride = info.receptiveFieldStride(1,end) ;
    
    opts.jitterAmount = max(round(stride / 4),1) ;

  else

    opts.jitterAmount = 1;

  end


  opts.normalize = @(im) bsxfun(@minus, im, net.meta.normalization.averageImage) ;
  opts.denormalize =  @(im) bsxfun(@plus, im, net.meta.normalization.averageImage) ;

  
  ref_size = [info.dataSize(1:3, end)', 1];
  
  % Mask
  mask = zeros(ref_size, 'single') ;
  sy = 1:ref_size(1) ;
  sx = 1:ref_size(2) ;
  sf = 1:ref_size(3) ;
  if ~isempty(exp.filterGroup)
    if exp.filterGroup == 1
      sf = vl_colsubset(sf, 0.5, 'beginning') ;
    elseif exp.filterGroup == 2 ;
      sf = vl_colsubset(sf, 0.5, 'ending') ;
    end
  end
  if ~isempty(exp.filterNeigh) ;
    nx = min(exp.filterNeigh, ref_size(2)) ;
    ny = min(exp.filterNeigh, ref_size(1)) ;
    sx = (0:nx-1) + ceil((ref_size(2)-nx+1)/2) ;
    sy = (0:ny-1) + ceil((ref_size(1)-ny+1)/2) ;
  end
  mask(sy,sx,sf) = 1 ;
  opts.mask = mask ;

 
  opts2 = opts;
  opts2.jitterAmount = 1;
  opts2.learningRate = def.slowLR;

  opts_all = {opts, opts2};




case 'enhance'

  def = get_defaults();
  
  opts.L.beta = expOrDef(exp, def, 'alpha') ;
  opts.TV.beta = expOrDef(exp, def, 'beta') ;
  
  def = get_defaults('alpha', opts.L.beta, 'beta', opts.TV.beta);

  opts.objective = 'inner' ;

  if ~isfield(exp, 'useJitter') || exp.useJitter
    info = vl_simplenn_display(net) ;
    rf = info.receptiveFieldSize(1,end) ;
    stride = info.receptiveFieldStride(1,end) ;
    opts.jitterAmount = max(round(stride / 4),1) ;
  else
    opts.jitterAmount = 1;
  end

  opts.normalize = @(im) bsxfun(@minus, im, net.meta.normalization.averageImage) ;
  opts.denormalize =  @(im) bsxfun(@plus, im, net.meta.normalization.averageImage) ;

  opts.bound = expOrDef(exp, def, 'x0_max') ;

  opts.L.lambda = expOrDef(exp, def, 'lambdaAlpha') ;

  opts.TV.lambda = expOrDef(exp, def, 'lambdaBeta') ;

  opts.learningRate = def.fastLR ;

  opts.imageSize = net.meta.normalization.imageSize ;

  opts_all  = opts;

otherwise
  error('Unrecognized mode for preimaging');

end

function val = expOrDef(exp, def, field)
if(~isfield(exp, field))
  val = def.(field);
else
  val = exp.(field);
end
