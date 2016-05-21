function exp = exp_noregularizer(pretend, subset)
% do all basic visualizations but without regularizers
% rsync -rav shallowtunnel:~/src/ijcv/data/stats data/

run dg_setup

models = {...
  'imagenet-caffe-alex', ...
         } ;
%models = {'imagenet-vgg-verydeep-16'} ;

if nargin == 0, pretend = false ; end

exp = {} ;

for i = 1:numel(models)
  modelPath = sprintf('matconvnet/data/models/%s.mat',models{i}) ;
  statsPath = sprintf('data/stats/%s-stats.mat',models{i}) ;

  exp = horzcat(exp, setup_enhance(modelPath, statsPath)) ;
  exp = horzcat(exp, setup_category(modelPath, statsPath)) ;
  exp = horzcat(exp, setup_maximization(modelPath, statsPath)) ;
  exp = horzcat(exp, setup_inversion(modelPath, statsPath, 'mode', 'val', ...
                                                'onlyConvolutional', false)) ;
end

if ~pretend
  if nargin < 2, subset = 1:numel(exp) ; end
  parfor f=subset
    try
      rng(0) ;
      switch exp{f}.mode
        case 'maximization', exp_maximization(exp{f}) ;
        case 'enhance', exp_enhance(exp{f}) ;
        case 'inversion', exp_inversion(exp{f}) ;
      end
      close all;
    catch e
      warning('******** ERROR') ;
      warning(sprintf('******* Exp %d',f)) ;
      fprintf('%s\n', getReport(e)) ;
    end
  end
end

% --------------------------------------------------------------------
function exp = setup_maximization(modelPath, statsPath)
% --------------------------------------------------------------------

prefix = 'view160-noreg';
exp0.mode = 'maximization' ;
exp0.modelPath = modelPath ;
exp0.statsPath = statsPath ;
exp0.layer = 1 ;
exp0.filters = 1 ;
exp0.resultPath = '' ;
exp0.useJitter = false; %true ;
exp0.leak = 0.05 ;

exp = {} ;
net = load(modelPath) ;
net = vl_simplenn_tidy(net);
info = vl_simplenn_display(net) ;
[~,name] = fileparts(modelPath) ;

for l = 1:numel(net.layers)-1
  exp0.layer = l ;
  if ~strcmp(net.layers{l}.type, 'conv'), continue ; end
  numFilters = info.dataSize(3,l+1) ;
  if numFilters > 512
    for f = 1:2
      exp0.filters = f ;
      exp0.resultPath = sprintf('data/%s/%s-l%02d/f%03d', prefix, name, l, f) ;
      exp{end+1} = exp0 ;
    end
  else
    exp0.filters = 1:25
    exp0.resultPath = sprintf('data/%s/%s-l%02d-mosaic', prefix, name, l) ;
    exp{end+1} = exp0 ;
  end
end

% --------------------------------------------------------------------
function exp = setup_category(modelPath, statsPath)
% --------------------------------------------------------------------

prefix = 'category160-noreg';
exp0.mode = 'maximization' ;
exp0.modelPath = modelPath ;
exp0.statsPath = statsPath ;
exp0.layer = [] ;
exp0.filters = 1 ;
exp0.resultPath = '' ;
exp0.useJitter = false; %true ;
exp0.leak = 0.01 ;
exp0.lambdaAlpha = 0;
exp0.lambdaBeta = 0;
exp0.x0_max = inf;

exp = {} ;
net = load(modelPath) ;
net = vl_simplenn_tidy(net);
info = vl_simplenn_display(net) ;
[~,name] = fileparts(modelPath) ;

classes = {...
  421, 'black swan', ...
  420, 'goose', ...
  500, 'tree frog', ...
  993, 'cheeseburger', ...
  996, 'coffee mug', ...
  930, 'comic book', ...
  558, 'vending machine'} ;

for i = 1:2:numel(classes)
  j = find(cellfun(@(x)isequal(x,1),...
                   strfind(net.meta.classes.description, classes{i+1}))) ;
  if isempty(j), keyboard; end
  classes{i} = j ;
end

exp0.layer = numel(net.layers) - 1 ;
exp0.imageSize = net.meta.normalization.imageSize ;
for i = 1:2:numel(classes)
  exp0.filters = classes{i} ;
  exp0.resultPath = sprintf('data/%s/%s/%s', prefix, name, classes{i+1}) ;
  exp{end+1} = exp0 ;
end

% --------------------------------------------------------------------
function exp = setup_enhance(modelPath, statsPath)
% --------------------------------------------------------------------

prefix = 'enhance160-noreg';
exp0.mode = 'enhance' ;
exp0.modelPath = modelPath ;
exp0.statsPath = statsPath ;
exp0.layer = 1 ;
exp0.filters = 1 ;
exp0.resultPath = '' ;
exp0.useJitter = false; %true ;
exp0.lambdaAlpha = 0;
exp0.lambdaBeta = 0;
exp0.x0_max = inf;

exp = {} ;
net = load(modelPath) ;
net = vl_simplenn_tidy(net);
info = vl_simplenn_display(net) ;
[~,name] = fileparts(modelPath) ;

images = {...
  'red-fox.jpg', ...
  'gong.jpg', ...
  'fish.jpg', ...
  'abstract.jpg', ...
  'val/ILSVRC2012_val_00000013.JPEG', ...
  'val/ILSVRC2012_val_00000043.JPEG', ...
  'val/ILSVRC2012_val_00000024.JPEG'} ;

exp0.imageSize = net.meta.normalization.imageSize ;
for i = 1:numel(images)
  for l = 1:numel(net.layers)-1
    exp0.layer = l ;
    exp0.resultPath = sprintf('data/%s/%s/%s/%s', ...
      prefix, name, sprintf('layer%d', exp0.layer), images{i}) ;
    exp0.initialImage = sprintf('data/pics/%s', images{i}) ;
    exp{end+1} = exp0 ;
  end
end

% --------------------------------------------------------------------
function exp = setup_inversion(modelPath, statsPath, varargin)
% --------------------------------------------------------------------
opts.mode = 'normal' ;
opts.onlyConvolutional = true ;
opts.group = 1 ;
opts = vl_argparse(opts, varargin) ;

exp0.mode = 'inversion' ;
exp0.modelPath = modelPath ;
exp0.statsPath = statsPath ;
exp0.layer = 1 ;
exp0.resultPath = '' ;
exp0.useJitter = false; %true ;
exp0.objectiveWeight = 30 ;
exp0.filterGroup = [] ;
exp0.filterNeigh = [] ;
exp0.numRepeats = 1 ;
exp0.beta = 2 ;

exp0.lambdaAlpha = 0;
exp0.lambdaBeta = 0;
exp0.x0_max = inf;

moreImages = {...
  'red-fox.jpg', ...
  'gong.jpg', ...
  'fish.jpg', ...
  'abstract.jpg', ...
  'val/ILSVRC2012_val_00000013.JPEG', ...
  'val/ILSVRC2012_val_00000043.JPEG', ...
  'val/ILSVRC2012_val_00000024.JPEG'} ;

images = {...
  'red-fox.jpg', ...
  'val/ILSVRC2012_val_00000043.JPEG', ...
  'val/ILSVRC2012_val_00000024.JPEG'} ;

images = moreImages ;

switch opts.mode
  case 'normal'
    prefix = 'inversion160';
    images = moreImages ;
  case 'val'
    prefix = 'inversion-val160-noreg';
    images = dir('data/pics/val/*.JPEG') ;
    images = fullfile('val', {images.name}) ;
  case 'multi'
    prefix = 'inversion-multi-160';
    exp0.numRepeats = 4 ;
  case 'neigh'
    prefix = 'inversion-neigh-160';
    exp0.filterNeigh = 5  ;
  case 'group'
    prefix = sprintf('inversion-group-160-grp%d', opts.group) ;
    exp0.filterGroup = opts.group ;
    images = {'red-fox.jpg', 'gong.jpg', 'abstract.jpg'} ;
end

exp = {} ;
net = load(modelPath) ;
net = vl_simplenn_tidy(net);
info = vl_simplenn_display(net) ;
[~,name] = fileparts(modelPath) ;

exp0.imageSize = net.meta.normalization.imageSize ;
for i = 1:numel(images)
  [~,base,ext] = fileparts(images{i}) ;
  for l = 1:numel(net.layers)-1
    if opts.onlyConvolutional && ...
        ~strcmp(net.layers{l}.type, 'conv')
      continue ;
    end
    if ~isempty(exp0.filterNeigh)
      if info.dataSize(1,l+1) < exp0.filterNeigh
        continue ;
      end
    end
    for s = [1]% 100 300 20]
      exp0.objectiveWeight = s ;
      exp0.layer = l ;
      exp0.resultPath = sprintf('data/%s/%s/%s/%s', ...
        prefix, name,  base, ...
        sprintf('layer%02d-str%04.1f', ...
        exp0.layer, exp0.objectiveWeight)) ;
      exp0.initialImage = sprintf('data/pics/%s', images{i}) ;
      exp{end+1} = exp0 ;
    end
  end
end

% --------------------------------------------------------------------
function exp = setup_inversion_variants(modelPath, statsPath)
% --------------------------------------------------------------------

prefix = 'inversion-variants-160';
exp0.mode = 'inversion' ;
exp0.modelPath = modelPath ;
exp0.statsPath = statsPath ;
exp0.layer = 10 ; % conv5 is 13 in AlexNet
exp0.filters = 1 ;
exp0.resultPath = '' ;
exp0.useJitter = true ;

exp0.objectiveWeight = 20 ;
exp0.filterGroup = [] ;
exp0.filterNeigh = [] ;
exp0.numRepeats = 1 ;
exp0.beta = 1.5 ;

exp0.lambdaAlpha = 0;
exp0.lambdaBeta = 0;
exp0.x0_max = inf;

exp = {} ;
net = load(modelPath) ;
net = vl_simplenn_tidy(net);
info = vl_simplenn_display(net) ;
[~,name] = fileparts(modelPath) ;

images = {...
  'val/ILSVRC2012_val_00000043.JPEG'} ;

exp0.imageSize = net.meta.normalization.imageSize ;

for i = 1:numel(images)
  [~,base,ext] = fileparts(images{i}) ;
  for jitter = [true false]
    for beta = [1 1.5 2]
      [~,base,ext] = fileparts(images{i}) ;
      exp0.useJitter = jitter ;
      exp0.beta = beta ;
      exp0.resultPath = sprintf('data/%s/%s/%s/%s', ...
        prefix, name, base, ...
        sprintf('jitter%d-beta%04.1f', ...
        exp0.useJitter, exp0.beta)) ;
      exp0.initialImage = sprintf('data/pics/%s', images{i}) ;
      exp{end+1} = exp0 ;
    end
  end
end

% --------------------------------------------------------------------
function exp = setup_category_variants(modelPath, statsPath)
% --------------------------------------------------------------------

prefix = 'category-variants-160';
exp0.mode = 'maximization' ;
exp0.modelPath = modelPath ;
exp0.statsPath = statsPath ;
exp0.layer = 2 ;
exp0.filters = 1 ;
exp0.resultPath = '' ;
exp0.useJitter = true ;

exp0.beta = 2 ;

exp0.lambdaAlpha = 0;
exp0.lambdaBeta = 0;
exp0.x0_max = inf;

exp = {} ;
net = load(modelPath) ;
net = vl_simplenn_tidy(net);
info = vl_simplenn_display(net) ;
[~,name] = fileparts(modelPath) ;

classes = {...
  500, 'tree frog'} ;

for i = 1:2:numel(classes)
  j = find(cellfun(@(x)isequal(x,1),strfind(net.meta.classes.description, classes{i+1}))) ;
  if isempty(j), keyboard; end
  classes{i} = j ;
end

exp0.layer = numel(net.layers) - 1 ;
exp0.imageSize = net.meta.normalization.imageSize ;
for i = 1:2:numel(classes)
  for jitter = [true false]
    for beta = [1 1.5 2]
      exp0.filters = classes{i} ;
      exp0.useJitter = jitter ;
      exp0.beta = beta ;
      exp0.resultPath = sprintf('data/%s/%s/%s/%s', ...
        prefix, name, classes{i+1}, ...
        sprintf('jitter%d-beta%04.1f', ...
        exp0.useJitter, exp0.beta)) ;
      exp{end+1} = exp0 ;
    end
  end
end



