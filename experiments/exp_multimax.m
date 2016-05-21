function exp = exp_multimax(pretend, subset)
% do all basic visualizations
% rsync -rav shallowtunnel:~/src/ijcv/data/stats data/

%models = {...
%  'imagenet-caffe-ref', ...
%  'imagenet-caffe-alex', ...
%  'imagenet-vgg-m', ....
%  'imagenet-vgg-verydeep-16', ...
%         } ;
models = {...
  'imagenet-caffe-alex', ...
  'imagenet-vgg-m', ....
  'imagenet-vgg-verydeep-16', ...
         } ;

if nargin == 0, pretend = false ; end

exp = {} ;

for i = 1:numel(models)
  modelPath = sprintf('networks/%s.mat',models{i}) ;
  statsPath = sprintf('networks/stats/%s-stats.mat',models{i}) ;

  more = strcmp(models{i}, 'imagenet-caffe-ref') || ...
         strcmp(models{i}, 'imagenet-caffe-alex') ;

%  exp = horzcat(exp, setup_enhance(modelPath, statsPath)) ;
%  exp = horzcat(exp, setup_category(modelPath, statsPath)) ;
%  exp = horzcat(exp, setup_maximization(modelPath, statsPath)) ;
  exp = horzcat(exp, setup_maximization(modelPath, statsPath, 'mode', 'multi')) ;
%  exp = horzcat(exp, setup_inversion(modelPath, statsPath, 'onlyConvolutional', ~more)) ;
%  exp = horzcat(exp, setup_inversion(modelPath, statsPath, 'mode', 'multi')) ;
%
  if ~more, continue ; end
%  exp = horzcat(exp, setup_inversion(modelPath, statsPath, 'mode', 'multi-val',...
%                                               'onlyConvolutional', false)) ;
%  exp = horzcat(exp, setup_inversion(modelPath, statsPath, 'mode', 'val', ...
%                                               'onlyConvolutional', false)) ;
%  exp = horzcat(exp, setup_inversion(modelPath, statsPath, 'mode', 'neigh', ...
%                                                 'onlyConvolutional', false)) ;
%  exp = horzcat(exp, setup_inversion(modelPath, statsPath, ...
%                                                'mode', 'group', ...
%                                                'group', 1, ...
%                                                'onlyConvolutional', false)) ;
%  exp = horzcat(exp, setup_inversion(modelPath, statsPath, ...
%                                                'mode', 'group', ...
%                                                'group', 2, ...
%                                                'onlyConvolutional', false)) ;
%  exp = horzcat(exp, setup_inversion_variants(modelPath, statsPath)) ;
%  exp = horzcat(exp, setup_category_variants(modelPath, statsPath)) ;
end

if ~pretend
  if nargin < 2, subset = 1:numel(exp) ; end
  for f=subset
      rng(0) ;
      switch exp{f}.mode
        case 'maximization', exp_maximization(exp{f}) ;
        case 'enhance', exp_enhance(exp{f}) ;
        case 'inversion', exp_inversion(exp{f}) ;
      end

  end
end

% --------------------------------------------------------------------
function exp = setup_maximization(modelPath, statsPath, varargin)
% --------------------------------------------------------------------

opts.mode = 'normal';
opts = vl_argparse(opts, varargin) ;

prefix = 'view160-noleak';
exp0.mode = 'maximization' ;
exp0.modelPath = modelPath ;
exp0.statsPath = statsPath ;
exp0.layer = 1 ;
exp0.filters = 1 ;
exp0.resultPath = '' ;
exp0.leak = 0;

switch opts.mode
  case 'normal'
    exp0.numRepeats = 1;

  case 'multi'
    prefix = 'view160-noleak-multi';
    exp0.numRepeats = 4;
    exp0.gpu = true;
 
  otherwise
    error('Invalid exp_maximization mode %s\n', opts.mode);
end

exp = {} ;
net = load(modelPath) ;
net = vl_simplenn_tidy(net);
info = vl_simplenn_display(net) ;
[~,name] = fileparts(modelPath) ;

switch opts.mode
  case 'normal'

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

  case 'multi'

    for l = ceil(numel(net.layers)/2):numel(net.layers)-7
      exp0.layer = l ;
      if ~strcmp(net.layers{l}.type, 'conv'), continue ; end
      numFilters = info.dataSize(3,l+1) ;
      selectedFilters = randsample(numFilters, 10);
      for f_no = 1:numel(selectedFilters)
          f = selectedFilters(f_no);
          exp0.filters = f ;
          exp0.resultPath = sprintf('data/%s/%s-l%02d/f%04d', prefix, name, l, f) ;
          exp{end+1} = exp0 ;
      end
    end
end

% --------------------------------------------------------------------
function exp = setup_category(modelPath, statsPath)
% --------------------------------------------------------------------

prefix = 'category160-noleak';
exp0.mode = 'maximization' ;
exp0.modelPath = modelPath ;
exp0.statsPath = statsPath ;
exp0.layer = [] ;
exp0.filters = 1 ;
exp0.resultPath = '' ;
exp0.leak = 0.0 ;

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

prefix = 'enhance160';
exp0.mode = 'enhance' ;
exp0.modelPath = modelPath ;
exp0.statsPath = statsPath ;
exp0.layer = 1 ;
exp0.filters = 1 ;
exp0.resultPath = '' ;

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

images = {'fish.jpg'};

for i = 1:numel(images)
  for l = [29, 36] %16:20%1:numel(net.layers)-1
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
exp0.filterGroup = [] ;
exp0.filterNeigh = [] ;

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
    prefix = 'inversion-val160';
    images = dir('data/pics/val/*.JPEG') ;
    images = fullfile('val', {images.name}) ;
  case 'multi'
    prefix = 'inversion-multi-160';
    exp0.numRepeats = 4 ;
  case 'multi-val'
    prefix = 'inversion-multi-val160';
    images = dir('data/pics/val/*.JPEG') ;
    images = fullfile('val', {images.name}) ;
    images = images(1:25);
    exp0.numRepeats = 25 ;
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
    for s = [1 100 300 20]
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
exp0.trackTarget = false ;
exp0.objectiveWeight = 20 ;
exp0.filterGroup = [] ;
exp0.filterNeigh = [] ;
exp0.numRepeats = 1 ;
exp0.beta = 1.5 ;

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

prefix = 'category-variants-160-noleak';
exp0.mode = 'maximization' ;
exp0.modelPath = modelPath ;
exp0.statsPath = statsPath ;
exp0.layer = 2 ;
exp0.filters = 1 ;
exp0.resultPath = '' ;
exp0.useJitter = true ;
exp0.beta = 2 ;
exp0.leak = 0;

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



