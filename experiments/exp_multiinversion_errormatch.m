function exp = exp_multiinversion_errormatch(pretend, subset)
% do all basic visualizations
% rsync -rav shallowtunnel:~/src/ijcv/data/stats data/

run dg_setup

models = {...
  'imagenet-caffe-alex', ...
  'imagenet-vgg-verydeep-16', ...
         } ;
%models = {'imagenet-vgg-verydeep-16'} ;

if nargin == 0, pretend = false ; end

exp = {} ;

for i = 1:numel(models)
  modelPath = sprintf('matconvnet/data/models/%s.mat',models{i}) ;
  statsPath = sprintf('data/stats/%s-stats.mat',models{i}) ;

  exp = horzcat(exp, setup_inversion(modelPath, statsPath, 'onlyConvolutional', false, 'mode', 'multi-val')) ;

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
    catch e
      warning('******** ERROR') ;
      warning(sprintf('******* Exp %d',f)) ;
      fprintf('%s\n', getReport(e)) ;
    end
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
exp0.useJitter = true ;
exp0.objectiveWeight = 30 ;
exp0.filterGroup = [] ;
exp0.filterNeigh = [] ;
exp0.numRepeats = 1 ;
exp0.beta = 2 ;

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

