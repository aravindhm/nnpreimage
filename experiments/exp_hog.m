function exp = exp_hog(pretend, subset)
% do all basic visualizations
% rsync -rav shallowtunnel:~/src/ijcv/data/stats data/

addpath(genpath('ihog')) ;

%models = {...
%  'ihog', 'hog', 'hogb', 'dsift'} ;
models = {...
  'hog', 'hogb', 'dsift'} ;

if nargin == 0, pretend = false ; end

exp = {} ;

for i = 1:numel(models)
  %exp = horzcat(exp, setup_inversion(models{i})) ;
  %exp = horzcat(exp, setup_inversion(models{i}, 'mode', 'val')) ;
  if ~strcmp(models{i}, 'hog'), continue ; end
  exp = horzcat(exp, setup_model_maximization(...
        models{i}, '', ...
            '/media/data/aravindh/CVPR2015/objtree/data/vhog-inria-person/model.mat'));
  %exp = horzcat(exp, setup_inversion_variants(models{i})) ;
end

if ~pretend
  if nargin < 2, subset = 1:numel(exp) ; end
  for f=subset
    rng(0) ;
    switch exp{f}.mode
      case 'inversion'
        exp_inversion_hog(exp{f}) ;
      case 'model-viz'
        exp_hog_model_viz(exp{f}) ;
    end
  end
end

% --------------------------------------------------------------------
function exp = setup_inversion(model, varargin)
% --------------------------------------------------------------------
opts.mode = 'normal' ;
opts.modelPath = '';
opts = vl_argparse(opts, varargin) ;

exp0.mode = 'inversion' ;
exp0.model = model ;
exp0.resultPath = '' ;
exp0.objectiveWeight = 100 ;

images = {...
  'hoggle/hoggle-orig-1.jpg'} ;

switch opts.mode
  case 'normal'
    prefix = 'inversion160';
  case 'val'
    prefix = 'inversion-val160';
    images = dir('data/pics/val/*.JPEG') ;
    images = fullfile('val', {images.name}) ;
  case 'model'
    prefix = 'inversion-model160-hack2';
    hogModel = load(opts.modelPath);
    exp0.ref = reshape(hogModel.w, [hogModel.size(2), hogModel.size(1), hogModel.hogDimension]);
    exp0.ref = exp0.ref(:,:,[1:27]);
    exp0.ref = exp0.ref .* (exp0.ref > 0);
    exp0.objectiveWeight = 100;
    'check the reference'
    keyboard;
end

exp = {} ;
switch opts.mode
  case {'normal', 'val'}
    for i = 1:numel(images)
      [~,base,ext] = fileparts(images{i}) ;
      exp0.resultPath = sprintf('data/%s/%s/%s/res', ...
                            prefix, model,  base) ;
      exp0.initialImage = sprintf('data/pics/%s', images{i}) ;
      exp{end+1} = exp0 ;
    end
  case {'model'}
    [~,base,ext] = fileparts(opts.modelPath) ;
    exp0.resultPath = sprintf('data/%s/%s/%s/res', ...
                            prefix, model,  base) ;
    exp{end+1} = exp0;
end

% --------------------------------------------------------------------
function exp = setup_model_maximization(modelPath, statsPath, hogModelPath)
% --------------------------------------------------------------------

prefix = 'hogModelViz';
exp0.mode = 'model-viz' ;
exp0.modelPath = modelPath ;
exp0.statsPath = statsPath ;
exp0.resultPath = '' ;
exp0.leak = 0;
exp0.model = 'hog';

exp = {} ;

hogModel = load(hogModelPath);
t = reshape(hogModel.w, [hogModel.size(2), hogModel.size(1), hogModel.hogDimension]);
exp0.hogModel = t(:,:,[1:27]);

[hogModelDir, name] = fileparts(hogModelPath);
[~, name2] = fileparts(hogModelDir);
exp0.resultPath = sprintf('data/%s/%s-%s', prefix, name2, name) ;
exp{end+1} = exp0 ;

% --------------------------------------------------------------------
function exp = setup_inversion_variants(model, varargin)
% --------------------------------------------------------------------
opts.mode = 'normal' ;
opts = vl_argparse(opts, varargin) ;

exp0.mode = 'inversion' ;
exp0.model = model ;
exp0.resultPath = '' ;
exp0.objectiveWeight = 100 ;

strs = [1 20 100 300] ;

prefix = 'inversion-variants-160' ;
images = {...
  'hoggle/hoggle-orig-1.jpg'} ;

exp = {} ;
for i = 1:numel(images)
  for s = 1:numel(strs)
    [~,base,ext] = fileparts(images{i}) ;
    exp0.resultPath = sprintf('data/%s/%s/%s/str%04.1f', ...
                              prefix, model,  base, strs(s)) ;
    exp0.initialImage = sprintf('data/pics/%s', images{i}) ;
    exp0.objectiveWeight = strs(s) ;
    exp{end+1} = exp0 ;
  end
end
