function exp_inversion(exp, stats)

[modelDir,modelName] = fileparts(exp.modelPath) ;
[resultDir,resultName] = fileparts(exp.resultPath) ;

if exist([exp.resultPath '.png'])
  fprintf('%s found skipping\n', exp.resultPath) ;
  return ;
end
vl_xmkdir(resultDir) ;

% load a network
net = load(exp.modelPath) ;
net = vl_simplenn_tidy(net);
net.meta.normalization.averageImage = mean(mean(net.meta.normalization.averageImage,1),2) ;
net.layers = net.layers(1:exp.layer) ;


% -------------------------------------------------------------------------
%                                                      Inversion parameters
% -------------------------------------------------------------------------

opts = get_preimage_opts('inversion', net, exp);
if(isfield(exp, 'gpu') && exp.gpu)
  opts.gpu = true;
end

% IMP: Note that opts.objectiveWeight is going to be normalized after the 
% reference signal is computed.

% -------------------------------------------------------------------------
%                                                          Reference Signal
% -------------------------------------------------------------------------

% Get reference signal
im = single(imread(exp.initialImage)) ;
im = imresize(im, opts{1}.imageSize(1:2)) ;
im = opts{1}.normalize(im) ;
res = vl_simplenn(net, im)
ref = res(end).x ;


% Normalize
for opts_i = 1:numel(opts)
  n = norm(ref(:) .* opts{opts_i}.mask(:)) ;
  opts{opts_i}.objectiveWeight = opts{opts_i}.objectiveWeight / n^2 ;
end

% -------------------------------------------------------------------------
%                                                        Run pre-image code
% -------------------------------------------------------------------------

% Get a first version with jitter
for opts_i = 1:numel(opts)
  if(opts_i == 1)
    res = nnpreimage(net, ref, opts{opts_i});
  else
    res = nnpreimage(net, ref, opts{opts_i}, ...
      'initialImage', res.output{end});
  end
end

if size(res.output{end},4) > 1
  out = vl_imarraysc(gather(res.output{end})) ;
else
  out = vl_imsc(gather(res.output{end})) ;
end
imwrite(out, [exp.resultPath, '.png']) ;

error = res.err / exp.objectiveWeight ;
save([exp.resultPath, '.mat'], 'error') ;
