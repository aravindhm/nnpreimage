function exp_enhance(exp, stats)

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

% increase stride of first fc layer if applied in a
% fully-convolutional model on a large image
%l = find(cellfun(@(x) strcmp('fc6', x.name), net.layers)) ;
%if ~isempty(l)
%  net.layers{l}.stride = [160 160] ;
%end

% -------------------------------------------------------------------------
%                                                      Inversion parameters
% -------------------------------------------------------------------------

opts = get_preimage_opts('enhance', net, exp);


% -------------------------------------------------------------------------
% How much space for a filter in the grid
% -------------------------------------------------------------------------

% Get reference signal
im = single(imread(exp.initialImage)) ;
im = imresize(im, opts.imageSize(1:2)) ;
res = vl_simplenn(net, im)
ref = res(end).x ;
ref = max(ref, 0) ;
ref = ref / norm(ref(:))^2 ;


% -------------------------------------------------------------------------
% Run inversion code
% -------------------------------------------------------------------------

res = nnpreimage(net, ref, opts, 'initialImage', im) ;

imwrite(vl_imsc(gather(res.output{end})), [exp.resultPath, '.png']) ;

% -------------------------------------------------------------------------
function net = set_leak_factor(net, leak)
% -------------------------------------------------------------------------
for l = 1:numel(net.layers)
  if strcmp(net.layers{l}.type,'relu')
    net.layers{l}.leak = leak ;
  end
end

% -------------------------------------------------------------------------
function imageSize = find_image_size(net, ref)
% -------------------------------------------------------------------------
for i=1:4096
  info = vl_simplenn_display(net, 'inputSize', [i i 3 1]);
  if info.dataSize(1,end) > size(ref,1)
    break ;
  end
end
info = vl_simplenn_display(net, 'inputSize', [i-1 i-1 3 1]);
imageSize = [info.dataSize(:,1) ; 1]' ;
