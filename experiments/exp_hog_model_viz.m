function exp_hog_model_viz(exp)

[modelDir,modelName] = fileparts(exp.modelPath) ;
[resultDir,resultName] = fileparts(exp.resultPath) ;

if exist([exp.resultPath '.png'])
  fprintf('%s found skipping\n', exp.resultPath) ;
  return ;
end
vl_xmkdir(resultDir) ;

% load a network
switch exp.model
  case {'ihog', 'hog'}
    net = hog_net(8) ;
  case 'hogb' ;
    net = hog_net(8, 'bilinearOrientations', 1) ;
  case 'dsift' ;
    net = dsift_net(8) ;
end
net = vl_simplenn_tidy(net);

% -------------------------------------------------------------------------
%                                                      Inversion parameters
% -------------------------------------------------------------------------

%load(exp.statsPath, 'stats') ;
%innerWeight = (10000 / rf^2) / numel(exp.filters) ./ stats.layers(exp.layer+1).max ;

def = get_defaults() ;
opts.L.beta = def.alpha ;
opts.L.lambda = def.lambdaAlpha ;
opts.TV.beta = def.beta ;
opts.TV.lambda = def.lambdaBeta ;
opts.numRepeats = 1 ;
opts.jitterAmount = 1 ;
opts.normalize = @(x) x ;
opts.denormalize = @(x) x ;

opts.bound = def.x0_max ;
opts.learningRate = def.fastLR;

opts.objective = 'inner';

ref = exp.hogModel;
sz = size(ref);
opts.imageSize = [(sz(1:2) + 2)*8, 1];

ly.type = 'custom';
ly.forward = @slice_hog_forward;
ly.backward = @slice_hog_backward;
net.layers{end+1} = ly;

net = vl_simplenn_tidy(net);

img_max = opts.bound * randn(opts.imageSize, 'single');
res = vl_simplenn(net, img_max);
M = res(end).x(:)' * ref(:);
innerWeight = (10000 / max(opts.imageSize)^2) / M ;

% -------------------------------------------------------------------------
% Run inversion code
% -------------------------------------------------------------------------

res = nnpreimage(net, ref, opts) ;

if size(res.output{end},4) > 1
  out = vl_imarraysc(gather(res.output{end})) ;
else
  out = vl_imsc(gather(res.output{end})) ;
end
imwrite(out, [exp.resultPath, '.png']) ;

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

function res_ = slice_hog_forward(ly, res, res_)
res_.x = res.x(2:end-1,2:end-1,:);

function res = slice_hog_backward(ly, res, res_)
res.dzdx = zeros(size(res.x), 'single');
res.dzdx(2:end-1,2:end-1,:) = res_.dzdx;
