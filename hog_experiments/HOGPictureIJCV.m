function im = HOGPictureIJCV(w, bs)
% Make picture of all HOG weights.
% im = HOGpicture(w, bs)

net = hog_net(8);

ly.type = 'custom';
ly.forward = @slice_hog_forward;
ly.backward = @slice_hog_backward;
net.layers{end+1} = ly;

net = vl_simplenn_tidy(net);

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

sz = size(w);
opts.imageSize = [(sz(1:2) + 2)*8, 1];

res = nnpreimage(net, w, opts);
im = vl_imsc(res.output{end});

% --------------------------------------------------
% Define the slice layers
% --------------------------------------------------

function res_ = slice_hog_forward(ly, res, res_)
res_.x = res.x(2:end-1,2:end-1,:);

function res = slice_hog_backward(ly, res, res_)
res.dzdx = zeros(size(res.x), 'single');
res.dzdx(2:end-1,2:end-1,:) = res_.dzdx;


% --------------------------------------------------
% get_defaults function pasted here for completeness
% --------------------------------------------------

function def = get_defaults(varargin)
opts.alpha = 6 ;
opts.beta = 2 ;
opts.objectiveWeight = 1 ;
opts.x0_max = 160 ;
opts.x0_sigma = 80 ;
opts = vl_argparse(opts, varargin) ;

lambdaAlpha = (1/opts.x0_sigma)^opts.alpha ;
lambdaBeta = (6.5/opts.x0_sigma)^opts.beta ;
LR = 0.05 * opts.x0_sigma^2 / opts.alpha ;

fastLR = LR * ones(1,300) ;
slowLR = 0.1 * LR * ones(1, 50) ;

def.x0_max = opts.x0_max ;
def.x0_sigma = opts.x0_sigma ;
def.alpha = opts.alpha ;
def.beta = opts.beta ;
def.lambdaAlpha = lambdaAlpha ;
def.lambdaBeta = lambdaBeta ;
def.fastLR = fastLR ;
def.slowLR = slowLR ;
def.objectiveWeight = opts.objectiveWeight ;
