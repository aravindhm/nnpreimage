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