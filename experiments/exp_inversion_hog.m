function exp_inversion_hog(exp)

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

def = get_defaults('objectiveWeight', exp.objectiveWeight) ;

opts.objective = 'l2' ;
opts.objectiveWeight = def.objectiveWeight ;
opts.bound = def.x0_max ;
opts.L.beta = def.alpha ;
opts.L.lambda = def.lambdaAlpha ;
opts.TV.beta = def.beta ;
opts.TV.lambda = def.lambdaBeta ;
opts.learningRate = def.fastLR ;
opts.normalize = @(im) im ;
opts.denormalize =  @(im) im ;

% -------------------------------------------------------------------------
%                                                           Filter geometry
% -------------------------------------------------------------------------

% Get reference signal
if(~isfield(exp, 'ref'))
  im = imread(exp.initialImage) ;
  if size(im,3) > 1, im = rgb2gray(im) ; end
  im = single(im) ;
  im = opts.normalize(im) ;
  res = vl_simplenn(net, im)
  ref = res(end).x ;
else
  % Find the image size
  ref = exp.ref;
  %for i=50:400
  %  fprintf(1, '%d\n', i);
  %  for j=50:i
  %    im = randn(i, j, 'single');
  %    res = vl_simplenn(net, im);
  %    hog = res(end).x(2:end-1,2:end-1,:,:);
  %    if(size(hog,1) == size(ref,1) && size(hog,2) == size(ref,2))
  %      break;
  %    end
  %  end
  %end
  im = randn(140,80,'single');
end

% Normalize
n = norm(ref(:)) ;
opts.objectiveWeight = exp.objectiveWeight / n^2 ;

% -------------------------------------------------------------------------
%                                                        Run pre-image code
% -------------------------------------------------------------------------

switch exp.model
  case {'ihog', 'hog'}
    hog = features(double(repmat(im,[1 1 3])),8) ;
end

switch exp.model
  case 'ihog'
    imrec = invertHOG(hog) ;
    figure(1) ; clf ;
    subplot(1,2,1);imagesc(im) ;
    subplot(1,2,2);imagesc(imrec) ;
    colormap gray ;
    drawnow ;

  otherwise
    res = nnpreimage(net, ref, opts, ...
                     'objectiveWeight', exp.objectiveWeight / n^2, ...
                     'imageSize', size(im)) ;
    imrec = double(gather(res.output{end})) ;
end

switch exp.model
  case 'ihog'
    errorCNN = 0 ;
  otherwise
    errorCNN = res.err / exp.objectiveWeight ;
end

switch exp.model
  case {'ihog', 'hog'}
    % for hog and ihog fair comparison using original HOG just in
    % case
    hog_ = features(double(repmat(imrec,[1 1 3])),8) ;
    error = norm(hog_(:) - hog(:))^2 / norm(hog(:))^2 ;
  otherwise
    error = errorCNN ;
end
save([exp.resultPath, '.mat'], 'error', 'errorCNN') ;

out = vl_imsc(imrec) ;
imwrite(out, [exp.resultPath, '.png']) ;
