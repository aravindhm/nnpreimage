function exp_maximization(exp, stats)

[modelDir,modelName] = fileparts(exp.modelPath) ;
[resultDir,resultName] = fileparts(exp.resultPath) ;

if exist([exp.resultPath '.png'])
  fprintf('%s found skipping\n', exp.resultPath) ;
  return ;
end
vl_xmkdir(resultDir) ;

% load a network
net = load(exp.modelPath) ;

if(isfield(net, 'net')), net = net.net;  end

net = vl_simplenn_tidy(net);
net.meta.normalization.averageImage = mean(mean(net.meta.normalization.averageImage,1),2) ;
net.layers = net.layers(1:exp.layer) ;
info = vl_simplenn_display(net) ;
rf = info.receptiveFieldSize(1,end) ;
stride = info.receptiveFieldStride(1,end) ;

% -------------------------------------------------------------------------
%                                                      Inversion parameters
% -------------------------------------------------------------------------

load(exp.statsPath, 'stats') ;
innerWeight = (10000 / rf^2) / numel(exp.filters) ./ stats.layers(exp.layer+1).max ;

opts = get_preimage_opts('maximization', net, exp);
if(isfield(exp, 'gpu') && exp.gpu)
  opts{1}.gpu = true;
end

% IMP: Note that opts{:}.imageSize is set later on as it uses ref below

% -------------------------------------------------------------------------
% How much space for a filter in the grid
% -------------------------------------------------------------------------

k = numel(exp.filters) ;
if k > 1
  reps = max(round(rf / stride * 1.1),4) ;
else
  reps = round(rf/stride * 1.1) ;
end
depth = info.dataSize(3,end) ;

% grid layout
kx = ceil(sqrt(k)) ;
ky = kx ;
if mod(reps,2) == 0, reps = reps + 1 ; end

ref = zeros(reps*kx,reps*ky,depth, 'single') ;
for i = 1:k
  b = fix((i-1)/ kx) ;
  a = i - 1 - b*kx ;
  a = a * reps + (reps-1)/2;
  b = b * reps + (reps-1)/2;
  ref(...
    1 + ...
    a + ...
    b * (reps*kx) + ...
    (exp.filters(i)-1)*(reps*reps*kx*ky)) = innerWeight(k) ;
end
ref = permute(ref, [2 1 3]) ;

% image size
for opts_i = 1:numel(opts)
  if isfield(exp,'imageSize')
    opts{opts_i}.imageSize = exp.imageSize ;
  else
    opts{opts_i}.imageSize = find_image_size(net, ref) ;
  end
end

% -------------------------------------------------------------------------
% Run inversion code
% -------------------------------------------------------------------------

for opts_i = 1:numel(opts)
  if(opts_i == 1)
    res = nnpreimage(net, ref, opts{opts_i}) ;
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
