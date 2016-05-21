function stats = get_neuron_stats()

run matconvnet/matlab/vl_setupnn
addpath matconvnet/examples/imagenet

opts.dataDir = fullfile('datasets', 'ILSVRC2012');
opts.expDir = fullfile('data/stats/') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.numFetchThreads = 12 ;

% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
  imdb = cnn_imagenet_setup_data('dataDir', opts.dataDir) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% -------------------------------------------------------------------------
%                                                        Network evaluation
% -------------------------------------------------------------------------

models =  {'imagenet-vgg-m', ...
           'imagenet-vgg-verydeep-16', ...
           'imagenet-caffe-alex', ...
           'imagenet-caffe-ref', ...
           'imagenet-vgg-f'}  ;

gpuDevice(1) ;

num = 0 ;
for i = 1:numel(models)
  models{i}
  net = load(sprintf('data/models/%s.mat', models{i})) ;

  if(isfield(net, 'net')), net = net.net;  end

  net = vl_simplenn_tidy(net);
  net = vl_simplenn_move(net,'gpu') ;

  bopts = net.meta.normalization ;
  bopts.numThreads = opts.numFetchThreads ;

  clear stats ;
  for j = 1:numel(net.layers)
    stats.layers(j).mean = 0 ;
    stats.layers(j).std = 0 ;
    stats.layers(j).max = 0 ;
  end

  train = find(imdb.images.set == 1) ;
  train = train(1: 1000: end);
  bs = 16 ;
  fn = getBatchSimpleNNWrapper(bopts) ;
  for t=1:bs:numel(train)
    batch_time = tic ;
    batch = train(t:min(t+bs-1, numel(train))) ;
    fprintf('collecting image stats: batch starting with image %d ...', batch(1)) ;
    temp = fn(imdb, batch) ;
    temp = gpuArray(temp) ;
    res = vl_simplenn(net, temp, [], [], 'conserveMemory', false) ;

    for l = 1:numel(net.layers)
      a = res(l).x ;
      u = squeeze(gather(mean(mean(mean(a,1),2),4))) ;
      s = squeeze(gather(std(std(std(a,0,1),0,2),0,4))) ;
      m = squeeze(gather(max(max(max(a,[],1),[],2),[],4))) ;
      stats.layers(l).mean = (num * stats.layers(l).mean + u) / (num + 1);
      stats.layers(l).std = (num *  stats.layers(l).std + s) / (num + 1);
      stats.layers(l).max = max(stats.layers(l).max, m) ;
    end
    num = num + 1 ;

    batch_time = toc(batch_time) ;
    fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
  end

  save(sprintf('data/stats/%s-stats.mat', models{i}), 'stats') ;
end



% -------------------------------------------------------------------------
function fn = getBatchSimpleNNWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchSimpleNN(imdb,batch,opts) ;

% -------------------------------------------------------------------------
function [im,labels] = getBatchSimpleNN(imdb, batch, opts)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im = cnn_imagenet_get_batch(images, opts, ...
                            'prefetch', nargout == 0) ;
labels = imdb.images.label(batch) ;
