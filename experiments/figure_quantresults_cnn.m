function figure_quantresults_cnn()

net = load('networks/imagenet-caffe-alex.mat') ;
ims = dir('data/pics/val/*.JPEG') ;
ims = {ims.name} ;
strs = [1 20 100 300] ;

for l = 1:numel(net.layers)-1
  for i = 1:numel(ims)
    for s = 1:numel(strs)
      [~,b] =fileparts(ims{i}) ;
      sPath = ...
          sprintf('data/inversion-val160/imagenet-caffe-alex/%s/layer%02d-str%04.1f.mat', ...
                  b, l, strs(s)) ;
      if ~exist(sPath),warning('missing %s', sPath); continue ; end
      a = load(sPath) ;
      table(s,l,i) = sqrt(a.error) ;
    end
  end
end

utable = mean(table,3) ;

plot_mean(permute(table, [3, 2, 1]), 'errorBars', true, ...
     'grid', 'on',...
     'xLabel', 'Layer', 'yLabel', 'Average normalized reconstruction error', ...
     'legend', {'C=1', 'C=20', 'C=100', 'C=300'}, ...
     'legendLocation', 'northeast',...
     'color', [0, 0, 0.9; 0, 0.9, 0; 0.9, 0, 0; 0.0, 0.0, 0]);
