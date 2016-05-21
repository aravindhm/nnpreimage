function same_label_all = exp_match_accuracy(subset, model)
% Evaluate the classification of inverted images

if(nargin < 1)
  subset = 1:100;
end
if(nargin < 2)
  model = 'imagenet-caffe-alex';
end

net = load(sprintf('networks/%s.mat', model));
net = vl_simplenn_tidy(net);

normalize_fn=@(x) (cnn_normalize(net.meta.normalization, ...
     x, true));

objective_coeffs = {'01.0', '20.0', '100.0', '300.0'};

same_label_all = zeros(100, 20, numel(objective_coeffs));

for img_no=subset

  original_filename = sprintf(...
            'data/pics/val/ILSVRC2012_val_%08d.JPEG',...
                img_no);
  img = imread(original_filename);
  img_pp = normalize_fn(img);
  
  res = vl_simplenn(net, img_pp);
  [~, label] = max(res(end).x(:));

  for layer=1:20
    
    for C_i=1:numel(objective_coeffs)
      inverse_filename = ...
          sprintf(...
            'data/inversion-val160/%s/ILSVRC2012_val_%08d/layer%02d-str%s.png',...
               model, img_no, layer, objective_coeffs{C_i});
      fprintf(1, '%s\n', inverse_filename);
      
      img_inverse = imread(inverse_filename);
      img_inverse_pp = normalize_fn(img_inverse);

      res_inverse = vl_simplenn(net, img_inverse_pp);

      [~, label_inverse] = max(res_inverse(end).x(:));

      same_label_all(img_no, layer, C_i) = (label == label_inverse);
      
    end


  end
end

info.model = model;
info.subset = subset;

save data/exp_match_accuracy.mat same_label_all info;

plot_mean(same_label_all(:, :, :), 'errorBars', false, ...
     'grid', 'on', ...
     'xLabel', 'Layer', 'yLabel', 'Classification Consistency', ...
     'legend', {'C=1','C=20','C=100','C=300'}, ...
     'legendLocation', 'southeast',...
     'color', [0, 0, 0.9; 0, 0.9, 0; 0.9, 0, 0; 0.0, 0.0, 0.0]);

end
