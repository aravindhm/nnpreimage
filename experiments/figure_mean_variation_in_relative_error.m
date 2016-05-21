function [mean_variation_in_error, mean_variation_in_relative_error, ...
               raw_error, relative_error] = figure_mean_variation_in_relative_error(subset, model) 
% This function generates a table that compares the error values for different images 
% obtained using different random initializations across the first 25 images in the ILSVRC dataset

net = load(sprintf('matconvnet/data/models/%s.mat', model));
net = vl_simplenn_tidy(net);

net_img_size = net.meta.normalization.imageSize;

normalize_fn=@(x) (cnn_normalize(net.meta.normalization, ...
     x, true));

objective_coeffs = {'01.0', '20.0', '100.0', '300.0'};

% Preallocating memory for some of the buffers
raw_error = zeros(numel(subset), numel(net.layers)-1, numel(objective_coeffs), 4);
variation_in_error = zeros(numel(subset), numel(net.layers)-1, numel(objective_coeffs));

for img_no = subset

  original_filename = sprintf(...
            'vedaldi-data/pics/val/ILSVRC2012_val_%08d.JPEG',...
                img_no);
  img = imread(original_filename);
  img_pp = normalize_fn(img);
  
  res = vl_simplenn(net, img_pp);

  for layer_i = 1:numel(net.layers)-1

    net_cur = net;
    net_cur.layers = net_cur.layers(1:layer_i);

    for C_i=1:numel(objective_coeffs)

      inverse_filename = ...
          sprintf(...
            'data/inversion-multi-val160/%s/ILSVRC2012_val_%08d/layer%02d-str%s.png',...
               model, img_no, layer_i, objective_coeffs{C_i});
      
      fprintf(1, '%s\n', inverse_filename);
      
      img_inverse = imread(inverse_filename);
      % This is going to be 4 images in a 2x2 grid
      img_inverse_4D = cat(4, ...
           img_inverse(1:net_img_size(1), 1:net_img_size(2), :),...
           img_inverse(1:net_img_size(1), net_img_size(2)+1:end, :),...
           img_inverse(net_img_size(1)+1:end, 1:net_img_size(2), :),...
           img_inverse(net_img_size(1)+1:end, net_img_size(2)+1:end, :));
      img_inverse_4D_pp = cat(4, ...
           normalize_fn(img_inverse_4D(:,:,:,1)),...
           normalize_fn(img_inverse_4D(:,:,:,2)),...
           normalize_fn(img_inverse_4D(:,:,:,3)),...
           normalize_fn(img_inverse_4D(:,:,:,4)));

      res_inverse = vl_simplenn(net_cur, img_inverse_4D_pp);

      reconstructed_features = reshape(res_inverse(end).x, [], 4);     
      for repeat_no = 1:4
        raw_error(img_no, layer_i, C_i, repeat_no) = ...
           norm(reconstructed_features(:,repeat_no) - res(layer_i+1).x(:)).^2;
        relative_error(img_no, layer_i, C_i, repeat_no) = ...
           norm(reconstructed_features(:,repeat_no) - res(layer_i+1).x(:)).^2 / norm(res(layer_i+1).x(:)).^2;
      end
      
      cur_error = squeeze(raw_error(img_no, layer_i, C_i, :));
      mean_raw_error = mean(cur_error);
      normalized_raw_error = cur_error / mean_raw_error;
   
      variation_in_error(img_no, layer_i, C_i) = std(normalized_raw_error);
      variation_in_relative_error(img_no, layer_i, C_i) = std(reshape(relative_error(img_no, layer_i, C_i, :), 4, 1));
    end

  end

end

% Now we should summarize the variation_in_error over the dataset of images 
%mean_variation_in_error = squeeze(mean(variation_in_error, 1));
%mean_variation_in_relative_error = squeeze(mean(variation_in_relative_error, 1));

plot_mean(variation_in_relative_error, 'errorBars', false, ...
     'xLabel', 'Layer', 'yLabel', 'Mean Standard Deviation in Normalized Error', ...
     'legend', {'C=1', 'C=20', 'C=100', 'C=300'}, ...
     'legendLocation', 'northwest',...
     'color', [0, 0, 0.9; 0, 0.9, 0; 0.9, 0, 0; 0.0, 0.0, 0]);
