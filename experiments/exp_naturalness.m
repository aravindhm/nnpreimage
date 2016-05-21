function exp_naturalness(subset, model)
% Measures the histogram of gradients and pixel intensities
% and compares these for the original image and the inversion
% with C=1,20,100,300 to see the effect of regularizer on these
% natural image statistics

run dg_setup;

if(nargin < 1)
  subset = 1:100;
end
if(nargin < 2)
  model = 'imagenet-caffe-alex';
end

net = load(sprintf('networks/%s.mat', model));
net = vl_simplenn_tidy(net);

objective_coeffs = {'01.0', '20.0', '100.0', '300.0'};

for img_no = subset
  original_filename = sprintf(...
            'data/pics/val/ILSVRC2012_val_%08d.JPEG',...
                img_no);
  
  img = imread(original_filename);
  img_pp = imresize(img, net.meta.normalization.imageSize(1:2));
  gradient_hist_original = computeGlobalHOG(img_pp); 

  for layer=1:20
    
    for C_i=1:numel(objective_coeffs)
      inverse_filename = ...
          sprintf(...
            'data/inversion-val160/%s/ILSVRC2012_val_%08d/layer%02d-str%s.png',...
               model, img_no, layer, objective_coeffs{C_i});
      fprintf(1, '%s\n', inverse_filename);

      img_inverse = imread(inverse_filename);

      gradient_hist_inverse = computeGlobalHOG(img_inverse);

      similarity_raw(img_no, layer, C_i) = ...
           histogram_intersection(gradient_hist_original, gradient_hist_inverse);

    end

    inverse_filename = ...
        sprintf(...
          'data/inversion-val160-noreg/%s/ILSVRC2012_val_%08d/layer%02d-str01.0.png',...
             model, img_no, layer);
    fprintf(1, '%s\n', inverse_filename);

    img_inverse = imread(inverse_filename);

    gradient_hist_inverse = computeGlobalHOG(img_inverse);

    similarity_raw(img_no, layer, numel(objective_coeffs) + 1) = ...
         histogram_intersection(gradient_hist_original, gradient_hist_inverse);

  end

end

plot_mean(similarity_raw(:, :, [1,3,5]), 'errorBars', true, ...
     'xLabel', 'Layer', 'yLabel', 'Gradient Histogram Intersection', ...
     'legend', {'C=1', 'C=100', 'No reg.'}, ...
     'legendLocation', 'northeast',...
     'color', [0, 0, 0.9; 0, 0.9, 0; 0.9, 0, 0]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function similarity = histogram_intersection(A, B)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

similarity = sum(min(A, B));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function gradient_hist = computeGlobalHOG(img)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(size(img, 3) == 3)
  img = rgb2gray(img);
end

[Gmag, Gdir] = imgradient(img);

gradient_hist = zeros(18,1);
o = linspace(-180, 180, 19);

for i=2:numel(o)
  if(i == 2)
    idx = find(Gdir(:) >= o(i-1) & Gdir(:) <= o(i));
  else
    idx = find(Gdir(:) > o(i-1) & Gdir(:) <= o(i));
  end
  gradient_hist(i-1) = sum(Gmag(idx));
end

gradient_hist = gradient_hist / sum(gradient_hist);
