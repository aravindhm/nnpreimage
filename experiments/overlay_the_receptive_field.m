function [viz_images, all_masks] = overlay_the_receptive_field()
% This function is meant to generate the figure where we study locality of neurons (the central 5x5 neuron field).
% I will load the relevant results from the data folder and load the network from the networks folder.

modelpathname = 'networks/imagenet-caffe-ref.mat';

net = load(modelpathname);
net = vl_simplenn_tidy(net);
info = vl_simplenn_display(net, 'inputSize', [227, 227, 3, 1]);

viz_images = cell(15,1);
all_masks = cell(15,1);

for layer_l=1:15
  imgfilename = sprintf(...
   'data/inversion-neigh-160/imagenet-caffe-ref/ILSVRC2012_val_00000013/layer%02d-str01.0.png', layer_l);

  sz = info.dataSize(:, layer_l + 1)';

  nx = min(5, sz(2)) ;
  ny = min(5, sz(1)) ;
  sx = (0:nx-1) + ceil((sz(2)-nx+1)/2) ;
  sy = (0:ny-1) + ceil((sz(1)-ny+1)/2) ;

  neuron_start_I = min(sy);
  neuron_start_J = min(sx);
  neuron_end_I = max(sy);
  neuron_end_J = max(sx);

  mask = zeros(sz(1), sz(2));
  mask(neuron_start_I:neuron_end_I, neuron_start_J:neuron_end_J) = 1;

  all_masks{layer_l} = mask;

  bbox_start = get_receptive_field(neuron_start_I, neuron_start_J, layer_l, info);
  bbox_end = get_receptive_field(neuron_end_I, neuron_end_J, layer_l, info);

  img = imread(imgfilename);
  shapeInserter = vision.ShapeInserter('Shape','Rectangles','LineWidth',3,'BorderColor','Custom','CustomBorderColor',uint8([255 0 0]));
  rectangle = int32([bbox_start(2), bbox_start(1), bbox_end(4)-bbox_start(2)+1, bbox_end(3)-bbox_start(1)+1]);
  img_overlaid = step(shapeInserter, img, rectangle);
 
  viz_images{layer_l} = img_overlaid;
   
  outimgfilename = sprintf(...
     'data/inversion-neigh-160-overlaid/imagenet-caffe-ref/ILSVRC2012_val_00000013/layer%02d-str01.0.png', layer_l);
  imwrite(img_overlaid, outimgfilename);

end



%--------------------------------------------------------------
function bbox = get_receptive_field(neuron_I, neuron_J, layer_l, info)
rf_start_pos = [neuron_I - 1; neuron_J - 1] .* ...
    info.receptiveFieldStride(:, layer_l) + ...
    info.receptiveFieldOffset(:, layer_l) - ...
    info.receptiveFieldSize(:, layer_l) / 2 + 1;

rf_end_pos = rf_start_pos + info.receptiveFieldSize(:, layer_l) - 1;

rf_start_pos = max(rf_start_pos, 1);
rf_end_pos = min(rf_end_pos, 227);

bbox = [rf_start_pos', rf_end_pos'];
