classdef JitterLayer < dagnn.Layer 

  properties
    jitterAmount = 1;
    sx = [];
    sy = [];
  end

  methods

    function outputs = forward(obj, inputs, ~)

      jx = randi(obj.jitterAmount) ;
      jy = randi(obj.jitterAmount) ;

      x0_size = size(inputs{1});
      x0_size(1:2) = x0_size(1:2) - obj.jitterAmount + 1;

      obj.sx = (1:x0_size(2)) + jx - 1 ;
      obj.sy = (1:x0_size(1)) + jy - 1 ;

      outputs{1} = inputs{1}(obj.sy, obj.sx, :, :);

    end

    function [derInputs, derParams] = backward(obj, inputs, ~, derOutputs)

       if(isa(inputs{1}, 'gpuArray'))
         derInputs{1} = zeros(size(inputs{1}), 'single', 'gpuArray');
       else
         derInputs{1} = zeros(size(inputs{1}), 'single');
       end

       derInputs{1}(obj.sy, obj.sx, :, :) = derOutputs{1};
      
       derParams = {};
          
       obj.sy = []; obj.sx = [];

    end

    function obj=JitterLayer(varargin)
      obj.load(varargin);
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = inputSizes{1};
      outputSizes{1}(1:2) = outputSizes{1}(1:2) - obj.jitterAmount + 1;
    end

  end % end methods

end  % end classdef
