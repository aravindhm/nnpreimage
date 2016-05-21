classdef L2Norm < dagnn.Loss

  properties
    beta = 2
  end

  methods

    function outputs = forward(obj, inputs, ~)

       outputs{1} = l2(inputs{1}, obj.beta);

    end

    function [derInputs, derParams] = backward(obj, inputs, ~, derOutputs)

       [~, dx] = l2(inputs{1}, obj.beta);
       derInputs{1} = dx * derOutputs{1};
       derParams = {};

    end

    function obj=L2Norm(varargin)
      obj.load(varargin);
      obj.loss = 'L2Norm';
    end

  end % end methods

end  % end classdef
    
