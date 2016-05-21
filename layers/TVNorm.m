classdef TVNorm < dagnn.Loss 

  properties
    beta = 2
  end

  methods

    function outputs = forward(obj, inputs, ~)

       outputs{1} = tv(inputs{1}, obj.beta);

    end

    function [derInputs, derParams] = backward(obj, inputs, ~, derOutputs)

       [~, dx] = tv(inputs{1}, obj.beta);
       derInputs{1} = dx * derOutputs{1};
       derParams = {};

    end

    function obj=TVNorm(varargin)
      obj.load(varargin);
      obj.loss = 'TVNorm';
    end

  end % end methods

end  % end classdef
