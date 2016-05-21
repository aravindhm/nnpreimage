classdef InnerProductLoss < dagnn.Loss 

  methods

    function outputs = forward(obj, inputs, ~)

       outputs{1} = sum(sum(sum(sum(inputs{2}.*inputs{1})))) ;

    end

    function [derInputs, derParams] = backward(obj, inputs, ~, derOutputs)

       derInputs{1} = derOutputs{1} * inputs{2} ;
       derInputs{2} = []
       derParams = {};

    end

    function obj=InnerProductLoss(varargin)
      obj.load(varargin);
      obj.loss = 'InnerProductLoss';
    end

  end % end methods

end  % end classdef
