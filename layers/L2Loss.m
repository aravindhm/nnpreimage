classdef L2Loss < dagnn.Loss 

  methods

    function outputs = forward(obj, inputs, ~)

       d = inputs{1} - inputs{2} ;
       outputs{1} = sum(sum(sum(sum(d.*d.*inputs{3})))) ;

    end

    function [derInputs, derParams] = backward(obj, inputs, ~, derOutputs)

       derInputs{1} = derOutputs{1} * 2 * (inputs{1} - inputs{2}) .* inputs{3} ;
       derInputs{2} = [];
       derInputs{3} = [];
       derParams = {};

    end

    function obj=L2Loss(varargin)
      obj.load(varargin);
      obj.loss = 'L2Loss';
    end

  end % end methods

end  % end classdef
