function plot_mean(data, varargin)
% This function is mean to plot the tables which give error values across
% different layers and objective coefficients C.

opts.errorBars = true;

opts.xLabel = '';
opts.yLabel = '';
opts.title = '';

opts.grid = 'on';

opts.legendLocation = 'northeast';
opts.legend = {'C=1', 'C=20', 'C=100', 'C=300'};

opts.xTick = [1, 5, 9, 13, 16, 20];
opts.xTickLabel = {'conv1', 'conv2', 'conv3', 'conv5', 'fc6', 'fc8'};

opts.color = rand(size(data,3), 3);
opts.lineStyles = {'-', ':', '-.', '--', 'o-'};

opts.fontSize = 15;

opts.xLimits = [0, size(data,2)+1];

opts = vl_argparse(opts, varargin) ;

figure;
hold on;

if(opts.errorBars)
  [dataErrorBars, dataMean] = calculateErrorBars(data);

  for i=1:size(dataMean,2)
    errorbar(dataMean(:,i), dataErrorBars(:,i), opts.lineStyles{i}, 'LineWidth', 2, 'Color', opts.color(i,:));
  end

else

  dataMean = squeeze(mean(data, 1));

  for i=1:size(dataMean,2)
    plot(dataMean(:,i), opts.lineStyles{i}, 'LineWidth', 2, 'Color', opts.color(i,:));
  end

end

ax = gca;
ax.XLim = opts.xLimits;
ax.FontSize = opts.fontSize;
ax.XTick = opts.xTick;
ax.XTickLabel = opts.xTickLabel;

xlabel(opts.xLabel, 'FontSize', opts.fontSize);
ylabel(opts.yLabel, 'FontSize', opts.fontSize);

legend(opts.legend{:}, 'Location', opts.legendLocation);

if(strcmp(opts.grid, 'on'))
  grid on;
elseif(strcmp(opts.grid, 'off'))
  grid off;
else
  error('Unrecognized grid option %s', opts.grid);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dataErrorBars, dataMean] = calculateErrorBars(data)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dataMean = squeeze(mean(data, 1));

standardError = squeeze(std(data, 0, 1)) / sqrt(size(data,1));

dataErrorBars = 1.96 * standardError;


