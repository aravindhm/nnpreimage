function table2()

ims = dir('data/pics/val/*.JPEG') ;
ims = {ims.name} ;
mods = {'ihog', 'hog', 'hogb', 'dsift'} ;

for m = 1:numel(mods)
  for i = 1:numel(ims)
    [~,b] =fileparts(ims{i}) ;
    sPath = ...
        sprintf('../vedaldi-data/inversion-val160/%s/%s/res.mat', ...
                mods{m}, b) ;
    if ~exist(sPath),warning('missing %s', sPath); continue ; end
    a = load(sPath) ;
    table(m,i) = sqrt(a.error) ;
  end
end

utable = mean(table,2) ;

data = table;
dataMean = mean(data, 2);
standardError = std(data, 0, 2) / sqrt(size(data,2));

stable = 1.96*standardError;


fprintf('error (\\%%)') ;
for m = 1:numel(mods)
  fprintf(' & ${%.1f}$', utable(m)*100) ;
end
fprintf('\\\\[-0.5em]\n') ;

for m = 1:numel(mods)
  fprintf(' & \\tiny $\\pm {%.1f}$', stable(m)*100) ;
end
fprintf('\\\\\n') ;
