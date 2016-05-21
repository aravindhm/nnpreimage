function exp_setup()

root = fileparts(mfilename('fullpath')) ;

addpath(fullfile(root));
addpath(fullfile(root, 'networks'));

% This will download the models and some of the data
curDir = pwd();
cd(root);
cmd = sprintf('bash exp_setup.sh', root)
system(cmd);
cd(curDir);

fprintf(1, '\n\n\n\n You may or may not still need to download the first 100 images of imagenet validation data into data/pics/val/\n\n\n');

