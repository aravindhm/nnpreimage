function dg_setup()
% Setup the toolbox dg_setup

root = fileparts(mfilename('fullpath')) ;

run(fullfile(root, 'vlfeat', 'toolbox', 'vl_setup.m')) ;
run(fullfile(root, 'matconvnet', 'matlab', 'vl_setupnn.m')) ;

addpath(fullfile(root, 'core'));
addpath(fullfile(root, 'helpers'));
addpath(fullfile(root, 'layers'));

