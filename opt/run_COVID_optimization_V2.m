clearvars
clc
format compact
close all

addpath MATLAB_blackbox
addpath MATLAB_blackbox\support_functions
addpath StoMADS_V2

global index
index = 0;

% clear history
d = dir('./data');
filenames = {d.name};
dirFlags = [d.isdir];
% Loop through the filenames-
for i = 1:numel(filenames)
    fn = filenames{i};
    if ~dirFlags(i)
        delete([d(1).folder,'\',fn])
    end
end

ceal = v_deux_Sto_Param(1);

diary([ceal{19},'/output.txt']) % save console output to window

diary on
V2_StoMADS_PB(34,ceal)
diary off
% pb_sto_algo(35)