%function to calculate the VIF between two images from two different folders

function [vif_mat] = calculateVIF(folder_dist, folder_ref, method)
%CALCULATEVIF Summary of this function goes here
%   Detailed explanation goes here
%folder_dist = 'C:\Users\moham\Desktop\2023\Enhancement\Violinplot\images';
%folder_ref = 'C:\Users\moham\Desktop\2023\Enhancement\Violinplot\images';

files = dir(fullfile(folder,'*.jpg')) | dir(fullfile(folder,'*.png')) | ...
    dir(fullfile(folder,'*.bmp')) | dir(fullfile(folder,'*.tiff')) | ...
    dir(fullfile(folder,'*.tif')) | dir(fullfile(folder,'*.jpeg')) | ...
    dir(fullfile(folder,'*.PNG')) | dir(fullfile(folder,'*_x1_SR.png'));
[~, ~, ext] = fileparts(files(1));
vif_all = zeros(1,length(files));
for i = 1:length(files)
    filename_dist = fullfile(folder_dist,files(i).name);
    filename_ref = fullfile(folder_ref,replace(files(i).name,ext, ".jpg"));
    vif_all(i) = vif(filename_dist, filename_ref);
end
vif_mat = vif_all;
name = strcat(method, '_vif.mat');
% save the vif metrics in a mat file with the name of the method.mat
save(name, 'vif_mat');
end