%function to calcualte the niqe metrics of all the images in the folder with one of these extensions: .jpg, .png, .bmp, .tiff, .tif 
function [niqe_mat] = calculateniqe(folder, method)
%CALCULATEMETRICS Summary of this function goes here
%   Detailed explanation goes here
%folder = 'C:\Users\moham\Desktop\2023\Enhancement\Violinplot\images';

%get all the images in the folder with one of these extensions: .jpg, .png, .bmp, .tiff, .tif, .jpeg, .PNG, _x1_SR.png
files = dir(fullfile(folder,'*.jpg')) | dir(fullfile(folder,'*.png')) | dir(fullfile(folder,'*.bmp')) | dir(fullfile(folder,'*.tiff')) | dir(fullfile(folder,'*.tif')) | dir(fullfile(folder,'*.jpeg')) | dir(fullfile(folder,'*.PNG')) | dir(fullfile(folder,'*_x1_SR.png'));
niqe_all = zeros(1,length(files));
for i = 1:length(files)
    filename = fullfile(folder,files(i).name);
    niqe_all(i) = niqe(filename);
end
niqe_mat = niqe_all;
name = strcat(method, '_niqe.mat');
// save the niqe metrics in a mat file with the name of the method.mat
save(name, 'niqe_mat');
end