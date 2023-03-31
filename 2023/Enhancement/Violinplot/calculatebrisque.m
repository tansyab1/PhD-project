%function to calcualte the brisque metrics of all the images in the folder
function [brisque_mat] = calculatebrisque(folder, method)
%CALCULATEMETRICS Summary of this function goes here
%   Detailed explanation goes here
%folder = 'C:\Users\moham\Desktop\2023\Enhancement\Violinplot\images';
files = dir(fullfile(folder,'*.jpg')) | dir(fullfile(folder,'*.png')) | dir(fullfile(folder,'*.bmp')) | dir(fullfile(folder,'*.tiff')) | dir(fullfile(folder,'*.tif')) | dir(fullfile(folder,'*.jpeg')) | dir(fullfile(folder,'*.PNG')) | dir(fullfile(folder,'*_x1_SR.png'));

brisque_all = zeros(1,length(files));
for i = 1:length(files)
    filename = fullfile(folder,files(i).name);
    brisque_all(i) = brisque(filename);
end
brisque_mat = brisque_all;
name = strcat(method, '_brisque.mat');
% save the brisque metrics in a mat file with the name of the method.mat
save(name, 'brisque_mat');
end