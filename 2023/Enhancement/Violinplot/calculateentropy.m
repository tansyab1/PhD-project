%function to calcualte the entropy metrics of all the images in the folder
function [entropy_mat] = calculateentropy(folder, method)
%CALCULATEMETRICS Summary of this function goes here
%   Detailed explanation goes here
%folder = 'C:\Users\moham\Desktop\2023\Enhancement\Violinplot\images';
files = dir(fullfile(folder,'*.jpg')) | dir(fullfile(folder,'*.png')) | dir(fullfile(folder,'*.bmp')) | dir(fullfile(folder,'*.tiff')) | dir(fullfile(folder,'*.tif')) | dir(fullfile(folder,'*.jpeg')) | dir(fullfile(folder,'*.PNG')) | dir(fullfile(folder,'*_x1_SR.png'));

entropy_all = zeros(1,length(files));
for i = 1:length(files)
    filename = fullfile(folder,files(i).name);
    entropy_all(i) = entropy(filename);
end
entropy_mat = entropy_all;
name = strcat(method, '_entropy.mat');
% save the entropy metrics in a mat file with the name of the method.mat
save(name, 'entropy_mat');
end
