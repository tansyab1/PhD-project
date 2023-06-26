%function to calculate the SSIM between two images from two different folders

function [ssim_mat] = calculateSSIM(folder_dist, folder_ref, method)
%CALCULATESSIM Summary of this function goes here
%   Detailed explanation goes here

%folder_dist = 'C:\Users\moham\Desktop\2023\Enhancement\Violinplot\images';
%folder_ref = 'C:\Users\moham\Desktop\2023\Enhancement\Violinplot\images';
files_jpg = dir(fullfile(folder_dist,'*.jpg'));
files_png = dir(fullfile(folder_dist,'*.png'));
files_PNG= dir(fullfile(folder_dist,'*.PNG')) ;
files = dir(fullfile(folder_dist,'*_x1_SR.png'));

files_length = max([length(files_x), length(files_PNG), length(files_png), length(files_jpg)]);

switch files_length
    case length(files_jpg)
        files = files_jpg;
    case length(files_png)
        files = files_png;
    case length(files_PNG)
        files = files_PNG;
end
[~, ~, ext] = fileparts(files(1));
ssim_all = zeros(1,length(files));
for i = 1:length(files)
    filename_dist = fullfile(folder_dist,files(i).name);
    filename_ref = fullfile(folder_ref,replace(files(i).name,ext, ".jpg"));
    ssim_all(i) = ssim(filename_dist, filename_ref);
end
ssim_mat = ssim_all;
name = strcat(method, '_ssim.mat');
% save the ssim metrics in a mat file with the name of the method.mat
save(name, 'ssim_mat');
end
