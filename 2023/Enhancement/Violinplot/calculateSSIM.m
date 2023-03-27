%function to calculate the SSIM between two images from two different folders

function [ssim_mat] = calculateSSIM(folder_dist, folder_ref, method)
%CALCULATESSIM Summary of this function goes here
%   Detailed explanation goes here

%folder_dist = 'C:\Users\moham\Desktop\2023\Enhancement\Violinplot\images';
%folder_ref = 'C:\Users\moham\Desktop\2023\Enhancement\Violinplot\images';

files = dir(fullfile(folder_dist,'*.jpg'));
ssim_all = zeros(1,length(files));
for i = 1:length(files)
    filename_dist = fullfile(folder_dist,files(i).name);
    filename_ref = fullfile(folder_ref,files(i).name);
    ssim_all(i) = ssim(filename_dist, filename_ref);
end
ssim_mat = ssim_all;
name = strcat(method, '_ssim.mat');
% save the ssim metrics in a mat file with the name of the method.mat
save(name, 'ssim_mat');
end
