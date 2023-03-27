%function to calculate the PSNR between two images from two different folders

function [psnr_mat] = calculatePSNR(folder_dist, folder_ref, method)
%CALCULATEPSNR Summary of this function goes here
%   Detailed explanation goes here

%folder_dist = 'C:\Users\moham\Desktop\2023\Enhancement\Violinplot\images';
%folder_ref = 'C:\Users\moham\Desktop\2023\Enhancement\Violinplot\images';
files = dir(fullfile(folder1,'*.jpg'));
psnr_all = zeros(1,length(files));
for i = 1:length(files)
    filename_dist = fullfile(folder_dist,files(i).name);
    filename_ref = fullfile(folder_ref,files(i).name);
    psnr_all(i) = psnr(filename_dist, filename_ref);
end
psnr_mat = psnr_all;
name = strcat(method, '_psnr.mat');
% save the psnr metrics in a mat file with the name of the method.mat
save(name, 'psnr_mat');
end
