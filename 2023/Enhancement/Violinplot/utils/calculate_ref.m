%function to calculate the PSNR between two images from two different folders

function [psnr_mat] = calculate_ref(input, folder_dist, method)
%CALCULATEPSNR Summary of this function goes here
%   Detailed explanation goes here

%folder_dist = 'C:\Users\moham\Desktop\2023\Enhancement\Violinplot\images';
folder_ref = 'D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\groundtruth';
files_jpg = dir(fullfile(folder_dist,'*.jpg'));
files_png = dir(fullfile(folder_dist,'*.png'));
files_PNG= dir(fullfile(folder_dist,'*.PNG')) ;
files = dir(fullfile(folder_dist,'*_x1_SR.png'));
filesSR = files;
filesDUAL = dir(fullfile(folder_dist,'*_DUAL_g0.6_l0.15.jpg'));
files_length = max([length(files), length(files_PNG), length(files_png), length(files_jpg)]);

if files_length < 20000
    switch files_length
    case length(files_jpg)
        files = files_jpg;
    case length(files_png)
        files = files_png;
    case length(files_PNG)
        files = files_PNG;
    end
end
[~, ~, ext] = fileparts(files(1).name);
if ~isempty(filesSR)
    ext = '_x1_SR.png';
end 
if ~isempty(filesDUAL)
    ext = '_DUAL_g0.6_l0.15.jpg';
end 
loe_all = zeros(1,length(files));
% lom_all = zeros(1,length(files));
% smo_all = zeros(1,length(files));
% psnr_all = zeros(1,length(files));
% ssim_all = zeros(1,length(files));
% vif_all = zeros(1,length(files));


for i = 1:length(files)
    filename_dist = fullfile(folder_dist,files(i).name);
%     filename_ref = fullfile(folder_ref,replace(files(i).name,ext, ".jpg"));
    filename_input = fullfile(input,replace(files(i).name,ext, ".jpg"));
    I_dist = imread(filename_dist);
%     I_ref = imread(filename_ref);
    I_input = imread(filename_input);
    I_input_gray = rgb2gray(I_input);
    I_dist_gray = rgb2gray(I_dist);

    loe_all(i) = LOE(I_input_gray, I_dist_gray);
%     lom_all(i) = LOM(I_input_gray, I_dist_gray);
%     smo_all(i) = SMO(I_input_gray, I_dist_gray);
% 
%     psnr_all(i) = psnr(I_dist, I_ref);
%     ssim_all(i) = ssim(I_dist, I_ref);
%     I_ref_gray = rgb2gray(I_ref);
%     vif_all(i) = vifvec(I_ref_gray,I_dist_gray);
end
loe_mat = loe_all;
name = strcat(method, '_loe.mat');
% save the psnr metrics in a mat file with the name of the method.mat
save(name, 'loe_mat');

% lom_mat = lom_all;
% name = strcat(method, '_lom.mat');
% % save the ssim metrics in a mat file with the name of the method.mat
% save(name, 'lom_mat');
% 
% smo_mat = smo_all;
% name = strcat(method, '_smo.mat');
% % save the ssim metrics in a mat file with the name of the method.mat
% save(name, 'smo_mat');
% 
% psnr_mat = psnr_all;
% name = strcat(method, '_psnr.mat');
% % save the ssim metrics in a mat file with the name of the method.mat
% save(name, 'psnr_mat');
% 
% ssim_mat = ssim_all;
% name = strcat(method, '_ssim.mat');
% % save the ssim metrics in a mat file with the name of the method.mat
% save(name, 'ssim_mat');
% 
% vif_mat = vif_all;
% name = strcat(method, '_vif.mat');
% % save the ssim metrics in a mat file with the name of the method.mat
% save(name, 'vif_mat');
end
