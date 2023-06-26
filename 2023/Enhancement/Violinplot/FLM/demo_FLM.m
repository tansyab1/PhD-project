%   The demo was written by Kun Zhan, Jicai Teng, Jinhui Shi
%   $Revision: 1.0.0.0 $  $Date: 2014/12/20 $ 10:30:18 $

%   Reference:
%   K. Zhan, J. Shi, J. Teng, Q. Li and M. Wang, 
%   "Feature-linking model for image enhancement," 
%   Neural Computation, vol. 28, no. 6, pp. 1072-1100, 2016.

close all;clc,clear
addpath(genpath(pwd));
K = 5;

folder_dist =['D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images' ...
    '\process\labelled_images\ExperimentalDATA\forRelatedWorks\UI_var\test\input'];
folder_afgt = ['D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process' ...
    '\labelled_images\ExperimentalDATA\forRelatedWorks\results\AFGT'];
folder_flm = ['D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process' ...
    '\labelled_images\ExperimentalDATA\forRelatedWorks\results\FLM'];

Contrast = ones(K,1); Spatial_frequency = Contrast; Gradient = Contrast;
files = dir(fullfile(folder_dist,'*.jpg'));

for i = 1:length(files)
    filename_dist = fullfile(folder_dist,files(i).name);
    I = imread(filename_dist);
    V = rgb2v(I);
    V_flm = FLM(V);
    I_flm = v2rgb(I,V_flm);
    I_afgt= AFGT_CR(I);
    imwrite(I_flm,fullfile(folder_flm,files(i).name));
    imwrite(I_afgt,fullfile(folder_afgt,files(i).name));
end