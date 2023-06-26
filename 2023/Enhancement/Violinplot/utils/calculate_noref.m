%function to calcualte the niqe metrics of all the images in the folder with one of these extensions: .jpg, .png, .bmp, .tiff, .tif 
function [niqe_mat,brisque_mat,entropy_mat] = calculate_noref(folder, method)
%CALCULATEMETRICS Summary of this function goes here
%   Detailed explanation goes here
%folder = 'C:\Users\moham\Desktop\2023\Enhancement\Violinplot\images';

%get all the images in the folder with one of these extensions: .jpg, .png, .bmp, .tiff, .tif, .jpeg, .PNG, _x1_SR.png
files_jpg = dir(fullfile(folder,'*.jpg'));
files_png = dir(fullfile(folder,'*.png'));
files_PNG= dir(fullfile(folder,'*.PNG')) ;
files = dir(fullfile(folder,'*_x1_SR.png'));

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
niqe_all = zeros(1,length(files));
brisque_all = zeros(1,length(files));
entropy_all = zeros(1,length(files));
for i = 1:length(files)
    filename = fullfile(folder,files(i).name);
    I = imread(filename);
    niqe_all(i) = niqe(I);
    brisque_all(i) = brisque(I);
    entropy_all(i) = entropy(I);
end
niqe_mat = niqe_all;
name = strcat(method, '_niqe.mat');
%save the niqe metrics in a mat file with the name of the method.mat
save(name, 'niqe_mat');

% brisque_mat = brisque_all;
% name = strcat(method, '_brisque.mat');
% % save the brisque metrics in a mat file with the name of the method.mat
% save(name, 'brisque_mat');
% 
% entropy_mat = entropy_all;
% name = strcat(method, '_entropy.mat');
% % save the entropy metrics in a mat file with the name of the method.mat
% save(name, 'entropy_mat');
end