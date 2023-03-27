%function to calcualte the niqe metrics of all the images in the folder
function [niqe_mat] = calculateniqe(folder, method)
%CALCULATEMETRICS Summary of this function goes here
%   Detailed explanation goes here
%folder = 'C:\Users\moham\Desktop\2023\Enhancement\Violinplot\images';
files = dir(fullfile(folder,'*.jpg'));
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