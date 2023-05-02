clear;
clear all 
close all

% read all images in the folder 
% and store them in a cell array

folder_ref = 'D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\groundtruth';
files = dir(fullfile(folder_dist,'*.jpg'));
folder_out = 'D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\labelled_images\ExperimentalDATA\forRelatedWorks\results\TV';

% read file and store them in a cell array
dict = 'blur_dict.mat';

% load the dictionary and save it in a variable
load(dict);
name = finaldict(1,:);
sigma = finaldict(2,:);


for i = 1:length(files)
    filename = files(i).name;
    % get the sigma value from the dictionary
    % find the file name in the list
    idx = find(strcmp(name,filename));
    sigma = sigma(idx);
    % convert the sigma from string to double
    sigma = str2double(sigma);
    % read the image
    Img = imread(fullfile(folder_dist,filename));
    X_out = deblur_tv_fista_demo(Img, sigma);

    % save the image
    imwrite(X_out,fullfile(folder_out,filename));
end
