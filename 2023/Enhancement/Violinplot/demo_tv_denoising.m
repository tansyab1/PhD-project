% read all images in the folder 
% and store them in a cell array
clear;

folder_dist = ['D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\' ...
    'labelled_images\ExperimentalDATA\forRelatedWorks\Noise_var\test\input'];
files = dir(fullfile(folder_dist,'*.jpg'));
folder_out = ['D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\' ...
    'labelled_images\ExperimentalDATA\forRelatedWorks\results\TV_noise'];
%% run the algorithm
rng(0)  % random seed, for reproducibility

lambda = 0.5e-1;      % regularization parameter
n_iters = 7;       % number of iteration



%%
for i = 1:length(files)
    filename = files(i).name;
    filessave = fullfile(folder_out, files(i).name);
    if exist(filessave,'file')
        disp('pass')
    else 
    % read the image
    Img = im2double(imread(fullfile(folder_dist,filename)));
    
    [X_out,runtime] = FGP_color2d(Img,lambda,n_iters);  % FPG
%     X_out = uint8(X_out);
    % save the image
    imwrite(X_out,fullfile(folder_out,filename));
    disp("image processed done")
    end 
    
end