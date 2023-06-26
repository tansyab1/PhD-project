% read all images in the folder 
% and store them in a cell array

folder_dist = ['D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\' ...
    'labelled_images\ExperimentalDATA\forRelatedWorks\Blur_var\test\input'];
files = dir(fullfile(folder_dist,'*.jpg'));
folder_out = ['D:\backup-data\PhD-work\Datasets\kvasir_capsule\labelled_images\process\' ...
    'labelled_images\ExperimentalDATA\forRelatedWorks\results\TV2'];
% read file and store them in a cell array
name = blurdict(:,1);
sigma = blurdict(:,2);

%%
for i = 1:length(files)
    filename = files(i).name;
    filessave = fullfile(folder_out, files(i).name);
    if exist(filessave,'file')
        disp('pass')
    else 
        % get the sigma value from the dictionary
    % find the file name in the list
    idx = find(name{:,:}==filename);
    sigmab = sigma{idx,:};
    
    % read the image
    Img = imread(fullfile(folder_dist,filename));
    
    X_out = deblur_tv_fista_demo(Img, sigmab);

    % save the image
    imwrite(X_out,fullfile(folder_out,filename));
    disp("image processed done")
    end 
    
end
