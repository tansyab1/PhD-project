clear;

clc;

% load all videos in ref_videos folder


ref_videos = dir('ref_videos/*.mp4');

% load all videos in dist_videos folder which has the structure as follows:
% dist_videos/kindofdist/levelofdist/vid_name.mp4

dist_videos = dir('dist_videos/*/*/*.mp4');

% load all levelofdist corresponding to each video in dist_videos folder without . and ..

levelofdist_noise = [5 10 15 30];
levelofdist_defocusblur = [1 2 3 5];
levelofdist_motionblur = [5 10 15 25];
levelofdist_ui = [100 150 200 250];
% load all kindofdist corresponding to each video in dist_videos folder

kindofdist =   dir('dist_videos/*');
levelofdist = levelofdist(3:end);

for i = 1:length(ref_videos)
    
    % read reference video
    
    ref_video = VideoReader(strcat('ref_videos/',ref_videos(i).name));

    psnr = zeros(length(kindofdist),length(levelofdist));
    ssim = zeros(length(kindofdist),length(levelofdist));
    vif = zeros(length(kindofdist),length(levelofdist));
    niqe = zeros(length(kindofdist),length(levelofdist));
    brisque = zeros(length(kindofdist),length(levelofdist));
    entropy = zeros(length(kindofdist),length(levelofdist));
    
    % read all distorted videos
    
    for j = 1:length(kindofdist)

        if strcmp(kindofdist(j).name,'Noise')
            levelofdist = levelofdist_noise;
        
        elseif strcmp(kindofdist(j).name,'Defocus Blur')
            levelofdist = levelofdist_defocusblur;

        elseif strcmp(kindofdist(j).name,'Motion Blur')
            levelofdist = levelofdist_motionblur;

        elseif strcmp(kindofdist(j).name,'Uneven Illumination')
            levelofdist = levelofdist_ui;
        end

        
        for k = 1:length(levelofdist)
            
            dist_video = VideoReader(strcat('dist_videos/',kindofdist(j).name,'/',levelofdist(k).name,'/',ref_videos(i).name));
            
            % calculate PSNR, SSIM, VIF, NIQE, BRISQUE, Decrete Entropy using calref and calnoref functions

            [psnr(j,k),ssim(j,k),vif(j,k)] = calref(ref_video,dist_video);

            [niqe(j,k),brisque(j,k),entropy(j,k)] = calnoref(dist_video);

            % open file csv and write filename, type of distortion, level of distortion, 
            % PSNR, SSIM, VIF, NIQE, BRISQUE, Decrete Entropy

            fid = fopen('result_cal.csv','a');
            with = [ref_videos(i).name,',',kindofdist(j).name,',',levelofdist(k).name,',',num2str(psnr(j,k)),',',num2str(ssim(j,k)),',',num2str(vif(j,k)),',',num2str(niqe(j,k)),',',num2str(brisque(j,k)),',',num2str(entropy(j,k)),'\n'];
            
            % save result to file csv

            fprintf(fid,with);
        end
        
        
        
    end
    
end