% function calculates the PSNR, SSIM, and VIF between two video sequences

% Input: ref - reference video sequence .mp4 file (example: 'ref.mp4')
%        dist - distorted video sequence .mp4 file (example: 'dist.mp4')

function [psnr, ssim, vif] = calref(ref, dist)
    
    % read the reference video sequence
    refVid = VideoReader(ref);
    % read the distorted video sequence
    distVid = VideoReader(dist);

    % get the number of frames in the video sequence
    numFrames = refVid.NumberOfFrames;

    % initialize the PSNR, SSIM, and VIF arrays
    psnr_list = zeros(1, numFrames);
    ssim_list = zeros(1, numFrames);
    vif_list = zeros(1, numFrames);

    % loop through each frame in the video sequence
    for i = 1:numFrames
        % read the reference frame
        refFrame = read(refVid, i);
        % read the distorted frame
        distFrame = read(distVid, i);

        % calculate the PSNR, SSIM, and VIF for the current frame
        
        % PSNR
        psnr_list(i) = psnr(refFrame, distFrame);

        % SSIM
        ssim_list(i) = ssim(refFrame, distFrame);

        % VIF

        % convert the reference frame to grayscale
        refFrameGray = rgb2gray(refFrame);
        % convert the distorted frame to grayscale
        distFrameGray = rgb2gray(distFrame);

        % calculate the VIF for the current frame
        vif_list(i) = vifvec(refFrameGray, distFrameGray);

    end

    % calculate the average PSNR, SSIM, and VIF for the video sequence
    psnr = mean(psnr_list);
    ssim = mean(ssim_list);
    vif = mean(vif_list);
    
end

