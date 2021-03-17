function add_smoke_to_video()



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Function to add smoke to a video using screen blending using a smoke-only video

%% Copyright (c) 2020, ZOHAIB AMJAD KHAN

%% All rights reserved.

%% Author: Zohaib Amjad Khan

%% Email: zohaibamjad.khan@univ-paris13.fr

%% Date: November 2019
%% Under the supervision of Prof. Azeddine Beghdadi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



inputFolder = 'D:\Zohaib\ref\';

outputFolder = 'D:\Zohaib\smoke\'

allfilepath = strcat(inputFolder ,'*.avi');



videofiles = dir(char(allfilepath));

smoke_obj = VideoReader('D:\Zohaib\MitchMartinez 512x288-2.avi');

smoke_vid = read(smoke_obj);



level = 3;

alpha = 0.85;

for k = 1 : length(videofiles)

    obj = VideoReader(char(strcat(inputFolder,videofiles(k).name)));

    str = regexp(videofiles(k).name,['\d*'],'match');

    writer = VideoWriter(char(strcat(outputFolder ,'video',str(1),'_',num2str(level),'.avi')),'Uncompressed AVI');

    ref_vid = read(obj);

    writer.FrameRate = obj.FrameRate;

    open(writer);

    vid = ref_vid;

    vid_out = vid;

    %% Apply distortion to complete video frame by frame

    nFrames = size(vid,4)



    for i = 1 : nFrames    

        vf = 1 - (1 - im2double(ref_vid(:,:,:,i))).*(1 - alpha*im2double(smoke_vid(:,:,:,i)));

        vid_out(:,:,:,i)= im2uint8(vf);

    end

    writeVideo(writer,vid_out);

    

    close(writer);

    clear obj 

end