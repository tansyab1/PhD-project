function add_dist_vid()

% Video distortion generation 
% code written by Zohaib Amjad Khan
% Under the supervision of Prof. Azeddine Beghdadi
% 2019-2020



inputFolder = 'D:\Zohaib\Database\ref\';

outputFolder = 'D:\Zohaib\Database\uneven_illum\';



%% Store path of all bmp files inside input folder %%

allfilepath = strcat(inputFolder ,'*.avi');





%% Store all mp4 files in an array %%

videofiles = dir(char(allfilepath));



%% Uncomment Parameters for awgn

% mean = 0;

% variance = 0.002;



%% Uncomment Parameters for motion blur %%

%  len = 14;

%  angle = 45;

%  blur_motion_filter = fspecial('motion',len,angle);



%% Uncomment Parameters for defocus blur %%

 %gaussian_size = [13,13];

 %gaussian_std = 2;

 %blur_defocus_filter = fspecial('gaussian', gaussian_size, gaussian_std);



%% Uncomment Parameters for uneven illum %%

 area_factor = 5;

 attenuation = 0.35;

 level =3;



%%% NON-UNIFROM ILLUMINATION %%%

%% Create image for use as mask for creating non-uniform illumination %%

for k = 1 : 1 : length(videofiles) 

    obj = VideoReader(char(strcat(inputFolder,videofiles(k).name)));

    str = regexp(videofiles(k).name,['\d*'],'match');

    

    writer = VideoWriter(char(strcat(outputFolder ,'video',str(1),'_',num2str(level),'.avi')),'Uncompressed AVI');

    vid = read(obj);

    

    writer.FrameRate = obj.FrameRate;

    open(writer);

    

    WIDTH = size(vid,1);

    HEIGHT = size(vid,2);

    p = zeros(WIDTH,HEIGHT,3);

    for i=1:WIDTH

        for j=1:HEIGHT

            %d = sqrt((i-WIDTH/2)^2+(j-HEIGHT/2)^2);  %% Distance from the center

            d = sqrt((i-350)^2+(j-350)^2); %%% for sideways illumination change center point as here

            %d = sqrt((i-550)^2+(j-275)^2); %%% for sideways illumination change center point as here

            %d = sqrt((i-550)^2+(j-625)^2); %%% for sideways illumination change center point as here

            if d < HEIGHT/area_factor

                p(i,j,:) = 1;

            elseif d > 2*HEIGHT/area_factor

                p(i,j,:) = attenuation;

            else

                p(i,j,:) = 1-( (1-attenuation)*(d-HEIGHT/area_factor )/(HEIGHT/area_factor )) ;

            end

        end

    end

    vid_out = zeros(size(vid));

%     imshow(double(p))

%     figure

    

    %% Apply distortion to complete video frame by frame

    nFrames = size(vid,4)

    for i = 1 : nFrames

        

       %% Uncomment for Noise Addition 

          %   vid(:,:,:,i)  = imnoise(vid(:,:,:,i),'gaussian',mean,variance);   % Add gaussian noise

       

        %% Uncomment for Motion blur Addition   

          %  vid(:,:,:,i)  = imfilter(vid(:,:,:,i),blur_motion_filter,'replicate');   % Add motion blur

          

        %% Uncomment for Defocus blur Addition    

         %    vid(:,:,:,i)  = imfilter(vid(:,:,:,i),blur_defocus_filter,'replicate');  % Add defocus blur



         imG = double(vid(:,:,:,i)).* double(p);

         vid_out(:,:,:,i) = (imG - min(imG(:))) ./ max(imG(:));

         



         

    end

     %% Uncomment for distortions other than uneven illumination

 %   writeVideo(writer,vid);

    writeVideo(writer,vid_out);

    

    close(writer);

    clear obj

end