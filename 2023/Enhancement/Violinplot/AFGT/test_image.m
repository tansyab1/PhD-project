 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %%adaptive fraction gamma transformation
 %%author£º Mingzhu Long
 %%date£º2018/04/03
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    
    imageName=strcat('dist.jpg');   
    in=imread(imageName);
    out=AFGT(in);%%%without color restoration
    imshow(out)
%     [out]=AFGT_CR(in);%%%with color restoration
%     imshow(out)