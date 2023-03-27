%get the input link and calculate the niq, brisque and entropy 
%metrics for all the images in the folder

folder = "C:\Users\moham\Desktop\2023\Enhancement\Violinplot\images";
method = "maincal";
calculateniqe(folder, method);
calculatebrisque(folder, method);
calculateentropy(folder, method);
