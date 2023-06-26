function lom_value = LOM(im_orig,im_enhanced)
%%% Lightness Order Measure for Over-enhancement
%%% Bai, C. and Reibman, A.R., 2018, October. Controllable Image 
%%% Illumination Enhancement with an Over-Enhancement Measure. 
%%% In 2018 25th IEEE International Conference on Image Processing (ICIP)
%%% Implementation by: Zohaib Amjad Khan
%%% larger value indicates more unnaturalness and worse enhancement

im_orig = im2double(im_orig);
im_enhanced = im2double(im_enhanced);

[H, W] = size(im_orig);

%% Apply local 31x31 mean filter
kernel = ones(31)/(31*31);
im_orig_filt = conv2(im_orig, kernel, 'same');
im_enhanced_filt = conv2(im_enhanced, kernel, 'same');

%% Subtract unfiltered from filtered
size(im_orig_filt);
diff_orig = im_orig_filt - im_orig;
diff_enhanced = im_enhanced_filt - im_enhanced;

lom_value = (1/W*H) * sum(sum(abs((diff_enhanced - diff_orig).*((sign(diff_enhanced) - sign(diff_orig))/2))));

end