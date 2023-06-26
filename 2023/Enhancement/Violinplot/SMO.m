function smo_val = SMO(im_orig,im_enhanced)
%%% Implementation of Structure Measure Operator (SMO) to detect over-enhancement.
%%% Cheng, H. D., & Zhang, Y. (2012, October). Detecting of contrast 
%%% over-enhancement. In 2012 19th IEEE international conference on image 
%%% processing (pp. 961-964). IEEE.
%%% Value closer to 0 means less overenhancement
%%% Implemented by: Zohaib Amjad Khan

%% Read images
im_orig = double(im_orig);
im_enhanced = double(im_enhanced);

edge_filter_x = fspecial('sobel');
edge_filter_y = edge_filter_x';

%% Apply 3x3 Sobel edge filters to original 
edge_x_orig = imfilter(im_orig,edge_filter_x);
edge_y_orig = imfilter(im_orig,edge_filter_y);

%% Edge_value of original
edge_value_orig = sqrt(edge_x_orig.^2 + edge_y_orig.^2);

%% 3x3 windowed standard deviation of original
std_orig = stdfilt(im_orig);

%% 3x3 windowed entropy of original
entropy_orig = entropyfilt(im_orig,true(3));

%% Apply sobel filters to enhanced
edge_x_enhanced = imfilter(im_enhanced,edge_filter_x);
edge_y_enhanced = imfilter(im_enhanced,edge_filter_y);

%% edge_value of enhanced
edge_value_enhanced = sqrt(edge_x_enhanced.^2 + edge_y_enhanced.^2);

%% windowed standard deviation of enhanced
std_enhanced = stdfilt(im_enhanced);

%% windowed entropy of enhanced
entropy_enhanced = entropyfilt(im_enhanced,true(3));

non_homogeneity_orig = edge_value_orig .* std_orig .* entropy_orig;
non_homogeneity_enhanced = edge_value_enhanced .* std_enhanced .* entropy_enhanced;

size(non_homogeneity_orig);
size(non_homogeneity_enhanced);

structure_difference = abs(non_homogeneity_orig - non_homogeneity_enhanced);

[M, N] = size(im_orig);
div_by_orig = structure_difference ./ non_homogeneity_orig;

div_by_orig(isnan(div_by_orig)) = 0;
div_by_orig(isinf(div_by_orig)) = 0;

smo_val = (1 / (M*N)) * sum(sum(div_by_orig));

end