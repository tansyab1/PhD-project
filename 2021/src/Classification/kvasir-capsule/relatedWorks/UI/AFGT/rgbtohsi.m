function [H, S, I] = rgbtohsi(rgb)
%Usage: [H, S, I] = rgbtohsi(rgb)
%Input: rgb -- 3d array of a color image
%Output: H, S, I
%Reference: Digital Image Processing using Matlab,pp 212-213
TwoPi = 2*pi;
rgb = double(rgb);
r = rgb(:,:,1);
g = rgb(:,:,2);
b = rgb(:,:,3);
theta = acos((0.5*((r-g)+(r-b)))./((sqrt((r-g).^2+(r-b).*(g-b)))+eps));
H = theta;
H(b > g) = TwoPi - H(b > g);
H = H/TwoPi;
I = (r + g + b);
S = 1 - 3.*(min(min(r,g),b))./(I + eps);
I = I/3;
