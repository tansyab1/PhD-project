function rgb = hsitorgb(H,S,I)
%Usage: rgb = hsitorgb(H,S,I)
%Input: H, S, I
%Output: rgb
%Reference: Digital Image Processing Using Matlab, pp213-214
%see also [H, S, I] = rgbtohsi(rgb)

H = H * 2 * pi;
R = zeros(size(H));
G = R;
B = R;

%RG Sector
id = find((0<=H)& (H<2*pi/3));
B(id) = I(id).*(1-S(id));
R(id) = I(id).*(1+S(id).*cos(H(id))./cos(pi/3-H(id)));
G(id) = 3*I(id)-(R(id)+B(id));

%BG Sector
id = find((2*pi/3 <= H) & (H < 4*pi/3));
R(id) = I(id).*(1-S(id));
G(id) = I(id).*(1 + S(id).*cos(H(id)-2*pi/3)./cos(pi-H(id)));
B(id) = 3*I(id)-(R(id)+G(id));

%BR Sector
id = find((4*pi/3 <= H)& (H <= 2*pi));
G(id) = I(id).*(1 - S(id));
B(id) = I(id).*(1 + S(id).*cos(H(id)-4*pi/3)./cos(5*pi/3-H(id)));
R(id) = 3*I(id)-(G(id)+B(id));
rgb =cat(3,R,G,B);
