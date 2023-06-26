function X_out = deblur_tv_fista_demo(Img, sigma)
    % deblur_tv_fista: FISTA algorithm for deblurring with TV regularization
    % take three channels of the image
    red = Img(:, :, 1);
    green = Img(:, :, 2);
    blue = Img(:, :, 3);
    red = double(red);
    green = double(green);
    blue = double(blue);

    w = 6*sigma+1;
    H = fspecial('gaussian', [w w], sigma);
    center = [ceil(abs(size(H, 1)) / 2) ceil(abs(size(H, 2)) / 2)];

%     %=============Blur the image=============
%     muR = 2.1534e-06;
%     muG = 1.7334e-06;
%     muB = 1.4575e-06;

%     muL = 0;
%     muU = 1e-04;
% 
%     mu = (muU+muL)/2;
% 
%     muR = (muU+muL)/2;
%     muG = (muU+muL)/2;
%     muB = (muU+muL)/2;
% % 
% %     lamR = 1 / muR;
% %     lamG = 1 / muG;
% %     lamB = 1 / muB;
% % 
    numinter = 1;
% 
%     sigman = 1e-04;
    X_outR = red;
    X_outG = green;
    X_outB = blue;

    for index = 1:numinter
        X_outR = deblur_tv_fista(X_outR, H, center,0.001, -Inf, Inf);
        X_outG = deblur_tv_fista(X_outG, H, center, 0.001, -Inf, Inf);
        X_outB = deblur_tv_fista(X_outB, H, center, 0.001, -Inf, Inf);
% 
%         normR = norm(X_outR, 'fro');
%         normG = norm(X_outG, 'fro');
%         normB = norm(X_outB, 'fro');
% 
%         normU = sqrt(normR ^ 2 + normG ^ 2 + normB ^ 2);
% 
%         if (normU^2 > w*w*sigman*sigman)
%             muU = mu;
%         else 
%             muL = mu;
%         end
%         mu = (muU+muL)/2;
% 
%         rR = normR / normU;
%         rG = normG / normU;
%         rB = normB / normU;
% 
%         muR = mu * rR;
%         muG = mu * rG;
%         muB = mu * rB;

%         lamR = 1 / muR;
%         lamG = 1 / muG;
%         lamB = 1 / muB;

%           X_out = cat(3, X_outR, X_outG, X_outB);
%           X_out = uint8(X_out);
    
%             figure(314)
%             imshow(X_out, [])

    end

    X_out = cat(3, X_outR, X_outG, X_outB);
    X_out = uint8(X_out);

    

end
