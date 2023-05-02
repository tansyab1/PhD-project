function X_out = deblur_tv_fista_demo(Img, sigma)
    % deblur_tv_fista: FISTA algorithm for deblurring with TV regularization
    % take three channels of the image
    red = Img(:, :, 1);
    green = Img(:, :, 2);
    blue = Img(:, :, 3);

    w,h = 6*sigma+1,6*sigma+1;
    H = fspecial('gaussian', [w h], sigma);
    center = [ceil(abs(size(H, 1)) / 2) ceil(abs(size(H, 2)) / 2)];

    %=============Blur the image=============
    muR = 2.1534e-06;
    muG = 1.7334e-06;
    muB = 1.4575e-06;

    lamR = 1 / muR;
    lamG = 1 / muG;
    lamB = 1 / muB;

    numinter = 100;

    for index = 1:numinter
        X_outR = deblur_tv_fista(red, H, center, lamR, -Inf, Inf);
        X_outG = deblur_tv_fista(green, H, center, lamG, -Inf, Inf);
        X_outB = deblur_tv_fista(blue, H, center, lamB, -Inf, Inf);

        normR = norm(X_outR, 'fro');
        normG = norm(X_outG, 'fro');
        normB = norm(X_outB, 'fro');

        normU = sqrt(normR ^ 2 + normG ^ 2 + normB ^ 2);

        rR = normR / normU;
        rG = normG / normU;
        rB = normB / normU;

        muR = muR * rR;
        muG = muG * rG;
        muB = muB * rB;

        lamR = 1 / muR;
        lamG = 1 / muG;
        lamB = 1 / muB;

    end

    X_out = cat(3, X_outR, X_outG, X_outB);

end
