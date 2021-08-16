import cv2


def detectBlur(img_path, threhold):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var = getVarianofLaplace(gray)
    if (var < threhold):
        return True
    else:
        return False


def getVarianofLaplace(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


if __name__ == '__main__':
    print(detectBlur("../../../../Downloads/blur.jpg", 0.01))
