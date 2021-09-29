
def read_endo(folder="/home/nguyentansy/PhD-work/PhD-project/2021/src/Pre-processing/Isotropic/data/labeled-images/"):
    images = []
    i = 0
    for filename in glob.glob("%s/*/pathological-findings/*/*" % folder):
        img = cv2.imread(filename)[:, :, ::-1]
        print(filename)
        if i > 5:
            break
        i += 1


if __name__ == "__main__":
    import glob
    import cv2
    print("hello")
    read_endo(
        "/home/nguyentansy/DATA/nguyentansy/PhD-work/Datasets/hyper-kvasir/labeled-images/")
