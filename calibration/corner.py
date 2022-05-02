import cv2
import numpy as np

def main():
    img_name = "board.jpg"
    img = cv2.imread(img_name)
    img = cv2.resize(img, (640, 640))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # img[dst>0.01*dst.max()]=[0,0,255]
    x = dst[dst>0.01*dst.max()]
    print(x.shape)
    # for corner in img[dst>0.01*dst.max()]:
    #     print(corner.shape)
        # img = cv2.circle(img, tuple(corner), 5, (0, 0, 255), 1)

    # print(dst.shape)
    # print(test.shape)
    cv2.imshow("img", img)
    cv2.waitKey(0)



if __name__ == "__main__":
    main()