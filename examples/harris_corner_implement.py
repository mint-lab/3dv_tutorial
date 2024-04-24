import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def cornerHarris(img, ksize=3, k=0.04):
    # Compute gradients and M matrix
    Ix = cv.Sobel(img, cv.CV_32F, 1, 0)
    Iy = cv.Sobel(img, cv.CV_32F, 0, 1)
    M11 = cv.GaussianBlur(Ix*Ix, (ksize, ksize), 0)
    M22 = cv.GaussianBlur(Iy*Iy, (ksize, ksize), 0)
    M12 = cv.GaussianBlur(Ix*Iy, (ksize, ksize), 0)

    # Compute Harris cornerness
    detM = M11 * M22 - M12 * M12
    traceM = M11 + M22
    cornerness = detM - k * traceM**2
    return cornerness

if __name__ == '__main__':
    video = cv.VideoCapture('../data/chessboard.avi')
    assert video.isOpened()

    # Run and show video stabilization
    while True:
        # Read an image from `video`
        valid, img = video.read()
        if not valid:
            break
        if img.ndim >= 3:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Extract Harris corners
        harris = cornerHarris(gray)
        corners = harris > 5e8

        # Show the corners on the image
        heatmap = np.dstack((np.zeros_like(corners), np.zeros_like(corners), corners*255))
        heatmap = (0.3 * img + 0.7 * heatmap).astype(np.uint8)
        cv.imshow('Harris Cornerness', heatmap)
        key = cv.waitKey(1)
        if key == ord(' '):
            key = cv.waitKey()
        if key == 27: # ESC
            break