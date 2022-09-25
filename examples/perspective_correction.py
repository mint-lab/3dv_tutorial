import cv2
import numpy as np
import copy

points_src = []

def mouse_event_handler(event, x, y, flags, param):
    global points_src
    if event == cv2.EVENT_LBUTTONDOWN:
        points_src.append((x,y))
        # points_src = np.append(points_src, np.array((x,y)))

def main():
    global points_src
    input = "../bin/data/sunglok_desk.jpg"
    window_name = "3DV Tutorial: Perspective Correction"
    # card_size = np.array([450, 250])
    card_size = (450,250)

    # Prepare the rectified points
    points_dst = np.array([[0,0], [card_size[0],0], [0, card_size[1]], [card_size[0], card_size[1]]])

    # Load an image
    original = cv2.imread(input)    
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_event_handler)
    while len(points_src) < 4:
        display = copy.deepcopy(original)
        display = cv2.rectangle(display, (10, 10), (10 + card_size[0], 10 + card_size[1]), (0, 0, 255), 2)
        idx = min(len(points_src), len(points_dst))
        display = cv2.circle(display, tuple(points_dst[idx]+10), 5, (0, 255, 0), -1)
        cv2.imshow(window_name, display)
        if cv2.waitKey(1) == ord('q'): break
    
    points_src = np.array(points_src, dtype=np.float32)
    H, inliner_mask = cv2.findHomography(points_src, points_dst, cv2.RANSAC)
    rectify = cv2.warpPerspective(original, H, card_size)
    

    cv2.imshow(window_name, rectify)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
