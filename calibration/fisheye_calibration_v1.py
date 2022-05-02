import cv2
import numpy as np
import pyrealsense2 as rs
from threading import Lock
import time

# This is cv2 and T265
frame_mutex = Lock()
frame_data = {"left"  : None,
              "right" : None,
              "timestamp_ms" : None
              }

def callback(frame):
    global frame_data
    if frame.is_frameset():
        frameset = frame.as_frameset()
        f1 = frameset.get_fisheye_frame(1).as_video_frame()
        f2 = frameset.get_fisheye_frame(2).as_video_frame()
        left_data = np.asanyarray(f1.get_data())
        right_data = np.asanyarray(f2.get_data())
        ts = frameset.get_timestamp()
        frame_mutex.acquire()
        frame_data["left"] = left_data
        frame_data["right"] = right_data
        frame_data["timestamp_ms"] = ts
        frame_mutex.release()

# How to see realsense with cv2
pipeline = rs.pipeline()
cfg = rs.config()

# Get Device Product
pipeline.start(cfg, callback)

try:
    WINDOW_TITLE = "Realsense"
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    profiles = pipeline.get_active_profile()
    streams = {"left": profiles.get_stream(rs.stream.fisheye,1).as_video_stream_profile(),
                "right": profiles.get_stream(rs.stream.fisheye,2).as_video_stream_profile()}
    t = 0
    while True:
        t = time.time()
        frame_mutex.acquire()
        valid = frame_data["timestamp_ms"] is not None
        frame_mutex.release()

        if valid:
            frame_mutex.acquire()
            frame_copy = {"left": frame_data["left"].copy(),
                          "right": frame_data["right"].copy()}
            frame_shape = frame_copy["left"].shape
            frame_mutex.release()
            try:
                t = time.time() - t
                fps = 1/t

                cv2.putText(frame_copy["left"], f"fps is {fps:.2}", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(frame_copy["right"], f"shape is {frame_shape[1]} x {frame_shape[0]}", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.imshow(WINDOW_TITLE, np.hstack((frame_copy["left"], frame_copy["right"])))
            except Exception as e:
                print(e)

        key = cv2.waitKey(1)
        if key == ord('q'): break

except Exception as e:
    print(e)

finally:
    pipeline.stop()