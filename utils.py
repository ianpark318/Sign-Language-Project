import time
import cv2
from mediapipe.framework.formats import landmark_pb2


def landmarkxy2list(
    landmarks_list: landmark_pb2.NormalizedLandmarkList,
):
    result_list = []
    for hand_num, landmark in enumerate(landmarks_list.landmark):
        landmark_list = [landmark.x, landmark.y]
        result_list.append(landmark_list)
    return result_list


def display_fps(pre_time, image):
    curr_time = time.time()
    t = curr_time - pre_time
    fps = 1 / t

    cv2.putText(
        image, f"fps: {fps}", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0)
    )
    return curr_time


def display_label(label, image):
    cv2.putText(
        image, f"{label}", (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3
    )
